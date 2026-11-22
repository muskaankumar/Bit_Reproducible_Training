#!/usr/bin/env python3
"""Bit-reproducible GPT-2 training script.

Usage:
  python train.py --mode repro --device cuda --dataset wikitext103 --epochs 1
  python train.py --mode stock  --device cuda --dataset wikitext103 --epochs 1

Modes:
  repro: our reproducible kernels — should give bit-identical results across GPUs
  stock: standard PyTorch — for accuracy/speed comparison (results will differ across GPUs)

Outputs:
  - Weight hash after each epoch (SHA-256 of all parameters)
  - Training/validation loss and perplexity
  - Timing per step
  - Everything saved to a JSON file for comparison across GPU types

Implementation:
  Uses repro_torch (the MNIST package) as the reproducible backend for:
    - Ozaki GEMM (Linear projections)
    - Superaccumulator cross-entropy
    - Adam with AdamClock
    - FiniteGuard (NaN/Inf scanning)
    - SplitMix64 (deterministic data shuffling)
    - reproducible_init (portable weight initialization)
  Extends with gpt2_repro layers for:
    - LayerNorm, GELU, Dropout, Embedding, Attention
"""

import argparse
import json
import math
import os
import platform
import statistics
import struct
import subprocess
import sys
import time

import numpy as np
import torch

# Ensure reproducible environment
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

# Import the shared reproducible toolkit (same package the MNIST path uses)
import repro_torch as rt
from gpt2_repro.model import GPT2, GPT2Config


def get_gpu_info():
    """Get GPU name and driver version."""
    if not torch.cuda.is_available():
        return platform.processor() or platform.machine(), "none"
    name = torch.cuda.get_device_name(0)
    try:
        driver = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=driver_version",
             "--format=csv,noheader", "-i", "0"],
            text=True).strip()
    except (OSError, subprocess.CalledProcessError):
        driver = "none"
    return name, driver


def load_wikitext103(data_dir, block_size=1024):
    """Load WikiText-103 as token IDs (GPT-2 BPE tokenizer).

    WikiText-103 is small enough to load entirely into memory and is a
    standard benchmark. For larger experiments, switch to OpenWebText.
    """
    try:
        import tiktoken
        enc = tiktoken.get_encoding("gpt2")
    except ImportError:
        raise ImportError("pip install tiktoken  (for GPT-2 BPE tokenizer)")

    cache_path = os.path.join(data_dir, "wikitext103_tokens.bin")
    if os.path.exists(cache_path):
        data = np.fromfile(cache_path, dtype=np.uint16)
        print(f"Loaded cached tokens: {len(data):,} tokens from {cache_path}")
    else:
        try:
            from datasets import load_dataset
        except ImportError:
            raise ImportError("pip install datasets  (for WikiText-103)")

        print("Downloading and tokenizing WikiText-103...")
        ds = load_dataset("Salesforce/wikitext", "wikitext-103-raw-v1",
                          split="train")
        text = "\n\n".join(ds["text"])
        tokens = enc.encode(text, allowed_special={"<|endoftext|>"})
        data = np.array(tokens, dtype=np.uint16)
        os.makedirs(data_dir, exist_ok=True)
        data.tofile(cache_path)
        print(f"Tokenized and cached: {len(data):,} tokens -> {cache_path}")

    # Split into train (98%) and val (2%)
    n = len(data)
    split = int(n * 0.98)
    train_data = data[:split]
    val_data = data[split:]
    print(f"Train: {len(train_data):,} tokens, Val: {len(val_data):,} tokens")
    return train_data, val_data


def get_batch(data, batch_size, block_size, step, rng):
    """Get a deterministic batch using SplitMix64 (from repro_torch).

    Same RNG as the MNIST path — identical sequence on every machine.
    """
    n = len(data) - block_size
    ix = []
    for _ in range(batch_size):
        ix.append(rng.next() % n)
    ix = np.array(ix, dtype=np.int64)

    x = np.stack([data[i:i + block_size].astype(np.int64) for i in ix])
    y = np.stack([data[i + 1:i + 1 + block_size].astype(np.int64) for i in ix])
    return torch.from_numpy(x), torch.from_numpy(y)


def sync(device):
    if device.startswith("cuda"):
        torch.cuda.synchronize()


def main():
    ap = argparse.ArgumentParser(description="Bit-reproducible GPT-2 training")
    ap.add_argument("--mode", default="repro", choices=["repro", "stock"])
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--slices", type=int, default=3,
                    help="GEMM slices (default 3 = 21-bit)")
    ap.add_argument("--batch-size", type=int, default=4)
    ap.add_argument("--block-size", type=int, default=256,
                    help="sequence length (default 256 for speed)")
    ap.add_argument("--max-steps", type=int, default=1000,
                    help="training steps (0 = full epoch)")
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--weight-decay", type=float, default=0.1)
    ap.add_argument("--grad-clip", type=float, default=1.0)
    ap.add_argument("--dropout", type=float, default=0.1)
    ap.add_argument("--eval-interval", type=int, default=100)
    ap.add_argument("--eval-steps", type=int, default=20)
    ap.add_argument("--data-dir",
                    default=os.path.join(os.path.dirname(__file__), "data"))
    ap.add_argument("--tag", default=os.environ.get("RESULT_TAG", ""))
    ap.add_argument("--out", default="")
    ap.add_argument("--profile", action="store_true")
    # Model size options (default = GPT-2 small)
    ap.add_argument("--n-layer", type=int, default=12)
    ap.add_argument("--n-head", type=int, default=12)
    ap.add_argument("--n-embd", type=int, default=768)
    args = ap.parse_args()

    lines = []

    def say(*a):
        text = " ".join(str(x) for x in a)
        print(text, flush=True)
        lines.append(text)

    gpu_name, driver = get_gpu_info()
    say(f"mode={args.mode} device={args.device} gpu={gpu_name} "
        f"torch={torch.__version__} cuda={torch.version.cuda} driver={driver}")
    say(f"seed={args.seed} slices={args.slices} batch={args.batch_size} "
        f"block={args.block_size} lr={args.lr} steps={args.max_steps}")

    if args.device.startswith("cuda"):
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False

    # Load data
    train_data, val_data = load_wikitext103(args.data_dir, args.block_size)

    # Build model
    config = GPT2Config(
        n_layer=args.n_layer, n_head=args.n_head, n_embd=args.n_embd,
        block_size=args.block_size, dropout=args.dropout,
    )

    if args.mode == "repro":
        # Use repro_torch's set_slices — same global as MNIST
        rt.set_slices(args.slices)
        model = GPT2(config)
        # Use repro_torch's reproducible_init — SplitMix64 + _uniform,
        # identical to the MNIST path (NOT torch.manual_seed + xavier)
        rt.reproducible_init(model, args.seed)
        rt.name_layers(model)
        # Use repro_torch's Adam — AdamClock + B.adam,
        # matching the MNIST optimizer exactly
        optimizer = rt.Adam(model.parameters(), lr=args.lr,
                            weight_decay=args.weight_decay, adamw=True)
    else:
        model = GPT2(config)
        # Same starting weights for fair comparison
        rt.reproducible_init(model, args.seed)
        optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr,
                                      weight_decay=args.weight_decay)

    model.to(args.device)
    # Use repro_torch's model_hash (same as MNIST)
    init_hash = rt.model_hash(model)
    say(f"hash init {init_hash}")
    say(f"parameters {sum(p.numel() for p in model.parameters()):,}")

    # Training loop
    # Use repro_torch's SplitMix64 for deterministic batching (same as MNIST)
    rng = rt.SplitMix64(args.seed + 1)
    # Use repro_torch's FiniteGuard (same as MNIST — scans for NaN/Inf
    # before every GEMM, check() reads flag once per step)
    guard = rt.guard()
    times = []
    step = 0
    best_val_loss = float("inf")
    hashes = {"init": init_hash}

    model.train()
    for step in range(1, args.max_steps + 1):
        x, y = get_batch(train_data, args.batch_size, args.block_size,
                         step, rng)
        x, y = x.to(args.device), y.to(args.device)

        sync(args.device)
        t0 = time.perf_counter()

        logits, loss = model(x, y)

        optimizer.zero_grad()
        loss.backward()

        # Gradient clipping
        if args.grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)

        optimizer.step()

        if args.mode == "repro":
            # FiniteGuard check — same pattern as MNIST's guard.check(step)
            guard.check(step)

        sync(args.device)
        t1 = time.perf_counter()
        times.append(t1 - t0)

        if step % 10 == 0:
            loss_val = loss.item()
            bits = struct.unpack("<I", struct.pack("<f", loss_val))[0]
            say(f"  step {step:5d} loss {loss_val:.4f} (bits {bits:08x}) "
                f"{1e3 * times[-1]:.1f} ms")

        # Evaluation
        if step % args.eval_interval == 0:
            model.eval()
            val_losses = []
            val_rng = rt.SplitMix64(0)  # fixed seed for val
            with torch.no_grad():
                for _ in range(args.eval_steps):
                    xv, yv = get_batch(val_data, args.batch_size,
                                       args.block_size, 0, val_rng)
                    xv, yv = xv.to(args.device), yv.to(args.device)
                    _, vloss = model(xv, yv)
                    val_losses.append(vloss.item())
            val_loss = sum(val_losses) / len(val_losses)
            ppl = math.exp(val_loss) if val_loss < 20 else float("inf")
            # Use repro_torch's model_hash (same as MNIST)
            h = rt.model_hash(model)
            hashes[f"step_{step}"] = h
            say(f"  eval step {step}: val_loss={val_loss:.4f} "
                f"ppl={ppl:.1f} hash={h}")
            model.train()

    # Final hash using repro_torch's model_hash
    final_hash = rt.model_hash(model)
    hashes["final"] = final_hash
    say(f"hash final {final_hash}")

    t = sorted(times[5:]) if len(times) > 10 else sorted(times)
    say(f"time total {sum(times):.1f}s; per step median "
        f"{1e3 * statistics.median(t):.1f}ms "
        f"mean {1e3 * statistics.mean(t):.1f}ms ({len(times)} steps)")

    peak = 0.0
    if args.device.startswith("cuda"):
        peak = torch.cuda.max_memory_allocated() / 2 ** 20
        say(f"peak GPU memory {peak:.0f} MiB")

        if args.profile:
            say("")
            say("=== MEMORY PROFILE ===")
            say(torch.cuda.memory_summary(abbreviated=True))

            # Detailed breakdown
            reserved = torch.cuda.memory_reserved() / 2 ** 20
            allocated = torch.cuda.max_memory_allocated() / 2 ** 20
            say(f"  Peak allocated:  {allocated:.0f} MiB")
            say(f"  Reserved (pool): {reserved:.0f} MiB")
            param_mib = sum(p.numel() * p.element_size()
                            for p in model.parameters()) / 2 ** 20
            grad_mib = sum(p.grad.numel() * p.grad.element_size()
                           for p in model.parameters()
                           if p.grad is not None) / 2 ** 20
            say(f"  Parameters:      {param_mib:.1f} MiB")
            say(f"  Gradients:       {grad_mib:.1f} MiB")
            # Optimizer states (each param has 2 fp32 buffers: m and v)
            optim_mib = 2 * param_mib  # rough estimate
            say(f"  Optimizer (est): {optim_mib:.1f} MiB")
            act_mib = allocated - param_mib - grad_mib - optim_mib
            say(f"  Activations+tmp: {max(0, act_mib):.1f} MiB (est)")

    # Save results
    if args.out:
        os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
        with open(args.out, "w") as f:
            f.write("\n".join(lines) + "\n")
        record = {
            "mode": args.mode, "tag": args.tag,
            "host": platform.node(), "device": gpu_name, "driver": driver,
            "torch": torch.__version__, "cuda": torch.version.cuda,
            "seed": args.seed, "slices": args.slices,
            "batch_size": args.batch_size, "block_size": args.block_size,
            "n_layer": args.n_layer, "n_head": args.n_head,
            "n_embd": args.n_embd,
            "lr": args.lr, "weight_decay": args.weight_decay,
            "max_steps": args.max_steps, "steps_completed": step,
            "hashes": hashes,
            "step_ms_median": 1e3 * statistics.median(t),
            "step_ms_mean": 1e3 * statistics.mean(t),
            "peak_memory_mib": peak,
            "total_time_s": sum(times),
        }
        with open(os.path.splitext(args.out)[0] + ".json", "w") as f:
            json.dump(record, f, indent=1)
        say(f"Results saved to {args.out}")


if __name__ == "__main__":
    main()