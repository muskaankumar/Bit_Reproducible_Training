#!/usr/bin/env python3
"""
Compare GPT-2 training results across GPUs, seeds, and modes.

Usage:
    python experiments/compare_gpt2_results.py results/gpt2/*.json
    python experiments/compare_gpt2_results.py results/gpt2/*.json --verbose
"""

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path


def load_results(paths):
    results = []
    for p in paths:
        with open(p) as f:
            r = json.load(f)
            r["_file"] = str(p)
            results.append(r)
    return results


def print_section(title):
    print(f"\n{'='*70}")
    print(f"  {title}")
    print(f"{'='*70}")


def compare_hashes(results, verbose=False):
    """Check if all repro runs with same seed produce identical hashes."""
    print_section("CROSS-GPU BIT-IDENTITY CHECK")

    # Group by (mode, seed)
    groups = defaultdict(list)
    for r in results:
        key = (r.get("mode", "unknown"), r.get("seed", "?"))
        groups[key].append(r)

    any_mismatch = False

    for (mode, seed), runs in sorted(groups.items()):
        if mode != "repro":
            continue
        if len(runs) < 2:
            print(f"\n  [{mode}] seed={seed}: only 1 run, need ≥2 for comparison")
            continue

        print(f"\n  [{mode}] seed={seed}: {len(runs)} runs")

        # Compare final weight hashes
        hashes = {}
        for r in runs:
            gpu = r.get("gpu", "unknown")
            cuda = r.get("cuda_version", "?")
            label = f"{gpu} (CUDA {cuda})"
            h = r.get("final_weight_hash", "N/A")
            hashes[label] = h
            if verbose:
                print(f"    {label}: {h}")

        unique_hashes = set(hashes.values())
        if len(unique_hashes) == 1:
            h = list(unique_hashes)[0]
            print(f"  ✓ BIT-IDENTICAL across all {len(runs)} GPUs")
            print(f"    Hash: {h[:16]}...")
        else:
            any_mismatch = True
            print(f"  ✗ MISMATCH! {len(unique_hashes)} distinct hashes:")
            for label, h in hashes.items():
                print(f"    {label}: {h}")

        # Compare per-step hashes if available
        if verbose and all("step_hashes" in r for r in runs):
            print(f"\n    Per-step hash comparison:")
            ref_run = runs[0]
            ref_label = f"{ref_run.get('gpu', '?')} (CUDA {ref_run.get('cuda_version', '?')})"
            for other in runs[1:]:
                other_label = f"{other.get('gpu', '?')} (CUDA {other.get('cuda_version', '?')})"
                ref_steps = ref_run["step_hashes"]
                other_steps = other["step_hashes"]
                first_diff = None
                for step_key in sorted(ref_steps.keys(), key=lambda x: int(x)):
                    if ref_steps.get(step_key) != other_steps.get(step_key):
                        first_diff = step_key
                        break
                if first_diff is None:
                    print(f"      {ref_label} vs {other_label}: identical at all steps")
                else:
                    print(f"      {ref_label} vs {other_label}: first divergence at step {first_diff}")

    if not any_mismatch:
        print(f"\n  ★ ALL REPRO RUNS ARE BIT-IDENTICAL ★")
    return not any_mismatch


def compare_stock(results, verbose=False):
    """Show that stock PyTorch does NOT produce identical hashes."""
    print_section("STOCK PyTorch NON-REPRODUCIBILITY")

    stock_runs = [r for r in results if r.get("mode") == "stock"]
    if len(stock_runs) < 2:
        print("\n  Need ≥2 stock runs for comparison (try different GPUs or re-runs)")
        return

    groups = defaultdict(list)
    for r in stock_runs:
        groups[r.get("seed", "?")].append(r)

    for seed, runs in sorted(groups.items()):
        print(f"\n  [stock] seed={seed}: {len(runs)} runs")
        hashes = {}
        for r in runs:
            gpu = r.get("gpu", "unknown")
            label = f"{gpu}"
            h = r.get("final_weight_hash", "N/A")
            hashes[label] = h

        unique = set(hashes.values())
        if len(unique) > 1:
            print(f"  ✓ EXPECTED: {len(unique)} distinct hashes (non-deterministic)")
        else:
            print(f"  ⚠ Unexpectedly identical (same GPU? single run?)")

        if verbose:
            for label, h in hashes.items():
                print(f"    {label}: {h}")


def timing_table(results, verbose=False):
    """Print timing comparison table."""
    print_section("PERFORMANCE COMPARISON")

    print(f"\n  {'GPU':<16} {'Mode':<8} {'Steps':<7} {'Total(s)':<10} {'Step(ms)':<10} {'Loss':<10}")
    print(f"  {'-'*16} {'-'*8} {'-'*7} {'-'*10} {'-'*10} {'-'*10}")

    for r in sorted(results, key=lambda x: (x.get("gpu", ""), x.get("mode", ""))):
        gpu = r.get("gpu", "?")[:15]
        mode = r.get("mode", "?")
        steps = r.get("max_steps", "?")
        total = r.get("total_time_s", 0)
        step_ms = r.get("avg_step_ms", 0)
        loss = r.get("final_loss", 0)

        print(f"  {gpu:<16} {mode:<8} {steps:<7} {total:<10.1f} {step_ms:<10.1f} {loss:<10.4f}")

    # Compute overhead
    repro_by_gpu = {}
    stock_by_gpu = {}
    for r in results:
        gpu = r.get("gpu", "?")
        mode = r.get("mode", "?")
        step_ms = r.get("avg_step_ms", 0)
        if mode == "repro":
            repro_by_gpu[gpu] = step_ms
        elif mode == "stock":
            stock_by_gpu[gpu] = step_ms

    if repro_by_gpu and stock_by_gpu:
        print(f"\n  Overhead (repro vs stock PyTorch):")
        for gpu in sorted(set(repro_by_gpu) & set(stock_by_gpu)):
            ratio = repro_by_gpu[gpu] / stock_by_gpu[gpu] if stock_by_gpu[gpu] > 0 else float('inf')
            print(f"    {gpu}: {ratio:.2f}x")


def main():
    parser = argparse.ArgumentParser(description="Compare GPT-2 training results")
    parser.add_argument("files", nargs="+", help="JSON result files")
    parser.add_argument("--verbose", "-v", action="store_true", help="Show per-step details")
    args = parser.parse_args()

    results = load_results(args.files)
    print(f"Loaded {len(results)} result files")

    all_identical = compare_hashes(results, args.verbose)
    compare_stock(results, args.verbose)
    timing_table(results, args.verbose)

    print_section("SUMMARY")
    gpus = sorted(set(r.get("gpu", "?") for r in results))
    modes = sorted(set(r.get("mode", "?") for r in results))
    print(f"  GPUs tested: {', '.join(gpus)}")
    print(f"  Modes: {', '.join(modes)}")
    print(f"  Cross-GPU bit-identity: {'YES ★' if all_identical else 'NO ✗'}")
    print()

    return 0 if all_identical else 1


if __name__ == "__main__":
    sys.exit(main())
