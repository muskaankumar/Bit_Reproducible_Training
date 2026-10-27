#!/usr/bin/env python3
"""
Bit-reproducible MNIST LeNet-5 training.

Every floating-point operation that involves a summation (GEMM, conv, loss,
optimizer) goes through either:
  1. repro_gemm: integer-exact Ozaki scheme, rounded once to FP32
  2. double-precision intermediates (element-wise ops, bias gradients, Adam)

Non-rounding ops (ReLU, MaxPool, Flatten, indexing) use plain PyTorch.

Reproducibility contract:
  - Same CUDA version + same code => bit-identical weights after every epoch,
    regardless of GPU model, number of runs, or split-K.
  - The training script prints an FNV-1a hash of all parameters after each epoch.
    If two runs on different GPUs produce the same hash, training is bit-reproducible.

Usage:
    cd mnist_pipeline
    pip install -e .
    python train_mnist.py [--epochs 10] [--batch-size 100] [--lr 0.001] [--slices 3] [--seed 42]

Verification:
    Run twice (possibly on different GPUs with the same CUDA version) and compare
    the per-epoch hashes. They must be identical.
"""

import argparse
import hashlib
import os
import struct
import sys
import time

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets, transforms

import repro_gemm_ext as _ext
from repro_nn import (
    ReproConv2d, ReproLinear, ReproCrossEntropyLoss, ReproAdam, SLICES
)
import repro_nn


# ============================================================================
# Deterministic seeding
# ============================================================================

def seed_everything(seed):
    """Make PyTorch, CUDA, and Python deterministic."""
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    # Disable cuDNN non-determinism (we don't use cuDNN for conv, but just in case)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    # Python hash seed
    os.environ['PYTHONHASHSEED'] = str(seed)


# ============================================================================
# Reproducible LeNet-5
# ============================================================================

class ReproLeNet(nn.Module):
    """
    LeNet-5 architecture matching the RepDL paper:
      conv1: 1 -> 6, kernel 5x5
      relu + maxpool 2x2
      conv2: 6 -> 16, kernel 5x5
      relu + maxpool 2x2
      flatten
      fc1: 400 -> 120
      relu
      fc2: 120 -> 84
      relu
      fc3: 84 -> 10

    All Linear and Conv2d layers use repro_gemm for bit-reproducibility.
    ReLU and MaxPool are plain PyTorch (exact: comparisons only).
    """
    def __init__(self):
        super().__init__()
        self.conv1 = ReproConv2d(1, 6, 5, padding=2)      # 28x28 -> 28x28
        self.conv2 = ReproConv2d(6, 16, 5)                 # 14x14 -> 10x10
        self.fc1 = ReproLinear(16 * 5 * 5, 120)
        self.fc2 = ReproLinear(120, 84)
        self.fc3 = ReproLinear(84, 10)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.max_pool2d(x, 2)
        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x, 2)
        x = x.reshape(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


# ============================================================================
# Hashing
# ============================================================================

def hash_parameters(model):
    """FNV-1a hash of all model parameters (weights and biases), deterministic."""
    h = hashlib.sha256()
    for name, param in sorted(model.named_parameters()):
        # Transfer to CPU and get raw bytes
        data = param.data.detach().cpu().numpy().tobytes()
        h.update(name.encode('utf-8'))
        h.update(data)
    return h.hexdigest()


def hash_optimizer_state(optimizer):
    """Hash optimizer state (m and v buffers) for full reproducibility check."""
    h = hashlib.sha256()
    for group in optimizer.param_groups:
        for p in group['params']:
            if p in optimizer.state:
                s = optimizer.state[p]
                for key in sorted(s.keys()):
                    data = s[key].detach().cpu().numpy().tobytes()
                    h.update(key.encode('utf-8'))
                    h.update(data)
    return h.hexdigest()


# ============================================================================
# Data loading (deterministic)
# ============================================================================

def get_data_loaders(batch_size, data_dir='./data'):
    transform = transforms.Compose([
        transforms.ToTensor(),
        # Normalize: (x - 0.1307) / 0.3081
        # Note: PyTorch's Normalize does x = (x - mean) / std element-wise.
        # This involves division by a constant and subtraction, both of which
        # are single FP32 operations with fixed order => reproducible.
        transforms.Normalize((0.1307,), (0.3081,)),
    ])

    train_dataset = datasets.MNIST(data_dir, train=True, download=True, transform=transform)
    test_dataset = datasets.MNIST(data_dir, train=False, transform=transform)

    # Use a seeded generator for the data loader's shuffling
    g = torch.Generator()
    g.manual_seed(0)

    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True,
        num_workers=0,  # deterministic: no multiprocess data loading
        generator=g,
        drop_last=False,
    )
    test_loader = torch.utils.data.DataLoader(
        test_dataset, batch_size=1000, shuffle=False,
        num_workers=0,
    )
    return train_loader, test_loader


# ============================================================================
# Training and evaluation
# ============================================================================

def train_one_epoch(model, train_loader, criterion, optimizer, device, epoch):
    model.train()
    total_loss = 0.0
    correct = 0
    total = 0

    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)

        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        pred = output.argmax(dim=1)
        correct += pred.eq(target).sum().item()
        total += target.size(0)

        if (batch_idx + 1) % 100 == 0:
            print(f'  Epoch {epoch} [{batch_idx+1}/{len(train_loader)}] '
                  f'Loss: {total_loss/(batch_idx+1):.4f} Acc: {100.*correct/total:.1f}%')

    return total_loss / len(train_loader), 100. * correct / total


@torch.no_grad()
def evaluate(model, test_loader, criterion, device):
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0

    for data, target in test_loader:
        data, target = data.to(device), target.to(device)
        output = model(data)
        loss = criterion(output, target)
        total_loss += loss.item()
        pred = output.argmax(dim=1)
        correct += pred.eq(target).sum().item()
        total += target.size(0)

    return total_loss / len(test_loader), 100. * correct / total


# ============================================================================
# Main
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description='Bit-reproducible MNIST LeNet-5')
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--batch-size', type=int, default=100)
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--slices', type=int, default=3, choices=[1, 2, 3],
                        help='repro_gemm slices (1=7bit, 2=14bit, 3=21bit/FP32)')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--data-dir', type=str, default='./data')
    parser.add_argument('--save-hashes', type=str, default='hashes.txt',
                        help='File to write per-epoch hashes')
    parser.add_argument('--device', type=str, default='cuda')
    args = parser.parse_args()

    # Set global slices
    repro_nn.SLICES = args.slices

    print(f'=== Bit-Reproducible MNIST Training ===')
    print(f'Seed: {args.seed}, Slices: {args.slices}, LR: {args.lr}, '
          f'Batch: {args.batch_size}, Epochs: {args.epochs}')

    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    if device.type == 'cuda':
        props = torch.cuda.get_device_properties(0)
        print(f'GPU: {props.name}, SM {props.major}.{props.minor}, '
              f'CUDA {torch.version.cuda}')
    else:
        print('WARNING: Running on CPU. GEMM will be slow but still reproducible.')

    # Seed everything
    seed_everything(args.seed)

    # Data
    train_loader, test_loader = get_data_loaders(args.batch_size, args.data_dir)

    # Model
    model = ReproLeNet().to(device)
    print(f'Parameters: {sum(p.numel() for p in model.parameters()):,}')

    # Print initial hash
    init_hash = hash_parameters(model)
    print(f'Initial param hash: {init_hash[:16]}...')

    # Optimizer and loss
    criterion = ReproCrossEntropyLoss()
    optimizer = ReproAdam(model.parameters(), lr=args.lr)

    # Training loop
    hash_log = []
    hash_log.append(f'epoch=init params={init_hash}')

    for epoch in range(1, args.epochs + 1):
        t0 = time.time()
        train_loss, train_acc = train_one_epoch(
            model, train_loader, criterion, optimizer, device, epoch)
        test_loss, test_acc = evaluate(model, test_loader, criterion, device)
        elapsed = time.time() - t0

        param_hash = hash_parameters(model)
        opt_hash = hash_optimizer_state(optimizer)

        print(f'\nEpoch {epoch}/{args.epochs} ({elapsed:.1f}s)')
        print(f'  Train: loss={train_loss:.4f} acc={train_acc:.1f}%')
        print(f'  Test:  loss={test_loss:.4f} acc={test_acc:.1f}%')
        print(f'  Param hash: {param_hash[:16]}...')
        print(f'  Optim hash: {opt_hash[:16]}...')
        print()

        hash_log.append(
            f'epoch={epoch} params={param_hash} optim={opt_hash} '
            f'train_loss={train_loss:.6f} test_acc={test_acc:.2f}'
        )

    # Write hashes
    with open(args.save_hashes, 'w') as f:
        for line in hash_log:
            f.write(line + '\n')
    print(f'Hashes written to {args.save_hashes}')
    print(f'\nTo verify reproducibility, run this script twice (possibly on '
          f'different GPUs with the same CUDA version) and diff the hash files:')
    print(f'  diff run1/hashes.txt run2/hashes.txt')


if __name__ == '__main__':
    main()
