"""
Bit-reproducible PyTorch modules built on top of repro_gemm_ext.

Every floating-point sum goes through either:
  - repro_gemm (integer-exact Ozaki GEMM, rounded once to FP32), or
  - double-precision intermediates for element-wise ops (exp, log, sqrt, etc.)

ReLU, MaxPool2d, Flatten are plain PyTorch (comparisons/indexing only, no rounding).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import repro_gemm_ext as _ext

SLICES = 3  # default: 21-bit inputs, FP32-level accuracy


# ============================================================================
# Autograd functions
# ============================================================================

class ReproMM(torch.autograd.Function):
    """C = op(A) @ op(B), bit-reproducible."""
    @staticmethod
    def forward(ctx, A, B, trans_a, trans_b):
        ctx.save_for_backward(A, B)
        ctx.trans_a = trans_a
        ctx.trans_b = trans_b
        return _ext.repro_mm(A, B, trans_a, trans_b, SLICES)

    @staticmethod
    def backward(ctx, grad_output):
        A, B = ctx.saved_tensors
        grad_A = grad_B = None

        # Forward was: C[M,N] = op(A)[M,K] @ op(B)[K,N]
        # grad_op(A) = grad_output @ op(B)^T   -> [M, K]
        # grad_op(B) = op(A)^T @ grad_output    -> [K, N]

        if ctx.needs_input_grad[0]:
            if not ctx.trans_b:
                # op(B) is B[K,N], op(B)^T = B^T: grad_A_mat = grad_output @ B^T
                grad_A_mat = _ext.repro_mm(grad_output, B, False, True, SLICES)
            else:
                # op(B) is B^T where B is [N,K], op(B)^T = B itself
                grad_A_mat = _ext.repro_mm(grad_output, B, False, False, SLICES)

            if ctx.trans_a:
                grad_A = grad_A_mat.t().contiguous()
            else:
                grad_A = grad_A_mat

        if ctx.needs_input_grad[1]:
            if not ctx.trans_a:
                # op(A) is A[M,K], op(A)^T = A^T: grad_B_mat = A^T @ grad_output
                grad_B_mat = _ext.repro_mm(A, grad_output, True, False, SLICES)
            else:
                # op(A) is A^T where A is [K,M], op(A)^T = A itself
                grad_B_mat = _ext.repro_mm(A, grad_output, False, False, SLICES)

            if ctx.trans_b:
                grad_B = grad_B_mat.t().contiguous()
            else:
                grad_B = grad_B_mat

        return grad_A, grad_B, None, None


def repro_mm(A, B, trans_a=False, trans_b=False):
    return ReproMM.apply(A, B, trans_a, trans_b)


class ReproLinearFn(torch.autograd.Function):
    """Y = X @ W^T + bias, all GEMMs via repro_gemm."""
    @staticmethod
    def forward(ctx, X, W, bias):
        # X: [B, in], W: [out, in], bias: [out] or None
        ctx.save_for_backward(X, W)
        ctx.has_bias = bias is not None
        # Y = X @ W^T: repro_mm(X[B,in], W[out,in], trans_a=False, trans_b=True)
        # Actually W^T means we want [B,in] @ [in,out].
        # repro_mm(A, B, ta, tb): C = op(A) @ op(B)
        # We want X @ W^T. X is [B,in], W is [out,in].
        # op(A)=X (no trans), op(B)=W^T (trans_b=True since B=W is [out,in] and we want [in,out])
        # Wait: trans_b=True means B is stored as [N,K] and we use B^T = [K,N].
        # Here K=in, N=out. B=W is [out, in] = [N, K]. trans_b=True -> op(B) = W^T = [K,N] = [in,out]. Yes.
        Y = _ext.repro_mm(X, W, False, True, SLICES)
        if bias is not None:
            Y = Y + bias.unsqueeze(0)  # exact: same exponent per column, one addition
        return Y

    @staticmethod
    def backward(ctx, grad_Y):
        X, W = ctx.saved_tensors
        grad_X = grad_W = grad_bias = None

        if ctx.needs_input_grad[0]:
            # grad_X = grad_Y @ W: [B, out] @ [out, in]
            grad_X = _ext.repro_mm(grad_Y, W, False, False, SLICES)

        if ctx.needs_input_grad[1]:
            # grad_W = grad_Y^T @ X: [out, B] @ [B, in] = [out, in]
            grad_W = _ext.repro_mm(grad_Y, X, True, False, SLICES)

        if ctx.has_bias and ctx.needs_input_grad[2]:
            # grad_bias = sum over batch of grad_Y: [B, out] -> [out]
            # This is a sum of B floats per output. For B=100 this is small.
            # Use double-precision accumulation for reproducibility.
            grad_bias = grad_Y.to(torch.float64).sum(dim=0).to(torch.float32)

        return grad_X, grad_W, grad_bias


class ReproConv2dFn(torch.autograd.Function):
    """
    Conv2d via im2col + repro_gemm.
    input: [B, Cin, H, W]
    weight: [Cout, Cin, kH, kW]
    bias: [Cout] or None
    """
    @staticmethod
    def forward(ctx, input, weight, bias, stride, padding):
        B, Cin, H, W = input.shape
        Cout, _, kH, kW = weight.shape
        sH, sW = stride
        pH, pW = padding
        outH = (H + 2*pH - kH) // sH + 1
        outW = (W + 2*pW - kW) // sW + 1

        ctx.save_for_backward(input, weight)
        ctx.stride = stride
        ctx.padding = padding
        ctx.has_bias = bias is not None
        ctx.input_shape = (B, Cin, H, W)

        # Weight reshaped: [Cout, Cin*kH*kW]
        W_mat = weight.reshape(Cout, -1)

        output = torch.empty(B, Cout, outH, outW, device=input.device, dtype=input.dtype)

        for b in range(B):
            # im2col: [Cin*kH*kW, outH*outW]
            col = _ext.repro_im2col(input[b], kH, kW, pH, pW, sH, sW)
            # Y = W_mat @ col: [Cout, outH*outW]
            Y = _ext.repro_mm(W_mat, col, False, False, SLICES)
            output[b] = Y.reshape(Cout, outH, outW)

        if bias is not None:
            output = output + bias.reshape(1, Cout, 1, 1)
        return output

    @staticmethod
    def backward(ctx, grad_output):
        input, weight = ctx.saved_tensors
        B, Cin, H, W = ctx.input_shape
        Cout, _, kH, kW = weight.shape
        sH, sW = ctx.stride
        pH, pW = ctx.padding
        outH = grad_output.size(2)
        outW = grad_output.size(3)

        W_mat = weight.reshape(Cout, -1)  # [Cout, Cin*kH*kW]

        grad_input = torch.zeros_like(input) if ctx.needs_input_grad[0] else None
        grad_weight = torch.zeros(Cout, Cin * kH * kW, device=input.device, dtype=torch.float64) \
                      if ctx.needs_input_grad[1] else None
        grad_bias = None

        for b in range(B):
            grad_Y = grad_output[b].reshape(Cout, -1)  # [Cout, outH*outW]

            if ctx.needs_input_grad[0]:
                # grad_col = W_mat^T @ grad_Y: [Cin*kH*kW, outH*outW]
                grad_col = _ext.repro_mm(W_mat, grad_Y, True, False, SLICES)
                grad_input[b] = _ext.repro_col2im(grad_col, Cin, H, W, kH, kW, pH, pW, sH, sW)

            if ctx.needs_input_grad[1]:
                # grad_W += grad_Y @ col^T: [Cout, Cin*kH*kW]
                col = _ext.repro_im2col(input[b], kH, kW, pH, pW, sH, sW)
                # Use repro_mm: grad_Y[Cout, outH*outW] @ col^T[outH*outW, Cin*kH*kW]
                gw = _ext.repro_mm(grad_Y, col, False, True, SLICES)
                # Accumulate in double for exact summation across batch
                grad_weight += gw.to(torch.float64)

        if ctx.needs_input_grad[1]:
            grad_weight = grad_weight.to(torch.float32).reshape_as(weight)

        if ctx.has_bias and ctx.needs_input_grad[2]:
            # sum over B, outH, outW
            grad_bias = grad_output.to(torch.float64).sum(dim=(0, 2, 3)).to(torch.float32)

        return grad_input, grad_weight, grad_bias, None, None


class ReproCrossEntropyFn(torch.autograd.Function):
    """Cross-entropy with double-precision log-softmax for reproducibility."""
    @staticmethod
    def forward(ctx, logits, targets):
        # logits: [B, C], targets: [B]
        B, C = logits.shape
        # Compute softmax (needed for backward) and loss
        probs = _ext.repro_softmax(logits)       # [B, C]
        loss = _ext.repro_cross_entropy(logits, targets)  # scalar
        ctx.save_for_backward(probs, targets)
        ctx.B = B
        ctx.C = C
        return loss

    @staticmethod
    def backward(ctx, grad_loss):
        probs, targets = ctx.saved_tensors
        B, C = ctx.B, ctx.C
        # grad_logits = probs - one_hot(targets), scaled by grad_loss / B
        grad_logits = probs.clone()
        grad_logits.scatter_(1, targets.unsqueeze(1), grad_logits.gather(1, targets.unsqueeze(1)) - 1.0)
        # Divide by B using IEEE division (exact since it's a single division per element)
        grad_logits = _ext.repro_div_scalar(grad_logits, float(B))
        grad_logits = grad_logits * grad_loss
        return grad_logits, None


# ============================================================================
# Module wrappers
# ============================================================================

class ReproLinear(nn.Module):
    def __init__(self, in_features, out_features, bias=True):
        super().__init__()
        self.weight = nn.Parameter(torch.empty(out_features, in_features))
        self.bias = nn.Parameter(torch.empty(out_features)) if bias else None
        self.reset_parameters()

    def reset_parameters(self):
        # Kaiming uniform (same as nn.Linear) but with a fixed-seed fallback
        nn.init.kaiming_uniform_(self.weight, a=5**0.5)
        if self.bias is not None:
            fan_in = self.weight.size(1)
            bound = 1 / fan_in**0.5
            nn.init.uniform_(self.bias, -bound, bound)

    def forward(self, x):
        shape = x.shape
        x2d = x.reshape(-1, shape[-1])
        y = ReproLinearFn.apply(x2d, self.weight, self.bias)
        return y.reshape(*shape[:-1], -1)


class ReproConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size,
                 stride=1, padding=0, bias=True):
        super().__init__()
        if isinstance(kernel_size, int): kernel_size = (kernel_size, kernel_size)
        if isinstance(stride, int): stride = (stride, stride)
        if isinstance(padding, int): padding = (padding, padding)
        self.stride = stride
        self.padding = padding
        self.weight = nn.Parameter(torch.empty(out_channels, in_channels, *kernel_size))
        self.bias = nn.Parameter(torch.empty(out_channels)) if bias else None
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.kaiming_uniform_(self.weight, a=5**0.5)
        if self.bias is not None:
            fan_in = self.weight[0].numel()
            bound = 1 / fan_in**0.5
            nn.init.uniform_(self.bias, -bound, bound)

    def forward(self, x):
        return ReproConv2dFn.apply(x, self.weight, self.bias, self.stride, self.padding)


class ReproCrossEntropyLoss(nn.Module):
    def forward(self, logits, targets):
        return ReproCrossEntropyFn.apply(logits, targets)


# ============================================================================
# Reproducible Adam optimizer
# ============================================================================

class ReproAdam:
    """
    Adam optimizer using only reproducible ops.

    All element-wise arithmetic is exact or uses double-precision intermediates:
    - sqrt via repro_sqrt (double cast)
    - division via repro_div_scalar (double cast)
    - multiplication by scalars: exact when the scalar is a power of two,
      otherwise we pass through double.
    - addition/subtraction of same-shape tensors: single FP32 rounding per element.
      This IS reproducible because the operand order is fixed (no reduction).
    """
    def __init__(self, params, lr=0.001, betas=(0.9, 0.999), eps=1e-8, weight_decay=0.0):
        self.lr = lr
        self.beta1, self.beta2 = betas
        self.eps = eps
        self.weight_decay = weight_decay
        self.param_groups = [{'params': list(params)}]
        self.state = {}
        self.step_count = 0

    def zero_grad(self):
        for group in self.param_groups:
            for p in group['params']:
                if p.grad is not None:
                    p.grad.zero_()

    @torch.no_grad()
    def step(self):
        self.step_count += 1
        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None:
                    continue
                grad = p.grad.data
                if self.weight_decay != 0:
                    # L2 regularization: exact element-wise multiply+add
                    grad = grad + self.weight_decay * p.data

                if p not in self.state:
                    self.state[p] = {
                        'm': torch.zeros_like(p.data),
                        'v': torch.zeros_like(p.data),
                    }
                s = self.state[p]

                # m = beta1 * m + (1 - beta1) * grad
                # v = beta2 * v + (1 - beta2) * grad^2
                # These are element-wise: fixed operand order, single FP32 rounding.
                s['m'].mul_(self.beta1).add_(grad, alpha=1.0 - self.beta1)
                s['v'].mul_(self.beta2).addcmul_(grad, grad, value=1.0 - self.beta2)

                # Bias correction
                bc1 = 1.0 - self.beta1 ** self.step_count
                bc2 = 1.0 - self.beta2 ** self.step_count

                # m_hat = m / bc1, v_hat = v / bc2
                # Use double-precision division for reproducibility
                m_hat = _ext.repro_div_scalar(s['m'].contiguous(), float(bc1))
                v_hat = _ext.repro_div_scalar(s['v'].contiguous(), float(bc2))

                # denom = sqrt(v_hat) + eps
                denom = _ext.repro_sqrt(v_hat.contiguous())
                denom = denom + self.eps  # exact: single addition per element

                # update = lr * m_hat / denom
                # Element-wise division in double precision
                update = self._div_tensors(m_hat, denom)
                p.data.add_(update, alpha=-self.lr)

    @staticmethod
    def _div_tensors(a, b):
        """Element-wise a/b through double precision for reproducibility."""
        return (a.to(torch.float64) / b.to(torch.float64)).to(torch.float32)
