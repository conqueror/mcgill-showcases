# Platform Notes

This showcase uses a unified learning flow, but the real upstream execution path depends on your hardware.

## macOS track

Use this when you have:

- Apple Silicon,
- MPS available in PyTorch,
- interest in `miolini/autoresearch-macos`.

Key ideas:

- the fork adds explicit macOS and MPS checks,
- SDPA is used instead of a hard FlashAttention-3 dependency,
- model compilation is treated more conservatively on MPS,
- batch sizing and memory expectations are different from CUDA.

Use the generated briefs:

- `artifacts/agent/codex_macos.md`
- `artifacts/agent/claude_macos.md`

## Unix track

Use this when you have:

- a Unix-like environment,
- a single NVIDIA GPU,
- interest in `karpathy/autoresearch`.

Key ideas:

- the original repo assumes CUDA execution,
- FlashAttention-3 is pulled through the `kernels` package,
- `torch.compile` is part of the main training path,
- results depend strongly on the GPU generation and memory budget.

Use the generated briefs:

- `artifacts/agent/codex_unix.md`
- `artifacts/agent/claude_unix.md`

## What stays the same

Across both tracks:

- `prepare.py` is fixed,
- `train.py` is the research surface,
- `program.md` defines the autonomous workflow,
- runs are fixed to 300 seconds,
- `val_bpb` is the primary decision metric.

## What should not be compared directly

Do not compare raw scores across different hardware as if they were leaderboard-equivalent.

The point of autoresearch is platform-local improvement under a fixed budget, not global benchmark fairness across every machine.
