# Testing

RL-Kernel uses focused tests for dispatch behavior and operator accuracy.

## Docker Images

Build the source-build images from the repository root:

```bash
docker build -f docker/Dockerfile.cuda -t rl-kernel-ci:cuda .
docker build -f docker/Dockerfile.rocm -t rl-kernel-ci:rocm .
```

The CUDA image is based on a CUDA-enabled PyTorch devel image and installs the
compiler, CMake, Ninja, and Python test tooling needed for editable source
builds. The ROCm image is based on the official ROCm PyTorch image and includes
the same source-build tooling plus common FlashAttention ROCm build helpers.

Run CUDA tests in the image with an NVIDIA runtime:

```bash
docker run --rm --gpus all \
  -v "$PWD:/workspace/RL-Kernel" \
  -w /workspace/RL-Kernel \
  rl-kernel-ci:cuda \
  bash -lc 'pip install -e ".[cuda,test]" && python -m pytest tests/test_kernel_registry.py -q'
```

Run ROCm tests on an AMD host with ROCm devices exposed:

```bash
docker run --rm \
  --device=/dev/kfd \
  --device=/dev/dri \
  --group-add video \
  --ipc=host \
  --security-opt seccomp=unconfined \
  -v "$PWD:/workspace/RL-Kernel" \
  -w /workspace/RL-Kernel \
  rl-kernel-ci:rocm \
  bash -lc 'bash ci/run_rocm_ci.sh'
```

Do not report a ROCm pass from a CUDA or CPU-only machine. `ci/run_rocm_ci.sh`
checks for a ROCm PyTorch build before running hardware tests.

The ROCm CI helper defaults to the PyTorch SDPA fallback path:

```bash
bash ci/run_rocm_ci.sh
```

To validate the external ROCm FlashAttention path on a machine where the longer
source build is acceptable:

```bash
RL_KERNEL_ROCM_ATTN_BACKEND=flash_attn bash ci/run_rocm_ci.sh
```

## Dispatch Tests

```bash
python -m pytest rl_engine/tests/test_dispatch.py -v
python -m pytest tests/test_kernel_registry.py -q
```

## Operator Accuracy

```bash
python tests/test_op_accuracy.py
```

## Documentation Build

```bash
pip install -r requirements-docs.txt
mkdocs build --strict -f mkdocs.yaml
```

Run the documentation build whenever adding a new operator page or changing navigation.

## Hardware CI

Default pull-request CI runs linting, documentation, CPU tests, and mocked
hardware dispatch tests. The Docker image workflow builds both CUDA and ROCm
images on pull requests and pushes `rl-kernel-ci:cuda` / `rl-kernel-ci:rocm` to
GHCR after merges to `main`.

Add `needs-gpu-ci` when a PR needs real NVIDIA validation. That label runs the
RunPod CUDA matrix through `ci/run_gpu_ci.sh`. In GitHub Actions the RunPod
helper defaults to the published GHCR CUDA CI image; set `CI_IMAGE` explicitly,
or set `RUNPOD_USE_GHCR_IMAGE=0`, to fall back to the RunPod PyTorch base image.
Add `needs-rocm-ci` only when a trusted ROCm self-hosted runner is online; that
job runs `ci/run_rocm_ci.sh` on real AMD hardware.

Fallback behavior for unavailable hardware-specific dependencies is covered by
`tests/test_kernel_registry.py`, including the case where SM90/TMA extension
symbols are missing or the current NVIDIA GPU is not Hopper-class.

### Required GitHub Actions secrets

The following secrets must be configured in the repository (or fork) settings
for hardware CI to work:

| Secret | Purpose |
|--------|---------|
| `RUNPOD_API_KEY` | Authenticates `runpodctl` pod creation/removal for CUDA CI |
| `RUNPOD_SSH_PRIVATE_KEY` | Ed25519 private key for SSH access to RunPod pods |

The **public key** counterpart of `RUNPOD_SSH_PRIVATE_KEY` must be registered in
your RunPod account under **Settings → SSH Public Keys** before GPU CI will be
able to connect to the provisioned pod.
