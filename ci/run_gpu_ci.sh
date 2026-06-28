#!/usr/bin/env bash
set -euo pipefail

# TP=2 by default. The GitHub workflow overrides these values per matrix row.
PRIMARY_GPU_ID="${RUNPOD_GPU_ID:-NVIDIA RTX A4000}"
PRIMARY_GPU_COUNT="${RUNPOD_GPU_COUNT:-2}"

# TP=1
FALLBACK_GPU_ID="${RUNPOD_FALLBACK_GPU_ID:-NVIDIA A40}"
FALLBACK_GPU_COUNT="${RUNPOD_FALLBACK_GPU_COUNT:-1}"

DEFAULT_CI_IMAGE="runpod/pytorch:0.7.2-dev-cu1241-torch241-ubuntu2204"
if [ -n "${GITHUB_REPOSITORY:-}" ] && [ "${RUNPOD_USE_GHCR_IMAGE:-1}" = "1" ]; then
  DEFAULT_CI_IMAGE="ghcr.io/${GITHUB_REPOSITORY,,}/rl-kernel-ci:cuda"
fi
CI_IMAGE="${CI_IMAGE:-$DEFAULT_CI_IMAGE}"
DISK_GB="${RUNPOD_DISK_GB:-40}"
PR_SHA="${PR_SHA:-main}"
PR_SHA_FOR_POD="${PR_SHA}"
if [ "$PR_SHA" = "main" ]; then
  PR_SHA_FOR_POD="$(date +%s)"
fi
PROFILE_SLUG=$(printf "%s" "${RUNPOD_PROFILE_NAME:-gpu}" | tr -c "[:alnum:]-" "-")
POD_NAME="rl-kernel-ci-${PR_SHA_FOR_POD:0:7}-${PROFILE_SLUG}"
READY_RETRIES="${RUNPOD_READY_RETRIES:-60}"
SSH_READY_RETRIES="${RUNPOD_SSH_READY_RETRIES:-30}"
PYTEST_ARGS="${PYTEST_ARGS:-tests/ rl_engine/tests/ -v}"
FLASHINFER_WHEEL_INDEX="${FLASHINFER_WHEEL_INDEX:-https://flashinfer.ai/whl/cu124/torch2.4/index.html}"
RUNPOD_MIN_CUDA_VERSION="${RUNPOD_MIN_CUDA_VERSION:-12.4}"
RUNPOD_TERMINATE_AFTER="${RUNPOD_TERMINATE_AFTER:-$(date -u -d '+2 hours' '+%Y-%m-%dT%H:%M:%SZ')}"

POD_ID=""

cleanup() {
  trap - EXIT INT TERM

  if [ -n "$POD_ID" ]; then
    echo ""
    echo "[ci] ========================================================"
    echo "[ci] === AUTOMATIC CLEANUP: Removing pod $POD_ID ==="
    echo "[ci] ========================================================"

    REMOVE_OUT=$(runpodctl pod remove "$POD_ID" 2>&1 || true)
    if echo "$REMOVE_OUT" | grep -qiE "unknown command|unknown subcommand"; then
      REMOVE_OUT=$(runpodctl pod delete "$POD_ID" 2>&1 || true)
    fi
    if echo "$REMOVE_OUT" | grep -qi "not found"; then
      echo "[ci] Pod $POD_ID was already cleared from the cloud. Safe to exit."
    else
      echo "$REMOVE_OUT"
    fi
  fi
}
trap cleanup EXIT INT TERM

GPU_ID=$PRIMARY_GPU_ID
GPU_COUNT=$PRIMARY_GPU_COUNT

echo "[ci] Attempt 1: create pod: ${GPU_COUNT}x ${GPU_ID}"
CREATE_STATUS=0
CREATE_OUT=$(runpodctl pod create \
  --name "$POD_NAME" \
  --gpu-id "$GPU_ID" \
  --gpu-count "$GPU_COUNT" \
  --image "$CI_IMAGE" \
  --container-disk-in-gb "$DISK_GB" \
  --cloud-type SECURE \
  --min-cuda-version "$RUNPOD_MIN_CUDA_VERSION" \
  --terminate-after "$RUNPOD_TERMINATE_AFTER" \
  --ports "22/tcp" 2>&1) || CREATE_STATUS=$?

# Fallback 触发
if [ "$CREATE_STATUS" -ne 0 ] && echo "$CREATE_OUT" | grep -qi "no longer any instances available"; then
  echo "[ci] WARN: ${GPU_COUNT}x ${GPU_ID} sold out! Triggering elastic Fallback..."

  GPU_ID=$FALLBACK_GPU_ID
  GPU_COUNT=$FALLBACK_GPU_COUNT

  echo "[ci] Attempt 2 (Fallback): create pod: ${GPU_COUNT}x ${GPU_ID}"
  CREATE_STATUS=0
  CREATE_OUT=$(runpodctl pod create \
    --name "$POD_NAME" \
    --gpu-id "$GPU_ID" \
    --gpu-count "$GPU_COUNT" \
    --image "$CI_IMAGE" \
    --container-disk-in-gb "$DISK_GB" \
    --cloud-type SECURE \
    --min-cuda-version "$RUNPOD_MIN_CUDA_VERSION" \
    --terminate-after "$RUNPOD_TERMINATE_AFTER" \
    --ports "22/tcp" 2>&1) || CREATE_STATUS=$?

  if [ "$CREATE_STATUS" -ne 0 ] && echo "$CREATE_OUT" | grep -qi "no longer any instances available"; then
    echo "[ci] FATAL: Alternatives (${GPU_COUNT}x ${GPU_ID}) have also been exhausted. Please try CI again later."
    exit 1
  fi
fi

if [ "$CREATE_STATUS" -ne 0 ]; then
  echo "[ci] ERROR: Failed to create pod. Output: $CREATE_OUT"
  exit "$CREATE_STATUS"
fi

POD_ID=$(echo "$CREATE_OUT" | grep -oE '"id":\s*"[a-z0-9]{8,}"' | cut -d '"' -f4 | head -1)
if [ -z "$POD_ID" ]; then
  POD_ID=$(echo "$CREATE_OUT" | grep -oE '"id":[[:space:]]*"([a-z0-9]{8,})"' | grep -oE '[a-z0-9]{8,}' | head -1)
fi

if [ -z "$POD_ID" ]; then
  echo "[ci] ERROR: Unable to resolve pod id. Output: $CREATE_OUT"
  exit 1
fi
echo "[ci] Successfully rented pod: $POD_ID"

echo "[ci] Waiting for pod network infrastructure to be fully ready..."
SSH_IP=""
SSH_PORT=""

for i in $(seq 1 "$READY_RETRIES"); do
  POD_INFO=$(runpodctl pod get "$POD_ID" -o json)

  SSH_IP=$(echo "$POD_INFO" | grep -iE '"ip"|"publicIp"|"address"' | grep -oE '[0-9]{1,3}\.[0-9]{1,3}\.[0-9]{1,3}\.[0-9]{1,3}' | head -1 || true)
  SSH_PORT=$(echo "$POD_INFO" | grep -iE '"port"|"externalPort"|"publicPort"' | grep -oE '[0-9]+' | grep -v '^22$' | head -1 || true)

  if [ -n "$SSH_IP" ] && [ -n "$SSH_PORT" ] && ! echo "$POD_INFO" | grep -qi "not ready"; then
    echo "[ci] Pod infrastructure is 100% READY!"
    break
  fi

  if [ "$i" -eq "$READY_RETRIES" ]; then
    echo "[ci] ERROR: Pod network/SSH infrastructure initialization timed out."
    exit 1
  fi

  echo "[ci] Pod layer status: RUNNING, but network routing is initializing... waiting 10s (Attempt $i/$READY_RETRIES)"
  sleep 10
done

echo "[ci] Target Establish -> root@$SSH_IP:$SSH_PORT"

RUNPOD_SSH_KEY_PATH="${RUNPOD_SSH_KEY_PATH:-}"
if [ -z "$RUNPOD_SSH_KEY_PATH" ] && [ -f "$HOME/.runpod/ssh/runpodctl-ssh-key" ]; then
  RUNPOD_SSH_KEY_PATH="$HOME/.runpod/ssh/runpodctl-ssh-key"
elif [ -z "$RUNPOD_SSH_KEY_PATH" ] && [ -f "$HOME/.ssh/id_ed25519" ]; then
  RUNPOD_SSH_KEY_PATH="$HOME/.ssh/id_ed25519"
fi

SSH_OPTIONS="-o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null -o LogLevel=ERROR -p $SSH_PORT"
if [ -n "$RUNPOD_SSH_KEY_PATH" ]; then
  SSH_OPTIONS="$SSH_OPTIONS -o IdentitiesOnly=yes -i $RUNPOD_SSH_KEY_PATH"
fi

echo "[ci] Verifying SSH daemon readiness..."
for i in $(seq 1 "$SSH_READY_RETRIES"); do
  if ssh $SSH_OPTIONS root@"$SSH_IP" true >/dev/null 2>&1; then
    echo "[ci] SSH daemon is ready."
    break
  fi

  if [ "$i" -eq "$SSH_READY_RETRIES" ]; then
    echo "[ci] ERROR: SSH daemon did not become ready."
    exit 1
  fi

  echo "[ci] SSH daemon is not ready yet... waiting 10s (Attempt $i/$SSH_READY_RETRIES)"
  sleep 10
done

printf -v REMOTE_ENV \
  "GPU_COUNT=%q PR_REPO_URL=%q PR_SHA=%q TORCH_CUDA_ARCH_LIST=%q FORCE_CUDA=%q MAX_JOBS=%q KERNEL_ALIGN_FORCE_SM90=%q PYTEST_ARGS=%q FLASHINFER_WHEEL_INDEX=%q RUNPOD_UPGRADE_BUILD_TOOLS=%q RUNPOD_INSTALL_FLASHINFER=%q" \
  "$GPU_COUNT" \
  "${PR_REPO_URL:-https://github.com/RL-Align/RL-Kernel.git}" \
  "$PR_SHA" \
  "${TORCH_CUDA_ARCH_LIST:-8.6}" \
  "${FORCE_CUDA:-1}" \
  "${MAX_JOBS:-8}" \
  "${KERNEL_ALIGN_FORCE_SM90:-0}" \
  "$PYTEST_ARGS" \
  "$FLASHINFER_WHEEL_INDEX" \
  "${RUNPOD_UPGRADE_BUILD_TOOLS:-0}" \
  "${RUNPOD_INSTALL_FLASHINFER:-0}"

echo "[ci] Launching remote test suite on GPU pod (TP=${GPU_COUNT})..."
ssh $SSH_OPTIONS root@"$SSH_IP" "$REMOTE_ENV bash -s" <<'REMOTE'
set -euo pipefail
PY=$(command -v python3.11 || command -v python3 || true)
if [ -z "$PY" ]; then echo "[remote] FATAL: python not found in PATH"; exit 127; fi
if ! "$PY" -c "import torch" >/dev/null 2>&1; then
  for cand in python3.11 python3.10 python3; do
    p=$(command -v "$cand" 2>/dev/null) || continue
    if "$p" -c "import torch" >/dev/null 2>&1; then PY="$p"; break; fi
  done
fi
echo "[remote] Using interpreter: $PY"
export TORCH_CUDA_ARCH_LIST
export FORCE_CUDA
export MAX_JOBS
export KERNEL_ALIGN_FORCE_SM90
PIP_INSTALL_ARGS=(--timeout 60 --retries 10 --resume-retries 5)
mkdir -p /workspace
cd /workspace
git clone "$PR_REPO_URL" repo
cd repo
git fetch origin "$PR_SHA"
git checkout --detach "$PR_SHA"
if [ "${RUNPOD_UPGRADE_BUILD_TOOLS:-0}" = "1" ]; then
  "$PY" -m pip install "${PIP_INSTALL_ARGS[@]}" -U pip setuptools wheel
else
  "$PY" -m pip --version
  "$PY" -m pip show setuptools wheel >/dev/null 2>&1 || true
fi
if [ "${RUNPOD_INSTALL_FLASHINFER:-0}" = "1" ]; then
  "$PY" -m pip install "${PIP_INSTALL_ARGS[@]}" flashinfer-python -f "$FLASHINFER_WHEEL_INDEX"
  "$PY" -m pip install "${PIP_INSTALL_ARGS[@]}" -e ".[cuda,test,hf]"
else
  "$PY" -m pip install "${PIP_INSTALL_ARGS[@]}" nvidia-ml-py
  "$PY" -m pip install "${PIP_INSTALL_ARGS[@]}" -e ".[test,hf]"
fi
"$PY" setup.py build_ext --inplace
nvidia-smi
"$PY" - <<'PY'
import sys

import torch

from rl_engine.kernels.registry import kernel_registry

if not torch.cuda.is_available():
    raise SystemExit("[remote] CUDA is unavailable after installation")

print(
    f"[remote] python={sys.version.split()[0]} "
    f"torch={torch.__version__} torch_cuda={torch.version.cuda}"
)
op = kernel_registry.get_op("logp")
backend = op.__class__.__name__
print(f"[remote] strict logp backend={backend}")
if not backend.startswith("FusedLogp"):
    raise SystemExit(
        "[remote] strict fused logp preflight failed: "
        f"dispatch selected {backend}, not a FusedLogp backend"
    )
PY
"$PY" examples/grpo_single_gpu.py \
  --device cuda \
  --require-fused-logp \
  --steps 2 \
  --num-prompts 1 \
  --samples-per-prompt 2 \
  --prompt-len 2 \
  --completion-len 3 \
  --vocab-size 16 \
  --hidden-dim 8

if [ "$GPU_COUNT" -gt 1 ]; then
  cat >/tmp/rl_kernel_nccl_smoke.py <<'PY'
import os

import torch
import torch.distributed as dist

local_rank = int(os.environ["LOCAL_RANK"])
torch.cuda.set_device(local_rank)
dist.init_process_group("nccl")
world_size = dist.get_world_size()
device = torch.device("cuda", local_rank)
value = torch.tensor([local_rank + 1], device=device, dtype=torch.float32)
dist.all_reduce(value, op=dist.ReduceOp.SUM)
expected = world_size * (world_size + 1) / 2
if value.item() != expected:
    raise SystemExit(f"unexpected all-reduce value: got {value.item()}, expected {expected}")
if dist.get_rank() == 0:
    print(f"[remote] NCCL all-reduce smoke passed on {world_size} GPUs")
dist.destroy_process_group()
PY
  "$PY" -m torch.distributed.run --nproc_per_node="$GPU_COUNT" /tmp/rl_kernel_nccl_smoke.py
fi

# PYTEST_ARGS is owned by CI and intentionally split into pytest argv here.
# shellcheck disable=SC2086
"$PY" -m pytest $PYTEST_ARGS
REMOTE
TEST_EXIT=$?

echo "[ci] Remote execution finished with exit code = $TEST_EXIT"
exit $TEST_EXIT
