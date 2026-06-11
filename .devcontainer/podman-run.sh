#!/usr/bin/env bash
# Build + run the egg dev container with rootless Podman, outside VS Code.
#
#   ./.devcontainer/podman-run.sh            # CPU-only (default)
#   ./.devcontainer/podman-run.sh rocm       # AMD GPU (ROCm/HIP) passthrough
#   ./.devcontainer/podman-run.sh cuda       # NVIDIA GPU (CUDA, via CDI)
#
# Mounts the repo at /workspace and keeps your UID so written files stay yours.
set -euo pipefail

BACKEND="${1:-cpu}"
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

build_args=()
gpu_args=()
case "$BACKEND" in
  cpu)
    IMAGE="egg-dev"
    ;;
  rocm)
    IMAGE="egg-rocm"
    build_args+=(--build-arg WITH_ROCM=1)
    # Requires CDI on the host (amd-container-toolkit):
    #   amd-ctk cdi generate --output=/etc/cdi/amd.yaml
    gpu_args+=(--device amd.com/gpu=all)
    ;;
  cuda)
    IMAGE="egg-cuda"
    build_args+=(--build-arg WITH_CUDA=1)
    # Requires CDI on the host (nvidia-container-toolkit):
    #   nvidia-ctk cdi generate --output=/etc/cdi/nvidia.yaml
    gpu_args+=(--device nvidia.com/gpu=all)
    ;;
  *)
    echo "usage: $0 [cpu|rocm|cuda]" >&2
    exit 2
    ;;
esac

echo ">> Building $IMAGE ($BACKEND) ..."
podman build "${build_args[@]}" -t "$IMAGE" "$REPO_ROOT/.devcontainer"

echo ">> Running $IMAGE ..."
exec podman run --rm -it \
  --userns=keep-id \
  -v "$REPO_ROOT":/workspace:Z \
  -w /workspace \
  "${gpu_args[@]}" \
  "$IMAGE" "${@:2}"
