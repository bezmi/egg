# Reproducible build environment

## Quick start (CLI, rootless Podman)

Make sure you have podman installed.

```bash
./.devcontainer/podman-run.sh # CPU-only
./.devcontainer/podman-run.sh rocm
./.devcontainer/podman-run.sh cuda
```

`postCreateCommand` runs the **standard editable build** for you. Just follow the editable workflow in [DEVELOPING.md](../DEVELOPING.md#a-devcontainer-workflow-recommended) from the container shell.

## VS Code / Dev Containers

Tell VS Code to use Podman once (User settings):

```json
"dev.containers.dockerPath": "podman"
```

Then **Reopen in Container**. The default build is CPU-only. For GPU, set the
build arg in `devcontainer.json` (`WITH_ROCM` / `WITH_CUDA`) and add the device
flags to `runArgs`, documented inline at the bottom of that file.

## GPU passthrough (rootless Podman)

- **AMD/ROCm** — host: `amd-container-toolkit` + `amdgpu` kernel driver.

  ```bash
  sudo amd-ctk cdi generate --output=/etc/cdi/amd.yaml
  amd-ctk cdi list # expect amd.com/gpu=all
  ```

  Run flag: `--device amd.com/gpu=all`. (You must be in the host `render`/`video` groups.)
- **NVIDIA/CUDA** — host: `nvidia-container-toolkit` + NVIDIA driver.

  ```bash
  sudo nvidia-ctk cdi generate --output=/etc/cdi/nvidia.yaml
  nvidia-ctk cdi list # expect nvidia.com/gpu=all
  ```

  Run flag: `--device nvidia.com/gpu=all`.

Verify devices are visible inside the container with `rocminfo` (AMD) or
`nvidia-smi` (NVIDIA), and SYCL devices with `acpp-info`.

## Version knobs (build args)

| Arg | Default | Notes |
|---|---|---|
| `ACPP_VERSION` | `54c31cd` | AdaptiveCpp git ref (tag/branch/commit). Pinned to a commit: released tags (incl. v25.10.0) don't yet compile against LLVM 21. |
| `CUDA_REPO_DEBIAN` | `debian13` | NVIDIA's CUDA apt repo for Debian 13. Only used with `WITH_CUDA=1`. |
| `CUDA_TOOLKIT_PKG` | `cuda-toolkit-13-3` | Versioned CUDA toolkit package to install. Only used with `WITH_CUDA=1`. |
| `ROCM_VERSION` | `7.2.4` | ROCm version from AMD's `repo.radeon.com` (Debian 13 → `noble` codename, installs to `/opt/rocm`). Only used with `WITH_ROCM=1`. |
| `UV_VERSION` | `0.11.19` | Pinned uv. |
