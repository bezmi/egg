# Reproducible build environment

## Quick start (CLI, rootless Podman)

Make sure you have podman installed.

```bash
./.devcontainer/podman-run.sh # CPU-only
./.devcontainer/podman-run.sh rocm
./.devcontainer/podman-run.sh cuda
./.devcontainer/podman-run.sh all  # both CUDA + ROCm in one image (egg-gpu)
```

Override the ROCm version with `--rocm-version` (before the backend), e.g.
`./.devcontainer/podman-run.sh --rocm-version 7.0.2 rocm`. It maps to the
Containerfile's `ROCM_VERSION` build arg and only affects `rocm`/`all` builds.

`all` builds acpp with the CUDA *and* ROCm backends in a single image — the
build needs no GPU, so it can produce a `core;cuda;hip` distribution for every
supported platform from one host. At run time it passes through whichever
vendor's GPU the host has a CDI spec for (`/etc/cdi/nvidia.yaml`,
`/etc/cdi/amd.yaml`).

On start, the container runs `uv sync` to build the project. This happens in both the CLI (`podman-run.sh`) and VS Code. The environment is then ready to use. For the next steps, see the editable workflow in [DEVELOPING.md](../DEVELOPING.md#a-devcontainer-workflow-recommended).

## Shell prompt (zsh + starship)

The container shell is **zsh** with a **starship** prompt (the pure preset).

The container has a persistent home folder at `.devcontainer/.home`. On first run, it seeds two files there:

- `.zshrc` — the default zsh config plus the starship setup.
- `.config/starship.toml` — the starship pure preset.

Edit these files to change your shell and prompt. To turn the prompt off, remove the starship line from `.zshrc`. You can also add other dotfiles in this folder.

## Clean rebuild

To rebuild from scratch, with no cache:

```bash
podman rmi -f egg-dev                          # drop the existing image
podman build --no-cache -t egg-dev .devcontainer
```

The `--no-cache` flag compiles **AdaptiveCpp from source**. This takes several minutes. To go faster, remove the `--no-cache` flag. Podman then reuses the cached layers. It rebuilds only from the first changed step.

To also clear the build cache and unused layers, and free disk space:

```bash
podman rmi -f egg-dev
podman system prune -af        # removes all unused images + build cache
podman build --no-cache -t egg-dev .devcontainer
```

Notes:

- The GPU builds are separate images. Remove them by name: `podman rmi -f egg-rocm`, `podman rmi -f egg-cuda`, or `podman rmi -f egg-gpu` (the `all` image).
- The Python packages live in the repo, not the image. To install them fresh, delete the virtual environment first: `rm -rf .venv`.

## VS Code / Dev Containers

First, tell VS Code to use Podman. Set this once in your User settings:

```json
"dev.containers.dockerPath": "podman"
```

Then run **Reopen in Container**. The default build is CPU-only.

To use a GPU, edit `devcontainer.json`. Set the build arg (`WITH_ROCM` or `WITH_CUDA`). Then add the device flag to `runArgs`. See the comments at the bottom of that file.

## GPU passthrough (rootless Podman)

**AMD / ROCm.** On the host, install `amd-container-toolkit` and the `amdgpu` kernel driver.

```bash
sudo amd-ctk cdi generate --output=/etc/cdi/amd.yaml
amd-ctk cdi list # expect amd.com/gpu=all
```

Run flag: `--device amd.com/gpu=all`. You must be in the host `render` and `video` groups.

**NVIDIA / CUDA.** On the host, install `nvidia-container-toolkit` and the NVIDIA driver.

```bash
sudo nvidia-ctk cdi generate --output=/etc/cdi/nvidia.yaml
nvidia-ctk cdi list # expect nvidia.com/gpu=all
```

Run flag: `--device nvidia.com/gpu=all`.

To check that the container sees the GPU, run `rocminfo` (AMD) or `nvidia-smi` (NVIDIA). To check the SYCL devices, run `acpp-info`.

## Build arguments

| Arg | Default | Notes |
|---|---|---|
| `ACPP_VERSION` | `54c31cd` | AdaptiveCpp git ref (tag, branch, or commit). Pinned to a commit. Released tags (including v25.10.0) do not yet compile against LLVM 21. |
| `CUDA_REPO_DEBIAN` | `debian13` | NVIDIA's CUDA apt repo for Debian 13. Only used with `WITH_CUDA=1`. |
| `CUDA_TOOLKIT_PKG` | `cuda-toolkit-13-3` | Versioned CUDA toolkit package to install. Only used with `WITH_CUDA=1`. |
| `ROCM_VERSION` | `7.2.4` | ROCm version from AMD's `repo.radeon.com`. Debian 13 uses the `noble` codename. Installs to `/opt/rocm`. Only used with `WITH_ROCM=1`. |
| `UV_VERSION` | `0.11.19` | Pinned uv version. |
</content>
</invoke>
