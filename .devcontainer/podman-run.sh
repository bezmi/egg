#!/usr/bin/env bash
# Build + run the egg dev container with rootless Podman, outside VS Code.
#
#   ./.devcontainer/podman-run.sh            # CPU-only (default)
#   ./.devcontainer/podman-run.sh rocm       # AMD GPU (ROCm/HIP) passthrough
#   ./.devcontainer/podman-run.sh cuda       # NVIDIA GPU (CUDA, via CDI)
#
# Mounts the repo at /egg-workspace and keeps your UID so written files stay yours.
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

# Persistent, git-ignored $HOME under .devcontainer/.home (it's inside the repo,
# already mounted at /egg-workspace). Keeps shell state (history, completions) out of
# the repo root, and is where you drop your own dotfiles to customise — e.g.
# .devcontainer/.home/.zshrc, .devcontainer/.home/.config/starship.toml. On first
# run $HOME/.zshrc is seeded from the image default (see below).
mkdir -p "$REPO_ROOT/.devcontainer/.home"

# Display + GPU passthrough for the GUI examples (PyVista/VTK, matplotlib). Share
# the host's X11 (XWayland) and Wayland sockets so windows appear on your desktop,
# and the DRM render node so mesa renders on the GPU (radeonsi). If there's no
# render node, force mesa's software path (llvmpipe). All best-effort: each piece
# is added only if present on the host, so this stays a no-op on headless hosts.
display_args=()
if [ -n "${DISPLAY:-}" ] && [ -d /tmp/.X11-unix ]; then
	display_args+=(-e "DISPLAY=$DISPLAY" -v /tmp/.X11-unix:/tmp/.X11-unix)
	if [ -n "${XAUTHORITY:-}" ] && [ -f "$XAUTHORITY" ]; then
		display_args+=(-v "$XAUTHORITY":/tmp/.Xauthority:ro -e XAUTHORITY=/tmp/.Xauthority)
	fi
fi
if [ -n "${WAYLAND_DISPLAY:-}" ] && [ -S "${XDG_RUNTIME_DIR:-}/$WAYLAND_DISPLAY" ]; then
	display_args+=(
		-e "WAYLAND_DISPLAY=$WAYLAND_DISPLAY"
		-e "XDG_RUNTIME_DIR=$XDG_RUNTIME_DIR"
		-v "$XDG_RUNTIME_DIR/$WAYLAND_DISPLAY":"$XDG_RUNTIME_DIR/$WAYLAND_DISPLAY"
	)
fi
if [ -e /dev/dri/renderD128 ]; then
	# keep-groups carries the host render/video gids in (in case the node isn't 0666).
	# SYS_PTRACE makes the default seccomp profile permit kcmp(2), which mesa uses to
	# dedupe DRM fds (os_same_file_description). Without it amdgpu warns and crashes.
	display_args+=(--device /dev/dri/renderD128 --group-add keep-groups --cap-add=SYS_PTRACE)
else
	display_args+=(-e LIBGL_ALWAYS_SOFTWARE=1)
fi

echo ">> Building $IMAGE ($BACKEND) ..."
podman build "${build_args[@]}" -t "$IMAGE" "$REPO_ROOT/.devcontainer"

run=(podman run --rm -it
	--userns=keep-id
	-v "$REPO_ROOT":/egg-workspace:Z
	-w /egg-workspace
	-e HOME=/egg-workspace/.devcontainer/.home
	"${display_args[@]}"
	"${gpu_args[@]}")

echo ">> Running $IMAGE ..."
if [ "$#" -gt 1 ]; then
	# Explicit command (e.g. `podman-run.sh cpu pytest -q`): run it as-is in /egg-workspace.
	exec "${run[@]}" "$IMAGE" "${@:2}"
fi
# Default: seed $HOME with default dotfiles (if missing), run the editable build
# (uv sync), then open the shell — all in /egg-workspace. uv sync is best-effort so a
# failure (e.g. offline) still drops you into the shell.
exec "${run[@]}" "$IMAGE" zsh -c 'seed-home; cd /egg-workspace; uv sync; egg-prewarm; exec zsh'
