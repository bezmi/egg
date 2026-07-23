#!/usr/bin/env bash
# Build + run the egg dev container with rootless Podman, outside VS Code.
#
#   ./.devcontainer/podman-run.sh            # CPU-only (default)
#   ./.devcontainer/podman-run.sh rocm       # AMD GPU (ROCm/HIP) passthrough
#   ./.devcontainer/podman-run.sh cuda       # NVIDIA GPU (CUDA, via CDI)
#   ./.devcontainer/podman-run.sh all        # both backends in one image
#                                            #   (alias: cuda+rocm)
#
# Options (must come before the backend):
#   --rocm-version VERSION   override the ROCm version (default: Containerfile's
#                            ROCM_VERSION). Only affects rocm/all builds.
#     e.g. ./.devcontainer/podman-run.sh --rocm-version 7.0.2 rocm
#
# Mounts the repo at /egg-workspace and keeps your UID so written files stay yours.
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

# Parse leading options, leaving "$@" = [backend] [explicit command...].
ROCM_VERSION=""
while [ "$#" -gt 0 ]; do
	case "$1" in
	--rocm-version)
		[ "$#" -ge 2 ] || { echo "error: --rocm-version needs a value" >&2; exit 2; }
		ROCM_VERSION="$2"
		shift 2
		;;
	--rocm-version=*)
		ROCM_VERSION="${1#*=}"
		shift
		;;
	--)
		shift
		break
		;;
	-*)
		echo "unknown option: $1" >&2
		echo "usage: $0 [--rocm-version VERSION] [cpu|rocm|cuda|all] [command...]" >&2
		exit 2
		;;
	*)
		break
		;;
	esac
done

BACKEND="${1:-cpu}"

build_args=()
gpu_args=()
case "$BACKEND" in
cpu)
	IMAGE="egg-dev"
	;;
rocm)
	IMAGE="egg-rocm"
	build_args+=(--build-arg WITH_ROCM=1)
	[ -n "$ROCM_VERSION" ] && build_args+=(--build-arg "ROCM_VERSION=$ROCM_VERSION")
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
all | cuda+rocm)
	# acpp built with BOTH the CUDA and ROCm backends (CPU/SSCP always on), so a
	# single generic build can deploy core;cuda;hip. The build needs no GPU; at
	# runtime we pass through whichever vendor the host has a CDI spec for —
	# best-effort, mirroring the display passthrough below. Each needs its own
	# host container toolkit:
	#   nvidia-ctk cdi generate --output=/etc/cdi/nvidia.yaml
	#   amd-ctk    cdi generate --output=/etc/cdi/amd.yaml
	IMAGE="egg-gpu"
	build_args+=(--build-arg WITH_CUDA=1 --build-arg WITH_ROCM=1)
	[ -n "$ROCM_VERSION" ] && build_args+=(--build-arg "ROCM_VERSION=$ROCM_VERSION")
	[ -f /etc/cdi/nvidia.yaml ] && gpu_args+=(--device nvidia.com/gpu=all)
	[ -f /etc/cdi/amd.yaml ] && gpu_args+=(--device amd.com/gpu=all)
	if [ "${#gpu_args[@]}" -eq 0 ]; then
		echo ">> warning: no CDI spec at /etc/cdi/{nvidia,amd}.yaml; building both backends but passing through no GPU" >&2
	fi
	;;
*)
	echo "usage: $0 [--rocm-version VERSION] [cpu|rocm|cuda|all] [command...]" >&2
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
# Render the GL window on whichever GPU actually drives the connected monitors.
# On a hybrid host the wrong GPU means GLX/DRI3 fails (e.g. mesa tries the nouveau
# driver against the proprietary nvidia kmod: "failed to create dri3 screen").
# This is a RENDER-ONLY concern, independent of the compute BACKEND above: `cpu`
# still opens a hardware-accelerated window.
#   - amd (radeonsi) / intel (iris): mesa is already in the image, so passing the
#     GPU's render node is all it needs.
#   - nvidia: GL needs the proprietary userspace (libGLX_nvidia/libEGL_nvidia),
#     which the *graphics* CDI device injects, matched to the host driver. (That
#     spec also mounts libcuda read-only, but nothing here uses it — the cpu image's
#     acpp has no CUDA backend. Regenerate the CDI spec graphics-only to trim it.)
render_node_for_displays() {
	local st conn card rd
	for st in /sys/class/drm/card*-*/status; do
		[ -e "$st" ] || continue
		[ "$(cat "$st" 2>/dev/null)" = connected ] || continue
		conn=$(basename "$(dirname "$st")"); card=${conn%%-*}
		for rd in "/sys/class/drm/$card/device/drm/"renderD*; do
			[ -e "$rd" ] && { basename "$rd"; return 0; }
		done
	done
	return 1
}
render_node=$(render_node_for_displays || true)
[ -z "$render_node" ] && [ -e /dev/dri/renderD128 ] && render_node=renderD128
render_drv=""
[ -n "$render_node" ] && render_drv=$(basename "$(readlink -f "/sys/class/drm/$render_node/device/driver" 2>/dev/null)")

# keep-groups carries the host render/video gids in (in case the node isn't 0666).
# SYS_PTRACE makes the default seccomp profile permit kcmp(2), which mesa uses to
# dedupe DRM fds (os_same_file_description). Without it amdgpu warns and crashes.
if [ "$render_drv" = nvidia ]; then
	if [ -f /etc/cdi/nvidia.yaml ]; then
		# Don't double-add if a gpu BACKEND (cuda/all) already passed it.
		case " ${gpu_args[*]} " in
		*" nvidia.com/gpu=all "*) : ;;
		*) gpu_args+=(--device nvidia.com/gpu=all) ;;
		esac
		display_args+=(--group-add keep-groups --cap-add=SYS_PTRACE)
	else
		echo ">> warning: display GPU is nvidia but /etc/cdi/nvidia.yaml is missing;" >&2
		echo ">>          the GL window will fall back to software (llvmpipe). Fix with:" >&2
		echo ">>          sudo nvidia-ctk cdi generate --output=/etc/cdi/nvidia.yaml" >&2
		display_args+=(-e LIBGL_ALWAYS_SOFTWARE=1)
	fi
elif [ -n "$render_node" ] && [ -e "/dev/dri/$render_node" ]; then
	display_args+=(--device "/dev/dri/$render_node" --group-add keep-groups --cap-add=SYS_PTRACE)
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
# (uv sync, with the web UI / desktop app and CAD import groups), then open the
# shell — all in /egg-workspace. uv sync is best-effort so a failure (e.g.
# offline) still drops you into the shell.
exec "${run[@]}" "$IMAGE" zsh -c 'seed-home; cd /egg-workspace; uv sync --group ui --group cad; egg-prewarm; exec zsh'
