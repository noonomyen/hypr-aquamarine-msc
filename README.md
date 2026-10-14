# Aquamarine Muxless Scanout Copy

A patched Aquamarine for **muxless laptops**, where the dGPU is the only usable
renderer and the iGPU is the only path to the panel: the dGPU composes, the iGPU
only scans out. Base: upstream `v0.15.1` - the unmodified readme is
[README.aquamarine.md](README.aquamarine.md). A one-page summary of what the patch changes
is in [PATCHES.md](PATCHES.md).

## What this is

- The dGPU composes the frame; the iGPU is only used to put it on the panel, and is never
  given a renderer.
- The finished frame is copied on the dGPU straight into a buffer the iGPU can display, so
  it never has to pass through the CPU.
- Opt-in: with `AQ_MSC` unset the behaviour is stock Aquamarine.

## How it works

```mermaid
flowchart LR
  A["dGPU<br/>composes the frame"] -->|"copy on the dGPU"| B["buffer the iGPU can display"]
  B --> C["iGPU<br/>scans it out"]
  C --> D["panel"]
```

The usual route - having the iGPU copy the finished frame - does not work on this
hardware: no format modifier suits both GPUs, so it falls back to a slow round trip through
the CPU. This patch copies the frame on the dGPU instead, into a buffer the iGPU can
display directly, and the iGPU only ever scans it out.

The copy only moves what changed on screen: a frame where nothing changed is skipped
entirely, and a small change moves only that part of the frame. It falls back to copying
the whole frame whenever it cannot be sure what changed.

## Usage

Build (Vulkan and its headers are required):

```sh
cmake -S . -B ./build -G Ninja -DCMAKE_BUILD_TYPE=Release
cmake --build ./build --parallel
```

Hook: Hyprland has to load the built library. It keeps the upstream soname
(`libaquamarine.so.14`), so pointing the session's `LD_LIBRARY_PATH` at the build
directory is enough. For a uwsm session, in `~/.config/uwsm/env`:

```sh
export LD_LIBRARY_PATH="$HOME/path/to/aquamarine-msc/build"
```

Environment - all roles are explicit, there is no auto-detection and no fallback:

```sh
export AQ_MSC=1                        # 0|1 switch
export AQ_MSC_COPY=vulkan              # how the copy is performed (vulkan is the only one)
export AQ_MSC_QUEUE=transfer           # transfer (async copy engine) | graphics
export AQ_MSC_DAMAGE=1                 # optional, default on: 0 copies whole frames
export AQ_MSC_DISPLAY=/dev/dri/card1   # gpu display (display only)
export AQ_MSC_COMPOSE=/dev/dri/card0   # gpu compose
```

The GPU paths are whatever your dGPU and iGPU card nodes are; `by-path` links work
too. An extra GPU is left unmanaged (the log says so).

Compositor side: `cursor:no_hardware_cursors = 0` is required. Without it Hyprland
chooses a software cursor on NVIDIA plus multi-GPU, which `AQ_MSC` makes true, and
the pointer is then composited into the frame.

## Limits

- the damage copy pays only for small changes: full-screen content (video, scrolling, games)
  still moves whole frames;
- a second output on the display GPU is untested;
- HDR / 10-bit, scaled modes and multi-GPU output are untested;
- there is no fallback: if the copy cannot be built, or the requested queue is not
  available, the session refuses to start and the log says why;
- Vulkan is linked unconditionally, so `libvulkan` is loaded even with `AQ_MSC` off.
