#!/usr/bin/env python3
"""
Create synthetic end-to-end models for GPU MVN 2-pass optimization benchmarking.

Builds 3 models using openvino.opset13 API and serializes to IR format:
  1. unet3d_encoder  — nnU-Net 3D-like encoder with InstanceNorm (MVN within-channels)
  2. segnet_highres  — 2D high-res segmentation with InstanceNorm
  3. segnet_control  — same as segnet_highres but without normalization (control)

Uses embedded subprocess pattern (requires python3.11 with openvino bindings).

Usage:
    uv run python create_models.py --build /path/to/build --output-dir models/
"""

import argparse
import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path


MODEL_BUILDER_CODE = r'''
import openvino as ov
from openvino import opset13 as opset
import numpy as np
import json
import sys
import os

output_dir = os.environ["OUTPUT_DIR"]
os.makedirs(output_dir, exist_ok=True)

core = ov.Core()


def make_conv_weights(in_ch, out_ch, kernel_size, ndim):
    """Generate deterministic random conv weights."""
    if ndim == 3:
        shape = (out_ch, in_ch, kernel_size, kernel_size, kernel_size)
    else:
        shape = (out_ch, in_ch, kernel_size, kernel_size)
    return (np.random.randn(*shape) * np.sqrt(2.0 / (in_ch * kernel_size**ndim))).astype(np.float32)


def make_conv_bias(out_ch):
    return np.zeros(out_ch, dtype=np.float32)


def build_unet3d_encoder():
    """
    nnU-Net 3D-like encoder: Conv3D + MVN(within-channels) + ReLU stages.

    Input: (1, 1, 128, 128, 128) — single-channel CT volume patch.
    3 stages with stride-2 downsampling between stages.
    Channel progression: 1 -> 32 -> 64 -> 128.
    MVN reductions: 128^3=2M (8 MB/WG), 64^3=262K (1 MB/WG), 32^3=32K (128 KB/WG).
    """
    np.random.seed(42)

    param = opset.parameter([1, 1, 128, 128, 128], dtype=np.float32, name="input")
    x = param

    stages = [
        (1, 32),    # stage 1: 1->32, spatial 128^3 -> MVN reduction 2M elements (8 MB)
        (32, 64),   # stage 2: 32->64, spatial 64^3 -> MVN reduction 262K elements (1 MB)
        (64, 128),  # stage 3: 64->128, spatial 32^3 -> MVN reduction 32K elements (128 KB)
    ]

    for i, (in_ch, out_ch) in enumerate(stages):
        # Conv3D (3x3x3, stride=1, pad=1)
        w = make_conv_weights(in_ch, out_ch, 3, ndim=3)
        b = make_conv_bias(out_ch)
        x = opset.convolution(x,
                              opset.constant(w),
                              strides=[1, 1, 1],
                              pads_begin=[1, 1, 1],
                              pads_end=[1, 1, 1],
                              dilations=[1, 1, 1])
        x = opset.add(x, opset.constant(b.reshape(1, out_ch, 1, 1, 1)))

        # MVN (within-channels: reduce spatial dims only)
        axes = opset.constant(np.array([2, 3, 4], dtype=np.int64))
        x = opset.mvn(x, axes, True, 1e-6, "inside_sqrt")

        # ReLU
        x = opset.relu(x)

        # Stride-2 downsample via max_pool (except after last stage)
        if i < len(stages) - 1:
            x = opset.max_pool(x,
                               strides=[2, 2, 2],
                               dilations=[1, 1, 1],
                               pads_begin=[0, 0, 0],
                               pads_end=[0, 0, 0],
                               kernel_shape=[2, 2, 2]).output(0)

    model = ov.Model([x], [param], "unet3d_encoder")
    return model


def build_segnet_highres():
    """
    2D high-res segmentation network: Conv2D + MVN(within-channels) + ReLU.

    Input: (1, 3, 512, 512) — RGB image.
    4 blocks with stride-2 downsampling between pairs.
    Channel progression: 3 -> 32 -> 32 -> 64 -> 64.
    MVN reductions: 512^2=262K (1 MB/WG) for first two layers.
    """
    np.random.seed(123)

    param = opset.parameter([1, 3, 512, 512], dtype=np.float32, name="input")
    x = param

    blocks = [
        (3, 32, False),     # block 1: 3->32, spatial 512x512, MVN 262K (1 MB)
        (32, 32, True),     # block 2: 32->32, spatial 512x512, MVN 262K (1 MB), then downsample
        (32, 64, False),    # block 3: 32->64, spatial 256x256, MVN 65K (256 KB)
        (64, 64, True),     # block 4: 64->64, spatial 256x256, MVN 65K (256 KB), then downsample
    ]

    for i, (in_ch, out_ch, downsample) in enumerate(blocks):
        # Conv2D (3x3, stride=1, pad=1)
        w = make_conv_weights(in_ch, out_ch, 3, ndim=2)
        b = make_conv_bias(out_ch)
        x = opset.convolution(x,
                              opset.constant(w),
                              strides=[1, 1],
                              pads_begin=[1, 1],
                              pads_end=[1, 1],
                              dilations=[1, 1])
        x = opset.add(x, opset.constant(b.reshape(1, out_ch, 1, 1)))

        # MVN (within-channels: reduce spatial dims only)
        axes = opset.constant(np.array([2, 3], dtype=np.int64))
        x = opset.mvn(x, axes, True, 1e-6, "inside_sqrt")

        # ReLU
        x = opset.relu(x)

        # Stride-2 downsample
        if downsample:
            x = opset.max_pool(x,
                               strides=[2, 2],
                               dilations=[1, 1],
                               pads_begin=[0, 0],
                               pads_end=[0, 0],
                               kernel_shape=[2, 2]).output(0)

    model = ov.Model([x], [param], "segnet_highres")
    return model


def build_segnet_control():
    """
    Control model: same as segnet_highres but WITHOUT MVN layers.

    Should show no change between baseline and optimized builds.
    Validates that any improvement in other models comes from MVN optimization.
    """
    np.random.seed(123)  # same seed as segnet_highres for comparable weights

    param = opset.parameter([1, 3, 512, 512], dtype=np.float32, name="input")
    x = param

    blocks = [
        (3, 32, False),
        (32, 32, True),
        (32, 64, False),
        (64, 64, True),
    ]

    for i, (in_ch, out_ch, downsample) in enumerate(blocks):
        # Conv2D (3x3, stride=1, pad=1)
        w = make_conv_weights(in_ch, out_ch, 3, ndim=2)
        b = make_conv_bias(out_ch)
        x = opset.convolution(x,
                              opset.constant(w),
                              strides=[1, 1],
                              pads_begin=[1, 1],
                              pads_end=[1, 1],
                              dilations=[1, 1])
        x = opset.add(x, opset.constant(b.reshape(1, out_ch, 1, 1)))

        # NO MVN — this is the control

        # ReLU
        x = opset.relu(x)

        # Stride-2 downsample
        if downsample:
            x = opset.max_pool(x,
                               strides=[2, 2],
                               dilations=[1, 1],
                               pads_begin=[0, 0],
                               pads_end=[0, 0],
                               kernel_shape=[2, 2]).output(0)

    model = ov.Model([x], [param], "segnet_control")
    return model


# Build and serialize all models
models = {
    "unet3d_encoder": build_unet3d_encoder,
    "segnet_highres": build_segnet_highres,
    "segnet_control": build_segnet_control,
}

results = {}
for name, builder_fn in models.items():
    print(f"Building {name}...", file=sys.stderr)
    model = builder_fn()

    xml_path = os.path.join(output_dir, f"{name}.xml")
    bin_path = os.path.join(output_dir, f"{name}.bin")
    ov.serialize(model, xml_path, bin_path)

    xml_size = os.path.getsize(xml_path)
    bin_size = os.path.getsize(bin_path)
    print(f"  Saved: {xml_path} ({xml_size} bytes) + {bin_path} ({bin_size} bytes)", file=sys.stderr)

    results[name] = {
        "xml_path": xml_path,
        "bin_path": bin_path,
        "xml_size": xml_size,
        "bin_size": bin_size,
    }

print(json.dumps(results))
'''


def main():
    parser = argparse.ArgumentParser(
        description="Create synthetic e2e models for GPU MVN benchmark"
    )
    parser.add_argument(
        "--build", required=True,
        help="Absolute path to build directory"
    )
    parser.add_argument(
        "--output-dir", required=True,
        help="Directory to write model IR files"
    )
    args = parser.parse_args()

    build_dir = Path(args.build)
    if not build_dir.is_absolute():
        print("ERROR: Build path must be absolute", file=sys.stderr)
        sys.exit(1)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    ov_dir = f"{build_dir}/openvino"

    print(f"{'=' * 60}")
    print(f"Create E2E Models for GPU MVN Benchmark")
    print(f"{'=' * 60}")
    print(f"Build:   {build_dir}")
    print(f"Output:  {output_dir}")
    print(f"{'=' * 60}\n")

    with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
        f.write(MODEL_BUILDER_CODE)
        script_path = f.name

    try:
        env = os.environ.copy()
        env["LD_LIBRARY_PATH"] = ":".join([
            f"{ov_dir}/bin/intel64/Release",
            f"{ov_dir}/temp/Linux_x86_64/tbb/lib",
            f"{ov_dir}/build/lib",
            env.get("LD_LIBRARY_PATH", ""),
        ])
        env["PYTHONPATH"] = f"{ov_dir}/bin/intel64/Release/python"
        env["OUTPUT_DIR"] = str(output_dir)

        result = subprocess.run(
            ["python3.11", script_path],
            env=env, timeout=120, capture_output=True, text=True,
        )

        if result.returncode != 0:
            print(f"STDERR:\n{result.stderr}", file=sys.stderr)
            print("FAILED to create models", file=sys.stderr)
            sys.exit(1)

        print(result.stderr, end="")

        data = json.loads(result.stdout.strip().split('\n')[-1])

        print(f"\n{'=' * 60}")
        print(f"Models created successfully:")
        for name, info in data.items():
            bin_mb = info['bin_size'] / (1024 * 1024)
            print(f"  {name}: {info['xml_path']} ({bin_mb:.1f} MB weights)")
        print(f"{'=' * 60}")

    finally:
        os.unlink(script_path)


if __name__ == "__main__":
    main()
