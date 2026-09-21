#!/usr/bin/env python3
"""
IMU JSON -> PNG

Creates ONE PNG with:
  - top graph being Accelerometer Ax/Ay/Az
  - bottom graph Gyroscope Gx/Gy/Gz
  - shared time axis between the two

Expected JSON keys:
    Timestamp, Ax, Ay, Az, Gx, Gy, Gz
- logger stores acceleration in g and gyroscope in deg/s.
"""

import argparse
import json
import math
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


AXIS_COLORS = {
    "x": "red",
    "y": "green",
    "z": "blue",
}


def load_json(path: Path):
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)

    required = ["Timestamp", "Ax", "Ay", "Az", "Gx", "Gy", "Gz"]
    missing = [key for key in required if key not in data]
    if missing:
        raise ValueError(
            "JSON is missing required field(s): " + ", ".join(missing)
        )

    arrays = {key: np.asarray(data[key], dtype=float) for key in required}
    lengths = {key: len(value) for key, value in arrays.items()}

    if len(set(lengths.values())) != 1:
        raise ValueError(f"JSON arrays are different lengths: {lengths}")

    if lengths["Timestamp"] == 0:
        raise ValueError("JSON contains no IMU samples.")

    # Always display relative recording time starting at zero.
    t = arrays["Timestamp"].copy()
    t = t - t[0]

    return data, t, arrays


def robust_limit(values, minimum_limit, percentile=99.5, headroom=1.15):
    combined = np.concatenate(values)
    finite = np.abs(combined[np.isfinite(combined)])
    if finite.size == 0:
        return minimum_limit
    p = float(np.percentile(finite, percentile))
    return max(minimum_limit, p * headroom)


def candidate_fall_times(t, ax, ay, az, gx, gy, gz,
                         impact_g=2.0, gyro_dps=200.0,
                         refractory_s=1.5):
    """
    Simple VISUAL marker only.
    A candidate requires both:
      |accel| >= impact_g
      |gyro|  >= gyro_dps
    near the same sample.

    This is not a validated clinical fall detector.
    """
    amag = np.sqrt(ax * ax + ay * ay + az * az)
    gmag = np.sqrt(gx * gx + gy * gy + gz * gz)

    hit_indices = np.flatnonzero((amag >= impact_g) & (gmag >= gyro_dps))
    if hit_indices.size == 0:
        return []

    times = []
    last = -1e99
    for idx in hit_indices:
        ti = float(t[idx])
        if ti - last >= refractory_s:
            times.append(ti)
            last = ti
    return times


def main():
    parser = argparse.ArgumentParser(
        description="Create a reference-style two-panel IMU PNG from one JSON recording."
    )
    parser.add_argument("json_file", type=Path, help="JSON recording to graph.")
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="PNG output path. Default: same filename as JSON, with .png extension.",
    )
    parser.add_argument(
        "--accel-limit",
        type=float,
        default=6.0,
        help="Fixed accelerometer Y limit in g (default: 6).",
    )
    parser.add_argument(
        "--gyro-limit",
        type=float,
        default=1000.0,
        help="Fixed gyroscope Y limit in deg/s (default: 1000, matching reference layout).",
    )
    parser.add_argument(
        "--auto-scale",
        action="store_true",
        help="Use robust automatic Y scaling instead of the fixed reference-style limits.",
    )
    parser.add_argument(
        "--mark-falls",
        action="store_true",
        help="Draw subtle vertical markers at simple fall-like impact candidates.",
    )
    parser.add_argument(
        "--impact-g",
        type=float,
        default=2.0,
        help="Acceleration-magnitude threshold used only with --mark-falls.",
    )
    parser.add_argument(
        "--gyro-dps",
        type=float,
        default=200.0,
        help="Gyro-magnitude threshold used only with --mark-falls.",
    )
    args = parser.parse_args()

    source, t, a = load_json(args.json_file)

    ax, ay, az = a["Ax"], a["Ay"], a["Az"]
    gx, gy, gz = a["Gx"], a["Gy"], a["Gz"]

    if args.output is None:
        output = args.json_file.with_suffix(".png")
    else:
        output = args.output
    output.parent.mkdir(parents=True, exist_ok=True)

    if args.auto_scale:
        accel_limit = robust_limit([ax, ay, az], minimum_limit=1.5)
        gyro_limit = robust_limit([gx, gy, gz], minimum_limit=100.0)
    else:
        accel_limit = abs(args.accel_limit)
        gyro_limit = abs(args.gyro_limit)

    # Large single PNG, matching the reference's clean white layout.
    fig, (accel_ax, gyro_ax) = plt.subplots(
        2,
        1,
        figsize=(16, 10),
        sharex=True,
        facecolor="white",
    )

    fig.suptitle("Sensor Data Visualization", fontsize=18, y=0.975)

    # --- Accelerometer panel ---
    accel_ax.plot(t, ax, color=AXIS_COLORS["x"], linewidth=1.0, label="Ax")
    accel_ax.plot(t, ay, color=AXIS_COLORS["y"], linewidth=1.0, label="Ay")
    accel_ax.plot(t, az, color=AXIS_COLORS["z"], linewidth=1.0, label="Az")
    accel_ax.set_title("Accelerometer Data", fontsize=13)
    accel_ax.set_ylabel("Acceleration (g)", fontsize=11)
    accel_ax.set_xlabel("Time (seconds)", fontsize=11)
    accel_ax.set_ylim(-accel_limit, accel_limit)
    accel_ax.grid(True, linewidth=0.7, alpha=0.55)
    accel_ax.legend(loc="upper right", frameon=True)

    # --- Gyroscope panel ---
    gyro_ax.plot(t, gx, color=AXIS_COLORS["x"], linewidth=1.0, label="Gx")
    gyro_ax.plot(t, gy, color=AXIS_COLORS["y"], linewidth=1.0, label="Gy")
    gyro_ax.plot(t, gz, color=AXIS_COLORS["z"], linewidth=1.0, label="Gz")
    gyro_ax.set_title("Gyroscope Data", fontsize=13)
    gyro_ax.set_ylabel("Angular Velocity (deg/s)", fontsize=11)
    gyro_ax.set_xlabel("Time (seconds)", fontsize=11)
    gyro_ax.set_ylim(-gyro_limit, gyro_limit)
    gyro_ax.grid(True, linewidth=0.7, alpha=0.55)
    gyro_ax.legend(loc="upper right", frameon=True)

    # Keep the two panels on exactly the same horizontal time scale.
    xmax = max(60.0, float(t[-1])) if float(t[-1]) > 0 else 60.0
    accel_ax.set_xlim(0.0, xmax)

    # Reference-like 10-second ticks for normal one-minute files.
    tick_end = math.ceil(xmax / 10.0) * 10.0
    ticks = np.arange(0.0, tick_end + 0.001, 10.0)
    gyro_ax.set_xticks(ticks)

    # Optional fall markers while preserving the same two-panel layout.
    fall_times = []
    if args.mark_falls:
        fall_times = candidate_fall_times(
            t, ax, ay, az, gx, gy, gz,
            impact_g=args.impact_g,
            gyro_dps=args.gyro_dps,
        )
        for i, ft in enumerate(fall_times):
            label = "fall candidate" if i == 0 else None
            accel_ax.axvline(ft, color="black", linestyle="--", linewidth=1.0, alpha=0.65, label=label)
            gyro_ax.axvline(ft, color="black", linestyle="--", linewidth=1.0, alpha=0.65)
        if fall_times:
            accel_ax.legend(loc="upper right", frameon=True)

    # Make spines/ticks visually similar to the supplied reference.
    for panel in (accel_ax, gyro_ax):
        panel.set_facecolor("white")
        for spine in panel.spines.values():
            spine.set_linewidth(0.8)
        panel.tick_params(axis="both", labelsize=9)

    fig.subplots_adjust(
        left=0.085,
        right=0.975,
        top=0.91,
        bottom=0.08,
        hspace=0.28,
    )

    fig.savefig(output, dpi=160, facecolor="white")
    plt.close(fig)

    sample_count = len(t)
    duration = float(t[-1]) if sample_count > 1 else 0.0
    print(f"Input:   {args.json_file}")
    print(f"Samples: {sample_count}")
    print(f"Time:    {duration:.3f} s")
    print(f"PNG:     {output}")
    if args.mark_falls:
        print(f"Fall-candidate marker(s): {len(fall_times)}")


if __name__ == "__main__":
    main()