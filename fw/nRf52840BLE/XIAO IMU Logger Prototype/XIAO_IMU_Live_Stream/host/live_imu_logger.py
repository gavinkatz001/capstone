import argparse
import asyncio
import json
import os
import struct
from datetime import datetime, timezone
from pathlib import Path

from bleak import BleakClient, BleakScanner

DEVICE_NAME = "XIAO-IMU-LIVE"
DATA_UUID = "7c8c1a10-2f31-4c84-a991-53c47aa10001"

# 20-byte packet sent by the XIAO:
#   uint32 raw_sensor_timestamp_ticks
#   int16  ax, ay, az, gx, gy, gz
#   uint32 sample_index
PACKET = struct.Struct("<IhhhhhhI")

ODR_HZ = 104.0
WINDOW_SECONDS = 60.0
SAMPLES_PER_WINDOW = int(round(ODR_HZ * WINDOW_SECONDS))  # 6240

ACCEL_G_PER_LSB = 0.000122
GYRO_DPS_PER_LSB = 0.0175


def new_window(window_start_index: int):
    return {
        "Units": {
            "Timestamp": "s",
            "Acceleration": "g",
            "Gyroscope": "deg/s",
            "RawSensorTimestampTicks": "25us ticks (diagnostic)",
        },
        "ConfiguredODR_Hz": ODR_HZ,
        "WindowSeconds": WINDOW_SECONDS,
        "WindowStartSampleIndex": window_start_index,
        "SampleIndex": [],
        "Timestamp": [],
        "RawSensorTimestampTicks": [],
        "Ax": [],
        "Ay": [],
        "Az": [],
        "Gx": [],
        "Gy": [],
        "Gz": [],
    }


def save_json(output_dir: Path, file_number: int, data: dict, partial=False):
    if not data["Timestamp"]:
        return None

    output_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S.%fZ")
    suffix = "_partial" if partial else ""
    path = output_dir / f"imu_{file_number:06d}_{stamp}{suffix}.json"
    temp_path = path.with_suffix(path.suffix + ".tmp")

    with temp_path.open("w", encoding="utf-8") as f:
        json.dump(data, f, separators=(",", ":"))

    os.replace(temp_path, path)
    return path


async def run(output_dir: Path):
    print(f"Scanning for BLE device '{DEVICE_NAME}'...")
    device = await BleakScanner.find_device_by_name(DEVICE_NAME, timeout=15.0)
    if device is None:
        raise RuntimeError(f"Could not find {DEVICE_NAME}")

    queue = asyncio.Queue(maxsize=4096)
    dropped_by_python_queue = 0

    def on_sample(_characteristic, payload: bytearray):
        nonlocal dropped_by_python_queue
        if len(payload) != PACKET.size:
            return
        try:
            queue.put_nowait(bytes(payload))
        except asyncio.QueueFull:
            dropped_by_python_queue += 1

    file_number = 1
    previous_index = None
    missing_samples = 0
    recording_start_index = None
    window_start_index = None
    window = None

    print(f"Connecting to {device.address}...")
    async with BleakClient(device) as client:
        print("Connected. Subscribing to live 20-byte IMU notifications...")
        await client.start_notify(DATA_UUID, on_sample)
        print(
            f"Streaming. One JSON is saved every {SAMPLES_PER_WINDOW} sample slots "
            f"(~{WINDOW_SECONDS:.0f}s at {ODR_HZ:.0f} Hz). Press Ctrl+C to stop."
        )
        print("NOTE: raw IMU FIFO timestamp is stored for diagnostics but is NOT used to split files.")

        try:
            while True:
                payload = await queue.get()
                raw_ts_ticks, ax_raw, ay_raw, az_raw, gx_raw, gy_raw, gz_raw, sample_index = PACKET.unpack(payload)

                if recording_start_index is None:
                    recording_start_index = sample_index
                    window_start_index = sample_index
                    window = new_window(window_start_index)

                # Detect missing BLE samples from the monotonic sequence number.
                if previous_index is not None and sample_index != previous_index + 1:
                    if sample_index > previous_index + 1:
                        gap = sample_index - previous_index - 1
                        missing_samples += gap
                        print(f"WARNING: missing {gap} BLE sample(s) before index {sample_index}")
                    elif sample_index <= previous_index:
                        print(
                            f"WARNING: sample index moved backwards/reset: "
                            f"previous={previous_index}, current={sample_index}"
                        )
                previous_index = sample_index

                # Close a window based ONLY on the sample sequence / configured ODR.
                # This prevents a bad/raw FIFO timestamp from creating dozens of tiny JSON files.
                while sample_index >= window_start_index + SAMPLES_PER_WINDOW:
                    if window["Timestamp"]:
                        path = save_json(output_dir, file_number, window, partial=False)
                        print(
                            f"Saved {path} | samples={len(window['Timestamp'])} | "
                            f"expected_slots={SAMPLES_PER_WINDOW} | "
                            f"stream missing so far={missing_samples} | "
                            f"python queue drops={dropped_by_python_queue}"
                        )
                        file_number += 1

                    window_start_index += SAMPLES_PER_WINDOW
                    window = new_window(window_start_index)

                # Stable per-window time axis. SampleIndex is generated by the XIAO
                # once per transmitted IMU sample, so this does not depend on BLE arrival jitter.
                timestamp_s = (sample_index - window_start_index) / ODR_HZ

                window["SampleIndex"].append(sample_index)
                window["Timestamp"].append(round(timestamp_s, 6))
                window["RawSensorTimestampTicks"].append(raw_ts_ticks)

                window["Ax"].append(round(ax_raw * ACCEL_G_PER_LSB, 6))
                window["Ay"].append(round(ay_raw * ACCEL_G_PER_LSB, 6))
                window["Az"].append(round(az_raw * ACCEL_G_PER_LSB, 6))
                window["Gx"].append(round(gx_raw * GYRO_DPS_PER_LSB, 6))
                window["Gy"].append(round(gy_raw * GYRO_DPS_PER_LSB, 6))
                window["Gz"].append(round(gz_raw * GYRO_DPS_PER_LSB, 6))

        finally:
            try:
                await client.stop_notify(DATA_UUID)
            except Exception:
                pass

            if window is not None and window["Timestamp"]:
                path = save_json(output_dir, file_number, window, partial=True)
                print(
                    f"Saved partial final window: {path} | "
                    f"samples={len(window['Timestamp'])}"
                )


async def async_main():
    parser = argparse.ArgumentParser(description="Continuous XIAO nRF52840 Sense IMU BLE logger")
    parser.add_argument("--output-dir", default="recordings", help="Directory for one-minute JSON files")
    args = parser.parse_args()
    await run(Path(args.output_dir))


if __name__ == "__main__":
    try:
        asyncio.run(async_main())
    except KeyboardInterrupt:
        print("\nStopped by Ctrl+C.")
    except Exception as exc:
        print(f"ERROR: {exc}")
        raise SystemExit(1)