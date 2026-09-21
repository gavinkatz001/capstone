# XIAO nRF52840 Sense - Simple Live IMU Stream
```
author: GBRK team
date: sept21 2026

```


## What runs where

- `XIAO_IMU_Live_Stream.ino` runs on the XIAO.
- `host/live_imu_logger.py` runs on the PC/Raspberry Pi.
- use the docx for more info on how to run the dataset gathering python script

## BLE packet

Exactly 20 bytes per IMU sample:

- 4 bytes: raw sensor timestamp field (uint32), containing the lower 24 bits returned from the FIFO timestamp, in 25 µs ticks. Used for diagnostics
- 2 bytes each: Ax, Ay, Az
- 2 bytes each: Gx, Gy, Gz
- 4 bytes: sample index


## Arduino libraries

Install:

1. Seeed nRF52 mbed-enabled Boards, select **Seeed XIAO nRF52840 Sense**.
2. `ArduinoBLE`.
3. `Seeed Arduino LSM6DS3` **v2.0.7 or newer**.

Open `XIAO_IMU_Live_Stream.ino`, Verify, then Upload.

## Host

Windows:

```powershell
cd host
py -m pip install -r requirements.txt
py live_imu_logger.py --output-dir recordings
```

Linux/macOS/Raspberry Pi:

```bash
cd host
python3 -m pip install -r requirements.txt
python3 live_imu_logger.py --output-dir recordings
```

Press Ctrl+C to stop. A partial final window is saved with `_partial` in its filename.

## JSON format (subject to change, dosent matter)

- Each complete file represents ~60 second window of ~6,240 sample slots at the configured 104 Hz output data rate. File boundaries and the main Timestamp array are derived from SampleIndex and the configured ODR; the raw IMU timestamp is retained only for diagnostics.

```json
{
  "Units": {"Timestamp":"s","Acceleration":"g","Gyroscope":"deg/s"},
  "ConfiguredODR_Hz": 104,
  "SampleIndex": [0,1,2],
  "Timestamp": [0.0,0.0096,0.0192],
  "Ax": [0.0,0.0,0.0],
  "Ay": [0.0,0.0,0.0],
  "Az": [1.0,1.0,1.0],
  "Gx": [0.0,0.0,0.0],
  "Gy": [0.0,0.0,0.0],
  "Gz": [0.0,0.0,0.0]
}
```
