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
# milestone 1 presentation demo

The Nordic nRF52840 runs our firmware and provides the Bluetooth Low Energy radio, while the onboard LSM6DS3TR-C is a separate six-axis motion sensor containing a three-axis accelerometer and three-axis gyroscope.

Our firmware configures both sensors for a nominal 104 Hz output rate, with the accelerometer at ±4 g and gyroscope at ±500 degrees per second. 
- Samples first enter the IMU's hardware FIFO. 
- The firmware drains this FIFO, combines the accelerometer, gyroscope, hardware timestamp and a monotonically increasing sample index into a fixed 20-byte binary packet, and sends that packet over BLE.


BLE stuff:
- On the BLE side, the XIAO operates as the peripheral and GATT server (ill explain this more in a bit). It advertises as XIAO-IMU-LIVE and exposes one custom IMU service containing one custom data characteristic. 
- That characteristic uses the Notify property because this is continuous streaming telemetry: the host subscribes once, and the XIAO then pushes new IMU samples as they become available rather than requiring the host to repeatedly poll the device.

At the application level the firmware thats flashed onto the xiao board uses ArduinoBLE. With the mbed-enabled XIAO board core used for this milestone, that API sits on top of the Mbed/Cordio BLE implementation.
-  Our firmware therefore deals with high-level GATT concepts such as services, characteristics and notifications 
- The host acts as the BLE central and GATT client. A Python program using Bleak scans for the XIAO, connects to it and subscribes to the IMU notification characteristic. Every 20-byte packet is unpacked, the raw sensor readings are converted into g and degrees per second, and the sample index is checked for gaps so that lost samples can be detected
- The host then groups the stream into approximately one minute windows based on the 104 Hz sample sequence. At 104 samples per second, a nominal one minute window contains 6,240 sample positions. Those recordings are saved as JSON for visualization and later algorithm development (this is where gavin uses the board to get his data and train it).



# What is BLE
- usefull links:
    - https://learn.adafruit.com/introduction-to-bluetooth-low-energy/gatt 
    


- if i have sensor data on a wearblae, and I need another device (laptop) to receive it wirelessly, in bLE, these two devices take on different roles
- before a connection exists, once this xiao baord is power on, it is broadcasting small BLE advertising messages. Essentially saying:

```
"Hi.
My name is XIAO-IMU-LIVE.
I'm available for a BLE connection."
```
- on the host end, the python script is doing the opposite. It is scanning for devices that are advertsing "XIAO IMU LIVE" and once it sees it (ie. in range) then it connects
    - therefor the XIAO is a peripheral and it advertises
    - the laptop is a host which is the central and scans/initiates connection
- GATT(Generic ATTribute Profile) is basically how a bLE device organizes the data and functions it it exposes to another device
    - it defines the way that two BLE devices transfer data abck and forth using concepts called "services" and "characteristics"
    - it makes use of a generic data protocol called Attribute Protocol (ATT) whcih is used to store services, characteriscs. 
- Services are used to break data up into logical entities, and contain specific chunks of data called characteristics. A service can have one or more characteristics, and each service distinguishes itself from other services by means of a unique numeric ID called a UUID, which can be 16-bit 
- characteristics encapsulates a single data point (though it may contain an array of related data, such as X/Y/Z values from a 3-axis accelerometer, etc). they are basicaly a defined piece of data or interface
    - Similarly to Services, each Characteristic distinguishes itself via a pre-defined 16-bit or 128-bit UUID
    - Characteristics are the main point that you will interact with your BLE peripheral

So essentially, here is what i am defining:
```
My custom IMU service

    "The purpose of this service is IMU streaming."

        │
        └── My custom data characteristic

             "This is the actual value through which
              IMU packets are delivered."

```
- now in the bLE world, there are different ways to interact with a characteristic. For example, one possuble architecture is READ, but this means the host would have to constantly poll (ie. give me the IMU value)
- instead, I chose the method of NOTIFY where the host basically says " I want updates from this characteristic" and the XIAO board says " Okay, here you go daddy". 

- What is happening on the host side?
    - Bleak is a python bluetooth library I used on the host side
    - this lets the python script to find the XIAO over bluetooth, connect to tit, and listen for the IMU data that the XIAO board is sending

