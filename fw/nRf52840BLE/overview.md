# Milestone 1: XIAO nRF52840 Sense One Minute IMU Logger and BLE Transfer Firmware Roadmap
```
author: dhruv gogna
date: sept21 2026

I will walk through this document with the team # TODO: remove after completed
```

## What is an IMU (TLDR: change in capacitance due to inertia)
- small bundle of MEMS sensors plus firmware that turns raw electrical signals into usable motion data
- accelerometer measures linear acceleration (including gravity) on 3 axes
    - Inside, a tiny mass is suspended on springs between capacitor plates. When the device accelerates, inertia makes the mass lag behind the frame, changing the gap between plates and thus the capacitance
    - That capacitance change is converted to a voltage, then digitized and so on etc
- Gyroscope measures angular velocity (rotation rate) on 3 axes, using the Coriolis effect.
    - the MEMS in a gyroscope contains a microscopic mass that is made to vibrate back and forth. Now imagine the chip (IMU chip) starts rotating while the mass is vibrating ; due to the coriolis effect, the vibrating mass experiences a small force in the other direction (experiences a Coriolis force perpendicular to both its vibration direction and the rotation axis). this motion is converted into angular velocity usually shown in degrees/sec
  


## What is a microcontroller?
- this is considered as bare metal programming. You flash firmware inside of itt (. like flashining an ino file inside the arduinos. There is no operating system, and memory is usually like 2kb -256kb). Therefore, things like arduino, and our xiao board are microcontrollers. when powered on, they immediatley execute one single program in a continuous predictabe loop (ie. flashing an LED, spinning a motor et
- hardware like raspberry pis / (or even laptops tbh) run a full operating stem . They can multirtask, run software applications, connect to monitors/keybpards etc. it has GBs worth of RAM 


- for our project, the XIAO Sense combines an nRF52840 MCU with onboard BLE, plus a separate LSM6DS3TR-C 6-axis IMU


## Requirements, Scope, Architecture, and Implementation Plan 

- Target board: Seeed Studio XIAO nRF52840 Sense
- Current hardware state: Board connected to a computer through USB-C + Arduino IDE installed 
- Intended use: gathering motion data (ie. fall detection) which will be sent to a host for further computation. For milestone 1, this consists of flashing .ino firmware onboard the XIAO seeed studio board


### Milestone 1 Goal (defn of "Done"):
- using the seeed studio xiao nRF52840 sense board, collect position data using the boards onboad IMU cihp. 
- Upon power up, the Seeed Studio XIAO nRF52840 Sense shall begin BLE advertsising immediatley. Once the host connects as BLE central, then the board will begin to acquire time ordered 6-axis IMU measurements from its onboard LSM6DS3TR-C IMU for around a 60 second acquisition window. After one minute, it will send over the data in packets . 

- As of monday september 21st:The XIAO samples the onboard IMU, packages each sample, sends it over BLE, and the host receives it, checks continuity, converts it, and stores roughly one-minute recordings.
---
</br>
</br>
</br>

### Scope

The architecture I listed should hopefully cover everything from the onboard IMU through the point at which bytes are transmitted by BLE:

```text
Physical motion
    ↓
LSM6DS3TR-C sensing elements
    ↓
LSM6DS3TR-C sampling, timestamp and FIFO
    ↓
Internal I²C connection
    ↓
nRF52840 firmware
    ↓
Capture validation
    ↓
Binary session record
    ↓
BLE GATT service
    ↓
BLE notification transmitted to central host
```



#### Simplified board architecture

```text
                         XIAO nRF52840 Sense

┌────────────────────────────────────────────────────────────────┐
│                                                                │
│  LSM6DS3TR-C                         nRF52840                  │
│                                                                │
│  Accelerometer ─┐                    ┌─────────────────────┐   │
│                 ├─ internal I²C ────>│ IMU driver          │   │
│  Gyroscope ─────┘                    │ FIFO service        │   │
│                                      │ capture engine      │   │
│  INT1 ──────────────────────────────>│ interrupt input     │   │
│                                      │                     │   │
│  Power <─────────────────────────────│ IMU power control   │   │
│                                      │                     │   │
│                                      │ sample storage      │   │
│                                      │ BLE GATT server     │   │
│                                      └──────────┬──────────┘   │
│                                                 │              │
│                                           BLE radio            │
│                                                 │              │
│                                          onboard antenna       │
└─────────────────────────────────────────────────┼──────────────┘
                                                  │
                                                  ▼
                                       PC / Raspberry Pi / SBC
```

---



## In depth Firmware stuff 
- (personal notes, not important for rest of team to know (yet*))
#### GPIO Pin Allocation

##### Internal pin map

| Function            |          nRF52840 physical pin | mbed-core definition/index             | Exposed on XIAO header? | Project use                              |
| ------------------- | -----------------------------: | -------------------------------------- | ----------------------- | ---------------------------------------- |
| External I²C SDA    |                          P0.04 | `PIN_WIRE_SDA`, Arduino D4             | Yes                     | Not used for onboard IMU                 |
| External I²C SCL    |                          P0.05 | `PIN_WIRE_SCL`, Arduino D5             | Yes                     | Not used for onboard IMU                 |
| Onboard IMU power   |                          P1.08 | `PIN_LSM6DS3TR_C_POWER`, core index 15 | No                      | Powers the LSM6DS3TR-C                   |
| Onboard IMU I²C SCL |                          P0.27 | `PIN_WIRE_SCL1`, core index 16         | No                      | Internal `Wire1` clock                   |
| Onboard IMU I²C SDA |                          P0.07 | `PIN_WIRE_SDA1`, core index 17         | No                      | Internal `Wire1` data                    |
| Onboard IMU INT1    |                          P0.11 | `PIN_LSM6DS3TR_C_INT1`, core index 18  | No                      | FIFO watermark/overrun interrupt         |
| BLE RF path         | Internal radio/antenna network | Managed by BLE stack/core              | No                      | BLE advertising and transfer             |
| USB                 |           USB-C D+/D− and VBUS | Managed by core/bootloader             | Connector               | Programming, diagnostics and bench power |

The official mbed board variant defines external `Wire` on D4/D5 and the onboard-sensor bus as `Wire1`; its variant mapping resolves the internal bus to P0.07/P0.27, IMU power to P1.08 and IMU interrupt to P0.11. The Seeed wiki separately identifies P1.08 as IMU power and P0.11 as IMU interrupt 

- The labels `SDA` and `SCL` on the board edge refer to the user-accessible `Wire` bus. The onboard IMU is connected to the second bus represented by `Wire1`.
- The current Seeed LSM6DS3 library contains a target-specific definition that redirects its internal `Wire` reference to `Wire1` when compiling for the XIAO nRF52840 Sense.


#### IMU address and identity:
- The schematic ties the IMU’s `SDO/SA0` address-selection pin low and holds `CS` in the I²C configuration. ST specifies that an SA0-low device uses binary address `1101010`, which is hexadecimal `0x6A`. ST also specifies that the read-only `WHO_AM_I` register at address `0x0F` returns `0x6A`.

These two values happen to be equal but represent different things:

```text
I²C device address:       0x6A
WHO_AM_I register value:  0x6A
WHO_AM_I register address: 0x0F
```

</br>
</br>
</br>




#### Sensor capabilities relevant to this design

The LSM6DS3TR-C provides:

* three-axis acceleration;
* three-axis angular rate;
* accelerometer ranges of ±2, ±4, ±8 and ±16 g;
* gyroscope ranges of ±125, ±250, ±500, ±1000 and ±2000 degrees per second;
* multiple output data rates;
* a hardware FIFO of up to 4 KB;
* configurable FIFO threshold and overrun reporting;
* interrupt routing to INT1;
* a 24 bit timestamp counter;




##### Interrupt Strategy

The INT1 interrupt service routine shall do only this (unless I can find anybevidence indicates it is preferefed otherwise):

```cpp
void onImuInterrupt() {
    imuInterruptPending = true;
}
```

It shall not:
- access I²C;
- drain the FIFO;
- write storage;
- update BLE;
- print to Serial;
- calculate CRC;
- allocate memory.

The main loop sees `imuInterruptPending`, clears it atomically and drains the FIFO.


##### Defining exactly “one minute”

“Exactly 6240 samples” and “exactly 60 seconds” are not necessarily the same requirement because a physical oscillator has tolerance.


- A forum thread from someone building almost exactly our architecture (IMU sampling + BLE, both active at once) reports that once ArduinoBLE is added into the loop, millis()/delayMicroseconds()-based timing becomes visibly erratic — sample timing that was clean in isolation degrades once the BLE stack is polling<cite index="22-1">. This is the single biggest risk to your "fixed-rate timer, independent of connection state" requirement, and it's fixable, but only if we build for it from the start rather than discovering it after integration. Two mitigations, both baked into the Stage 3 sketch below:
    - Use a hardware timer interrupt (mbed's Ticker) purely to flag "sample due," and do the actual I2C read outside the interrupt, in the main loop — never block inside the ISR.
Timestamp every sample with micros() at the moment it's actually read, rather than assuming perfect 50Hz spacing. Since all real fall-detection math happens host-side anyway, the host can handle small timing irregularities in software

Binary recording format:

- `RawImuSampleV1` is exactly 16 bytes:

| Byte offset | Size | Field             | Type            |
| ----------: | ---: | ----------------- | --------------- |
|           0 |    4 | `sensor_time_tag` | unsigned 32-bit |
|           4 |    2 | `accel_x_raw`     | signed 16-bit   |
|           6 |    2 | `accel_y_raw`     | signed 16-bit   |
|           8 |    2 | `accel_z_raw`     | signed 16-bit   |
|          10 |    2 | `gyro_x_raw`      | signed 16-bit   |
|          12 |    2 | `gyro_y_raw`      | signed 16-bit   |
|          14 |    2 | `gyro_z_raw`      | signed 16-bit   |



#### BLE architecture

```text
XIAO nRF52840 Sense = peripheral / GATT server
PC or Raspberry Pi  = central / GATT client
```

ArduinoBLE supports peripheral and central roles. In this design, the XIAO advertises a custom service; the central connects, writes commands, reads metadata and subscribes to notifications. ArduinoBLE describes notifications as suitable for sensor data, while indications include an ATT-level confirmation.
- The XIAO does not stream IMU packets until a host subscribes to the BLE data characteristic. On each new subscription, the firmware clears the IMU FIFO, resets SampleIndex to zero, discards the first FIFO pattern after the reset, and then begins streaming.
