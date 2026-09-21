#include <ArduinoBLE.h>
#include <LSM6DS3.h>
#include <Wire.h>

// ------------------------------------------------------------
// XIAO nRF52840 Sense: continuous IMU -> BLE live stream
// Packet = exactly 20 bytes:
//   uint32 timestamp_ticks_25us
//   int16  ax, ay, az
//   int16  gx, gy, gz
//   uint32 sample_index
// ------------------------------------------------------------

static const char* DEVICE_NAME = "XIAO-IMU-LIVE";
static const char* SERVICE_UUID = "7c8c1a10-2f31-4c84-a991-53c47aa10000";
static const char* DATA_UUID    = "7c8c1a10-2f31-4c84-a991-53c47aa10001";

static constexpr uint8_t PACKET_BYTES = 20;
static constexpr uint16_t FIFO_WORDS_PER_PATTERN = 12; // G(3) + A(3) + DS3(3) + timestamp(3)

LSM6DS3 imu(I2C_MODE, 0x6A);
BLEService imuService(SERVICE_UUID);
BLECharacteristic dataCharacteristic(DATA_UUID, BLENotify, PACKET_BYTES, true);

bool imuReady = false;
bool streaming = false;
bool discardFirstPattern = false;
uint32_t sampleIndex = 0;
uint32_t fifoOverruns = 0;

static void putU32LE(uint8_t* dst, uint32_t value) {
  dst[0] = (uint8_t)(value & 0xFF);
  dst[1] = (uint8_t)((value >> 8) & 0xFF);
  dst[2] = (uint8_t)((value >> 16) & 0xFF);
  dst[3] = (uint8_t)((value >> 24) & 0xFF);
}

static void putI16LE(uint8_t* dst, int16_t value) {
  uint16_t u = (uint16_t)value;
  dst[0] = (uint8_t)(u & 0xFF);
  dst[1] = (uint8_t)((u >> 8) & 0xFF);
}

static void configureImuSettings() {
  // Gyroscope: +/-500 dps, 104 Hz.
  imu.settings.gyroEnabled = 1;
  imu.settings.gyroRange = 500;
  imu.settings.gyroSampleRate = 104;
  imu.settings.gyroBandWidth = 100;
  imu.settings.gyroFifoEnabled = 1;
  imu.settings.gyroFifoDecimation = 1;

  // Accelerometer: +/-4 g, 104 Hz.
  imu.settings.accelEnabled = 1;
  imu.settings.accelRange = 4;
  imu.settings.accelSampleRate = 104;
  imu.settings.accelBandWidth = 100;
  imu.settings.accelFifoEnabled = 1;
  imu.settings.accelFifoDecimation = 1;

  imu.settings.tempEnabled = 0;
  imu.settings.commMode = 1;

  // Hardware timestamp in FIFO, 25 us/tick.
  imu.settings.timestampEnabled = 1;
  imu.settings.timestampFifoEnabled = 1;
  imu.settings.timestampResolution = 1;

  // Seeed's FIFO API names this setting 100 Hz; it maps to the
  // LSM6DS3TR-C FIFO ODR register setting used with 104 Hz sensors.
  // Threshold is four complete FIFO patterns.
  imu.settings.fifoThreshold = FIFO_WORDS_PER_PATTERN * 4;
  imu.settings.fifoSampleRate = 100;
  imu.settings.fifoModeWord = 6; // continuous mode
}

static bool beginImu() {
  configureImuSettings();

  // Seeed's library handles the XIAO Sense-specific Wire1 routing and
  // IMU power-enable sequence internally.
  if (imu.begin() != 0) {
    Serial.println("ERROR: IMU begin() failed");
    return false;
  }

  uint8_t who = 0;
  if (imu.readRegister(&who, LSM6DS3_ACC_GYRO_WHO_AM_I_REG) != 0 || who != 0x6A) {
    Serial.print("ERROR: WHO_AM_I expected 0x6A, got 0x");
    Serial.println(who, HEX);
    return false;
  }

  imu.fifoBegin();
  imu.fifoClear();

  Serial.println("IMU ready: LSM6DS3TR-C @ 0x6A");
  Serial.println("Accel: +/-4 g, 0.122 mg/LSB");
  Serial.println("Gyro:  +/-500 dps, 17.50 mdps/LSB");
  Serial.println("ODR:   nominal 104 Hz");
  return true;
}

static bool beginBle() {
  if (!BLE.begin()) {
    Serial.println("ERROR: BLE.begin() failed");
    return false;
  }

  BLE.setLocalName(DEVICE_NAME);
  BLE.setDeviceName(DEVICE_NAME);
  BLE.setAdvertisedService(imuService);

  imuService.addCharacteristic(dataCharacteristic);
  BLE.addService(imuService);

  uint8_t zeroPacket[PACKET_BYTES] = {0};
  dataCharacteristic.writeValue(zeroPacket, PACKET_BYTES);

  BLE.advertise();
  Serial.print("BLE advertising as ");
  Serial.println(DEVICE_NAME);
  return true;
}

static void startStreaming() {
  // fifoClear() switches through BYPASS and restores continuous mode.
  // Discard the first complete pattern after that transition.
  imu.fifoClear();
  sampleIndex = 0;
  fifoOverruns = 0;
  discardFirstPattern = true;
  streaming = true;
  Serial.println("Host subscribed: live IMU streaming started.");
}

static void stopStreaming() {
  streaming = false;
  Serial.print("Streaming stopped. FIFO overruns observed: ");
  Serial.println(fifoOverruns);
}

static void sendOneFifoPattern() {
  // FIFO order follows Seeed's official timestamp-enabled FifoExample:
  // gyro XYZ -> accel XYZ -> third FIFO dataset -> timestamp dataset.
  int16_t gx = imu.fifoRead();
  int16_t gy = imu.fifoRead();
  int16_t gz = imu.fifoRead();

  int16_t ax = imu.fifoRead();
  int16_t ay = imu.fifoRead();
  int16_t az = imu.fifoRead();

  // Consume the third FIFO dataset. We do not transmit it.
  (void)imu.fifoTimestamp();

  // Timestamp dataset: lower 24 bits are the sensor timestamp.
  uint32_t timestampTicks = imu.fifoTimestamp() & 0x00FFFFFFUL;

  if (discardFirstPattern) {
    discardFirstPattern = false;
    return;
  }

  uint8_t packet[PACKET_BYTES];
  putU32LE(packet + 0, timestampTicks);
  putI16LE(packet + 4, ax);
  putI16LE(packet + 6, ay);
  putI16LE(packet + 8, az);
  putI16LE(packet + 10, gx);
  putI16LE(packet + 12, gy);
  putI16LE(packet + 14, gz);
  putU32LE(packet + 16, sampleIndex);

  // Notification: no per-sample acknowledgement. The IMU itself naturally
  // paces this at ~104 samples/s instead of dumping thousands at once.
  dataCharacteristic.writeValue(packet, PACKET_BYTES);
  ++sampleIndex;
}

static void serviceImuStream() {
  // Limit each pass so BLE.poll() keeps getting CPU time even if FIFO has built up.
  static constexpr uint8_t MAX_PATTERNS_PER_LOOP = 4;
  uint8_t sent = 0;

  while (sent < MAX_PATTERNS_PER_LOOP) {
    uint16_t status = imu.fifoGetStatus();

    // FIFO_STATUS2: OVR=bit 6 -> combined mask 0x4000.
    if (status & 0x4000) {
      ++fifoOverruns;
      Serial.println("WARNING: IMU FIFO overrun; clearing FIFO.");
      imu.fifoClear();
      discardFirstPattern = true;
      return;
    }

    uint16_t unreadWords = status & 0x0FFF;
    if (unreadWords < FIFO_WORDS_PER_PATTERN) {
      return;
    }

    sendOneFifoPattern();
    ++sent;
    BLE.poll();
  }
}

void setup() {
  Serial.begin(115200);

  // Do not wait for Serial; the board must work headless.
  if (!beginBle()) {
    while (true) { delay(1000); }
  }

  imuReady = beginImu();
  if (!imuReady) {
    Serial.println("IMU failed; BLE remains available but no samples will stream.");
  }

  Serial.println("Ready. Run live_imu_logger.py on the host.");
}

void loop() {
  BLE.poll();

  const bool subscribed = dataCharacteristic.subscribed();

  if (subscribed && !streaming && imuReady) {
    startStreaming();
  } else if (!subscribed && streaming) {
    stopStreaming();
  }

  if (streaming) {
    serviceImuStream();
  }
}
