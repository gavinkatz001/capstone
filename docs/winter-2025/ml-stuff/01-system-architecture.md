# System Architecture

## The Full Pipeline

The fall detection system has three physical components. The ML model lives in component 3 (cloud).

```
[1. Wearable]  ──BLE──>  [2. Gateway]  ──WiFi/Cell──>  [3. Cloud App]
   IMU sensor               Range extender                ML inference
   Records 60s packets       Relays data                   Runs trained model
   accel + gyro              Buffers if offline             Flags anomalies as falls
```

### 1. Wearable Device
- Worn by elderly residents in long-term care (LTC) homes
- Contains an IMU (Inertial Measurement Unit) with a 3-axis accelerometer and 3-axis gyroscope
- Records motion data continuously
- Packages data into **60-second packets** and transmits via BLE
- The 60-second window size is an architectural constant -- it is the fundamental unit of data the entire ML pipeline is built around

### 2. Gateway
- A bedside or room-mounted relay device
- Extends the BLE range of the wearable
- Forwards packets to the cloud over WiFi or cellular
- May buffer packets if connectivity is temporarily lost

### 3. Cloud Application
- Receives 60-second IMU packets from gateways
- Runs the trained PyTorch model on each packet
- Classifies each packet as **fall** or **no-fall** with a confidence score
- If a fall is detected above the operating threshold, triggers an alert to nursing staff
- **No compute constraints** -- the model runs on cloud hardware, so model size and complexity are not limited

## What the ML Pipeline Does

The ML pipeline (everything inside `ml/`) is responsible for:

1. **Training**: Taking public fall detection datasets, preprocessing them into 60-second windows, and training a binary classifier
2. **Evaluation**: Measuring the trained model against the engineering targets from the QFD (TPR >= 95%, FPR <= 10%)
3. **Inference**: Taking a single 60-second packet and outputting a fall probability

The ML pipeline does NOT handle:
- Wearable firmware or BLE communication
- Gateway relay logic
- Cloud application server code, alert routing, or nurse notification
- Real-time streaming -- it processes one 60s packet at a time

## Key Constraint: The 60-Second Window

Everything in the data pipeline is built around the 60-second window:

- All training data is segmented (or padded) to exactly 60 seconds
- At 50 Hz sampling rate, this means every model input is exactly **3000 samples x 6 channels**
- The model receives a tensor of shape `(batch, 6, 3000)` in channels-first format
- A window is labeled as a "fall" if a fall event occurs anywhere within those 60 seconds
- At inference time, the cloud app will feed each incoming 60-second packet directly into the model

This is not an arbitrary choice -- it matches the physical architecture of how the wearable transmits data.

## Engineering Targets

These come from the QFD analysis done in BME 362 (3B term). They are the measurable success criteria:

| Metric | Marginal (minimum acceptable) | Ideal (target) |
|--------|-------------------------------|----------------|
| True Positive Rate (TPR) | >= 90% | >= 95% |
| False Positive Rate (FPR) | <= 15% | <= 10% |
| Fall types detected | >= 3 | >= 4 |
| System response latency | <= 60s | <= 35s |
| Device unit cost | <= $200 CAD | <= $150 CAD |

The ML pipeline is directly responsible for TPR, FPR, and fall type coverage. Response latency is partially ML (inference time) and partially system (transmission + alert routing).
