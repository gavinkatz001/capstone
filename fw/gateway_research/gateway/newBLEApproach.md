Summary of the New BLE Architecture
The core problem was that the D-Bus communication layer, which all standard Python BLE libraries use to talk to the system's Bluetooth daemon, was too slow to handle the rapid burst of 15 packets from the sensor.
The new solution completely bypasses this bottleneck by adopting a two-tiered, hybrid C/Python architecture. We have separated the real-time "Data Plane" from the complex "Control Plane".
Here is the high-level data flow from sensor to application:
1. The C Daemon (Data Plane)
Responsibility: Its only job is to handle the high-speed, real-time BLE communication.
How it Works: A small, lightweight C program uses the libbluetooth system library to talk directly to the Linux kernel's Bluetooth stack. It acts as the BLE peripheral, advertises as "Fallyx", and accepts connections.
Action: When a sensor connects and sends its burst of 15 packets, the C daemon receives the raw bytes of each packet instantly, without any D-Bus overhead. As soon as a packet is received, the C daemon immediately writes those raw bytes into a Unix Domain Socket (/tmp/fallyx_ble.sock).
2. The Unix Domain Socket (The Bridge)
Responsibility: To act as an extremely fast, low-latency pipe between the C daemon and the Python application.
How it Works: This is a standard, in-memory communication channel provided by the Linux kernel. It is measured in Gigabytes/sec and will never be a bottleneck for our data rate.
3. The Python BLEModule (Control Plane)
New Responsibility: The BLEModule no longer handles BLE at all. It has been completely rewritten to become a simple socket server.
How it Works: It opens and listens on the server end of the same Unix Domain Socket (/tmp/fallyx_ble.sock).
Action: When the C daemon forwards data into the socket, the BLEModule receives the raw bytes. It immediately puts this data onto the same internal queue.Queue that we've always had, handing it off to the existing data processing worker thread.
In Short:
We are using C for what it's best at: raw speed and low-level system interaction. We are using Python for what it's best at: high-level application logic, data processing, and cloud connectivity.
The C daemon acts as a highly efficient "shock absorber," effortlessly handling the high-speed data burst and feeding it smoothly into the Python application through a high-capacity pipe, allowing the main application to process it at its own pace without data loss.