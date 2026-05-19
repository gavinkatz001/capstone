
import asyncio
import json
import os
from datetime import datetime
from bleak import BleakClient, BleakScanner

# ========== Configuration ==========
DEVICE_NAME = "FALLYX_XIAO_SENSOR"
UART_RX_UUID = "6e400003-b5a3-f393-e0a9-e50e24dcca9e"  # Bluefruit BLE UART RX UUID
SAMPLES_PER_BATCH = 3000
OUTPUT_DIR = "final_data"

os.makedirs(OUTPUT_DIR, exist_ok=True)

# ========== Global Data ==========
data_batch = []

# ========== BLE Notification Callback ==========
rx_buffer = ""  # Global buffer

def handle_rx(_, data: bytearray):
    global rx_buffer, data_batch

    rx_buffer += data.decode('utf-8')

    while "\n" in rx_buffer:
        line, rx_buffer = rx_buffer.split("\n", 1)
        try:
            json_obj = json.loads(line.strip())
            data_batch.append(json_obj)

            if len(data_batch) >= SAMPLES_PER_BATCH:
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                file_path = os.path.join(OUTPUT_DIR, f"batch_{timestamp}.json")
                with open(file_path, 'w') as f:
                    json.dump({
                        "metadata": {
                            "device": DEVICE_NAME,
                            "timestamp": timestamp,
                            "samples": len(data_batch)
                        },
                        "data": data_batch
                    }, f, indent=2)

                print(f"✅ Saved {len(data_batch)} samples to '{file_path}'")
                data_batch.clear()

        except json.JSONDecodeError:
            print("⚠️ Invalid JSON fragment (ignored):", line.strip())





# ========== Main BLE Task ==========
async def main():
    print("🔍 Scanning for BLE devices...")

    device = await BleakScanner.find_device_by_filter(
        lambda d, _: d.name is not None and DEVICE_NAME in d.name
    )

    if not device:
        print("❌ Device not found. Please check if it is powered on and advertising.")
        return

    print(f"🔗 Connecting to {device.name} ({device.address})...")
    async with BleakClient(device) as client:
        print("✅ Connected. Subscribing to UART RX notifications...")

        await client.start_notify(UART_RX_UUID, handle_rx)
        print("📡 Listening for JSON data. Waiting for batches of 3000 samples...")

        try:
            while True:
                await asyncio.sleep(1)  # Keep the loop alive
        except asyncio.CancelledError:
            print("🛑 Logging cancelled.")

# ========== Entry Point ==========
if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\n👋 Logger stopped by user.")
