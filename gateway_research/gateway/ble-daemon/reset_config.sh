#!/bin/bash

# Reset Fallyx BLE Daemon Configuration
# This script removes the devices.txt file so it will be auto-regenerated
# with the correct local Bluetooth adapter address on next run

echo "Resetting Fallyx BLE daemon configuration..."

if [ -f "gateway/ble-daemon/devices.txt" ]; then
    rm gateway/ble-daemon/devices.txt
    echo "Removed existing devices.txt file"
else
    echo "No devices.txt file found"
fi

echo "Configuration reset complete."
echo "The daemon will auto-generate a new devices.txt file with the correct"
echo "local Bluetooth adapter address when it starts next time."