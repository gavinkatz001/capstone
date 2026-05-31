#!/bin/sh

# Fallyx Gateway BLE Daemon Compilation Script
# This script compiles the BLE daemon and configures the gateway location

echo "=== Fallyx Gateway BLE Daemon Setup ==="
echo ""

# Function to validate BLE name length
validate_ble_name() {
    local name="$1"
    local length=${#name}
    if [ $length -gt 20 ]; then
        return 1
    fi
    return 0
}

# Prompt for gateway location
echo "Gateway Location Configuration:"
echo "This will set both the BLE advertised name and the location for telemetry data."
echo ""
echo "BLE Name Rules:"
echo "- Must contain 'Fallyx' (will be automatically prepended)"
echo "- Total name must be 20 characters or less"
echo "- This is the name that sensors will see when connecting"
echo ""

while true; do
    printf "Enter location/suffix for this gateway (e.g., 'Office', 'Kitchen', 'Bedroom'): "
    read location_input
    
    # Remove any spaces and special characters for BLE name
    location_clean=$(echo "$location_input" | tr -d ' ' | tr -cd '[:alnum:]')
    
    if [ -z "$location_clean" ]; then
        echo "Error: Location cannot be empty. Please try again."
        continue
    fi
    
    # Create BLE name with Fallyx prefix
    ble_name="Fallyx${location_clean}"
    
    # Validate length
    if validate_ble_name "$ble_name"; then
        echo ""
        echo "Configuration Summary:"
        echo "  Location for telemetry: $location_input"
        echo "  BLE advertised name: $ble_name (${#ble_name} characters)"
        echo ""
        printf "Is this correct? (y/n): "
        read confirm
        if [ "$confirm" = "y" ] || [ "$confirm" = "Y" ]; then
            break
        fi
    else
        echo "Error: BLE name '$ble_name' is ${#ble_name} characters (max 20 allowed)."
        echo "Please use a shorter location name."
        echo ""
    fi
done

# Detect the actual Bluetooth MAC address of this device
echo ""
echo "Detecting Bluetooth MAC address..."

ble_mac_address=""

# Try multiple methods to get the Bluetooth MAC address
if command -v hciconfig >/dev/null 2>&1; then
    # Method 1: Use hciconfig (most reliable for Bluetooth)
    ble_mac_address=$(hciconfig hci0 2>/dev/null | grep "BD Address" | awk '{print $3}' | tr '[:lower:]' '[:upper:]')
fi

if [ -z "$ble_mac_address" ] && command -v bluetoothctl >/dev/null 2>&1; then
    # Method 2: Use bluetoothctl
    ble_mac_address=$(bluetoothctl show 2>/dev/null | grep "Controller" | awk '{print $2}' | tr '[:lower:]' '[:upper:]')
fi

if [ -z "$ble_mac_address" ] && [ -f /sys/class/bluetooth/hci0/address ]; then
    # Method 3: Read from sysfs
    ble_mac_address=$(cat /sys/class/bluetooth/hci0/address 2>/dev/null | tr '[:lower:]' '[:upper:]')
fi

# Fallback to a default MAC if detection fails
if [ -z "$ble_mac_address" ]; then
    echo "Warning: Could not detect Bluetooth MAC address, using placeholder"
    ble_mac_address="00:00:00:00:00:00"
else
    echo "Detected Bluetooth MAC address: $ble_mac_address"
fi

# Generate devices.txt with the custom BLE name and actual MAC address
echo ""
echo "Generating devices.txt with BLE name: $ble_name and MAC: $ble_mac_address"
cat > devices.txt << EOF
DEVICE=$ble_name  TYPE=MESH  NODE=1  CHANNEL=1 ADDRESS=$ble_mac_address
  PRIMARY_SERVICE = 1800
    LECHAR = Device name  PERMIT=06  SIZE=16  UUID=2A00   ; index 0
  PRIMARY_SERVICE = 55535343fe7d4ae58fa99fafd205e455
    LECHAR = DataTx  PERMIT=06  SIZE=16 UUID=49535343884143f4a8d4ecbe34729bb3        ; index 1
    LECHAR = DataRx  PERMIT=06  SIZE=16 UUID=495353431e4d4bd9ba6123c647249616        ; index 2
EOF

# Create a configuration file for the Python gateway to read
echo ""
echo "Creating gateway configuration..."
cat > gateway_config.txt << EOF
GATEWAY_LOCATION=$location_input
BLE_NAME=$ble_name
EOF

echo "Configuration saved to gateway_config.txt"
echo ""

# Compile the fallyx_daemon.c with btlib and cJSON for BLE functionality
echo "Compiling fallyx_daemon.c with cJSON and threading support..."
gcc fallyx_daemon.c btlib.c cJSON.c -o fallyx_daemon -lm -lpthread

if [ $? -eq 0 ]; then
    echo "Compilation successful! Executable: fallyx_daemon"
    echo ""
    echo "=== Setup Complete ==="
    echo "BLE Name: $ble_name"
    echo "Location: $location_input"
    echo ""
    echo "To run the daemon:"
    echo "  sudo ./fallyx_daemon"
    echo ""
    echo "Note: The daemon requires sudo privileges for Bluetooth access"
    echo ""
    echo "To update the Python gateway with this location, set the environment variable:"
    echo "  export GATEWAY_NAME=\"$location_input\""
    echo "Or add to your .env file:"
    echo "  GATEWAY_NAME=$location_input"
else
    echo "Compilation failed!"
    exit 1
fi

echo ""
echo "Legacy gateway.c compilation (for reference)..."
gcc gateway.c btlib.c -o gateway

if [ $? -eq 0 ]; then
    echo "Legacy gateway compilation successful! Executable: gateway"
else
    echo "Legacy gateway compilation failed!"
fi