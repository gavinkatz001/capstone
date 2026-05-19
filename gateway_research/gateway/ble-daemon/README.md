# Fallyx BLE Daemon Configuration

This directory contains the BLE daemon that handles Bluetooth Low Energy communication with Fallyx sensors.

## Quick Start

1. **Compile and Configure the Daemon:**
   ```bash
   cd gateway/ble-daemon
   ./compile_gateway.sh
   ```
   
   The script will prompt you for a location name that will be used for:
   - BLE advertised name (visible to sensors)
   - Location identifier in telemetry data

2. **Run the Daemon:**
   ```bash
   sudo ./fallyx_daemon
   ```

## Configuration Details

### BLE Name Configuration

When you run `./compile_gateway.sh`, you'll be prompted to enter a location for your gateway:

- **Input:** Location name (e.g., "Office", "Kitchen", "Bedroom")
- **BLE Name:** Automatically prefixed with "Fallyx" (e.g., "FallyxOffice")
- **Constraints:** 
  - Total BLE name must be 20 characters or less
  - Only alphanumeric characters allowed
  - Must contain "Fallyx" prefix

### Generated Files

The compilation script creates:

1. **`devices.txt`** - BLE device configuration with your custom name
2. **`gateway_config.txt`** - Configuration file containing:
   ```
   GATEWAY_LOCATION=Your Location
   BLE_NAME=FallyxYourLocation
   ```

### Environment Variables

To ensure the Python gateway uses the same location for telemetry:

```bash
export GATEWAY_NAME="Your Location"
```

Or add to your `.env` file:
```
GATEWAY_NAME=Your Location
```

## Examples

### Example 1: Office Gateway
- **Input:** "Office"
- **BLE Name:** "FallyxOffice" (12 characters)
- **Telemetry Location:** "Office"

### Example 2: Kitchen Gateway
- **Input:** "Kitchen"
- **BLE Name:** "FallyxKitchen" (13 characters)
- **Telemetry Location:** "Kitchen"

### Example 3: Long Location Name
- **Input:** "MasterBedroom"
- **BLE Name:** "FallyxMasterBedroom" (19 characters - OK)
- **Telemetry Location:** "MasterBedroom"

### Example 4: Too Long (Error)
- **Input:** "VeryLongLocationName"
- **BLE Name:** "FallyxVeryLongLocationName" (26 characters - ERROR)
- **Action:** Script will ask for a shorter name

## Files

- **`fallyx_daemon.c`** - Main BLE daemon source code
- **`compile_gateway.sh`** - Interactive compilation and configuration script
- **`devices.txt`** - Generated BLE device configuration (auto-created)
- **`gateway_config.txt`** - Generated gateway configuration (auto-created)
- **`btlib.c/h`** - Bluetooth library
- **`cJSON.c/h`** - JSON parsing library

## Troubleshooting

### Permission Issues
```bash
sudo ./fallyx_daemon
```
The daemon requires root privileges for Bluetooth access.

### Reconfiguration
To change the location/BLE name:
1. Delete `devices.txt` and `gateway_config.txt`
2. Run `./compile_gateway.sh` again
3. Restart the daemon

### Default Configuration
If you run the daemon without using the compile script, it will auto-generate a default configuration with:
- **BLE Name:** "FallyxGateway"
- **Location:** Not configured (uses default)

## Integration with Python Gateway

The Python gateway reads the `GATEWAY_NAME` environment variable to set the location field in ThingsBoard telemetry. Make sure this matches your BLE daemon configuration for consistency.