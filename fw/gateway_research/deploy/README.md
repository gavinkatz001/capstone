# Fallyx Gateway Deployment

This directory contains scripts for deploying the Fallyx Gateway system on a fresh Armbian installation as a stopgap solution for build pipeline limitations.

## Quick Start

1. **Flash Armbian** to your device and complete initial setup
2. **Copy the project** to your device (e.g., via git clone or scp)
3. **Prepare all deployment scripts** (fix permissions and line endings):
   ```bash
   cd /path/to/Fallyx-Gateway/deploy
   # Fix all deployment scripts
   for script in *.sh; do
       chmod +x "$script"
       sed -i 's/\r$//' "$script"
   done
   
   # Fix BLE daemon scripts
   cd ../gateway/ble-daemon
   for script in *.sh; do
       chmod +x "$script"
       sed -i 's/\r$//' "$script"
   done
   cd ../../deploy
   ```
4. **Run the installation script** (without starting services):
   ```bash
   sudo bash ./install_fallyx_gateway.sh --no-start
   ```
5. **Configure your environment** (required before starting services):
   ```bash
   sudo ./configure_env.sh --interactive
   ```
6. **Start the services**:
   ```bash
   sudo ./manage_services.sh start
   ```
7. **Verify installation**:
   ```bash
   sudo ./manage_services.sh status
   ```

## Installation Troubleshooting

If you encounter issues during installation, these are the most common problems and solutions:

### Script Permission/Execution Issues

**Problem:** `sudo: unable to execute ./install_fallyx_gateway.sh: No such file or directory`

**Solution:** The script needs execute permissions and proper line endings:
```bash
chmod +x ./install_fallyx_gateway.sh
sed -i 's/\r$//' ./install_fallyx_gateway.sh
sudo bash ./install_fallyx_gateway.sh --no-start
```

**Problem:** `$'\r': command not found` errors when running any script

**Solution:** The scripts have Windows line endings. Convert them to Unix format:
```bash
# Fix the main installation script
sed -i 's/\r$//' ./install_fallyx_gateway.sh

# Fix other scripts if needed
sed -i 's/\r$//' ./configure_env.sh
sed -i 's/\r$//' ./manage_services.sh
```

Alternatively, if you have `dos2unix` available:
```bash
dos2unix ./install_fallyx_gateway.sh
dos2unix ./configure_env.sh
dos2unix ./manage_services.sh
```

**Note:** The installation script now automatically fixes line endings for all BLE daemon scripts during installation, but you should fix the main deployment scripts before running the installation as shown in step 3 above.

### Alternative Installation Method

If you continue to have issues with the script execution, you can run it directly with bash:
```bash
sudo bash ./install_fallyx_gateway.sh --no-start
```

## Scripts Overview

### `install_fallyx_gateway.sh`
Main installation script that:
- Updates system packages
- Installs system dependencies (Python, Bluetooth, audio libraries, etc.)
- Installs Python packages using `pip install --break-system-packages`
- Compiles the BLE daemon from C source with unique advertised name
- Copies application files to `/opt/fallyx-gateway`
- Sets up systemd services
- Configures Bluetooth and audio systems

**Usage:**
```bash
sudo ./install_fallyx_gateway.sh           # Full installation (starts services immediately)
sudo ./install_fallyx_gateway.sh --no-start # Install without starting services (recommended)
sudo ./install_fallyx_gateway.sh --help    # Show help
```

**Note:** It's recommended to use `--no-start` during initial setup so you can configure the environment variables before starting the services. Services started without proper configuration will fail or run with default values.

### `configure_env.sh`
Environment configuration script that helps set up the `.env` file with actual values.

**Usage:**
```bash
sudo ./configure_env.sh --interactive      # Interactive configuration
sudo ./configure_env.sh --show            # Show current config (masked)
sudo ./configure_env.sh --validate        # Validate required variables
sudo ./configure_env.sh --tb-host myhost.com --tb-token abc123  # Command line config
```

**Required Configuration Variables:**
- `THINGSBOARD_HOST` - Your ThingsBoard server hostname
- `THINGSBOARD_ACCESS_TOKEN` - Device access token from ThingsBoard
- `OPENAI_API_KEY` - OpenAI API key for audio transcription
- `PICOVOICE_ACCESS_KEY` - Picovoice access key for wake word detection
- `EMERGENCY_NUMBER` - Emergency phone number (E.164 format)
- `GATEWAY_BUILDING` - Building name where the gateway is located
- `GATEWAY_ROOM` - Room/area where the gateway is located
- `GATEWAY_NAME` - Human-readable name for this gateway device

### `manage_services.sh`
Service management script for controlling the Fallyx Gateway services.

**Usage:**
```bash
sudo ./manage_services.sh status          # Show service status
sudo ./manage_services.sh start           # Start services
sudo ./manage_services.sh stop            # Stop services
sudo ./manage_services.sh restart         # Restart services
sudo ./manage_services.sh enable          # Enable auto-start
sudo ./manage_services.sh disable         # Disable auto-start
sudo ./manage_services.sh logs gateway    # Show gateway logs
sudo ./manage_services.sh logs ble -f     # Follow BLE daemon logs
```

## Installation Details

### System Dependencies
The installation script installs these system packages:
- **Python**: `python3`, `python3-pip`, `python3-dev`
- **Build tools**: `gcc`, `make`, `build-essential`
- **Bluetooth**: `libbluetooth-dev`, `bluetooth`, `bluez`, `bluez-tools`
- **Audio**: `libasound2-dev`, `portaudio19-dev`, `pulseaudio`, `ffmpeg`
- **System**: `systemd`, `curl`, `wget`, `git`

### Python Dependencies
Installed via `pip install --break-system-packages`:
- `pydantic>=1.9.0` - Data validation
- `numpy>=1.22.0`, `scipy>=1.8.0` - Audio processing
- `paho-mqtt>=1.6.1` - MQTT communication
- `requests>=2.27.1` - HTTP requests
- `psutil>=5.9.0` - System monitoring
- `sounddevice>=0.4.4` - Audio I/O
- `openai>=1.0.0` - OpenAI API client
- `pvporcupine>=3.0.0` - Wake word detection
- `python-dotenv>=0.19.0` - Environment variable loading
- `python-json-logger>=2.0.2` - Structured logging
- `pyyaml>=6.0` - Configuration management

### File Structure After Installation
```
/opt/fallyx-gateway/
├── gateway/                    # Python application code
│   ├── main.py                # Main entry point
│   ├── modules/               # Core modules
│   ├── coordinators/          # Coordination logic
│   ├── adapters/              # External API adapters
│   └── keywords/              # Wake word models
├── .env                       # Environment configuration
└── README.md                  # Documentation

/usr/local/bin/
└── fallyx_daemon              # Compiled BLE daemon

/etc/systemd/system/
├── fallyx-gateway.service     # Main gateway service
└── fallyx-ble-daemon.service  # BLE daemon service
```

### Services
Two systemd services are installed:

1. **fallyx-gateway.service**
   - Runs the main Python gateway application
   - Working directory: `/opt/fallyx-gateway`
   - Command: `python3 -m gateway.main`
   - Auto-restart on failure

2. **fallyx-ble-daemon.service**
   - Runs the compiled BLE daemon
   - Depends on fallyx-gateway.service
   - Command: `/usr/local/bin/fallyx_daemon`
   - Auto-restart on failure

### BLE Advertised Name
The installation script automatically creates a unique BLE advertised name to prevent conflicts when multiple gateways are deployed:

- **Default**: `GatewayFallyx`
- **With USER_FIRST_NAME**: `GatewayFallyx-{USER_FIRST_NAME}` (e.g., `GatewayFallyx-John`)
- **With GATEWAY_ID**: `GatewayFallyx-{suffix}` (e.g., `GatewayFallyx-001`)

The name is automatically injected into the C code before compilation, ensuring each gateway broadcasts a unique identifier. Names are limited to 20 characters for BLE compatibility.

## Environment Configuration

The application uses environment variables loaded from `/opt/fallyx-gateway/.env`. Here are the key variables:

### ThingsBoard Configuration
```bash
# Hostname or IP address of your ThingsBoard server
THINGSBOARD_HOST=localhost
# MQTT port for ThingsBoard (default is 1883)
THINGSBOARD_PORT=1883
# Access token for the Fallyx Gateway device in ThingsBoard
THINGSBOARD_ACCESS_TOKEN=YOUR_THINGSBOARD_DEVICE_ACCESS_TOKEN
# MQTT Quality of Service level (0, 1, or 2)
THINGSBOARD_QOS=1
```

### Audio Device Configuration
```bash
# Optional: Part of the name of your preferred audio input device (microphone)
# If blank, the system's default input device will be used
# Example: AUDIO_INPUT_DEVICE_NAME_CONTAINS=USB PnP Sound Device
AUDIO_INPUT_DEVICE_NAME_CONTAINS=
# Optional: Part of the name of your preferred audio output device (speaker)
# If blank, the system's default output device will be used
# Example: AUDIO_OUTPUT_DEVICE_NAME_CONTAINS=USB PnP Sound Device
AUDIO_OUTPUT_DEVICE_NAME_CONTAINS=
```

### OpenAI Configuration
```bash
# Your OpenAI API key for audio transcription and intent interpretation
OPENAI_API_KEY=YOUR_OPENAI_API_KEY
```

### Voice Assistant Configuration
```bash
# Your Picovoice access key (required for pvporcupine)
PICOVOICE_ACCESS_KEY=YOUR_PICOVOICE_ACCESS_KEY
# Configurable wake word for voice assistant
WAKE_WORD=hey fallyx
# Boolean flag to enable/disable the voice assistant feature
VOICE_ASSISTANT_ENABLED=true
# Backend API endpoint for chat functionality
CHAT_API_BASE_URL=https://your-backend.com/v1/chat
# Unique identifier for this gateway device
GATEWAY_ID=fallyx-gateway-001
```

### Emergency Contact Information
```bash
# Emergency phone number to call (e.g., Twilio E.164 format)
EMERGENCY_NUMBER=+14166299094
# Building name where the gateway is located (used for emergency location info)
GATEWAY_BUILDING=Main Building
# Room/area where the gateway is located (used for emergency location info)
GATEWAY_ROOM=Living Room
# Name of the emergency contact person (for personalized messages)
EMERGENCY_CONTACT_NAME=SRR
```

### Audio Sample Rate Configuration
```bash
# Default sample rate for all audio recording and processing (Hz)
# This should be 16000Hz for optimal compatibility with speech recognition
AUDIO_DEFAULT_SAMPLE_RATE=16000
# Sample rate for audio playback compatibility (Hz)
AUDIO_PLAYBACK_SAMPLE_RATE=16000
```

### Gateway Identification
```bash
# Human-readable name for this gateway device (used in data manager)
GATEWAY_NAME=FallyxGateway
```

### General Application Settings
```bash
# Log level for the application (e.g., DEBUG, INFO, WARNING, ERROR)
# (Currently hardcoded to INFO in main.py, but good to make configurable)
# LOG_LEVEL=INFO
```

## Troubleshooting

### Check Service Status
```bash
sudo systemctl status fallyx-gateway.service
sudo systemctl status fallyx-ble-daemon.service
```

### View Logs
```bash
# Real-time logs
sudo journalctl -u fallyx-gateway.service -f
sudo journalctl -u fallyx-ble-daemon.service -f

# Recent logs
sudo journalctl -u fallyx-gateway.service --since "1 hour ago"
```

### Common Issues

1. **Service fails to start**
   - Check configuration: `sudo ./configure_env.sh --validate`
   - Check logs: `sudo journalctl -u fallyx-gateway.service`
   - Verify .env file exists: `ls -la /opt/fallyx-gateway/.env`

2. **BLE daemon fails**
   - Check if Bluetooth is enabled: `sudo systemctl status bluetooth`
   - Verify binary exists: `ls -la /usr/local/bin/fallyx_daemon`
   - Check permissions: `sudo chmod +x /usr/local/bin/fallyx_daemon`

3. **Audio issues**
   - List audio devices: `aplay -l` and `arecord -l`
   - Check PulseAudio: `pulseaudio --check`
   - Test audio: `speaker-test -t sine -f 1000 -l 1`

4. **Python dependency issues**
   - Reinstall packages: `sudo pip3 install --break-system-packages -r /opt/fallyx-gateway/gateway/requirements.txt --force-reinstall`

### Manual Service Control
```bash
# Stop services
sudo systemctl stop fallyx-ble-daemon.service
sudo systemctl stop fallyx-gateway.service

# Start services
sudo systemctl start fallyx-gateway.service
sudo systemctl start fallyx-ble-daemon.service

# Restart services
sudo systemctl restart fallyx-gateway.service
sudo systemctl restart fallyx-ble-daemon.service
```

## Development and Updates

### Updating the Application
1. Stop services: `sudo ./manage_services.sh stop`
2. Update code in `/opt/fallyx-gateway/`
3. Recompile BLE daemon if needed:
   ```bash
   cd /opt/fallyx-gateway/gateway/ble-daemon
   ./compile_gateway.sh
   sudo cp fallyx_daemon /usr/local/bin/
   ```
4. Start services: `sudo ./manage_services.sh start`

### Adding New Dependencies
1. Update `gateway/requirements.txt`
2. Install: `sudo pip3 install --break-system-packages -r /opt/fallyx-gateway/gateway/requirements.txt`
3. Restart services: `sudo ./manage_services.sh restart`

## Security Considerations

- The `.env` file contains sensitive API keys and is set to mode 600 (owner read/write only)
- Services run as root due to Bluetooth and audio hardware requirements
- Consider using a dedicated user account for production deployments
- Regularly update system packages: `sudo apt update && sudo apt upgrade`

## Support

For issues with the deployment scripts or configuration, check:
1. Service logs: `sudo ./manage_services.sh logs gateway`
2. System logs: `sudo journalctl --since "1 hour ago"`
3. Configuration validation: `sudo ./configure_env.sh --validate`