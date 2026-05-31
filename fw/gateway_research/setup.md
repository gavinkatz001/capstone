# Fallyx Gateway Setup Guide for Orange Pi Zero 2W

This guide provides complete step-by-step instructions for setting up the Fallyx Gateway on an Orange Pi Zero 2W device from scratch.

## Prerequisites

Before starting, ensure you have:
- Orange Pi Zero 2W device
- MicroSD card (16GB or larger, Class 10 recommended)
- SD card reader/writer
- Computer with internet connection
- USB speaker and microphone (for voice assistant functionality)
- WiFi network credentials
- ThingsBoard server access
- OpenAI API account
- Picovoice account

## Step 1: Flash Armbian Image

### 1.1 Download Armbian Image
1. Visit the official Armbian download page: https://dl.armbian.com/orangepizero2w/Bookworm_current_minimal
2. Download the latest **Bookworm_current_minimal** image for Orange Pi Zero 2W
3. The file will be named something like `Armbian_XX.XX.X_Orangepizero2w_bookworm_current_X.X.X_minimal.img.xz`

### 1.2 Flash the Image
1. **Using Balena Etcher (Recommended):**
   - Download and install [Balena Etcher](https://www.balena.io/etcher/)
   - Insert your microSD card into your computer
   - Open Balena Etcher
   - Select the downloaded Armbian image file
   - Select your microSD card as the target
   - Click "Flash" and wait for completion

2. **Using Raspberry Pi Imager (Alternative):**
   - Download [Raspberry Pi Imager](https://www.raspberrypi.org/software/)
   - Use "Use custom image" option to select the Armbian image

### 1.3 First Boot Preparation
1. Insert the flashed microSD card into the Orange Pi Zero 2W
2. Connect a USB keyboard and monitor (or use SSH after initial setup)
3. Power on the device

## Step 2: Initial Armbian Setup

### 2.1 First Boot Configuration
When you first boot Armbian, you'll be prompted to:

1. **Create Root Password:**
   - Set a strong root password when prompted
   - Remember this password for future access

2. **Create User Account:**
   - Create a regular user account (recommended)
   - Add the user to sudo group when prompted

3. **System Locale:**
   - Select your preferred language/locale
   - Choose your timezone
   - Configure keyboard layout if needed
   
4. **Wifi Setup:**
   - Configure the wifi you want to use

### 2.2 Connect to WiFi
1. **Using nmtui (Network Manager Text UI):**
   ```bash
   sudo nmtui
   ```
   - Select "Activate a connection"
   - Choose your WiFi network
   - Enter WiFi password
   - Select "Activate"

2. **Using command line (alternative):**
   ```bash
   # Scan for networks
   sudo nmcli dev wifi list
   
   # Connect to your network
   sudo nmcli dev wifi connect "YourWiFiName" password "YourWiFiPassword"
   ```

3. **Verify connection:**
   ```bash
   ping -c 4 google.com
   ```

### 2.3 Update System
```bash
sudo apt update
sudo apt upgrade -y
```

### 2.4 Enable SSH (Optional but Recommended)
```bash
sudo systemctl enable ssh
sudo systemctl start ssh

# Find your IP address
ip addr show wlan0
```

Now you can continue setup via SSH from another computer:
```bash
ssh username@your-pi-ip-address
```

## Step 3: Register Device in ThingsBoard

### 3.1 Access ThingsBoard
1. Log into your ThingsBoard server dashboard
2. Navigate to **Devices** section

### 3.2 Create New Device
1. Click **"+"** (Add Device) button
2. Fill in device details:
   - **Name:** `Fallyx Gateway - [Location]` (e.g., "Fallyx Gateway - Living Room")
   - **Device Profile:** Select appropriate profile or use default
   - **Label:** Optional descriptive label
3. Click **"Add"** to create the device

### 3.3 Copy Device Access Token
1. Click on the newly created device
2. Go to **"Details"** tab
3. Find **"Access Token"** field
4. **IMPORTANT:** Copy this token and save it securely - you'll need it for configuration
5. The token looks like: `A1_07DaYWzOiAiQXcvP2sbHFP2hbrCuDwkWs`

## Step 4: Prepare API Keys

### 4.1 Get OpenAI API Key
1. Visit [OpenAI Platform](https://platform.openai.com/)
2. Sign in to your account (create one if needed)
3. Navigate to **API Keys** section
4. Click **"Create new secret key"**
5. Copy and save the key securely (starts with `sk-`)
6. **Note:** You'll need billing set up for API usage

### 4.2 Get Picovoice Access Key
1. Visit [Picovoice Console](https://console.picovoice.ai/)
2. Sign in to your account (create one if needed)
3. Go to **"Access Keys"** section
4. Copy your access key
5. **Important:** Ensure your account is configured for Raspberry Pi platform

### 4.3 Configure Wake Word (Optional)
1. In Picovoice Console, go to **"Porcupine"** section
2. Create a custom wake word or use default "hey fallyx"
3. Download the `.ppn` file for **Raspberry Pi** platform
4. Save the file - you'll copy it to the device later

## Step 5: Download and Prepare Fallyx Gateway

You have two options for getting the Fallyx Gateway code onto your device:

### Option A: Clone Repository Directly on Device
```bash
# Install git if not already installed
sudo apt install git -y

# Clone the repository
cd /home/$(whoami)
git clone https://github.com/your-repo/Fallyx-Gateway.git
cd Fallyx-Gateway
```

### Option B: Clone on Computer and Transfer via SFTP (Alternative)

**On your computer:**
1. Clone the repository:
   ```bash
   git clone https://github.com/your-repo/Fallyx-Gateway.git
   ```

**Transfer to Orange Pi using SFTP:**

2. **Using command line SFTP:**
   ```bash
   # Connect to your Pi (replace with your actual username and IP)
   sftp username@pi-ip-address
   
   # Create directory on Pi
   mkdir Fallyx-Gateway
   
   # Transfer the entire directory recursively
   put -r Fallyx-Gateway
   
   # Exit SFTP
   quit
   ```

3. **Using SCP (alternative to SFTP):**
   ```bash
   # Transfer entire directory using SCP
   scp -r Fallyx-Gateway username@pi-ip-address:/home/username/
   ```

**Alternative GUI methods:**
- **WinSCP (Windows):** Download WinSCP, connect using SFTP protocol, drag and drop the `Fallyx-Gateway` folder
- **FileZilla:** Use SFTP protocol to connect and transfer the entire folder
- **VS Code with SFTP extension:** Install "SFTP" extension to sync files directly from your editor

**On the Orange Pi:**
4. Navigate to the transferred directory:
   ```bash
   cd /home/$(whoami)/Fallyx-Gateway
   ```

### 5.2 Prepare Deployment Scripts
```bash
cd deploy

# Fix permissions and line endings for all scripts
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

## Step 6: Install Fallyx Gateway

### 6.1 Run Installation Script
```bash
# Run installation without starting services (recommended)
sudo bash ./install_fallyx_gateway.sh --no-start
```

This script will:
- Update system packages
- Install Python dependencies
- Install Bluetooth and audio libraries
- Compile the BLE daemon
- Set up systemd services
- Create application directory at `/opt/fallyx-gateway`

### 6.2 Verify Installation
```bash
# Check if installation directory exists
ls -la /opt/fallyx-gateway

# Check if BLE daemon was compiled
ls -la /usr/local/bin/fallyx_daemon

# Check if services were installed
sudo systemctl list-unit-files | grep fallyx
```

## Step 7: Configure Environment

### 7.1 Interactive Configuration (Recommended)
```bash
sudo ./configure_env.sh --interactive
```

You'll be prompted for:

**ThingsBoard Configuration:**
- ThingsBoard Host: `your-thingsboard-server.com`
- ThingsBoard Port: `1883` (default)
- ThingsBoard Access Token: `[paste token from Step 3.3]`
- ThingsBoard QoS: `1` (default)

**OpenAI Configuration:**
- OpenAI API Key: `[paste key from Step 4.1]`

**Picovoice Configuration:**
- Picovoice Access Key: `[paste key from Step 4.2]`
- Wake Word: `hey fallyx` (default)

**Voice Assistant Configuration:**
- Enable Voice Assistant: `true`
- Chat API Base URL: `https://your-backend.com/v1/chat`
- Gateway ID: `fallyx-gateway-001` (or unique identifier)
- Gateway Name: `[descriptive name like "Living Room Gateway"]`

**Emergency Configuration:**
- Emergency Phone Number: `+1234567890` (E.164 format)
- Gateway Building: `[building name where gateway is located]`
- Gateway Room: `[room/area where gateway is located]`
- Emergency Contact Name: `[emergency contact person]`

**Audio Configuration:**
- Audio Input Device: `[leave blank for default or specify USB device]`
- Audio Output Device: `[leave blank for default or specify USB device]`
- Audio Sample Rate: `16000` (default)
- Audio Playback Sample Rate: `16000` (default)

### 7.2 Alternative: Command Line Configuration
```bash
# Configure essential settings via command line
sudo ./configure_env.sh \
    --tb-host "your-thingsboard.com" \
    --tb-token "YOUR_THINGSBOARD_TOKEN" \
    --openai-key "YOUR_OPENAI_KEY" \
    --picovoice-key "YOUR_PICOVOICE_KEY" \
    --emergency-number "+1234567890" \
    --gateway-building "Main Building" \
    --gateway-room "Living Room" \
    --gateway-name "Living Room Gateway"
```

### 7.3 Validate Configuration
```bash
sudo ./configure_env.sh --validate
```

## Step 8: Start Services

### 8.1 Start Fallyx Gateway Services
```bash
sudo ./manage_services.sh start
```

### 8.2 Enable Auto-Start on Boot
```bash
sudo ./manage_services.sh enable
```

## Step 9: Validate Installation

### 9.1 Check Service Status
```bash
sudo ./manage_services.sh status
```

You should see both services as "active (running)":
- `fallyx-gateway.service`
- `fallyx-ble-daemon.service`

### 9.2 Monitor Logs
```bash
# View gateway logs
sudo ./manage_services.sh logs gateway

# View BLE daemon logs
sudo ./manage_services.sh logs ble

# Follow logs in real-time
sudo ./manage_services.sh logs gateway -f
```

### 9.3 Test Audio Devices
```bash
# List audio devices
aplay -l    # List playback devices
arecord -l  # List recording devices

# Test speaker
speaker-test -t sine -f 1000 -l 1

# Test microphone (record 5 seconds)
arecord -d 5 -f cd test.wav
aplay test.wav
```

### 9.4 Test Bluetooth
```bash
# Check Bluetooth status
sudo systemctl status bluetooth

# Scan for BLE devices
sudo bluetoothctl
# In bluetoothctl:
# scan on
# (wait a few seconds)
# scan off
# exit
```

### 9.5 Verify ThingsBoard Connection
1. Check gateway logs for ThingsBoard connection messages:
   ```bash
   sudo journalctl -u fallyx-gateway.service | grep -i thingsboard
   ```

2. In ThingsBoard dashboard:
   - Go to your device
   - Check **"Latest Telemetry"** tab for incoming data
   - Look for device activity indicators

## Step 10: Final Configuration and Testing

### 10.1 Connect USB Audio Devices
1. Connect USB microphone and speaker
2. Restart audio services:
   ```bash
   sudo systemctl restart pulseaudio
   sudo ./manage_services.sh restart
   ```

### 10.2 Test Voice Assistant
1. Say the wake word: "Hey Fallyx"
2. Check logs for wake word detection:
   ```bash
   sudo ./manage_services.sh logs gateway -f
   ```

### 10.3 Test BLE Connectivity
1. Ensure BLE sensors are nearby and powered on
2. Check BLE daemon logs for device discovery:
   ```bash
   sudo ./manage_services.sh logs ble -f
   ```

## Troubleshooting

### Common Issues and Solutions

#### Services Won't Start
```bash
# Check configuration
sudo ./configure_env.sh --validate

# Check detailed service status
sudo systemctl status fallyx-gateway.service
sudo systemctl status fallyx-ble-daemon.service

# View error logs
sudo journalctl -u fallyx-gateway.service --since "10 minutes ago"
sudo journalctl -u fallyx-ble-daemon.service --since "10 minutes ago"
```

#### Audio Issues
```bash
# Restart audio system
sudo systemctl restart pulseaudio
pulseaudio --kill
pulseaudio --start

# Check audio devices
pactl list sources short  # Microphones
pactl list sinks short    # Speakers
```

#### Bluetooth Issues
```bash
# Restart Bluetooth
sudo systemctl restart bluetooth

# Reset Bluetooth adapter
sudo hciconfig hci0 down
sudo hciconfig hci0 up
```

#### Network Connectivity Issues
```bash
# Check network status
nmcli connection show
nmcli device status

# Restart network
sudo systemctl restart NetworkManager
```

### Log Locations
- Gateway logs: `sudo journalctl -u fallyx-gateway.service`
- BLE daemon logs: `sudo journalctl -u fallyx-ble-daemon.service`
- System logs: `sudo journalctl --since "1 hour ago"`

### Configuration Files
- Main config: `/opt/fallyx-gateway/.env`
- Service files: `/etc/systemd/system/fallyx-*.service`
- Application: `/opt/fallyx-gateway/`
- BLE Daemon devices.txt: `/opt/fallyx-gateway/gateway/ble-module/devices.txt`

## Maintenance

### Regular Updates
```bash
# Update system packages
sudo apt update && sudo apt upgrade -y

# Update Fallyx Gateway (when new versions available)
cd /path/to/Fallyx-Gateway
git pull
sudo ./manage_services.sh stop
sudo bash ./deploy/install_fallyx_gateway.sh --no-start
sudo ./manage_services.sh start
```

### Monitoring
```bash
# Check service status
sudo ./manage_services.sh status

# View recent logs
sudo ./manage_services.sh logs gateway --since "1 hour ago"

# Monitor system resources
htop
df -h
```

## Security Considerations

1. **Change default passwords** for all accounts
2. **Keep system updated** with security patches
3. **Secure .env file** (already set to 600 permissions)
4. **Use strong WiFi passwords**
5. **Consider firewall configuration** for production deployments
6. **Regularly backup configuration** files

## Support

If you encounter issues:

1. **Check logs** first using the troubleshooting section
2. **Validate configuration** with `sudo ./configure_env.sh --validate`
3. **Review service status** with `sudo ./manage_services.sh status`
4. **Consult the deploy/README.md** for additional troubleshooting steps

---

**Congratulations!** Your Fallyx Gateway should now be fully operational and ready to monitor BLE sensors, process voice commands, and communicate with your ThingsBoard server.
