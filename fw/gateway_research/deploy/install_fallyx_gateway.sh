#!/bin/bash

# Fallyx Gateway Installation Script
# This script installs the Fallyx Gateway system on a fresh Armbian installation

set -e  # Exit on any error

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Configuration
INSTALL_DIR="/opt/fallyx-gateway"
SERVICE_USER="root"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

# Function to print colored output
print_status() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

print_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Function to check if running as root
check_root() {
    if [[ $EUID -ne 0 ]]; then
        print_error "This script must be run as root (use sudo)"
        exit 1
    fi
}

# Function to update system packages
update_system() {
    print_status "Updating system packages..."
    apt-get update
    apt-get upgrade -y
    print_success "System packages updated"
}

# Function to install system dependencies
install_system_dependencies() {
    print_status "Installing system dependencies..."
    
    # Essential build tools and libraries
    apt-get install -y \
        python3 \
        python3-pip \
        python3-dev \
        python3-venv \
        gcc \
        make \
        build-essential \
        git \
        curl \
        wget \
        unzip \
        libbluetooth-dev \
        bluetooth \
        bluez \
        bluez-tools \
        libasound2-dev \
        portaudio19-dev \
        libportaudio2 \
        libportaudiocpp0 \
        ffmpeg \
        alsa-utils \
        pulseaudio \
        pulseaudio-utils \
        libpulse-dev \
        libffi-dev \
        libssl-dev \
        pkg-config \
        systemd
    
    print_success "System dependencies installed"
}

# Function to install Python dependencies
install_python_dependencies() {
    print_status "Installing Python dependencies..."
    
    # Install pip packages with --break-system-packages for system-wide installation
    pip3 install --break-system-packages --upgrade pip
    
    # Install requirements from requirements.txt
    if [[ -f "$PROJECT_ROOT/gateway/requirements.txt" ]]; then
        pip3 install --break-system-packages -r "$PROJECT_ROOT/gateway/requirements.txt"
        print_success "Python dependencies installed from requirements.txt"
    else
        print_warning "requirements.txt not found, installing essential packages manually"
        pip3 install --break-system-packages \
            numpy>=1.22.0 \
            scipy>=1.8.0 \
            paho-mqtt>=1.6.1 \
            requests>=2.27.1 \
            psutil>=5.9.0 \
            sounddevice>=0.4.4 \
            openai>=1.0.0 \
            pvporcupine>=3.0.0 \
            python-dotenv>=0.19.0
    fi
    
    print_success "Python dependencies installed"
}

# Function to compile BLE daemon
compile_ble_daemon() {
    print_status "Compiling BLE daemon..."
    
    cd "$PROJECT_ROOT/gateway/ble-daemon"
    
    # Create a unique device name based on .env configuration
    local device_name="GatewayFallyx"
    
    # Try to get a unique identifier from .env file if it exists
    if [[ -f "$INSTALL_DIR/.env" ]]; then
        # Extract GATEWAY_BUILDING and GATEWAY_ID from .env
        local gateway_building=$(grep "^GATEWAY_BUILDING=" "$INSTALL_DIR/.env" 2>/dev/null | cut -d'=' -f2 | tr -d '"' | tr -d "'" | sed 's/[^a-zA-Z0-9]//g')
        local gateway_id=$(grep "^GATEWAY_ID=" "$INSTALL_DIR/.env" 2>/dev/null | cut -d'=' -f2 | tr -d '"' | tr -d "'" | sed 's/[^a-zA-Z0-9-]//g')
        
        # Construct unique name: GatewayFallyx-{GATEWAY_BUILDING or GATEWAY_ID}
        if [[ -n "$gateway_building" ]]; then
            device_name="GatewayFallyx-${gateway_building}"
        elif [[ -n "$gateway_id" ]]; then
            # Remove common prefixes and use last part
            gateway_suffix=$(echo "$gateway_id" | sed 's/.*-//' | sed 's/fallyx-gateway-//' | sed 's/gateway-//')
            if [[ -n "$gateway_suffix" ]]; then
                device_name="GatewayFallyx-${gateway_suffix}"
            fi
        fi
    fi
    
    # Truncate to max 20 characters for BLE advertising name limits
    if [[ ${#device_name} -gt 20 ]]; then
        device_name="${device_name:0:20}"
    fi
    
    print_status "Setting BLE advertised name to: $device_name"
    
    # Detect the actual Bluetooth MAC address of this device
    local ble_mac_address=""
    
    # Try multiple methods to get the Bluetooth MAC address
    if command -v hciconfig >/dev/null 2>&1; then
        # Method 1: Use hciconfig (most reliable for Bluetooth)
        ble_mac_address=$(hciconfig hci0 2>/dev/null | grep "BD Address" | awk '{print $3}' | tr '[:lower:]' '[:upper:]')
    fi
    
    if [[ -z "$ble_mac_address" ]] && command -v bluetoothctl >/dev/null 2>&1; then
        # Method 2: Use bluetoothctl
        ble_mac_address=$(bluetoothctl show 2>/dev/null | grep "Controller" | awk '{print $2}' | tr '[:lower:]' '[:upper:]')
    fi
    
    if [[ -z "$ble_mac_address" ]] && [[ -f /sys/class/bluetooth/hci0/address ]]; then
        # Method 3: Read from sysfs
        ble_mac_address=$(cat /sys/class/bluetooth/hci0/address 2>/dev/null | tr '[:lower:]' '[:upper:]')
    fi
    
    # Fallback to a default MAC if detection fails
    if [[ -z "$ble_mac_address" ]]; then
        print_warning "Could not detect Bluetooth MAC address, using default"
        ble_mac_address="00:00:00:00:00:00"
    else
        print_success "Detected Bluetooth MAC address: $ble_mac_address"
    fi
    
    # Configure devices.txt with the dynamic device name and actual MAC address
    print_status "Configuring devices.txt with BLE name: $device_name and MAC: $ble_mac_address"
    cat > devices.txt << EOF
DEVICE=$device_name  TYPE=MESH  NODE=1  CHANNEL=1 ADDRESS=$ble_mac_address
  PRIMARY_SERVICE = 1800
    LECHAR = Device name  PERMIT=06  SIZE=16  UUID=2A00   ; index 0
  PRIMARY_SERVICE = 55535343fe7d4ae58fa99fafd205e455
    LECHAR = DataTx  PERMIT=06  SIZE=16 UUID=49535343884143f4a8d4ecbe34729bb3        ; index 1
    LECHAR = DataRx  PERMIT=06  SIZE=16 UUID=495353431e4d4bd9ba6123c647249616        ; index 2
EOF
    
    # Fix line endings and make all shell scripts executable
    for script in *.sh; do
        if [[ -f "$script" ]]; then
            chmod +x "$script"
            sed -i 's/\r$//' "$script"
            print_status "Fixed line endings for $script"
        fi
    done
    
    # Compile the daemon using direct gcc (no interactive prompts)
    print_status "Compiling BLE daemon binary..."
    gcc fallyx_daemon.c btlib.c cJSON.c -o fallyx_daemon -lm
    
    if [[ -f "fallyx_daemon" ]]; then
        # Install the compiled binary to system location
        cp fallyx_daemon /usr/local/bin/fallyx_daemon
        chmod +x /usr/local/bin/fallyx_daemon
        print_success "BLE daemon compiled and installed to /usr/local/bin/fallyx_daemon"
        print_success "BLE advertised name set to: $device_name"
        print_success "BLE MAC address configured as: $ble_mac_address"
    else
        print_error "BLE daemon compilation failed"
        exit 1
    fi
    
    cd "$PROJECT_ROOT"
}

# Function to create installation directory and copy files
setup_application_files() {
    print_status "Setting up application files..."
    
    # Create installation directory
    mkdir -p "$INSTALL_DIR"
    
    # Copy gateway Python code
    cp -r "$PROJECT_ROOT/gateway" "$INSTALL_DIR/"
    
    # Copy other necessary files
    if [[ -f "$PROJECT_ROOT/README.md" ]]; then
        cp "$PROJECT_ROOT/README.md" "$INSTALL_DIR/"
    fi
    
    # Set proper permissions
    chown -R $SERVICE_USER:$SERVICE_USER "$INSTALL_DIR"
    chmod -R 755 "$INSTALL_DIR"
    
    print_success "Application files copied to $INSTALL_DIR"
}

# Function to configure environment variables
configure_environment() {
    print_status "Configuring environment variables..."
    
    # Check if .env already exists
    if [[ -f "$INSTALL_DIR/.env" ]]; then
        print_warning ".env file already exists, backing up to .env.backup"
        cp "$INSTALL_DIR/.env" "$INSTALL_DIR/.env.backup"
    fi
    
    # Copy template if .env doesn't exist
    if [[ ! -f "$INSTALL_DIR/.env" ]] && [[ -f "$PROJECT_ROOT/.env.template" ]]; then
        cp "$PROJECT_ROOT/.env.template" "$INSTALL_DIR/.env"
        print_warning "Created .env from template. Please edit $INSTALL_DIR/.env with your actual configuration values."
    elif [[ -f "$PROJECT_ROOT/.env" ]]; then
        cp "$PROJECT_ROOT/.env" "$INSTALL_DIR/.env"
        print_success "Copied existing .env file"
    else
        print_warning "No .env or .env.template found. You'll need to create $INSTALL_DIR/.env manually."
    fi
    
    # Set proper permissions for .env file
    if [[ -f "$INSTALL_DIR/.env" ]]; then
        chown $SERVICE_USER:$SERVICE_USER "$INSTALL_DIR/.env"
        chmod 600 "$INSTALL_DIR/.env"  # Restrict access to owner only
    fi
}

# Function to install systemd services
install_systemd_services() {
    print_status "Installing systemd services..."
    
    # Copy service files
    if [[ -f "$PROJECT_ROOT/deploy/systemd/fallyx-gateway.service" ]]; then
        cp "$PROJECT_ROOT/deploy/systemd/fallyx-gateway.service" /etc/systemd/system/
        print_success "Installed fallyx-gateway.service"
    else
        print_error "fallyx-gateway.service not found"
        exit 1
    fi
    
    if [[ -f "$PROJECT_ROOT/deploy/systemd/fallyx-ble-daemon.service" ]]; then
        cp "$PROJECT_ROOT/deploy/systemd/fallyx-ble-daemon.service" /etc/systemd/system/
        print_success "Installed fallyx-ble-daemon.service"
    else
        print_error "fallyx-ble-daemon.service not found"
        exit 1
    fi
    
    # Reload systemd and enable services
    systemctl daemon-reload
    systemctl enable fallyx-gateway.service
    systemctl enable fallyx-ble-daemon.service
    
    print_success "Systemd services installed and enabled"
}

# Function to configure Bluetooth
configure_bluetooth() {
    print_status "Configuring Bluetooth..."
    
    # Enable and start Bluetooth service
    systemctl enable bluetooth
    systemctl start bluetooth
    
    # Add user to bluetooth group (if not root)
    if [[ "$SERVICE_USER" != "root" ]]; then
        usermod -a -G bluetooth $SERVICE_USER
    fi
    
    print_success "Bluetooth configured"
}

# Function to configure audio
configure_audio() {
    print_status "Configuring audio system..."
    
    # Add user to audio group (if not root)
    if [[ "$SERVICE_USER" != "root" ]]; then
        usermod -a -G audio $SERVICE_USER
    fi
    
    # Start PulseAudio system service
    systemctl --global enable pulseaudio.service
    systemctl --global enable pulseaudio.socket
    
    print_success "Audio system configured"
}

# Function to start services
start_services() {
    print_status "Starting Fallyx Gateway services..."
    
    # Start the main gateway service first
    systemctl start fallyx-gateway.service
    sleep 5  # Give it time to start
    
    # Start the BLE daemon service
    systemctl start fallyx-ble-daemon.service
    
    # Check service status
    if systemctl is-active --quiet fallyx-gateway.service; then
        print_success "fallyx-gateway.service is running"
    else
        print_error "fallyx-gateway.service failed to start"
        systemctl status fallyx-gateway.service
    fi
    
    if systemctl is-active --quiet fallyx-ble-daemon.service; then
        print_success "fallyx-ble-daemon.service is running"
    else
        print_error "fallyx-ble-daemon.service failed to start"
        systemctl status fallyx-ble-daemon.service
    fi
}

# Function to display post-installation information
display_post_install_info() {
    echo ""
    print_success "=== Fallyx Gateway Installation Complete ==="
    echo ""
    echo "Installation Directory: $INSTALL_DIR"
    echo "Configuration File: $INSTALL_DIR/.env"
    echo "BLE Daemon Binary: /usr/local/bin/fallyx_daemon"
    echo ""
    echo "Services:"
    echo "  - fallyx-gateway.service"
    echo "  - fallyx-ble-daemon.service"
    echo ""
    echo "Useful Commands:"
    echo "  Check service status: sudo systemctl status fallyx-gateway.service"
    echo "  View logs: sudo journalctl -u fallyx-gateway.service -f"
    echo "  Restart services: sudo systemctl restart fallyx-gateway.service"
    echo "  Stop services: sudo systemctl stop fallyx-gateway.service fallyx-ble-daemon.service"
    echo ""
    print_warning "IMPORTANT: Please edit $INSTALL_DIR/.env with your actual configuration values!"
    echo ""
    print_warning "Required configuration:"
    echo "  - THINGSBOARD_HOST"
    echo "  - THINGSBOARD_ACCESS_TOKEN"
    echo "  - OPENAI_API_KEY"
    echo "  - PICOVOICE_ACCESS_KEY"
    echo "  - EMERGENCY_NUMBER"
    echo "  - GATEWAY_BUILDING"
    echo "  - GATEWAY_ROOM"
    echo ""
}

# Main installation function
main() {
    echo ""
    print_status "=== Fallyx Gateway Installation Script ==="
    echo ""
    
    check_root
    update_system
    install_system_dependencies
    install_python_dependencies
    setup_application_files
    configure_environment
    compile_ble_daemon
    install_systemd_services
    configure_bluetooth
    configure_audio
    start_services
    display_post_install_info
    
    print_success "Installation completed successfully!"
}

# Handle script arguments
case "${1:-}" in
    --help|-h)
        echo "Fallyx Gateway Installation Script"
        echo ""
        echo "Usage: $0 [options]"
        echo ""
        echo "Options:"
        echo "  --help, -h     Show this help message"
        echo "  --no-start     Install but don't start services"
        echo ""
        echo "This script must be run as root (use sudo)"
        exit 0
        ;;
    --no-start)
        # Run installation but skip starting services
        check_root
        update_system
        install_system_dependencies
        install_python_dependencies
        setup_application_files
        configure_environment
        compile_ble_daemon
        install_systemd_services
        configure_bluetooth
        configure_audio
        display_post_install_info
        print_success "Installation completed (services not started)"
        ;;
    "")
        # Run full installation
        main
        ;;
    *)
        print_error "Unknown option: $1"
        echo "Use --help for usage information"
        exit 1
        ;;
esac