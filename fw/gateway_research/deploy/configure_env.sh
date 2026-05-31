#!/bin/bash

# Fallyx Gateway Environment Configuration Script
# This script helps configure the .env file with actual values

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Configuration
INSTALL_DIR="/opt/fallyx-gateway"
ENV_FILE="$INSTALL_DIR/.env"

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

# Function to update environment variable
update_env_var() {
    local var_name="$1"
    local var_value="$2"
    local env_file="$3"
    
    if grep -q "^${var_name}=" "$env_file"; then
        # Variable exists, update it
        sed -i "s|^${var_name}=.*|${var_name}=${var_value}|" "$env_file"
        print_success "Updated $var_name"
    else
        # Variable doesn't exist, add it
        echo "${var_name}=${var_value}" >> "$env_file"
        print_success "Added $var_name"
    fi
}

# Function to prompt for configuration values
configure_interactively() {
    print_status "=== Interactive Configuration ==="
    echo ""
    
    # ThingsBoard Configuration
    echo "=== ThingsBoard Configuration ==="
    read -p "ThingsBoard Host (e.g., your-thingsboard.com): " tb_host
    read -p "ThingsBoard Port [1883]: " tb_port
    tb_port=${tb_port:-1883}
    read -p "ThingsBoard Access Token: " tb_token
    read -p "ThingsBoard QoS [1]: " tb_qos
    tb_qos=${tb_qos:-1}
    
    # OpenAI Configuration
    echo ""
    echo "=== OpenAI Configuration ==="
    read -p "OpenAI API Key: " openai_key
    
    # Picovoice Configuration
    echo ""
    echo "=== Picovoice Configuration ==="
    read -p "Picovoice Access Key: " picovoice_key
    read -p "Wake Word [hey fallyx]: " wake_word
    wake_word=${wake_word:-"hey fallyx"}
    
    # Voice Assistant Configuration
    echo ""
    echo "=== Voice Assistant Configuration ==="
    read -p "Enable Voice Assistant? [true]: " voice_enabled
    voice_enabled=${voice_enabled:-true}
    read -p "Chat API Base URL: " chat_api_url
    read -p "Gateway ID [fallyx-gateway-001]: " gateway_id
    gateway_id=${gateway_id:-"fallyx-gateway-001"}
    read -p "Gateway Name/Location [FallyxGateway]: " gateway_name
    gateway_name=${gateway_name:-"FallyxGateway"}
    echo ""
    echo "Note: This is the location present in the telemetry upload"
    
    # Emergency Configuration
    echo ""
    echo "=== Emergency Configuration ==="
    read -p "Emergency Phone Number (E.164 format, e.g., +14166299094): " emergency_number
    read -p "Gateway Building Name: " gateway_building
    read -p "Gateway Room/Area: " gateway_room
    read -p "Emergency Contact Name: " emergency_contact_name
    
    # Audio Configuration
    echo ""
    echo "=== Audio Configuration ==="
    read -p "Audio Input Device Name Contains (leave blank for default): " audio_input
    read -p "Audio Output Device Name Contains (leave blank for default): " audio_output
    read -p "Audio Sample Rate [16000]: " sample_rate
    sample_rate=${sample_rate:-16000}
    read -p "Audio Playback Sample Rate [16000]: " playback_rate
    playback_rate=${playback_rate:-16000}
    
    # Update .env file
    print_status "Updating .env file..."
    
    update_env_var "THINGSBOARD_HOST" "$tb_host" "$ENV_FILE"
    update_env_var "THINGSBOARD_PORT" "$tb_port" "$ENV_FILE"
    update_env_var "THINGSBOARD_ACCESS_TOKEN" "$tb_token" "$ENV_FILE"
    update_env_var "THINGSBOARD_QOS" "$tb_qos" "$ENV_FILE"
    
    update_env_var "OPENAI_API_KEY" "$openai_key" "$ENV_FILE"
    
    update_env_var "PICOVOICE_ACCESS_KEY" "$picovoice_key" "$ENV_FILE"
    update_env_var "WAKE_WORD" "$wake_word" "$ENV_FILE"
    
    update_env_var "VOICE_ASSISTANT_ENABLED" "$voice_enabled" "$ENV_FILE"
    update_env_var "CHAT_API_BASE_URL" "$chat_api_url" "$ENV_FILE"
    update_env_var "GATEWAY_ID" "$gateway_id" "$ENV_FILE"
    update_env_var "GATEWAY_NAME" "$gateway_name" "$ENV_FILE"
    
    update_env_var "EMERGENCY_NUMBER" "$emergency_number" "$ENV_FILE"
    update_env_var "GATEWAY_BUILDING" "$gateway_building" "$ENV_FILE"
    update_env_var "GATEWAY_ROOM" "$gateway_room" "$ENV_FILE"
    update_env_var "EMERGENCY_CONTACT_NAME" "$emergency_contact_name" "$ENV_FILE"
    
    update_env_var "AUDIO_INPUT_DEVICE_NAME_CONTAINS" "$audio_input" "$ENV_FILE"
    update_env_var "AUDIO_OUTPUT_DEVICE_NAME_CONTAINS" "$audio_output" "$ENV_FILE"
    update_env_var "AUDIO_DEFAULT_SAMPLE_RATE" "$sample_rate" "$ENV_FILE"
    update_env_var "AUDIO_PLAYBACK_SAMPLE_RATE" "$playback_rate" "$ENV_FILE"
    
    print_success "Configuration completed!"
}

# Function to configure from command line arguments
configure_from_args() {
    while [[ $# -gt 0 ]]; do
        case $1 in
            # ThingsBoard Configuration
            --tb-host)
                update_env_var "THINGSBOARD_HOST" "$2" "$ENV_FILE"
                shift 2
                ;;
            --tb-port)
                update_env_var "THINGSBOARD_PORT" "${2:-1883}" "$ENV_FILE"
                shift 2
                ;;
            --tb-token)
                update_env_var "THINGSBOARD_ACCESS_TOKEN" "$2" "$ENV_FILE"
                shift 2
                ;;
            --tb-qos)
                update_env_var "THINGSBOARD_QOS" "${2:-1}" "$ENV_FILE"
                shift 2
                ;;
            # OpenAI Configuration
            --openai-key)
                update_env_var "OPENAI_API_KEY" "$2" "$ENV_FILE"
                shift 2
                ;;
            # Picovoice Configuration
            --picovoice-key)
                update_env_var "PICOVOICE_ACCESS_KEY" "$2" "$ENV_FILE"
                shift 2
                ;;
            --wake-word)
                update_env_var "WAKE_WORD" "${2:-hey fallyx}" "$ENV_FILE"
                shift 2
                ;;
            # Voice Assistant Configuration
            --voice-assistant)
                update_env_var "VOICE_ASSISTANT_ENABLED" "${2:-true}" "$ENV_FILE"
                shift 2
                ;;
            --chat-api-url)
                update_env_var "CHAT_API_BASE_URL" "$2" "$ENV_FILE"
                shift 2
                ;;
            --gateway-id)
                update_env_var "GATEWAY_ID" "${2:-fallyx-gateway-001}" "$ENV_FILE"
                shift 2
                ;;
            --gateway-name)
                update_env_var "GATEWAY_NAME" "${2:-FallyxGateway}" "$ENV_FILE"
                shift 2
                ;;
            # Emergency Configuration
            --emergency-number)
                update_env_var "EMERGENCY_NUMBER" "$2" "$ENV_FILE"
                shift 2
                ;;
            --gateway-building)
                update_env_var "GATEWAY_BUILDING" "$2" "$ENV_FILE"
                shift 2
                ;;
            --gateway-room)
                update_env_var "GATEWAY_ROOM" "$2" "$ENV_FILE"
                shift 2
                ;;
            --emergency-contact-name)
                update_env_var "EMERGENCY_CONTACT_NAME" "$2" "$ENV_FILE"
                shift 2
                ;;
            # Audio Configuration
            --audio-input)
                update_env_var "AUDIO_INPUT_DEVICE_NAME_CONTAINS" "$2" "$ENV_FILE"
                shift 2
                ;;
            --audio-output)
                update_env_var "AUDIO_OUTPUT_DEVICE_NAME_CONTAINS" "$2" "$ENV_FILE"
                shift 2
                ;;
            --sample-rate)
                update_env_var "AUDIO_DEFAULT_SAMPLE_RATE" "${2:-16000}" "$ENV_FILE"
                shift 2
                ;;
            --playback-rate)
                update_env_var "AUDIO_PLAYBACK_SAMPLE_RATE" "${2:-16000}" "$ENV_FILE"
                shift 2
                ;;
            *)
                print_error "Unknown option: $1"
                exit 1
                ;;
        esac
    done
}

# Function to display current configuration
show_config() {
    if [[ ! -f "$ENV_FILE" ]]; then
        print_error ".env file not found at $ENV_FILE"
        exit 1
    fi
    
    print_status "Current configuration in $ENV_FILE:"
    echo ""
    
    # Show non-empty, non-comment lines
    grep -v '^#' "$ENV_FILE" | grep -v '^$' | while IFS= read -r line; do
        if [[ "$line" == *"TOKEN"* ]] || [[ "$line" == *"KEY"* ]]; then
            # Mask sensitive values
            var_name=$(echo "$line" | cut -d'=' -f1)
            echo "$var_name=***MASKED***"
        else
            echo "$line"
        fi
    done
}

# Function to validate configuration
validate_config() {
    if [[ ! -f "$ENV_FILE" ]]; then
        print_error ".env file not found at $ENV_FILE"
        exit 1
    fi
    
    print_status "Validating configuration..."
    
    # Required variables based on your code analysis
    required_vars=(
        "THINGSBOARD_HOST"
        "THINGSBOARD_ACCESS_TOKEN"
        "OPENAI_API_KEY"
        "PICOVOICE_ACCESS_KEY"
        "EMERGENCY_NUMBER"
        "GATEWAY_BUILDING"
        "GATEWAY_ROOM"
        "GATEWAY_NAME"
        "CHAT_API_BASE_URL"
    )
    
    missing_vars=()
    
    for var in "${required_vars[@]}"; do
        if ! grep -q "^${var}=..*" "$ENV_FILE"; then
            missing_vars+=("$var")
        fi
    done
    
    if [[ ${#missing_vars[@]} -eq 0 ]]; then
        print_success "All required configuration variables are set"
        return 0
    else
        print_error "Missing required configuration variables:"
        for var in "${missing_vars[@]}"; do
            echo "  - $var"
        done
        return 1
    fi
}

# Main function
main() {
    # Check if .env file exists
    if [[ ! -f "$ENV_FILE" ]]; then
        print_error ".env file not found at $ENV_FILE"
        print_error "Please run the installation script first"
        exit 1
    fi
    
    # Check if running as root
    if [[ $EUID -ne 0 ]]; then
        print_error "This script must be run as root (use sudo)"
        exit 1
    fi
    
    case "${1:-}" in
        --help|-h)
            echo "Fallyx Gateway Environment Configuration Script"
            echo ""
            echo "Usage: $0 [options]"
            echo ""
            echo "Options:"
            echo "  --help, -h              Show this help message"
            echo "  --interactive, -i       Interactive configuration"
            echo "  --show                  Show current configuration"
            echo "  --validate              Validate configuration"
            echo ""
            echo "Command line configuration:"
            echo ""
            echo "ThingsBoard Configuration:"
            echo "  --tb-host HOST          ThingsBoard host"
            echo "  --tb-port PORT          ThingsBoard port [default: 1883]"
            echo "  --tb-token TOKEN        ThingsBoard access token"
            echo "  --tb-qos QOS            ThingsBoard QoS [default: 1]"
            echo ""
            echo "OpenAI Configuration:"
            echo "  --openai-key KEY        OpenAI API key"
            echo ""
            echo "Picovoice Configuration:"
            echo "  --picovoice-key KEY     Picovoice access key"
            echo "  --wake-word WORD        Wake word [default: 'hey fallyx']"
            echo ""
            echo "Voice Assistant Configuration:"
            echo "  --voice-assistant BOOL  Enable voice assistant [default: true]"
            echo "  --chat-api-url URL      Chat API base URL"
            echo "  --gateway-id ID         Gateway ID [default: 'fallyx-gateway-001']"
            echo "  --gateway-name NAME     Gateway display name [default: 'FallyxGateway']"
            echo ""
            echo "Emergency Configuration:"
            echo "  --emergency-number NUM  Emergency phone number (E.164 format)"
            echo "  --gateway-building NAME Gateway building name"
            echo "  --gateway-room NAME     Gateway room/area"
            echo "  --emergency-contact-name NAME Emergency contact name"
            echo ""
            echo "Audio Configuration:"
            echo "  --audio-input DEVICE    Audio input device name contains"
            echo "  --audio-output DEVICE   Audio output device name contains"
            echo "  --sample-rate RATE      Audio sample rate [default: 16000]"
            echo "  --playback-rate RATE    Audio playback sample rate [default: 16000]"
            echo ""
            exit 0
            ;;
        --interactive|-i)
            configure_interactively
            ;;
        --show)
            show_config
            ;;
        --validate)
            validate_config
            ;;
        --tb-host|--tb-port|--tb-token|--tb-qos|--openai-key|--picovoice-key|--wake-word|--voice-assistant|--chat-api-url|--gateway-id|--gateway-name|--emergency-number|--gateway-building|--gateway-room|--emergency-contact-name|--audio-input|--audio-output|--sample-rate|--playback-rate)
            configure_from_args "$@"
            ;;
        "")
            configure_interactively
            ;;
        *)
            print_error "Unknown option: $1"
            echo "Use --help for usage information"
            exit 1
            ;;
    esac
}

main "$@"