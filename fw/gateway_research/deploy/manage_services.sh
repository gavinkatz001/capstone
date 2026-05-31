#!/bin/bash

# Fallyx Gateway Service Management Script
# This script provides easy management of Fallyx Gateway services

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Service names
GATEWAY_SERVICE="fallyx-gateway.service"
BLE_SERVICE="fallyx-ble-daemon.service"

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

# Function to show service status
show_status() {
    print_status "=== Fallyx Gateway Service Status ==="
    echo ""
    
    echo "Gateway Service:"
    systemctl status $GATEWAY_SERVICE --no-pager -l
    echo ""
    
    echo "BLE Daemon Service:"
    systemctl status $BLE_SERVICE --no-pager -l
}

# Function to start services
start_services() {
    print_status "Starting Fallyx Gateway services..."
    
    systemctl start $GATEWAY_SERVICE
    sleep 3
    systemctl start $BLE_SERVICE
    
    print_success "Services started"
    show_status
}

# Function to stop services
stop_services() {
    print_status "Stopping Fallyx Gateway services..."
    
    systemctl stop $BLE_SERVICE
    systemctl stop $GATEWAY_SERVICE
    
    print_success "Services stopped"
}

# Function to restart services
restart_services() {
    print_status "Restarting Fallyx Gateway services..."
    
    systemctl restart $GATEWAY_SERVICE
    sleep 3
    systemctl restart $BLE_SERVICE
    
    print_success "Services restarted"
    show_status
}

# Function to enable services
enable_services() {
    print_status "Enabling Fallyx Gateway services..."
    
    systemctl enable $GATEWAY_SERVICE
    systemctl enable $BLE_SERVICE
    
    print_success "Services enabled for auto-start"
}

# Function to disable services
disable_services() {
    print_status "Disabling Fallyx Gateway services..."
    
    systemctl disable $GATEWAY_SERVICE
    systemctl disable $BLE_SERVICE
    
    print_success "Services disabled from auto-start"
}

# Function to show logs
show_logs() {
    local service="$1"
    local follow="$2"
    
    if [[ "$service" == "gateway" ]]; then
        if [[ "$follow" == "follow" ]]; then
            journalctl -u $GATEWAY_SERVICE -f
        else
            journalctl -u $GATEWAY_SERVICE --no-pager -l
        fi
    elif [[ "$service" == "ble" ]]; then
        if [[ "$follow" == "follow" ]]; then
            journalctl -u $BLE_SERVICE -f
        else
            journalctl -u $BLE_SERVICE --no-pager -l
        fi
    else
        print_error "Unknown service: $service"
        echo "Use 'gateway' or 'ble'"
        exit 1
    fi
}

# Function to show help
show_help() {
    echo "Fallyx Gateway Service Management Script"
    echo ""
    echo "Usage: $0 <command> [options]"
    echo ""
    echo "Commands:"
    echo "  status              Show service status"
    echo "  start               Start services"
    echo "  stop                Stop services"
    echo "  restart             Restart services"
    echo "  enable              Enable services for auto-start"
    echo "  disable             Disable services from auto-start"
    echo "  logs <service>      Show logs for service (gateway|ble)"
    echo "  logs <service> -f   Follow logs for service (gateway|ble)"
    echo "  help                Show this help message"
    echo ""
    echo "Examples:"
    echo "  $0 status"
    echo "  $0 restart"
    echo "  $0 logs gateway"
    echo "  $0 logs ble -f"
    echo ""
}

# Main function
main() {
    case "${1:-}" in
        status)
            show_status
            ;;
        start)
            check_root
            start_services
            ;;
        stop)
            check_root
            stop_services
            ;;
        restart)
            check_root
            restart_services
            ;;
        enable)
            check_root
            enable_services
            ;;
        disable)
            check_root
            disable_services
            ;;
        logs)
            if [[ -z "$2" ]]; then
                print_error "Please specify service: gateway or ble"
                exit 1
            fi
            show_logs "$2" "$3"
            ;;
        help|--help|-h)
            show_help
            ;;
        "")
            show_help
            ;;
        *)
            print_error "Unknown command: $1"
            echo "Use 'help' for usage information"
            exit 1
            ;;
    esac
}

main "$@"