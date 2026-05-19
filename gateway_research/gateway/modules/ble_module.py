# gateway/modules/ble_module.py

import logging
import socket
import os
import threading
import json
import time
from typing import Any, Dict, Optional

logger = logging.getLogger("BLEModule")
SOCKET_PATH = "/tmp/fallyx_ble.sock"

class BLEModule:
    """
    Receives BLE data from a dedicated C daemon over a Unix socket.
    The daemon now sends newline-delimited JSON objects.
    """
    def __init__(self):
        logger.info("Initializing BLE Module (JSON Socket Listener)")
        self.data_manager = None
        self.command_module = None
        self.running = False
        # Socket-based implementation attributes
        self.reassembly_buffers = {}  # For tracking incomplete packets
        self.lock = threading.Lock()  # For thread-safe operations

    def set_data_manager(self, data_manager):
        self.data_manager = data_manager
        logger.info("Data manager set")
    
    def set_command_module(self, command_module):
        """Set the command module reference for configuration updates."""
        self.command_module = command_module
        logger.info("Command module set")
    
    def run(self):
        """Runs the socket server to listen for data from the C++ daemon."""
        self.running = True
        logger.info("BLE module starting with single-threaded reassembly.")

        if os.path.exists(SOCKET_PATH):
            os.remove(SOCKET_PATH)

        server_sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        server_sock.bind(SOCKET_PATH)
        server_sock.listen(1)
        logger.info(f"BLE module listening on Unix socket: {SOCKET_PATH}")

        while self.running:
            try:
                logger.info("Waiting for C BLE daemon to connect...")
                conn, addr = server_sock.accept()
                logger.info(f"C BLE daemon connected to socket from {addr}")
                with conn:
                    self._handle_client_connection(conn)
                logger.info("C BLE daemon session ended. Ready for new connection.")
            except Exception as e:
                if not self.running:
                    break
                logger.error(f"Socket server error: {e}", exc_info=True)
                time.sleep(5) # Prevent rapid-fire error loops
        
        logger.info("BLE socket listener shutting down.")
        server_sock.close()
        if os.path.exists(SOCKET_PATH):
            os.remove(SOCKET_PATH)

    def stop(self):
        if not self.running:
            return
        logger.info("Stopping BLE Module")
        self.running = False
        
        try:
            with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as s:
                s.connect(SOCKET_PATH)
        except Exception:
            pass
            
    def _handle_client_connection(self, conn):
        """
        Handle the client connection with newline-delimited JSON protocol.
        """
        # Wrap the socket connection with a file-like object for readline()
        socket_file = conn.makefile('r')
        
        try:
            while self.running:
                try:
                    # Read lines from the socket
                    line = socket_file.readline()
                    if not line:
                        logger.warning("C BLE daemon disconnected.")
                        break
                    
                    line = line.strip()
                    if not line:
                        continue
                    
                    # Parse the JSON string
                    try:
                        json_data = json.loads(line)
                        logger.info(f"Received JSON data: {len(line)} characters")
                        
                        # Extract MAC and packet data
                        mac = json_data.get('mac')
                        packet_data = json_data.get('packet')
                        
                        if mac and packet_data:
                            logger.info(f"Processing data for MAC {mac}")
                            # Pass the pre-parsed data to the data manager
                            if self.data_manager:
                                self.data_manager.process_sensor_data(mac, packet_data)
                        else:
                            logger.warning("JSON data missing 'mac' or 'packet' fields")
                            
                    except json.JSONDecodeError as e:
                        logger.error(f"Failed to parse JSON: {e}")
                        logger.error(f"Raw line: {line[:200]}...")  # Log first 200 chars
                        continue
                        
                except Exception as e:
                    logger.error(f"Error reading from socket: {e}", exc_info=True)
                    break
                    
        finally:
            socket_file.close()

    # Removed obsolete methods:
    # - _recv_exact
    # - _append_and_check_complete_packet
    # - _parse_fallyx_packet
    # The module is now a lightweight bridge that receives JSON and passes it on

    def send_response_to_device(self, device_path, response):
        """Send a response to a connected device."""
        logger.info(f"Sending response to device: {device_path}, response: {response}")
        
        try:
            # In the new architecture, we would need to send this back through the C++ daemon
            # For now, we'll just log it as the C++ daemon doesn't implement bidirectional communication yet
            logger.info(f"Response would be sent to device: {device_path}")
            return True
                
        except Exception as e:
            logger.error(f"Error sending response to device {device_path}: {e}")
            return False

    def _format_response(self, response):
        """Format a response for BLE transmission."""
        import json
        if isinstance(response, dict):
            return json.dumps(response).encode('utf-8')
        elif isinstance(response, str):
            return response.encode('utf-8')
        elif isinstance(response, bytes):
            return response
        else:
            return str(response).encode('utf-8')

    def get_connected_device_count(self):
        """Get the number of currently connected devices."""
        with self.lock:
            return len(self.reassembly_buffers)

    def get_connected_devices(self):
        """Get a list of currently connected device paths."""
        with self.lock:
            return list(self.reassembly_buffers.keys())

    def is_device_connected(self, device_id):
        """
        Check if a device is connected.
        NOTE: In peripheral mode, 'connected' means we have received data from it.
               The device_id here should correspond to the C++ daemon device identifier.
        """
        with self.lock:
            return device_id in self.reassembly_buffers

    def get_device_battery_level(self, device_id):
        """
        Placeholder for getting a device's battery level.
        This would need to be implemented if battery data is sent.
        """
        logger.warning(f"get_device_battery_level is not yet implemented for device {device_id}.")
        return None

    def send_command_to_device(self, target_device, command):
        """
        Sends a command to a device.
        In the new architecture, this would need to be sent through the C++ daemon.
        """
        logger.info(f"Attempting to send command to device {target_device}: {command}")
        
        # For now, we'll just log it as the C++ daemon doesn't implement bidirectional communication yet
        logger.info(f"Command would be sent to device: {target_device}")
        return True
