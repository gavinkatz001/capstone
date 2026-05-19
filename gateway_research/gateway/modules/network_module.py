"""
Network Module for Fallyx Gateway

This module handles connectivity and communication with cloud services,
including sending sensor data and receiving commands via ThingsBoard IoT platform.
"""

import logging
import time
import threading
import os
from gateway.modules.thingsboard_client import ThingsBoardClient

logger = logging.getLogger("NetworkModule")

class NetworkModule:
    """
    Network Module for handling ThingsBoard communication.
    
    This module is responsible for:
    - Managing ThingsBoard connectivity
    - Sending sensor data to ThingsBoard
    - Receiving commands from ThingsBoard
    - Routing commands to appropriate handlers
    """
    
    def __init__(self, config=None):
        """
        Initialize the Network Module.
        
        Args:
            config: Optional configuration dictionary for ThingsBoard connection
        """
        logger.info("Initializing Network Module")
        self.config = config or {}
        self.running = False
        self.connected = False
        self.retry_queue = []  # Queue for failed transmissions
        self.lock = threading.Lock()  # For thread-safe operations
        
        # Initialize ThingsBoard client
        self.tb_client = ThingsBoardClient(self.config)
    
    @property
    def thingsboard_client(self):
        """Property to access ThingsBoard client for compatibility."""
        return self.tb_client
    
    def run(self):
        """Run the Network Module main loop."""
        logger.info("Starting Network Module")
        self.running = True
        
        # Start ThingsBoard client in a separate thread
        # Remove direct _connect() call to avoid double connection logic
        self.tb_thread = threading.Thread(target=self.tb_client.run, name="ThingsBoard-Thread")
        self.tb_thread.daemon = True
        self.tb_thread.start()
        
        while self.running:
            # Process retry queue
            self._process_retry_queue()
            
            # Update connection status from ThingsBoard client
            self.connected = self.tb_client.connected
            
            logger.debug("Network Module running...")
            time.sleep(5)  # Sleep to prevent CPU hogging
    
    def stop(self):
        """Stop the Network Module."""
        logger.info("Stopping Network Module")
        self.running = False
        self._disconnect()
        
        # Wait for ThingsBoard thread to complete
        if hasattr(self, 'tb_thread') and self.tb_thread.is_alive():
            self.tb_thread.join(timeout=5.0)
            if self.tb_thread.is_alive():
                logger.warning("ThingsBoard thread did not terminate gracefully")
    
    def _connect(self):
        """Connect to ThingsBoard."""
        logger.info("Connecting to ThingsBoard")
        
        # Connect using ThingsBoard client
        if self.tb_client.connect():
            self.connected = True
            logger.info("Connected to ThingsBoard successfully")
        else:
            self.connected = False
            logger.warning("Failed to connect to ThingsBoard")
    
    def _disconnect(self):
        """Disconnect from ThingsBoard."""
        logger.info("Disconnecting from ThingsBoard")
        
        # Disconnect using ThingsBoard client
        self.tb_client.disconnect()
        self.connected = False
    
    def _process_retry_queue(self):
        """Process queued messages that failed to send."""
        if not self.retry_queue or not self.connected:
            return
            
        logger.info(f"Processing retry queue: {len(self.retry_queue)} items")
        
        with self.lock:
            # Make a copy of the queue and clear the original
            queue_copy = self.retry_queue.copy()
            self.retry_queue = []
        
        # Try to send each item in the queue
        for telemetry_data in queue_copy:
            success = self.send_data_to_thingsboard(telemetry_data)
            if not success:
                logger.warning(f"Failed to send queued telemetry data, re-queuing")
                with self.lock:
                    self.retry_queue.append(telemetry_data)
    
    def send_data_to_thingsboard(self, telemetry_data):
        """
        Send telemetry data to ThingsBoard.
        
        Args:
            telemetry_data: Dictionary containing complete telemetry data ready for publication
            
        Returns:
            True if successfully sent or queued, False otherwise
        """
        logger.info("Sending telemetry data to ThingsBoard")
        
        # Check if connected
        if not self.connected:
            logger.warning("Not connected to ThingsBoard, queuing data for retry")
            with self.lock:
                self.retry_queue.append(telemetry_data)
            return False
        
        # Send data using ThingsBoard client
        success = self.tb_client.publish_telemetry(telemetry_data)
        
        if success:
            logger.info("Successfully sent telemetry data")
        else:
            logger.warning("Failed to send telemetry data")
            with self.lock:
                self.retry_queue.append(telemetry_data)
        
        return success
    
    def register_command_handler(self, command_type, handler_function):
        """
        Register a handler function for a specific command type.
        
        Args:
            command_type: The type of command to handle
            handler_function: Function to call when command is received
        """
        logger.info(f"Registering handler for command type: {command_type}")
        
        # Register handler with ThingsBoard client
        self.tb_client.register_command_handler(command_type, handler_function)
    
    def register_config_new_handler(self, handler_function):
        """
        Register a handler function for new configuration messages.
        
        Args:
            handler_function: Function to call when new config is received
        """
        logger.info("Registering config new handler")
        self.tb_client.register_config_new_handler(handler_function)
    
    def register_config_completed_handler(self, handler_function):
        """
        Register a handler function for configuration completed messages.
        
        Args:
            handler_function: Function to call when config completion is received
        """
        logger.info("Registering config completed handler")
        self.tb_client.register_config_completed_handler(handler_function)
    
    def publish_config_completed(self, sensor_mac: str, config_id: str, gateway_id: str):
        """
        Publish configuration completion message.
        
        Args:
            sensor_mac: MAC address of the sensor
            config_id: ID of the configuration that was applied
            gateway_id: ID of the gateway that applied the configuration
            
        Returns:
            True if successfully published, False otherwise
        """
        if self.connected:
            return self.tb_client.publish_config_completed(sensor_mac, config_id, gateway_id)
        else:
            logger.warning("Not connected to ThingsBoard, can't publish config completion")
            return False
    
    def publish_status(self, status):
        """
        Publish gateway status to ThingsBoard.
        
        Args:
            status: Dictionary containing gateway status information
            
        Returns:
            True if successfully published, False otherwise
        """
        logger.info(f"Publishing gateway status: {status}")
        
        # Publish status as device attributes
        if self.connected:
            return self.tb_client.publish_attributes(status)
        else:
            logger.warning("Not connected to ThingsBoard, can't publish status")
            return False
