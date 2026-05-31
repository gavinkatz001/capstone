"""
ThingsBoard Client Module for Fallyx Gateway

This module handles the MQTT communication with ThingsBoard IoT platform,
including sending telemetry data and receiving RPC commands.
"""

import logging
import json
import time
import threading
import os
import paho.mqtt.client as mqtt

logger = logging.getLogger("ThingsBoardClient")

class ThingsBoardClient:
    """
    ThingsBoard Client for handling MQTT-based communication with ThingsBoard IoT platform.
    
    This module is responsible for:
    - Connecting to ThingsBoard via MQTT
    - Publishing telemetry data to ThingsBoard
    - Subscribing to RPC requests
    - Handling and routing received commands
    - Managing reconnection and error handling
    """
    
    def __init__(self, config=None):
        """
        Initialize the ThingsBoard client.
        
        Args:
            config: Configuration dictionary with ThingsBoard connection parameters
        """
        logger.info("Initializing ThingsBoard Client")
        
        self.config = config or {}
        self.host = self.config.get("host", os.environ.get("THINGSBOARD_HOST", "3.99.30.21"))
        self.port = int(self.config.get("port", os.environ.get("THINGSBOARD_PORT", 1883)))
        self.access_token = self.config.get("access_token", os.environ.get("THINGSBOARD_ACCESS_TOKEN", ""))
        self.qos = int(self.config.get("qos", os.environ.get("THINGSBOARD_QOS", 1)))
        self.topic_prefix = "v1/devices/me"
        
        # Create MQTT client and configure callbacks
        self.client = mqtt.Client()
        self.client.username_pw_set(self.access_token)
        self.client.on_connect = self._on_connect
        self.client.on_disconnect = self._on_disconnect
        self.client.on_message = self._on_message
        
        # State management
        self.connected = False
        self.connecting = False  # Track if connection is in progress
        self.running = False
        self.command_handlers = {}
        self.config_new_handler = None
        self.config_completed_handler = None
        self.telemetry_queue = []
        self.lock = threading.Lock()  # For thread-safe operations
        
        # Reconnection management
        self.reconnect_delay = 1
        self.max_reconnect_delay = 60
        self.reconnect_timer = None
        
        logger.info(f"ThingsBoard client initialized with host: {self.host}, port: {self.port}")
    
    def run(self):
        """Run the ThingsBoard client main loop."""
        logger.info("Starting ThingsBoard Client")
        self.running = True
        
        # Connect to ThingsBoard
        self.connect()
        
        while self.running:
            # Process telemetry queue
            self._process_telemetry_queue()
            
            logger.debug("ThingsBoard Client running...")
            time.sleep(5)  # Sleep to prevent CPU hogging
    
    def stop(self):
        """Stop the ThingsBoard client."""
        logger.info("Stopping ThingsBoard Client")
        self.running = False
        self.disconnect()
    
    def connect(self):
        """Connect to ThingsBoard MQTT broker."""
        # Check if we're already connected or in the process of connecting
        with self.lock:
            if self.connected:
                logger.debug("Already connected to ThingsBoard")
                return True
            if self.connecting:
                logger.debug("Connection already in progress")
                return True
            self.connecting = True
        
        if not self.access_token:
            logger.error("Access token not provided, cannot connect to ThingsBoard")
            with self.lock:
                self.connecting = False
            return False
        
        logger.info(f"Connecting to ThingsBoard at {self.host}:{self.port}")
        try:
            self.client.connect(self.host, self.port, keepalive=60)
            self.client.loop_start()
            return True
        except Exception as e:
            logger.error(f"Failed to connect to ThingsBoard: {e}")
            with self.lock:
                self.connecting = False
            self._schedule_reconnect()
            return False
    
    def disconnect(self):
        """Disconnect from ThingsBoard MQTT broker."""
        if self.reconnect_timer:
            self.reconnect_timer.cancel()
            self.reconnect_timer = None
            
        logger.info("Disconnecting from ThingsBoard")
        try:
            self.client.loop_stop()
            self.client.disconnect()
        except Exception as e:
            logger.error(f"Error disconnecting from ThingsBoard: {e}")
        
        with self.lock:
            self.connected = False
            self.connecting = False
    
    def _on_connect(self, client, userdata, flags, rc):
        """Callback when connected to MQTT broker."""
        with self.lock:
            self.connecting = False
            
        if rc == 0:
            logger.info(f"Connected to ThingsBoard at {self.host}:{self.port}")
            with self.lock:
                self.connected = True
            self.reconnect_delay = 1  # Reset reconnect delay
            
            # Subscribe to RPC requests
            self._subscribe_to_rpc()
            
            # Subscribe to shared attributes (for broadcast config updates)
            self._subscribe_to_shared_attributes()
            
            # Publish any queued telemetry
            self._process_telemetry_queue()
        else:
            logger.error(f"Failed to connect to ThingsBoard, return code: {rc}")
            with self.lock:
                self.connected = False
            self._schedule_reconnect()
    
    def _on_disconnect(self, client, userdata, rc):
        """Callback when disconnected from MQTT broker."""
        logger.warning(f"Disconnected from ThingsBoard with code: {rc}")
        with self.lock:
            self.connected = False
            self.connecting = False
        
        if self.running:
            self._schedule_reconnect()
    
    def _schedule_reconnect(self):
        """Schedule a reconnection attempt with exponential backoff."""
        with self.lock:
            if not self.running or self.reconnect_timer or self.connecting:
                return
            
        logger.info(f"Scheduling reconnection in {self.reconnect_delay} seconds")
        
        self.reconnect_timer = threading.Timer(self.reconnect_delay, self._reconnect)
        self.reconnect_timer.daemon = True
        self.reconnect_timer.start()
        
        # Apply exponential backoff
        self.reconnect_delay = min(self.reconnect_delay * 2, self.max_reconnect_delay)
    
    def _reconnect(self):
        """Attempt to reconnect to the MQTT broker."""
        logger.info("Attempting to reconnect to ThingsBoard")
        self.reconnect_timer = None
        
        # Check if we're already connected or connecting
        with self.lock:
            if self.connected or self.connecting:
                logger.debug("Already connected or connecting, skipping reconnect")
                return
            self.connecting = True
        
        try:
            # Ensure we stop any existing loop before reconnecting
            self.client.loop_stop()
            self.client.reconnect()
            self.client.loop_start()
        except Exception as e:
            logger.error(f"Reconnection attempt failed: {e}")
            with self.lock:
                self.connecting = False
            self._schedule_reconnect()
    
    def _subscribe_to_rpc(self):
        """Subscribe to RPC requests from ThingsBoard."""
        topic = f"{self.topic_prefix}/rpc/request/+"
        self.client.subscribe(topic, qos=self.qos)
        logger.info(f"Subscribed to RPC requests: {topic}")
    
    def _subscribe_to_shared_attributes(self):
        """Subscribe to shared attributes updates from ThingsBoard."""
        topic = f"{self.topic_prefix}/attributes"
        self.client.subscribe(topic, qos=self.qos)
        logger.info(f"Subscribed to shared attributes updates: {topic}")
    
    def _on_message(self, client, userdata, msg):
        """Callback when message is received."""
        try:
            logger.info(f"Received message on topic {msg.topic} with payload: {msg.payload.decode()}")
            
            # Check if this is a shared attributes update
            if msg.topic == f"{self.topic_prefix}/attributes":
                self._handle_shared_attributes_message(msg)
                return
            
            # Handle RPC messages (existing logic)
            # Extract request ID from topic
            topic_parts = msg.topic.split('/')
            if len(topic_parts) < 5:
                logger.warning(f"Invalid topic format: {msg.topic}")
                return
                
            request_id = topic_parts[-1]
            
            # Parse message
            try:
                message = json.loads(msg.payload.decode())
                logger.debug(f"Parsed message: {message}")
            except json.JSONDecodeError as e:
                logger.error(f"Error decoding message: {e}")
                return
                
            # Process command
            method = message.get('method')
            params = message.get('params')
            
            if not method:
                logger.warning("No method field in RPC request")
                return
                
            # Route to appropriate handler
            self._process_command(method, params, request_id)
                
        except Exception as e:
            logger.error(f"Error processing message: {e}")
    
    def _handle_shared_attributes_message(self, msg):
        """Handle shared attributes update message from ThingsBoard."""
        try:
            payload = json.loads(msg.payload.decode())
            logger.info(f"Received shared attributes update from server: {payload}")

            # 1. Handle broadcasted NEW configuration requests
            if "broadcast_config_request" in payload:
                config_request_data = payload["broadcast_config_request"]
                # It's good practice to ensure the data is a dictionary
                if isinstance(config_request_data, dict):
                    logger.info(f"Processing 'broadcast_config_request' from shared attributes: {config_request_data}")
                    if self.config_new_handler:
                        # Pass the inner dictionary containing sensor_mac, config_id, etc.
                        self.config_new_handler(config_request_data)
                    else:
                        logger.warning("No config_new_handler registered to process 'broadcast_config_request'")
                else:
                    logger.warning(f"'broadcast_config_request' payload is not a dictionary: {config_request_data}")

            # 2. Handle broadcasted CONFIGURATION COMPLETION notices
            if "broadcast_config_completion" in payload:
                completion_notice_data = payload["broadcast_config_completion"]
                # Ensure data is a dictionary
                if isinstance(completion_notice_data, dict):
                    logger.info(f"Processing 'broadcast_config_completion' notice from shared attributes: {completion_notice_data}")
                    if self.config_completed_handler:
                        # Pass the inner dictionary containing sensor_mac, config_id, completed_by_gateway_id
                        self.config_completed_handler(completion_notice_data)
                    else:
                        logger.warning("No config_completed_handler registered to process 'broadcast_config_completion'")
                else:
                    logger.warning(f"'broadcast_config_completion' payload is not a dictionary: {completion_notice_data}")
            
        except json.JSONDecodeError as e:
            logger.error(f"Error decoding shared attributes message: {e}")
        except Exception as e:
            logger.error(f"Error handling shared attributes message: {e}")
    
    def _process_command(self, method, params, request_id):
        """Process a command received from ThingsBoard."""
        logger.info(f"Processing command: {method}, request_id: {request_id}")
        
        # Route to appropriate handler
        if method in self.command_handlers:
            try:
                handler = self.command_handlers[method]
                result = handler(params)
                
                # Send response
                self._send_rpc_response(request_id, result)
            except Exception as e:
                logger.error(f"Error executing command handler: {e}")
                self._send_rpc_response(request_id, {"error": str(e)})
        else:
            logger.warning(f"No handler for method: {method}")
            self._send_rpc_response(request_id, {"error": f"Unsupported method: {method}"})
    
    def _send_rpc_response(self, request_id, result):
        """Send RPC response to ThingsBoard."""
        if not self.connected:
            logger.warning("Not connected to ThingsBoard, can't send RPC response")
            return False
            
        topic = f"{self.topic_prefix}/rpc/response/{request_id}"
        payload = json.dumps(result)
        
        try:
            self.client.publish(topic, payload, qos=self.qos)
            logger.debug(f"Sent RPC response to {topic}: {payload}")
            return True
        except Exception as e:
            logger.error(f"Error sending RPC response: {e}")
            return False
    
    def publish_telemetry(self, data):
        """
        Publish telemetry data to ThingsBoard.
        
        Args:
            data: Dictionary containing telemetry data
            
        Returns:
            True if successfully published or queued, False otherwise
        """
        # Add to queue if not connected
        if not self.connected:
            logger.warning("Not connected to ThingsBoard, queueing telemetry data")
            with self.lock:
                self.telemetry_queue.append(data)
            return True
            
        try:
            topic = f"{self.topic_prefix}/telemetry"
            payload = json.dumps(data)
            result = self.client.publish(topic, payload, qos=self.qos)
            result.wait_for_publish()
            
            if result.rc == mqtt.MQTT_ERR_SUCCESS:
                logger.debug(f"Published telemetry data: {data}")
                return True
            else:
                logger.warning(f"Failed to publish telemetry, queuing data. Error code: {result.rc}")
                with self.lock:
                    self.telemetry_queue.append(data)
                return False
        except Exception as e:
            logger.error(f"Error publishing telemetry: {e}")
            with self.lock:
                self.telemetry_queue.append(data)
            return False
    
    def _process_telemetry_queue(self):
        """Process queued telemetry data."""
        if not self.telemetry_queue or not self.connected:
            return
            
        logger.info(f"Processing telemetry queue: {len(self.telemetry_queue)} items")
        
        with self.lock:
            # Make a copy of the queue and clear the original
            queue_copy = self.telemetry_queue.copy()
            self.telemetry_queue = []
        
        # Try to publish each item
        for data in queue_copy:
            try:
                topic = f"{self.topic_prefix}/telemetry"
                payload = json.dumps(data)
                result = self.client.publish(topic, payload, qos=self.qos)
                result.wait_for_publish()
                
                if result.rc != mqtt.MQTT_ERR_SUCCESS:
                    logger.warning(f"Failed to publish queued telemetry, re-queuing. Error code: {result.rc}")
                    with self.lock:
                        self.telemetry_queue.append(data)
            except Exception as e:
                logger.error(f"Error publishing queued telemetry: {e}")
                with self.lock:
                    self.telemetry_queue.append(data)
    
    def register_command_handler(self, command_type, handler_function):
        """
        Register a handler function for a specific command type.
        
        Args:
            command_type: The type of command to handle
            handler_function: Function to call when command is received
        """
        logger.info(f"Registering handler for command type: {command_type}")
        self.command_handlers[command_type] = handler_function
    
    def register_config_new_handler(self, handler_function):
        """
        Register a handler function for new configuration messages.
        
        Args:
            handler_function: Function to call when new config is received
        """
        logger.info("Registering config new handler")
        self.config_new_handler = handler_function
    
    def register_config_completed_handler(self, handler_function):
        """
        Register a handler function for configuration completed messages.
        
        Args:
            handler_function: Function to call when config completion is received
        """
        logger.info("Registering config completed handler")
        self.config_completed_handler = handler_function
    
    def publish_config_completed(self, sensor_mac: str, config_id: str, gateway_id: str):
        """
        Publish a configuration completion EVENT to ThingsBoard.
        This event signals that THIS gateway has completed the configuration.
        It's intended to be picked up by a Rule Engine rule for further processing (e.g., broadcasting).

        Args:
            sensor_mac: MAC address of the sensor
            config_id: ID of the configuration that was applied
            gateway_id: ID of THIS gateway that applied the configuration

        Returns:
            True if successfully published, False otherwise
        """
        if not self.connected:
            logger.warning("Not connected to ThingsBoard, can't publish config completion event")
            return False

        try:
            # Create a payload clearly identifying this as an event reported by this gateway
            event_payload = {
                "config_completion_event": {  # Key indicating an event for Rule Engine
                    "sensor_mac": sensor_mac,
                    "config_id": config_id,
                    "gateway_id": gateway_id  # ID of this gateway
                },
                "event_timestamp": int(time.time() * 1000) # Timestamp of the event
            }

            topic = f"{self.topic_prefix}/attributes" # Publish as client-side attributes
            payload_json = json.dumps(event_payload)
            result = self.client.publish(topic, payload_json, qos=self.qos) # Ensure self.qos is appropriate (e.g., 1)
            result.wait_for_publish() # Important for ensuring message is sent

            if result.rc == mqtt.MQTT_ERR_SUCCESS:
                logger.info(f"Published config completion event for sensor {sensor_mac}, config_id {config_id} by gateway {gateway_id}")
                return True
            else:
                logger.warning(f"Failed to publish config completion event. Error code: {result.rc}")
                return False
        except Exception as e:
            logger.error(f"Error publishing config completion event: {e}")
            return False
    
    def publish_attributes(self, attributes):
        """
        Publish device attributes to ThingsBoard.
        
        Args:
            attributes: Dictionary containing device attributes
            
        Returns:
            True if successfully published, False otherwise
        """
        if not self.connected:
            logger.warning("Not connected to ThingsBoard, can't publish attributes")
            return False
            
        try:
            topic = f"{self.topic_prefix}/attributes"
            payload = json.dumps(attributes)
            result = self.client.publish(topic, payload, qos=self.qos)
            result.wait_for_publish()
            
            if result.rc == mqtt.MQTT_ERR_SUCCESS:
                logger.debug(f"Published attributes: {attributes}")
                return True
            else:
                logger.warning(f"Failed to publish attributes. Error code: {result.rc}")
                return False
        except Exception as e:
            logger.error(f"Error publishing attributes: {e}")
            return False