"""
Command Module for Fallyx Gateway

This module handles commands from the server and ThingsBoard RPC requests.
It also handles emergency calls when triggered by other modules.
"""

import logging
import time
import threading
import json
import requests
import os

logger = logging.getLogger("CommandModule")

class CommandModule:
    """
    Command Module for handling server commands, ThingsBoard RPC requests, and emergency calls.
    
    This module is responsible for:
    - Receiving commands from the server and ThingsBoard
    - Relaying commands to connected sensors
    - Processing and reporting command responses
    - Handling emergency calls
    - Responding to ThingsBoard RPC requests
    """
    
    def __init__(self, fall_detection_coordinator=None, voice_assistant_coordinator=None):
        """Initialize the Command Module."""
        logger.info("Initializing Command Module")
        self.ble_module = None
        self.network_module = None
        self.fall_detection_coordinator = fall_detection_coordinator
        self.voice_assistant_coordinator = voice_assistant_coordinator
        self.running = False
        self.command_queue = []  # Queue for pending commands
        self.lock = threading.Lock()  # For thread-safe operations
        self.keyword_detector = None 
        
        # Gateway ID
           
        self.gateway_id = os.environ.get("GATEWAY_ID", f"gateway-test")
        # In-memory pending configurations queue
        self.pending_sensor_configs = {}  # sensor_mac -> {"config_id": str, "config_data": dict, "received_at": timestamp}
        self.pending_configs_lock = threading.Lock()  # For thread-safe access to pending configs
        
        # Emergency call configuration
        self.emergency_number = None
        self.emergency_api_url = "https://724cu8r3wk.execute-api.ca-central-1.amazonaws.com/Prod/outcall"
        self.emergency_user_info = {
            "emergencyFirstname": "SRR",
            "userFirstName": "Unknown Building",  # Gateway building location
            "userLastName": "Unknown Room"        # Gateway room location
        }
    
    def set_ble_module(self, ble_module):
        """Set the BLE module reference for sending commands to sensors."""
        self.ble_module = ble_module
        logger.info("BLE module set")
    
    def set_network_module(self, network_module):
        """Set the Network module reference for ThingsBoard integration."""
        self.network_module = network_module
        logger.info("Network module set")
        self._register_thingsboard_handlers()
    
    def set_fall_detection_coordinator(self, fall_detection_coordinator):
        """Set the Fall Detection Coordinator reference."""
        self.fall_detection_coordinator = fall_detection_coordinator
        logger.info("Fall Detection Coordinator set")
    
    def set_voice_assistant_coordinator(self, voice_assistant_coordinator):
        """Set the Voice Assistant Coordinator reference."""
        self.voice_assistant_coordinator = voice_assistant_coordinator
        logger.info("Voice Assistant Coordinator set")
    
    def set_keyword_detector(self, keyword_detector):
        """Set the Keyword Detector reference."""
        self.keyword_detector = keyword_detector
        logger.info("Keyword Detector set")
    
    def _register_thingsboard_handlers(self):
        """Register handlers for ThingsBoard RPC commands."""
        if not self.network_module:
            logger.warning("Cannot register ThingsBoard handlers - Network module not set")
            return
        
        logger.info("Registering ThingsBoard command handlers")
        
        # Register specific command handlers
        self.network_module.register_command_handler("fallEvent", self.handle_fall_event)
        self.network_module.register_command_handler("getDeviceStatus", self.handle_get_device_status)
        self.network_module.register_command_handler("checkConnectivity", self.handle_check_connectivity)
        
        # Register voice assistant RPC handlers
        self.network_module.register_command_handler("play_response", self.handle_play_response)
        self.network_module.register_command_handler("play_and_record", self.handle_play_and_record)
        
        # Register MQTT configuration handlers
        self.network_module.register_config_new_handler(self.handle_new_config_request)
        self.network_module.register_config_completed_handler(self.handle_config_completed_notification)
        
        # Register voice assistant configuration handler
        self.network_module.register_command_handler("updateVoiceAssistantConfig", self.handle_update_voice_assistant_config)
        
        logger.info("ThingsBoard command handlers registered")
    
    def set_emergency_number(self, number):
        """Set the emergency contact number."""
        self.emergency_number = number
        logger.info(f"Emergency number set: {number}")
    
    def set_emergency_user_info(self, first_name, last_name="", emergency_contact_name="SRR"):
        """
        Set the location information for emergency calls.
        
        Args:
            first_name: Building/high-level location where the gateway is located
            last_name: Room/specific location where the gateway is located
            emergency_contact_name: Name of the emergency contact person
        """
        self.emergency_user_info = {
            "emergencyFirstname": emergency_contact_name,
            "userFirstName": first_name,   # Building location
            "userLastName": last_name      # Room location
        }
        logger.info(f"Emergency location info set - Building: {first_name}, Room: {last_name}, Contact: {emergency_contact_name}")
    
    def run(self):
        """Run the Command Module main loop."""
        logger.info("Starting Command Module")
        self.running = True
        
        while self.running:
            # Process command queue
            self._process_command_queue()
            
            logger.debug("Command Module running...")
            time.sleep(5)  # Sleep to prevent CPU hogging
    
    def stop(self):
        """Stop the Command Module."""
        logger.info("Stopping Command Module")
        self.running = False
    
    def _process_command_queue(self):
        """Process queued commands."""
        if not self.command_queue:
            return
            
        logger.info(f"Processing command queue: {len(self.command_queue)} items")
        
        with self.lock:
            if self.command_queue:
                command = self.command_queue.pop(0)
                self._execute_command(command)
    
    def _execute_command(self, command):
        """Execute a command by relaying it to the appropriate sensor."""
        logger.info(f"Executing command: {command}")
        
        # Extract target sensor MAC address and action
        target = command.get("target")
        action = command.get("action")
        
        if not target or not action:
            logger.error("Invalid command format: missing target or action")
            return False
        
        # Relay command to sensor via BLE module
        if self.ble_module:
            logger.info(f"Relaying command to sensor {target}: {action}")
            return self.ble_module.send_command_to_device(target, command)
        else:
            logger.error("BLE module not available, cannot relay command")
            return False
    
    def handle_command(self, command):
        """Handle a command received from the server."""
        logger.info(f"Received command from server: {command}")
        
        # Validate command
        if not self._validate_command(command):
            logger.error("Invalid command received")
            return False
        
        # Add to command queue
        with self.lock:
            self.command_queue.append(command)
        
        logger.info("Command queued for execution")
        return True
    
    def _validate_command(self, command):
        """Validate a command structure."""
        logger.info("Validating command")
        # This would validate the command against a schema
        return True
    
    def report_command_result(self, command_id, success, response=None):
        """Report the result of a command execution back to the server."""
        logger.info(f"Reporting command result: {command_id}, success: {success}")
        # This would send the command result back to the server
    
    # Emergency call functionality
    
    def handle_emergency_call(self):
        """
        Handle an emergency call request.
        This method is called by other modules when an emergency is detected.
        """
        logger.info("Handling emergency call request")
        
        if self.emergency_number is None:
            logger.error("Emergency number not set, cannot make emergency call")
            return False
        
        # Make the emergency call
        return self._call_emergency_service()
    
    def _call_emergency_service(self):
        """
        Call the emergency service API.
        Returns True if the call was successful, False otherwise.
        """
        logger.info(f"Calling emergency service: {self.emergency_number}")
        
        try:
            # Prepare the request data
            headers = {'Content-Type': 'application/json'}
            data = {
                "emergencyPhoneNumber": self.emergency_number,
                **self.emergency_user_info
            }
            
            # Make the API request
            response = requests.post(self.emergency_api_url, json=data, headers=headers)
            status_code = response.status_code
            
            logger.info(f"Emergency request sent, response status code: {status_code}")
            
            # Check if the request was successful
            if status_code == 200:
                logger.info("Emergency call successful")
                return True
            else:
                logger.error(f"Emergency call failed with status code: {status_code}")
                return False
        except Exception as e:
            logger.error(f"Error making emergency call: {e}")
            return False
        
    # ThingsBoard RPC command handlers
    
    def handle_fall_event(self, params):
        """
        Handle a fall event command from ThingsBoard.
        
        This handler is called when a 'fallEvent' RPC command is received from ThingsBoard.
        It uses the FallDetectionCoordinator to orchestrate the fall detection process.
        
        Args:
            params: Parameters from the RPC call, may include device ID and reason
            
        Returns:
            Dictionary with the result of the fall detection and potential emergency call
        """
        logger.info(f"Handling fallEvent command from ThingsBoard with params: {params}")
        
        # Extract parameters
        device_id = params.get("deviceId", "unknown")
        reason = params.get("reason", "ThingsBoard trigger")
        
        # Check if fall detection coordinator is available
        if not self.fall_detection_coordinator:
            logger.error("Fall Detection Coordinator not available, cannot run fall detection")
            return {
                "success": False,
                "deviceId": device_id,
                "timestamp": int(time.time() * 1000),
                "intent": "unclear",
                "message": "Fall Detection Coordinator not available"
            }
        
        try:
            # Configure paths for audio files if available
            response_audio_ok_path = params.get("responseAudioOkPath")
            response_audio_not_ok_path = params.get("responseAudioNotOkPath")
            recording_duration = params.get("recordingDuration", 5.0)
            
            # Start fall detection verification in a separate thread
            logger.info("Initiating fall detection process from RPC fallEvent")
            verification_thread = self.fall_detection_coordinator.start_fall_verification_async(
                response_audio_ok_path=response_audio_ok_path,
                response_audio_not_ok_path=response_audio_not_ok_path,
                recording_duration=recording_duration
            )
            
            # Wait for the fall detection to complete with a timeout
            verification_thread.join(timeout=45.0)  # 45 seconds timeout (updated for TTS step)
            
            # Get the result from the coordinator
            result_data = self.fall_detection_coordinator.get_last_verification_result()
            
            if verification_thread.is_alive():
                logger.warning("Fall detection verification thread did not complete in time")
                # Update status to indicate timeout if still pending
                if result_data.get("status") == "pending":
                    with self.fall_detection_coordinator.lock:
                        self.fall_detection_coordinator.last_verification_result["status"] = "timeout"
                        self.fall_detection_coordinator.last_verification_result["message"] = "Fall detection verification timed out"
                
                return {
                    "success": False,
                    "deviceId": device_id,
                    "timestamp": int(time.time() * 1000),
                    "intent": result_data.get("intent", "unclear"),
                    "message": "Fall detection verification timed out",
                    "steps_completed": result_data.get("steps_completed", [])
                }
            
            # Extract key information for the response
            intent = result_data.get("intent", "unclear")
            emergency_call_triggered = result_data.get("emergency_call_triggered", False)
            emergency_call_successful = result_data.get("emergency_call_successful")
            message = result_data.get("message", "Fall detection completed")
            
            # Determine overall success
            if intent == "ok":
                success = True
            elif intent == "not_ok":
                success = emergency_call_successful if emergency_call_successful is not None else False
            else:
                success = True  # Unclear is handled gracefully
            
            # Return comprehensive response
            return {
                "success": success,
                "deviceId": device_id,
                "timestamp": int(time.time() * 1000),
                "intent": intent,
                "message": message,
                "emergency_call_triggered": emergency_call_triggered,
                "emergency_call_successful": emergency_call_successful,
                "transcribed_text": result_data.get("transcribed_text"),
                "steps_completed": result_data.get("steps_completed", [])
            }
            
        except Exception as e:
            logger.error(f"Error during fall detection: {e}")
            return {
                "success": False,
                "deviceId": device_id,
                "timestamp": int(time.time() * 1000),
                "intent": "unclear",
                "message": f"Error during fall detection: {str(e)}",
                "error": str(e)
            }
    
    def handle_get_device_status(self, params):
        """
        Handle a device status request from ThingsBoard.
        
        Args:
            params: Parameters from the RPC call, including the device ID
            
        Returns:
            Dictionary with the device status
        """
        logger.info(f"Handling getDeviceStatus command from ThingsBoard with params: {params}")
        
        # Extract device ID
        device_id = params.get("deviceId")
        
        if not device_id:
            logger.error("Missing deviceId in getDeviceStatus command")
            return {"error": "Missing deviceId parameter"}
        
        # Get device status from BLE module
        status = {"connected": False, "batteryLevel": None}
        
        if self.ble_module:
            # Check if device is connected
            connected = self.ble_module.is_device_connected(device_id)
            status["connected"] = connected
            
            # Get battery level if available
            if connected:
                try:
                    battery_level = self.ble_module.get_device_battery_level(device_id)
                    status["batteryLevel"] = battery_level
                except:
                    logger.warning(f"Could not retrieve battery level for device {device_id}")
        
        # Return status with timestamp
        status["timestamp"] = int(time.time() * 1000)
        status["deviceId"] = device_id
        
        return status
    
    def handle_check_connectivity(self, params):
        """
        Handle a connectivity check request from ThingsBoard.
        
        Args:
            params: Parameters from the RPC call
            
        Returns:
            Dictionary with connectivity status
        """
        logger.info("Handling checkConnectivity command from ThingsBoard")
        
        # Check BLE module status
        ble_status = self.ble_module is not None and self.ble_module.running
        
        # Return status
        return {
            "gateway": "online",
            "bleModule": "online" if ble_status else "offline",
            "timestamp": int(time.time() * 1000)
        }
    
    def handle_play_response(self, params):
        """
        Handle a play_response RPC command from ThingsBoard for voice assistant.
        
        This handler processes final response messages that end the chat session.
        It delegates to the VoiceAssistantCoordinator to handle the actual response.
        
        Args:
            params: Parameters from the RPC call, expected to contain:
                   - chat_session_id: UUID of the chat session
                   - text_to_speak: Final response text to be spoken
                   
        Returns:
            Dictionary with the result of the play response operation
        """
        logger.info(f"Handling play_response command from ThingsBoard with params: {params}")
        
        # Validate that voice assistant coordinator is available
        if not self.voice_assistant_coordinator:
            logger.error("Voice Assistant Coordinator not available, cannot handle play_response")
            return {
                "success": False,
                "timestamp": int(time.time() * 1000),
                "message": "Voice Assistant Coordinator not available"
            }
        
        # Extract and validate required parameters
        chat_session_id = params.get("chat_session_id")
        text_to_speak = params.get("text_to_speak")
        
        if not chat_session_id:
            logger.error("Missing chat_session_id in play_response command")
            return {
                "success": False,
                "timestamp": int(time.time() * 1000),
                "message": "Missing required parameter: chat_session_id"
            }
        
        if not text_to_speak:
            logger.error("Missing text_to_speak in play_response command")
            return {
                "success": False,
                "timestamp": int(time.time() * 1000),
                "message": "Missing required parameter: text_to_speak"
            }
        
        try:
            # Delegate to voice assistant coordinator
            logger.info(f"Delegating play_response to coordinator for session {chat_session_id}")
            success = self.voice_assistant_coordinator.handle_play_response(chat_session_id, text_to_speak)
            
            # Return response based on success
            return {
                "success": success,
                "chat_session_id": chat_session_id,
                "timestamp": int(time.time() * 1000),
                "message": "Play response completed successfully" if success else "Play response failed"
            }
            
        except Exception as e:
            logger.error(f"Error handling play_response for session {chat_session_id}: {e}")
            return {
                "success": False,
                "chat_session_id": chat_session_id,
                "timestamp": int(time.time() * 1000),
                "message": f"Error handling play_response: {str(e)}",
                "error": str(e)
            }
    
    def handle_play_and_record(self, params):
        """
        Handle a play_and_record RPC command from ThingsBoard for voice assistant.
        
        This handler processes response messages followed by recording for continued conversation.
        It delegates to the VoiceAssistantCoordinator to handle the actual response and recording.
        
        Args:
            params: Parameters from the RPC call, expected to contain:
                   - chat_session_id: UUID of the chat session
                   - text_to_speak: Response text to be spoken before recording
                   
        Returns:
            Dictionary with the result of the play and record operation
        """
        logger.info(f"Handling play_and_record command from ThingsBoard with params: {params}")
        
        # Validate that voice assistant coordinator is available
        if not self.voice_assistant_coordinator:
            logger.error("Voice Assistant Coordinator not available, cannot handle play_and_record")
            return {
                "success": False,
                "timestamp": int(time.time() * 1000),
                "message": "Voice Assistant Coordinator not available"
            }
        
        # Extract and validate required parameters
        chat_session_id = params.get("chat_session_id")
        text_to_speak = params.get("text_to_speak")
        
        if not chat_session_id:
            logger.error("Missing chat_session_id in play_and_record command")
            return {
                "success": False,
                "timestamp": int(time.time() * 1000),
                "message": "Missing required parameter: chat_session_id"
            }
        
        if not text_to_speak:
            logger.error("Missing text_to_speak in play_and_record command")
            return {
                "success": False,
                "timestamp": int(time.time() * 1000),
                "message": "Missing required parameter: text_to_speak"
            }
        
        try:
            # Delegate to voice assistant coordinator
            logger.info(f"Delegating play_and_record to coordinator for session {chat_session_id}")
            success = self.voice_assistant_coordinator.handle_play_and_record(chat_session_id, text_to_speak)
            
            # Return response based on success
            return {
                "success": success,
                "chat_session_id": chat_session_id,
                "timestamp": int(time.time() * 1000),
                "message": "Play and record completed successfully" if success else "Play and record failed"
            }
            
        except Exception as e:
            logger.error(f"Error handling play_and_record for session {chat_session_id}: {e}")
            return {
                "success": False,
                "chat_session_id": chat_session_id,
                "timestamp": int(time.time() * 1000),
                "message": f"Error handling play_and_record: {str(e)}",
                "error": str(e)
            }
    
    def handle_update_voice_assistant_config(self, params):
        """
        Handles the RPC to dynamically configure the voice assistant.
        
        Args:
            params (dict): RPC parameters, e.g., {"enabled": bool, "chatApiBaseUrl": str}
            
        Returns:
            dict: A result dictionary for the RPC response.
        """
        logger.info(f"Handling updateVoiceAssistantConfig command with params: {params}")
        
        updated_settings = []
        errors = []

        # --- Handle enabling/disabling the keyword detector ---
        if 'enabled' in params:
            is_enabled = params.get('enabled')
            if self.keyword_detector:
                try:
                    if is_enabled:
                        self.keyword_detector.resume()
                        logger.info("Voice assistant keyword detector resumed.")
                        updated_settings.append("voice_assistant_enabled")
                    else:
                        self.keyword_detector.pause()
                        logger.info("Voice assistant keyword detector paused.")
                        updated_settings.append("voice_assistant_disabled")
                except Exception as e:
                    error_msg = f"Failed to change keyword detector state: {e}"
                    logger.error(error_msg)
                    errors.append(error_msg)
            else:
                error_msg = "Keyword detector not available to be configured."
                logger.warning(error_msg)
                errors.append(error_msg)

        # --- Handle updating the backend URL ---
        if 'chatApiBaseUrl' in params:
            new_url = params.get('chatApiBaseUrl')
            if self.voice_assistant_coordinator:
                if isinstance(new_url, str) and new_url.startswith('http'):
                    self.voice_assistant_coordinator.backend_base_url = new_url
                    logger.info(f"Voice assistant backend URL updated to: {new_url}")
                    updated_settings.append("chatApiBaseUrl_updated")
                else:
                    error_msg = f"Invalid chatApiBaseUrl provided: {new_url}"
                    logger.warning(error_msg)
                    errors.append(error_msg)
            else:
                error_msg = "Voice assistant coordinator not available to be configured."
                logger.warning(error_msg)
                errors.append(error_msg)

        # --- Prepare the RPC response ---
        if not errors:
            return {
                "success": True,
                "message": "Voice assistant configuration updated.",
                "updated_settings": updated_settings,
                "timestamp": int(time.time() * 1000)
            }
        else:
            return {
                "success": False,
                "message": "Failed to update voice assistant configuration.",
                "errors": errors,
                "timestamp": int(time.time() * 1000)
            }
    
    def handle_generic_command(self, params):
        """
        Generic handler for commands that don't have a specific handler.
        
        Args:
            params: Command parameters
            
        Returns:
            Dictionary with the result
        """
        logger.info(f"Handling generic command with params: {params}")
        
        # Extract command details
        command_type = params.get("type")
        device_id = params.get("deviceId")
        
        if not command_type or not device_id:
            logger.error("Missing type or deviceId in generic command")
            return {"error": "Missing required parameters (type, deviceId)"}
        
        # Create command for queue
        command = {
            "target": device_id,
            "action": command_type,
            "params": params
        }
        
        # Queue command for execution
        with self.lock:
            self.command_queue.append(command)
        
        return {
            "success": True,
            "deviceId": device_id,
            "commandType": command_type,
            "timestamp": int(time.time() * 1000),
            "message": f"Command {command_type} queued for execution"
        }
    
    # New configuration management handlers
    
    def handle_new_config_request(self, payload: dict):
        """
        Handle a new configuration request (potentially from a broadcasted shared attribute).
        
        Args:
            payload: Dictionary expected to contain sensor_mac, config_id, and config_data.
                     This comes from the 'broadcast_config_request' shared attribute.
        """
        logger.info(f"Handling new config request (from shared attribute or RPC): {payload}")
        
        # Validate payload keys (these should match what's set in 'broadcast_config_request')
        sensor_mac = payload.get("sensor_mac")
        config_id = payload.get("config_id")
        config_data = payload.get("config_data")
        # gateway_id = payload.get("gateway_id") # This might be originator, or not present if broadcast
        # source = payload.get("source") # Might be useful for debugging

        if not sensor_mac or not config_id or not config_data:
            logger.error(f"Invalid new config payload: missing sensor_mac, config_id, or config_data. Payload: {payload}")
            return
        
        # Add/update the entry in pending configurations
        with self.pending_configs_lock:
            self.pending_sensor_configs[sensor_mac.lower()] = {
                "config_id": config_id,
                "config_data": config_data,
                "received_at": time.time()
            }
        
        logger.info(f"Queued configuration {config_id} for sensor {sensor_mac}")
    
    def handle_config_completed_notification(self, payload: dict):
        """
        Handle a configuration completed notification (potentially from a broadcasted shared attribute).
        This removes the completed configuration from the local pending queue.
        
        Args:
            payload: Dictionary expected to contain sensor_mac, config_id,
                     and an identifier for the gateway that completed the task (e.g., 'completed_by_gateway_id').
                     This comes from the 'broadcast_config_completion' shared attribute.
        """
        logger.info(f"Handling config completed notification (from shared attribute or RPC): {payload}")
        
        # Validate payload keys (these should match what's set in 'broadcast_config_completion')
        sensor_mac = payload.get("sensor_mac")
        config_id = payload.get("config_id")
        # Key for the completing gateway's ID might be 'completed_by_gateway_id', 'gateway_id', etc.
        # Let's assume 'completed_by_gateway_id' for clarity, but also check 'gateway_id' for backward compatibility
        completing_gateway_id = payload.get("completed_by_gateway_id") or payload.get("gateway_id")
        # source = payload.get("source") # Might be useful for debugging

        if not sensor_mac or not config_id: # Completing gateway ID is useful but not strictly needed to clear queue
            logger.error(f"Invalid config completed payload: missing sensor_mac or config_id. Payload: {payload}")
            return
        
        # Remove the completed configuration from our pending queue
        with self.pending_configs_lock:
            sensor_key = sensor_mac.lower()
            if sensor_key in self.pending_sensor_configs:
                pending_config = self.pending_sensor_configs[sensor_key]
                
                # Verify it's the same config_id before removing
                if pending_config["config_id"] == config_id:
                    del self.pending_sensor_configs[sensor_key]
                    logger.info(f"Removed completed configuration {config_id} for sensor {sensor_mac} (completed by gateway {completing_gateway_id or 'unknown'})")
                else:
                    logger.warning(f"Config ID mismatch for sensor {sensor_mac}: local pending={pending_config['config_id']}, reported completed={config_id}. Notification from {completing_gateway_id or 'unknown'}.")
            else:
                logger.debug(f"No local pending configuration found for sensor {sensor_mac} to remove (config_id: {config_id}, completed by {completing_gateway_id or 'unknown'}). Already processed or not for this gateway.")
    
    async def process_sensor_connection_async(self, sensor_mac: str):
        """
        Process a sensor connection and attempt to push any pending configuration asynchronously.
        This method is called by BLEModule when a sensor connects.
        
        Args:
            sensor_mac: MAC address of the connected sensor
        """
        logger.info(f"Processing sensor connection (async): {sensor_mac}")
        
        # Check for pending configuration
        with self.pending_configs_lock:
            logger.info(f"available pending configurations: {self.pending_sensor_configs.keys()}")
            sensor_key = sensor_mac.lower()
            if sensor_key not in self.pending_sensor_configs:
                logger.info(f"No pending configuration found for sensor {sensor_mac}")
                return
            
            pending_config = self.pending_sensor_configs[sensor_key].copy()
        
        logger.info(f"Found pending configuration {pending_config['config_id']} for sensor {sensor_mac}")
        
        # Check if BLE module is available
        if not self.ble_module:
            logger.error("BLE module not available, cannot push configuration")
            return
        
        # Construct the configuration command
        config_command = {
            "action": "updateSensorConfig",
            "config_id": pending_config["config_id"],
            "payload": pending_config["config_data"]
        }
        
        logger.info(f"Attempting to push configuration {pending_config['config_id']} to sensor {sensor_mac}")
        
        # Send command to device asynchronously
        success = await self.ble_module.send_command_to_device_async(sensor_mac, config_command)
        
        if success:
            logger.info(f"Successfully sent configuration {pending_config['config_id']} to sensor {sensor_mac}")
            
            # Publish completion notification
            if self.network_module:
                self.network_module.publish_config_completed(sensor_mac, pending_config["config_id"], self.gateway_id)
            
            # Remove the configuration from our queue (re-check config_id before deleting)
            with self.pending_configs_lock:
                if (sensor_key in self.pending_sensor_configs and
                    self.pending_sensor_configs[sensor_key]["config_id"] == pending_config["config_id"]):
                    del self.pending_sensor_configs[sensor_key]
                    logger.info(f"Removed completed configuration {pending_config['config_id']} for sensor {sensor_mac}")
        else:
            logger.warning(f"Failed to send configuration {pending_config['config_id']} to sensor {sensor_mac}, keeping in queue")

    def process_sensor_connection(self, sensor_mac: str):
        """
        Process a sensor connection and attempt to push any pending configuration.
        This method is called by BLEModule when a sensor connects.
        
        DEPRECATED: This synchronous method is kept for compatibility but may cause deadlocks
        when called from the BLE module's event loop. Use process_sensor_connection_async instead.
        
        Args:
            sensor_mac: MAC address of the connected sensor
        """
        logger.info(f"Processing sensor connection: {sensor_mac}")
        
        # Check for pending configuration
        with self.pending_configs_lock:
            logger.info(f"available pending configurations: {self.pending_sensor_configs.keys()}")
            sensor_key = sensor_mac.lower()
            if sensor_key not in self.pending_sensor_configs:
                logger.info(f"No pending configuration found for sensor {sensor_mac}")
                return
            
            pending_config = self.pending_sensor_configs[sensor_key].copy()
        
        logger.info(f"Found pending configuration {pending_config['config_id']} for sensor {sensor_mac}")
        
        # Check if BLE module is available
        if not self.ble_module:
            logger.error("BLE module not available, cannot push configuration")
            return
        
        # Construct the configuration command
        config_command = {
            "action": "updateSensorConfig",
            "config_id": pending_config["config_id"],
            "payload": pending_config["config_data"]
        }
        
        logger.info(f"Attempting to push configuration {pending_config['config_id']} to sensor {sensor_mac}")
        
        # Send command to device
        success = self.ble_module.send_command_to_device(sensor_mac, config_command)
        
        if success:
            logger.info(f"Successfully sent configuration {pending_config['config_id']} to sensor {sensor_mac}")
            
            # Publish completion notification
            if self.network_module:
                self.network_module.publish_config_completed(sensor_mac, pending_config["config_id"], self.gateway_id)
            
            # Remove the configuration from our queue (re-check config_id before deleting)
            with self.pending_configs_lock:
                if (sensor_key in self.pending_sensor_configs and
                    self.pending_sensor_configs[sensor_key]["config_id"] == pending_config["config_id"]):
                    del self.pending_sensor_configs[sensor_key]
                    logger.info(f"Removed completed configuration {pending_config['config_id']} for sensor {sensor_mac}")
        else:
            logger.warning(f"Failed to send configuration {pending_config['config_id']} to sensor {sensor_mac}, keeping in queue")
