#!/usr/bin/env python3
"""
Fallyx Gateway - Main Controller

This is the main entry point for the Fallyx Gateway application.
It initializes and manages all the modules required for the gateway functionality.
"""

import threading
import time
import logging
import sys
import os
import signal
import enum
from typing import Optional
from dotenv import load_dotenv

# Load environment variables from .env file
# Look for .env file in the parent directory (project root)
load_dotenv(dotenv_path=os.path.join(os.path.dirname(__file__), '..', '.env'))

# Import modules
from gateway.modules.ble_module import BLEModule
from gateway.modules.data_module import DataManager
from gateway.modules.network_module import NetworkModule
from gateway.modules.command_module import CommandModule
from gateway.modules.audio_module import AudioModule
from gateway.modules.monitoring_module import MonitoringService
from gateway.modules.keyword_detector import KeywordDetector

# Import new components
from gateway.adapters.openai_adapter import OpenAIAdapter
from gateway.coordinators.fall_detection_coordinator import FallDetectionCoordinator
from gateway.coordinators.voice_assistant_coordinator import VoiceAssistantCoordinator
from gateway.audio_types import AudioState

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler('gateway.log')
    ]
)

logger = logging.getLogger("MainController")

class MainController:
    """
    Main controller class for the Fallyx Gateway.
    Initializes and manages all modules with audio state management.
    """
    
    def __init__(self):
        """Initialize the main controller and all modules."""
        logger.info("Initializing Fallyx Gateway...")
        
        # Audio state management
        self.audio_state = AudioState.IDLE
        self.audio_lock = threading.Lock()
        self.audio_lock_holder = None  # Track which component holds the lock
        
        # Configure network settings for ThingsBoard
        network_config = self._configure_network_settings()
        
        # Configure audio device settings
        audio_config = self._configure_audio_settings()
        
        # Configure voice assistant settings
        voice_config = self._configure_voice_assistant_settings()
        
        # Initialize core modules
        self.ble_module = BLEModule()
        self.data_manager = DataManager(os.environ.get("GATEWAY_NAME", "FallyxGateway"))
        self.network_module = NetworkModule(config=network_config)
        self.audio_module = AudioModule(
            target_input_name_fragment=audio_config["input_name"],
            target_output_name_fragment=audio_config["output_name"],
            main_controller=self
        )
        self.monitoring_service = MonitoringService()
        
        # Initialize new components
        self.openai_adapter = OpenAIAdapter()
        
        # Initialize fall detection coordinator with dependencies
        self.fall_detection_coordinator = FallDetectionCoordinator(
            audio_module=self.audio_module,
            openai_adapter=self.openai_adapter,
            command_module=None,  # Will be set after command module is created
            main_controller=self  # Pass self to the coordinator for lock management
        )
        
        # Initialize command module with fall detection coordinator
        self.command_module = CommandModule(
            fall_detection_coordinator=self.fall_detection_coordinator
        )
        
        # Set command module reference in fall detection coordinator
        self.fall_detection_coordinator.command_module = self.command_module
        
        # Initialize voice assistant components (conditionally)
        self.keyword_detector = None
        self.voice_assistant_coordinator = None
        
        if voice_config["enabled"]:
            try:
                # Initialize keyword detector with wake word callback
                self.keyword_detector = KeywordDetector(
                    access_key=voice_config["picovoice_access_key"],
                    wake_word=voice_config["wake_word"],
                    callback=self._on_wake_word_detected
                )
                
                # Initialize voice assistant coordinator
                self.voice_assistant_coordinator = VoiceAssistantCoordinator(
                    audio_module=self.audio_module,
                    openai_adapter=self.openai_adapter,
                    thingsboard_client=self.network_module.thingsboard_client,
                    main_controller=self
                )
                
                # Update backend base URL if configured
                if voice_config["chat_api_base_url"]:
                    self.voice_assistant_coordinator.backend_base_url = voice_config["chat_api_base_url"]
                
                # Update gateway ID if configured
                if voice_config["gateway_id"]:
                    self.voice_assistant_coordinator.gateway_id = voice_config["gateway_id"]
                
                # Connect voice assistant coordinator to command module
                self.command_module.set_voice_assistant_coordinator(self.voice_assistant_coordinator)
                
                logger.info("Voice assistant components initialized and connected successfully")
                
            except Exception as e:
                logger.error(f"Failed to initialize voice assistant components: {e}")
                self.keyword_detector = None
                self.voice_assistant_coordinator = None
        else:
            logger.info("Voice assistant functionality is disabled")
        
        # Connect modules (dependency injection)
        self.ble_module.set_data_manager(self.data_manager)
        self.ble_module.set_command_module(self.command_module)
        self.data_manager.set_network_module(self.network_module)
        self.command_module.set_ble_module(self.ble_module)
        self.command_module.set_network_module(self.network_module)
        
        # Connect keyword detector to command module if available
        if self.keyword_detector:
            self.command_module.set_keyword_detector(self.keyword_detector)
        
        # Configure emergency settings
        self._configure_emergency_settings()
        
        # Thread references
        self.threads = []
        self.running = False
        
        logger.info("Fallyx Gateway initialized")
        
    def _configure_network_settings(self):
        """Configure network settings from environment variables or config files."""
        logger.info("Configuring ThingsBoard settings")
        
        # Prepare ThingsBoard configuration
        thingsboard_config = {
            "host": os.environ.get("THINGSBOARD_HOST", "localhost"),
            "port": int(os.environ.get("THINGSBOARD_PORT", 1883)),
            "access_token": os.environ.get("THINGSBOARD_ACCESS_TOKEN", ""),
            "qos": int(os.environ.get("THINGSBOARD_QOS", 1))
        }
        
        # Log configuration (hide sensitive information)
        safe_config = thingsboard_config.copy()
        if "access_token" in safe_config and safe_config["access_token"]:
            safe_config["access_token"] = "****"
        logger.info(f"ThingsBoard configuration: {safe_config}")
        
        return thingsboard_config
    
    def _configure_audio_settings(self):
        """Configure audio settings from environment variables."""
        logger.info("Configuring audio device settings")
        
        # Get audio device name fragments from environment variables
        audio_input_name = os.environ.get("AUDIO_INPUT_DEVICE_NAME_CONTAINS")
        audio_output_name = os.environ.get("AUDIO_OUTPUT_DEVICE_NAME_CONTAINS")
        
        # Log the configuration
        if audio_input_name:
            logger.info(f"Target input device name fragment: '{audio_input_name}'")
        else:
            logger.info("No specific input device name configured - will use first available device")
            
        if audio_output_name:
            logger.info(f"Target output device name fragment: '{audio_output_name}'")
        else:
            logger.info("No specific output device name configured - will use first available device")
        
        return {
            "input_name": audio_input_name,
            "output_name": audio_output_name
        }
    
    def _configure_voice_assistant_settings(self):
        """Configure voice assistant settings from environment variables."""
        logger.info("Configuring voice assistant settings")
        
        # Check if voice assistant is enabled
        voice_assistant_enabled = os.environ.get("VOICE_ASSISTANT_ENABLED", "false").lower() == "true"
        
        config = {
            "enabled": voice_assistant_enabled,
            "picovoice_access_key": None,
            "wake_word": "hey fallyx",
            "chat_api_base_url": None,
            "gateway_id": None
        }
        
        if voice_assistant_enabled:
            # Get Picovoice access key
            picovoice_access_key = os.environ.get("PICOVOICE_ACCESS_KEY")
            if picovoice_access_key:
                config["picovoice_access_key"] = picovoice_access_key
                logger.info("Picovoice access key configured from environment")
            else:
                logger.error("PICOVOICE_ACCESS_KEY not found in environment - voice assistant will be disabled")
                config["enabled"] = False
            
            # Get wake word
            wake_word = os.environ.get("WAKE_WORD", "hey glenn")
            config["wake_word"] = wake_word
            logger.info(f"Wake word configured: '{wake_word}'")
            
            # Get chat API base URL
            chat_api_base_url = os.environ.get("CHAT_API_BASE_URL")
            if chat_api_base_url:
                config["chat_api_base_url"] = chat_api_base_url
                logger.info(f"Chat API base URL configured: {chat_api_base_url}")
            else:
                logger.info("Using default chat API base URL (localhost:8000)")
            
            # Get gateway ID
            gateway_id = os.environ.get("GATEWAY_ID")
            if gateway_id:
                config["gateway_id"] = gateway_id
                logger.info(f"Gateway ID configured: {gateway_id}")
            else:
                logger.info("Using auto-generated gateway ID")
        else:
            logger.info("Voice assistant functionality is disabled")
        
        return config
    
    def _configure_emergency_settings(self):
        """Configure emergency settings from environment variables or config files."""
        logger.info("Configuring emergency settings")
        
        # Set OpenAI API key for OpenAI adapter
        openai_api_key = os.environ.get("OPENAI_API_KEY")
        if openai_api_key:
            logger.info("Setting OpenAI API key from environment")
            self.openai_adapter.set_api_key(openai_api_key)
        else:
            logger.warning("OpenAI API key not found in environment")
        
        # Set emergency number for command module
        emergency_number = os.environ.get("EMERGENCY_NUMBER", "+14166299094")
        if emergency_number:
            logger.info(f"Setting emergency number: {emergency_number}")
            self.command_module.set_emergency_number(emergency_number)
        
        # Set emergency location info (using first/last name fields for location data)
        gateway_building = os.environ.get("GATEWAY_BUILDING", "Unknown Building")
        gateway_room = os.environ.get("GATEWAY_ROOM", "Unknown Room")
        emergency_contact_name = os.environ.get("EMERGENCY_CONTACT_NAME", "SRR")
        
        self.command_module.set_emergency_user_info(
            first_name=gateway_building,
            last_name=gateway_room,
            emergency_contact_name=emergency_contact_name
        )
        
        logger.info(f"Emergency location configured - Building: {gateway_building}, Room: {gateway_room}")
    
    def request_audio_lock(self, requester: str, priority_state: AudioState) -> bool:
        """
        Request audio resource lock with priority handling.
        
        Args:
            requester: Name of the component requesting the lock
            priority_state: Audio state to transition to
            
        Returns:
            bool: True if lock was acquired, False otherwise
        """
        with self.audio_lock:
            # Fall detection has absolute priority
            if priority_state == AudioState.FALL_VERIFICATION_ACTIVE:
                # Always grant to fall detection, pausing other activities.
                # Preempt voice assistant if it's active.
                if self.audio_state == AudioState.VOICE_ASSISTANT_ACTIVE:
                    logger.info("Fall detection preempting voice assistant session")
                    # Note: Voice assistant coordinator will be notified via its session cleanup
                
                # --- FIX START ---
                # Pause the keyword detector if it's running. This is the crucial fix, as
                # the detector holds the hardware lock in the IDLE state.
                if self.keyword_detector and self.keyword_detector.is_running() and not self.keyword_detector.is_paused():
                    logger.info(f"Pausing keyword detector for high-priority task: {requester}")
                    self.keyword_detector.pause()
                # --- FIX END ---
                
                self.audio_state = priority_state
                self.audio_lock_holder = requester
                logger.info(f"Audio lock granted to {requester} (fall detection priority)")
                return True
            
            # For voice assistant requests
            elif priority_state == AudioState.VOICE_ASSISTANT_ACTIVE:
                # Can only acquire if currently idle
                if self.audio_state == AudioState.IDLE:
                    self.audio_state = priority_state
                    self.audio_lock_holder = requester
                    
                    # Pause keyword detection during voice session (this part is already correct)
                    if self.keyword_detector and self.keyword_detector.is_running() and not self.keyword_detector.is_paused():
                        self.keyword_detector.pause()
                    
                    logger.info(f"Audio lock granted to {requester} (voice assistant)")
                    return True
                else:
                    logger.warning(f"Audio lock denied to {requester} - currently in state: {self.audio_state.value}")
                    return False
            
            # Invalid state request
            else:
                logger.error(f"Invalid audio state requested: {priority_state}")
                return False
    
    def release_audio_lock(self, requester: str = None) -> bool:
        """
        Release audio resource lock.
        
        Args:
            requester: Name of the component releasing the lock (optional for validation)
            
        Returns:
            bool: True if lock was released, False otherwise
        """
        with self.audio_lock:
            if self.audio_state == AudioState.IDLE:
                logger.debug("Audio lock release requested but already idle")
                return True
            
            # Validate requester if provided
            if requester and self.audio_lock_holder != requester:
                logger.warning(f"Lock release denied: {requester} does not hold the lock (held by {self.audio_lock_holder})")
                return False
            
            previous_state = self.audio_state
            self.audio_state = AudioState.IDLE
            self.audio_lock_holder = None
            
            # Resume keyword detection when returning to idle state
            if self.keyword_detector and self.keyword_detector.is_running() and self.keyword_detector.is_paused():
                try:
                    self.keyword_detector.resume()
                    logger.debug("Keyword detection resumed after audio lock release")
                except Exception as e:
                    logger.error(f"Failed to resume keyword detection: {e}")
            
            logger.info(f"Audio lock released (was in state: {previous_state.value})")
            return True
    
    def _on_wake_word_detected(self):
        """
        Callback function called when wake word is detected by KeywordDetector.
        Initiates a voice assistant session.
        """
        logger.info("Wake word detected - initiating voice assistant session")
        
        if not self.voice_assistant_coordinator:
            logger.warning("Wake word detected but voice assistant coordinator not available")
            return
        
        try:
            # Start voice session in a separate thread to avoid blocking keyword detection
            session_thread = threading.Thread(
                target=self._handle_voice_session,
                name="VoiceSessionThread",
                daemon=True
            )
            session_thread.start()
            
        except Exception as e:
            logger.error(f"Error starting voice session thread: {e}")
    
    def _handle_voice_session(self):
        """
        Handle voice assistant session in separate thread.
        This is called from the wake word detection callback.
        """
        try:
            session_id = self.voice_assistant_coordinator.start_voice_session()
            if session_id:
                logger.info(f"Voice session started successfully: {session_id}")
            else:
                logger.warning("Failed to start voice session")
                
        except Exception as e:
            logger.error(f"Error in voice session handling: {e}")
    
    def start(self):
        """Start all modules in separate threads."""
        if self.running:
            logger.warning("Gateway is already running")
            return
        
        logger.info("Starting Fallyx Gateway...")
        self.running = True
        
        # Start all modules in separate threads
        self.threads = [
            threading.Thread(target=self.ble_module.run, name="BLE-Thread"),
            threading.Thread(target=self.data_manager.run, name="Data-Thread"),
            threading.Thread(target=self.network_module.run, name="Network-Thread"),
            threading.Thread(target=self.command_module.run, name="Command-Thread"),
            threading.Thread(target=self.audio_module.run, name="Audio-Thread"),
            threading.Thread(target=self.monitoring_service.run, name="Monitoring-Thread")
        ]
        
        # Start keyword detector if available
        if self.keyword_detector:
            self.keyword_detector.start()
            logger.info("Keyword detector started")
        
        # Start threads
        for thread in self.threads:
            thread.daemon = True  # Allow the program to exit even if threads are running
            thread.start()
            logger.info(f"Started thread: {thread.name}")
        
        logger.info("All modules started")
        
        # Main loop
        try:
            while self.running:
                time.sleep(1)
        except KeyboardInterrupt:
            logger.info("Keyboard interrupt received in main thread.")
        finally:
            logger.info("Main loop stopped. Initiating gateway shutdown.")
            if self.running:
                self.stop()
    
    def stop(self):
        """Stop all modules and threads."""
        if not self.running:
            return
        
        logger.info("Stopping Fallyx Gateway...")
        self.running = False
        
        # Stop keyword detector first
        if self.keyword_detector:
            self.keyword_detector.stop()
            logger.info("Keyword detector stopped")
        
        # Signal all modules to stop
        self.ble_module.stop()
        self.data_manager.stop()
        self.network_module.stop()
        self.command_module.stop()
        self.audio_module.stop()
        self.monitoring_service.stop()
        
        # Wait for threads to complete
        for thread in self.threads:
            if thread is not threading.current_thread():
                thread.join(timeout=5.0)
                if thread.is_alive():
                    logger.warning(f"Thread {thread.name} did not terminate gracefully")
        
        logger.info("Fallyx Gateway stopped")
    
    def signal_handler(self, sig, frame):
        """Handle termination signals."""
        logger.info(f"Received signal {sig}, shutting down...")
        self.stop()

def main():
    """Main entry point for the application."""
    controller = MainController()
    
    # Set up signal handlers after controller is created to bind to its methods
    signal.signal(signal.SIGINT, controller.signal_handler)
    signal.signal(signal.SIGTERM, controller.signal_handler)
    
    # Run the controller
    try:
        controller.start()
    except KeyboardInterrupt:
        logger.info("Keyboard interrupt received in main.")
    except Exception as e:
        logger.error(f"Unexpected error in main: {e}", exc_info=True)
    finally:
        logger.info("Fallyx Gateway has shut down.")
        sys.exit(0)

if __name__ == "__main__":
    main()
