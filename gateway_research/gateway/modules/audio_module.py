"""
Audio Module for Fallyx Gateway

This module handles audio I/O operations including
speaker/microphone connections and audio recording/playback.
"""

import logging
import time
import threading
import os
import numpy as np
import sounddevice as sd
from scipy.io.wavfile import read
from scipy import signal
from typing import Optional

logger = logging.getLogger("AudioModule")

class AudioModule:
    """
    Audio Module for handling audio I/O operations.
    
    This module is responsible for:
    - Managing USB speaker/microphone connections
    - Audio recording and playback
    - Managing audio device settings
    """
    
    def __init__(self,
                 target_input_name_fragment: Optional[str] = None,
                 target_output_name_fragment: Optional[str] = None,
                 main_controller=None):
        """
        Initialize the Audio Module.
        
        Args:
            target_input_name_fragment: Optional name fragment to match input devices against
            target_output_name_fragment: Optional name fragment to match output devices against
            main_controller: Reference to MainController for audio lock state checking
        """
        logger.info("Initializing Audio Module")
        self.running = False
        self.voice_assistant_active = False
        self.lock = threading.Lock()  # For thread-safe operations
        self.main_controller = main_controller  # Reference for audio lock state checking
        
        # Audio configuration - load from environment variables
        # Use a single sample rate for all recording/processing for consistency
        self.sample_rate = int(os.environ.get("AUDIO_DEFAULT_SAMPLE_RATE", "16000"))
        # Playback sample rate may be different due to hardware requirements
        self.playback_sample_rate = int(os.environ.get("AUDIO_PLAYBACK_SAMPLE_RATE", "16000"))
        self.audio_channels = 1  # Mono audio
        self.audio_dtype = 'int16'  # Audio data type
        
        logger.info(f"Audio configuration loaded from environment:")
        logger.info(f"  Recording/Processing sample rate: {self.sample_rate}Hz")
        logger.info(f"  Playback sample rate: {self.playback_sample_rate}Hz")
        logger.info(f"  Audio channels: {self.audio_channels}")
        logger.info(f"  Audio data type: {self.audio_dtype}")
        
        # Target device name fragments for device selection
        self.target_input_name_fragment = 'ReSpeaker'
        self.target_output_name_fragment = 'ReSpeaker'
        
        # Initialize audio device lists
        self.available_input_devices = []
        self.available_output_devices = []
        
        # Setup devices first, then prepare audio recording settings
        self.device_setup_success = self._setup_audio_devices()
        self._prepare_audio_recording()
        
        # Track last device check time for retry mechanism
        self.last_device_check = time.time()
        self.device_check_interval = 60.0  # Check every 60 seconds
        
        if not self.device_setup_success:
            logger.warning("Audio device setup completed with warnings - some devices may not be available")
            logger.info("Will retry audio device detection every 60 seconds")
    
    def run(self):
        """Run the Audio Module main loop with periodic device checking."""
        logger.info("Starting Audio Module")
        self.running = True
        
        while self.running:
            logger.debug("Audio Module running...")
            
            # Check if it's time to retry audio device detection
            current_time = time.time()
            if current_time - self.last_device_check >= self.device_check_interval:
                self._periodic_device_check()
                self.last_device_check = current_time
            
            time.sleep(5)  # Sleep to prevent CPU hogging
    
    def stop(self):
        """Stop the Audio Module."""
        logger.info("Stopping Audio Module")
        self.running = False
    
    def _setup_audio_devices(self):
        """
        Setup audio devices with optional target device name matching.
        
        This method queries available audio devices and attempts to select devices
        based on target name fragments. Falls back to first available device if
        target names are not found or not provided.
        
        Returns:
            bool: True if at least one input and one output device are available, False otherwise
        """
        logger.info("Starting audio device setup")
        
        try:
            # Get all available audio devices
            devices = sd.query_devices()
            
            # Filter for input and output devices
            input_devices = [device for device in devices if device['max_input_channels'] > 0]
            output_devices = [device for device in devices if device['max_output_channels'] > 0]
            
            # Log the available devices
            logger.info(f"Found {len(input_devices)} input devices and {len(output_devices)} output devices")
            
            # Log details at debug level for each device
            for device in input_devices:
                logger.debug(f"Found Input Device {device['index']}: {device['name']}")
            
            for device in output_devices:
                logger.debug(f"Found Output Device {device['index']}: {device['name']}")
            
            # Check if we have any devices
            if not input_devices:
                logger.warning("No input devices found")
            
            if not output_devices:
                logger.warning("No output devices found")
            
            # Device selection logic
            selected_input_idx = None
            selected_output_idx = None
            
            # Input device selection
            if self.target_input_name_fragment:
                logger.debug(f"Looking for input device containing: '{self.target_input_name_fragment}'")
                for device in input_devices:
                    if self.target_input_name_fragment.lower() in device['name'].lower():
                        selected_input_idx = device['index']
                        logger.info(f"Selected input device by name match: {device['name']} (index {selected_input_idx})")
                        break
                
                if selected_input_idx is None:
                    logger.warning(f"No input device found containing '{self.target_input_name_fragment}'")
            
            # Fallback to first available input device if none selected yet
            if selected_input_idx is None and input_devices:
                selected_input_idx = input_devices[0]['index']
                logger.info(f"Using first available input device as fallback: {input_devices[0]['name']} (index {selected_input_idx})")
            
            # Output device selection
            if self.target_output_name_fragment:
                logger.debug(f"Looking for output device containing: '{self.target_output_name_fragment}'")
                for device in output_devices:
                    if self.target_output_name_fragment.lower() in device['name'].lower():
                        selected_output_idx = device['index']
                        logger.info(f"Selected output device by name match: {device['name']} (index {selected_output_idx})")
                        break
                
                if selected_output_idx is None:
                    logger.warning(f"No output device found containing '{self.target_output_name_fragment}'")
            
            # Fallback to first available output device if none selected yet
            if selected_output_idx is None and output_devices:
                selected_output_idx = output_devices[0]['index']
                logger.info(f"Using first available output device as fallback: {output_devices[0]['name']} (index {selected_output_idx})")
            
            # Get current sounddevice defaults
            try:
                current_default_input_idx, current_default_output_idx = sd.default.device
            except (TypeError, ValueError):
                # sd.default.device might be None or a single int
                current_default_input_idx = None
                current_default_output_idx = None
            
            # Determine final device indices
            final_input_idx = selected_input_idx if selected_input_idx is not None else current_default_input_idx
            final_output_idx = selected_output_idx if selected_output_idx is not None else current_default_output_idx
            
            # Set sounddevice defaults
            try:
                if final_input_idx is not None or final_output_idx is not None:
                    sd.default.device = (final_input_idx, final_output_idx)
                    
                    # Log the successfully set devices
                    input_name = "System Default"
                    output_name = "System Default"
                    
                    if final_input_idx is not None:
                        try:
                            input_device_info = sd.query_devices(final_input_idx)
                            input_name = input_device_info['name']
                        except:
                            input_name = f"Device {final_input_idx}"
                    
                    if final_output_idx is not None:
                        try:
                            output_device_info = sd.query_devices(final_output_idx)
                            output_name = output_device_info['name']
                        except:
                            output_name = f"Device {final_output_idx}"
                    
                    logger.info(f"Successfully set sounddevice defaults: Input={input_name}, Output={output_name}")
                else:
                    logger.warning("No specific input or output devices were selected to update system defaults")
                    
            except Exception as e:
                logger.error(f"Error setting sounddevice default devices: {e}. Current defaults remain: Input={current_default_input_idx}, Output={current_default_output_idx}")
            
            # Store the device lists for diagnostic purposes
            self.available_input_devices = input_devices
            self.available_output_devices = output_devices
            
            return len(input_devices) > 0 and len(output_devices) > 0
            
        except Exception as e:
            logger.error(f"Error during audio device setup: {e}")
            return False
    
    def check_audio_devices(self):
        """
        Public method to manually check for audio devices.
        
        This can be called externally if there's a need to refresh the audio device list,
        for example if a device was disconnected and reconnected.
        
        Returns:
            bool: True if audio devices are available, False otherwise
        """
        return self._setup_audio_devices()
    
    def is_audio_available(self) -> bool:
        """
        Check if audio devices are currently available.
        
        Returns:
            bool: True if audio devices are available, False otherwise
        """
        return self.device_setup_success
    
    def _periodic_device_check(self):
        """
        Periodic background check for audio devices.
        
        This method runs every minute to:
        1. Check if devices are still connected (if previously successful)
        2. Look for newly connected devices (if no devices were found at boot)
        
        This is a low-priority task that won't interrupt other operations.
        """
        try:
            logger.debug("Performing periodic audio device check")
            
            # Check if it's safe to refresh device cache (no ongoing audio operations)
            if self._is_audio_operation_safe():
                # Attempt to refresh device list with cache refresh for better detection
                new_device_status = self._setup_audio_devices_with_refresh()
            else:
                # Skip cache refresh if audio is active, just do a basic check
                logger.debug("Audio operation in progress, skipping device cache refresh")
                new_device_status = self._setup_audio_devices()
            
            # Log status changes and notify if devices become available
            if not self.device_setup_success and new_device_status:
                logger.info("Audio devices detected! Audio functionality is now available.")
                self.device_setup_success = True
                self._notify_device_availability_change(True)
            elif self.device_setup_success and not new_device_status:
                logger.warning("Audio devices no longer detected. Audio functionality may be limited.")
                self.device_setup_success = False
                self._notify_device_availability_change(False)
            else:
                # No change in status - only log at debug level to avoid spam
                if self.device_setup_success:
                    logger.debug("Audio devices still available")
                else:
                    logger.debug("No audio devices found during periodic check")
                    
        except Exception as e:
            # Don't let device checking errors crash the audio module
            logger.error(f"Error during periodic device check: {e}")
    
    def _notify_device_availability_change(self, devices_available: bool):
        """
        Notify about audio device availability changes.
        
        This method can be extended in the future to notify other components
        (like KeywordDetector or VoiceAssistantCoordinator) when audio devices
        become available or unavailable.
        
        Args:
            devices_available: True if devices are now available, False if lost
        """
        # For now, just log the change. In the future, this could:
        # - Restart KeywordDetector if devices become available
        # - Notify MainController to update audio lock status
        # - Send device status to monitoring service
        
        if devices_available:
            logger.info("Audio device availability changed: AVAILABLE - voice features can be enabled")
        else:
            logger.info("Audio device availability changed: UNAVAILABLE - voice features may be disabled")
    
    def _is_audio_operation_safe(self) -> bool:
        """
        Check if it's safe to refresh audio device cache without interrupting operations.
        
        This checks the audio lock system to ensure no fall detection or voice assistant
        operations are currently using the audio hardware.
        
        Returns:
            bool: True if safe to refresh device cache, False if audio operations are active
        """
        try:
            # If we don't have a main controller reference, be conservative
            if not self.main_controller:
                logger.debug("No main controller reference, assuming audio operations may be active")
                return False
            
            # Import AudioState to check current state
            from gateway.audio_types import AudioState
            
            # Check the current audio state through the main controller
            current_audio_state = getattr(self.main_controller, 'audio_state', AudioState.IDLE)
            
            # Only safe to refresh when audio is idle
            if current_audio_state == AudioState.IDLE:
                logger.debug("Audio state is IDLE, device refresh is safe")
                return True
            else:
                logger.debug(f"Audio state is {current_audio_state.value}, device refresh not safe")
                return False
                
        except Exception as e:
            logger.debug(f"Error checking audio operation safety: {e}")
            # If we can't determine safety, err on the side of caution
            return False
    
    def _setup_audio_devices_with_refresh(self):
        """
        Setup audio devices with device cache refresh for better detection of new devices.
        
        This method should only be called when it's safe to refresh the device cache
        (i.e., no ongoing audio operations).
        
        Returns:
            bool: True if at least one input and one output device are available, False otherwise
        """
        logger.debug("Setting up audio devices with cache refresh")
        
        try:
            # Force refresh of sounddevice's internal device cache
            # This helps detect newly plugged devices
            try:
                # Reset sounddevice's cached device list
                sd._terminate()
                sd._initialize()
                logger.debug("Successfully refreshed sounddevice device cache")
            except Exception as cache_refresh_error:
                # If cache refresh fails, continue anyway - query_devices might still work
                logger.debug(f"Device cache refresh failed (continuing anyway): {cache_refresh_error}")
            
            # Now call the regular device setup method
            return self._setup_audio_devices()
            
        except Exception as e:
            logger.error(f"Error during audio device setup with refresh: {e}")
            # Fall back to regular device setup if refresh fails
            return self._setup_audio_devices()
    
    def _prepare_audio_recording(self):
        """
        Prepare audio recording settings.
        
        This method sets the global sounddevice defaults for sample rate, channels, and data type.
        Should be called after _setup_audio_devices() has potentially set sd.default.device.
        """
        logger.debug("Preparing audio recording settings")
        
        try:
            sd.default.samplerate = self.sample_rate
            sd.default.channels = self.audio_channels
            sd.default.dtype = self.audio_dtype
            logger.debug(f"Audio recording settings configured: {self.sample_rate}Hz, {self.audio_channels} channels, {self.audio_dtype}")
        except Exception as e:
            logger.error(f"Error setting audio recording defaults: {e}. Audio recording may not work properly.")
        
    def play_audio_data(self, audio_data, sample_rate=None):
        """
        Play audio data from a numpy array.
        
        Args:
            audio_data: Audio data as numpy array
            sample_rate: Sample rate for playback (uses default if None)
        """
        if sample_rate is None:
            sample_rate = self.sample_rate
        
        # Check if resampling is needed for compatibility
        target_sample_rate = self._get_compatible_sample_rate(sample_rate)
        
        if target_sample_rate != sample_rate:
            logger.info(f"Resampling audio from {sample_rate}Hz to {target_sample_rate}Hz for compatibility")
            audio_data = self._resample_audio(audio_data, sample_rate, target_sample_rate)
            sample_rate = target_sample_rate
            
        logger.debug(f"Playing audio data at {sample_rate}Hz")
        sd.play(audio_data, sample_rate)
        sd.wait()  # Wait until audio is finished playing
    
    def play_audio_file(self, file_path):
        """
        Play audio from a file.
        
        Args:
            file_path: Path to the audio file
            
        Returns:
            bool: True if successful, False otherwise
        """
        logger.debug(f"Playing audio file: {file_path}")
        
        try:
            # Read the audio file
            sample_rate, audio_data = read(file_path)
            # Play the audio
            self.play_audio_data(audio_data, sample_rate)
            return True
        except Exception as e:
            logger.error(f"Error playing audio file: {e}")
            return False
    
    def record_audio(self, duration=5):
        """
        Record audio and return the audio data.
        
        Args:
            duration: Duration in seconds to record
            
        Returns:
            numpy.ndarray: Audio data or None if recording failed
        """
        logger.info(f"Recording audio for {duration} seconds")
        
        try:
            # Record audio
            audio_data = sd.rec(
                int(duration * self.sample_rate),
                samplerate=self.sample_rate,
                channels=self.audio_channels,
                dtype=self.audio_dtype
            )
            sd.wait()  # Wait until recording is finished
            
            logger.info("Audio recording complete")
            return audio_data
        except Exception as e:
            logger.error(f"Error recording audio: {e}")
            return None
    
    def _get_compatible_sample_rate(self, requested_rate: int) -> int:
        """
        Get a compatible sample rate for the audio hardware.
        Uses the configured sample rate from environment variables for consistency.
        
        Args:
            requested_rate: The requested sample rate
            
        Returns:
            int: The configured sample rate for consistency across the app
        """
        # Use the main configured sample rate for consistency across all audio processing
        return self.sample_rate
    
    def _resample_audio(self, audio_data: np.ndarray, original_rate: int, target_rate: int) -> np.ndarray:
        """
        Resample audio data to a different sample rate.
        
        Args:
            audio_data: Original audio data
            original_rate: Original sample rate
            target_rate: Target sample rate
            
        Returns:
            Resampled audio data
        """
        try:
            # Calculate the number of samples in the resampled audio
            num_samples = int(len(audio_data) * target_rate / original_rate)
            
            # Use scipy.signal.resample for high-quality resampling
            resampled_audio = signal.resample(audio_data, num_samples)
            
            # Ensure the output is the same data type as input
            if audio_data.dtype == np.int16:
                resampled_audio = np.clip(resampled_audio, -32768, 32767).astype(np.int16)
            
            logger.debug(f"Audio resampled from {original_rate}Hz to {target_rate}Hz")
            return resampled_audio
            
        except Exception as e:
            logger.error(f"Error resampling audio: {e}")
            # Return original audio if resampling fails
            return audio_data
