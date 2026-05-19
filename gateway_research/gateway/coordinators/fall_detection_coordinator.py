"""
Fall Detection Coordinator for Fallyx Gateway

This module orchestrates the fall detection workflow by coordinating
between AudioModule, OpenAIAdapter, and CommandModule.
"""

import logging
import threading
import time
import io
import numpy as np
from scipy.io.wavfile import write
from typing import Dict, Any, Optional
from gateway.audio_types import AudioState

logger = logging.getLogger("FallDetectionCoordinator")

class FallDetectionCoordinator:
    """
    Fall Detection Coordinator for orchestrating the fall detection workflow.
    
    This coordinator is responsible for:
    - Orchestrating the complete fall detection verification process
    - Coordinating between AudioModule, OpenAIAdapter, and CommandModule
    - Managing threading and result storage for async operations
    - Providing comprehensive result reporting
    """
    
    def __init__(self, audio_module, openai_adapter, command_module, main_controller):
        """
        Initialize the Fall Detection Coordinator.
        
        Args:
            audio_module: Instance of AudioModule for audio I/O operations
            openai_adapter: Instance of OpenAIAdapter for AI processing
            command_module: Instance of CommandModule for emergency calls
            main_controller: Instance of MainController for audio locking
        """
        logger.info("Initializing Fall Detection Coordinator")
        
        # Store module references
        self.audio_module = audio_module
        self.openai_adapter = openai_adapter
        self.command_module = command_module
        self.main_controller = main_controller
        
        # Threading and state management
        self.lock = threading.Lock()
        self.last_verification_result = {"status": "idle"}
        
        logger.info("Fall Detection Coordinator initialized")
    
    def verify_potential_fall(self, response_audio_ok_path: Optional[str] = None,
                            response_audio_not_ok_path: Optional[str] = None,
                            recording_duration: float = 5.0) -> Dict[str, Any]:
        """
        Execute the complete fall detection verification process.
        
        Args:
            response_audio_ok_path: Path to the audio file for 'ok' response
            response_audio_not_ok_path: Path to the audio file for 'not_ok' response
            recording_duration: Duration in seconds to record user response
            
        Returns:
            Dictionary containing comprehensive verification results
        """
        logger.info("Starting fall detection verification process")
        
        result = {
            "intent": "unclear",
            "emergency_call_triggered": False,
            "emergency_call_successful": None,
            "message": "Fall detection process started",
            "transcribed_text": None,
            "timestamp": int(time.time() * 1000),
            "steps_completed": []
        }
        
        # Request audio lock for fall verification
        lock_acquired = self.main_controller.request_audio_lock("FallDetectionCoordinator", AudioState.FALL_VERIFICATION_ACTIVE)
        if not lock_acquired:
            logger.error("Failed to acquire audio lock for fall detection verification")
            result["message"] = "Failed to acquire audio lock for fall detection verification"
            result["error"] = "Audio lock acquisition failed"
            return result
        
        try:
            # Step 1: Play prompt text via TTS
            prompt_to_speak = "Have you fallen?"
            prompt_action_taken = False
            
            if prompt_to_speak and self.openai_adapter and self.audio_module:
                logger.info(f"Synthesizing and playing TTS prompt: '{prompt_to_speak}'")
                tts_output = self.openai_adapter.synthesize_speech(prompt_to_speak)
                
                if tts_output is not None:
                    audio_np_array, sample_rate = tts_output
                    try:
                        self.audio_module.play_audio_data(audio_np_array, sample_rate)
                        result["steps_completed"].append("prompt_spoken_via_tts")
                        logger.info("TTS prompt played successfully")
                        prompt_action_taken = True
                    except Exception as e:
                        logger.warning(f"Failed to play TTS audio: {e}")
                        result["steps_completed"].append("prompt_tts_playback_failed")
                else:
                    logger.warning("TTS synthesis failed")
                    result["steps_completed"].append("prompt_tts_synthesis_failed")
            else:
                logger.info("TTS modules not available, skipping prompt step")
                result["steps_completed"].append("prompt_tts_skipped")
            
            if not prompt_action_taken:
                logger.warning("Continuing fall detection process without spoken prompt")
            
            # Brief delay after TTS prompt to give user time to react
            if "prompt_spoken_via_tts" in result["steps_completed"]:
                time.sleep(1)
            
            # Step 2: Record user response
            logger.info(f"Recording user response for {recording_duration} seconds")
            recorded_audio_data_np_array = self.audio_module.record_audio(duration=recording_duration)
            
            if recorded_audio_data_np_array is None:
                logger.error("Failed to record audio")
                result["message"] = "Failed to record audio response"
                return result
            
            result["steps_completed"].append("audio_recorded")
            logger.info("Audio recording completed successfully")
            
            # Step 3: Convert recorded audio data to format suitable for OpenAI
            logger.info("Converting recorded audio data for transcription")
            audio_bytes_io_for_transcription = self._convert_audio_to_bytes_io(
                recorded_audio_data_np_array,
                self.audio_module.sample_rate
            )
            
            if audio_bytes_io_for_transcription is None:
                logger.error("Failed to convert recorded audio data")
                result["message"] = "Failed to process recorded audio data"
                return result
            
            result["steps_completed"].append("recorded_audio_converted_for_transcription")
            
            # Step 4: Transcribe audio
            logger.info("Transcribing recorded audio using OpenAI")
            transcribed_text = self.openai_adapter.transcribe_audio(
                audio_bytes_io_for_transcription, self.audio_module.sample_rate
            )
            
            if transcribed_text is None:
                logger.warning("Failed to transcribe audio, proceeding with empty text")
                transcribed_text = ""
                result["steps_completed"].append("audio_transcription_failed")
            else:
                result["transcribed_text"] = transcribed_text
                result["steps_completed"].append("audio_transcribed")
                logger.info(f"Audio transcribed successfully: '{transcribed_text}'")
            
            # Step 5: Interpret intent
            logger.info("Interpreting user intent")
            intent = self.openai_adapter.interpret_user_response(
                transcribed_text if transcribed_text else "", context="fall_check"
            )
            
            result["intent"] = intent
            result["steps_completed"].append("intent_interpreted")
            logger.info(f"Intent interpreted: {intent}")
            
            # Step 6: Handle intent-based responses
            if intent == "ok":
                logger.info("Fall detection: False alarm - No action needed")
                result["message"] = "Fall detection completed: False alarm - No action needed"
                
                # Play 'ok' response audio
                if response_audio_ok_path:
                    play_success = self.audio_module.play_audio_file(response_audio_ok_path)
                    if play_success:
                        result["steps_completed"].append("ok_response_played")
                        logger.info("'OK' response audio played successfully")
                    else:
                        logger.warning("Failed to play 'OK' response audio")
                
            elif intent == "not_ok":
                logger.info("Fall detection: Emergency confirmed - Help needed")
                result["emergency_call_triggered"] = True
                
                # Play 'not ok' response audio
                if response_audio_not_ok_path:
                    play_success = self.audio_module.play_audio_file(response_audio_not_ok_path)
                    if play_success:
                        result["steps_completed"].append("not_ok_response_played")
                        logger.info("'Not OK' response audio played successfully")
                    else:
                        logger.warning("Failed to play 'Not OK' response audio")
                
                # Trigger emergency call
                logger.info("Initiating emergency call")
                emergency_call_success = self.command_module.handle_emergency_call()
                result["emergency_call_successful"] = emergency_call_success
                result["steps_completed"].append("emergency_call_attempted")
                
                if emergency_call_success:
                    result["message"] = "Fall detection completed: Emergency confirmed - Emergency call successful"
                    logger.info("Emergency call completed successfully")
                else:
                    result["message"] = "Fall detection completed: Emergency confirmed - Emergency call failed"
                    logger.error("Emergency call failed")
                    
            else:  # intent == "unclear"
                logger.info("Fall detection: Unclear response - No action taken")
                result["message"] = "Fall detection completed: Unclear response - No action taken"
                result["steps_completed"].append("unclear_response_handled")
            
            result["steps_completed"].append("process_completed")
            logger.info("Fall detection verification process completed successfully")
            
        except Exception as e:
            logger.error(f"Error during fall detection verification: {e}")
            result["message"] = f"Error during fall detection: {str(e)}"
            result["error"] = str(e)
        finally:
            # Always release the audio lock
            self.main_controller.release_audio_lock("FallDetectionCoordinator")
            logger.info("Audio lock released for fall detection verification")
        
        return result
    
    def start_fall_verification_async(self, response_audio_ok_path: Optional[str] = None,
                                    response_audio_not_ok_path: Optional[str] = None,
                                    recording_duration: float = 5.0) -> threading.Thread:
        """
        Start the fall detection verification process in a separate thread.
        
        Args:
            response_audio_ok_path: Path to the audio file for 'ok' response
            response_audio_not_ok_path: Path to the audio file for 'not_ok' response
            recording_duration: Duration in seconds to record user response
            
        Returns:
            Thread object for the fall detection process
        """
        logger.info("Starting fall detection verification in a separate thread")
        
        # Initialize result state
        with self.lock:
            self.last_verification_result = {
                "status": "pending",
                "timestamp": int(time.time() * 1000)
            }
        
        def verification_wrapper():
            """Wrapper function to capture results in the thread."""
            try:
                result = self.verify_potential_fall(
                    response_audio_ok_path=response_audio_ok_path,
                    response_audio_not_ok_path=response_audio_not_ok_path,
                    recording_duration=recording_duration
                )
                
                # Store the result
                with self.lock:
                    result["status"] = "completed"
                    self.last_verification_result = result
                    
                logger.info("Fall detection verification thread completed successfully")
                
            except Exception as e:
                logger.error(f"Error in fall detection verification thread: {e}")
                with self.lock:
                    self.last_verification_result = {
                        "status": "error",
                        "intent": "unclear",
                        "emergency_call_triggered": False,
                        "emergency_call_successful": None,
                        "message": f"Error during fall detection: {str(e)}",
                        "error": str(e),
                        "timestamp": int(time.time() * 1000)
                    }
        
        # Create and start the thread
        thread = threading.Thread(
            target=verification_wrapper,
            name="FallDetectionVerification-Thread"
        )
        thread.daemon = True
        thread.start()
        
        logger.info("Fall detection verification thread started")
        return thread
    
    def get_last_verification_result(self) -> Dict[str, Any]:
        """
        Get the result of the most recent fall detection verification process.
        
        Returns:
            Dictionary containing the verification result, protected by lock
        """
        with self.lock:
            return self.last_verification_result.copy()
    
    def _convert_audio_to_bytes_io(self, audio_data_np_array: np.ndarray,
                                 sample_rate: int) -> Optional[io.BytesIO]:
        """
        Convert numpy audio data to BytesIO WAV format for OpenAI API.
        
        Args:
            audio_data_np_array: Audio data as numpy array
            sample_rate: Sample rate of the audio
            
        Returns:
            BytesIO object containing WAV data, or None if conversion failed
        """
        try:
            # Create a BytesIO buffer
            audio_buffer = io.BytesIO()
            
            # Write audio data as WAV to the buffer
            write(audio_buffer, sample_rate, audio_data_np_array)
            
            # Reset buffer position to start
            audio_buffer.seek(0)
            
            logger.debug("Audio data converted to BytesIO WAV format successfully")
            return audio_buffer
            
        except Exception as e:
            logger.error(f"Error converting audio data to BytesIO: {e}")
            return None
