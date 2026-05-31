"""
Voice Assistant Coordinator for Fallyx Gateway

This module orchestrates the voice chat session workflow by coordinating
between AudioModule, OpenAIAdapter, backend API, and managing session lifecycle.
"""

import logging
import threading
import time
import uuid
import io
import requests
import json
import numpy as np
from scipy.io.wavfile import write
from typing import Dict, Any, Optional

logger = logging.getLogger("VoiceAssistantCoordinator")

class VoiceAssistantCoordinator:
    """
    Voice Assistant Coordinator for orchestrating voice chat sessions.
    
    This coordinator is responsible for:
    - Managing chat session lifecycle with unique UUID generation
    - Coordinating with MainController for audio resource locking
    - Recording user commands using existing AudioModule
    - Transcribing audio using existing OpenAIAdapter
    - Communicating with backend API (/start and /continue endpoints)
    - Handling RPC commands (play_response, play_and_record) from backend
    - Reusing existing openai_adapter.synthesize_speech() and audio_module.play_audio_data()
    - Releasing audio lock when session is complete
    """
    
    def __init__(self, audio_module, openai_adapter, thingsboard_client, main_controller):
        """
        Initialize the Voice Assistant Coordinator.
        
        Args:
            audio_module: Instance of AudioModule for audio I/O operations
            openai_adapter: Instance of OpenAIAdapter for AI processing
            thingsboard_client: Instance of ThingsBoardClient for backend communication
            main_controller: Instance of MainController for resource management
        """
        logger.info("Initializing Voice Assistant Coordinator")
        
        # Store module references
        self.audio_module = audio_module
        self.openai_adapter = openai_adapter
        self.thingsboard_client = thingsboard_client
        self.main_controller = main_controller
        
        # Threading and state management
        self.lock = threading.Lock()
        self.active_sessions = {}  # session_id -> session_info
        self.backend_base_url = "http://localhost:5000"  # TODO: make configurable
        self.gateway_id = str(uuid.uuid4())  # Unique gateway identifier
        
        logger.info(f"Voice Assistant Coordinator initialized with gateway_id: {self.gateway_id}")
    
    def start_voice_session(self) -> Optional[str]:
        """
        Begin a new voice chat session.
        
        Returns:
            str: Session ID if successful, None if failed
        """
        session_id = str(uuid.uuid4())
        logger.info(f"Starting voice session: {session_id}")
        
        try:
            # Acquire audio lock from main controller
            if hasattr(self.main_controller, 'request_audio_lock'):
                # Import AudioState from types module
                from gateway.audio_types import AudioState
                if not self.main_controller.request_audio_lock("VoiceAssistantCoordinator", AudioState.VOICE_ASSISTANT_ACTIVE):
                    logger.warning("Failed to acquire audio lock for voice session")
                    return None
            
            # Initialize session state
            with self.lock:
                self.active_sessions[session_id] = {
                    "created_at": time.time(),
                    "status": "active",
                    "audio_locked": True
                }
            
            # Record initial user input
            audio_data = self._record_user_input()
            if audio_data is None:
                self._cleanup_session(session_id)
                return None
            
            # Transcribe audio
            transcript = self._transcribe_audio(audio_data)
            if transcript is None:
                self._cleanup_session(session_id)
                return None
            
            # Send to backend API
            success = self._send_to_backend(session_id, transcript, is_start=True)
            if not success:
                self._cleanup_session(session_id)
                return None
            
            logger.info(f"Voice session started successfully: {session_id}")
            return session_id
            
        except Exception as e:
            logger.error(f"Error starting voice session {session_id}: {e}")
            self._cleanup_session(session_id)
            return None
    
    def handle_play_response(self, session_id: str, text_to_speak: str) -> bool:
        """
        Handle final response RPC from backend.
        
        Args:
            session_id: Session identifier
            text_to_speak: Text to synthesize and play
            
        Returns:
            bool: True if successful, False otherwise
        """
        logger.info(f"Handling play_response for session {session_id}: '{text_to_speak}'")
        
        try:
            # Verify session exists
            with self.lock:
                if session_id not in self.active_sessions:
                    logger.warning(f"Session {session_id} not found")
                    return False
            
            # Synthesize speech
            tts_result = self.openai_adapter.synthesize_speech(text_to_speak)
            if tts_result is None:
                logger.error("Failed to synthesize speech")
                return False
            
            audio_data, sample_rate = tts_result
            
            # Play audio
            self.audio_module.play_audio_data(audio_data, sample_rate)
            
            # Clean up session (final response)
            self._cleanup_session(session_id)
            
            logger.info(f"Successfully played final response for session {session_id}")
            return True
            
        except Exception as e:
            logger.error(f"Error handling play_response for session {session_id}: {e}")
            self._cleanup_session(session_id)
            return False
    
    def handle_play_and_record(self, session_id: str, text_to_speak: str) -> bool:
        """
        Handle continue conversation RPC from backend.
        
        Args:
            session_id: Session identifier
            text_to_speak: Text to synthesize and play before recording
            
        Returns:
            bool: True if successful, False otherwise
        """
        logger.info(f"Handling play_and_record for session {session_id}: '{text_to_speak}'")
        
        try:
            # Verify session exists
            with self.lock:
                if session_id not in self.active_sessions:
                    logger.warning(f"Session {session_id} not found")
                    return False
            
            # Synthesize and play speech
            tts_result = self.openai_adapter.synthesize_speech(text_to_speak)
            if tts_result is None:
                logger.error("Failed to synthesize speech")
                return False
            
            audio_data, sample_rate = tts_result
            self.audio_module.play_audio_data(audio_data, sample_rate)
            
            # Brief pause between playback and recording
            time.sleep(0.5)
            
            # Record user response
            audio_data = self._record_user_input()
            if audio_data is None:
                self._cleanup_session(session_id)
                return False
            
            # Transcribe audio
            transcript = self._transcribe_audio(audio_data)
            if transcript is None:
                self._cleanup_session(session_id)
                return False
            
            # Send continue request to backend
            success = self._send_to_backend(session_id, transcript, is_start=False)
            if not success:
                self._cleanup_session(session_id)
                return False
            
            logger.info(f"Successfully handled play_and_record for session {session_id}")
            return True
            
        except Exception as e:
            logger.error(f"Error handling play_and_record for session {session_id}: {e}")
            self._cleanup_session(session_id)
            return False
    
    def _record_user_input(self, duration: float = 2.0) -> Optional[np.ndarray]:
        """
        Record audio from user.
        
        Args:
            duration: Recording duration in seconds
            
        Returns:
            numpy.ndarray: Audio data or None if recording failed
        """
        logger.info(f"Recording user input for {duration} seconds")
        
        try:
            audio_data = self.audio_module.record_audio(duration=duration)
            if audio_data is None:
                logger.error("Failed to record audio")
                return None
            
            logger.info("Audio recording completed successfully")
            return audio_data
            
        except Exception as e:
            logger.error(f"Error recording user input: {e}")
            return None
    
    def _transcribe_audio(self, audio_data: np.ndarray) -> Optional[str]:
        """
        Transcribe audio using OpenAI.
        
        Args:
            audio_data: Audio data as numpy array
            
        Returns:
            str: Transcribed text or None if transcription failed
        """
        logger.info("Transcribing audio using OpenAI")
        
        try:
            # Convert audio data to BytesIO format for OpenAI
            audio_bytes_io = self._convert_audio_to_bytes_io(
                audio_data, 
                self.audio_module.sample_rate
            )
            
            if audio_bytes_io is None:
                logger.error("Failed to convert audio data for transcription")
                return None
            
            # Transcribe using OpenAI adapter
            transcript = self.openai_adapter.transcribe_audio(
                audio_bytes_io, 
                self.audio_module.sample_rate
            )
            
            if transcript is None:
                logger.warning("Failed to transcribe audio")
                return None
            
            logger.info(f"Audio transcribed successfully: '{transcript}'")
            return transcript
            
        except Exception as e:
            logger.error(f"Error transcribing audio: {e}")
            return None
    
    def _send_to_backend(self, session_id: str, transcript: str, is_start: bool = False) -> bool:
        """
        Send transcript to backend chat API.
        
        Args:
            session_id: Session identifier
            transcript: Transcribed text
            is_start: Whether this is the start of a new session
            
        Returns:
            bool: True if successful, False otherwise
        """
        endpoint = "/v1/chat/start" if is_start else "/v1/chat/continue"
        url = f"{self.backend_base_url}{endpoint}"
        
        payload = {
            "chat_session_id": session_id,
            "gateway_device_id": self.gateway_id,
            "transcript": transcript
        }
        
        logger.info(f"Sending {'start' if is_start else 'continue'} request to backend: {url}")
        logger.debug(f"Payload: {payload}")
        
        try:
            headers = {
                "Content-Type": "application/json",
                "Accept": "application/json"
            }
            
            response = requests.post(
                url,
                json=payload,
                headers=headers,
                timeout=30  # 30 second timeout
            )
            
            response.raise_for_status()
            
            logger.info(f"Backend request successful: {response.status_code}")
            return True
            
        except requests.exceptions.RequestException as e:
            logger.error(f"Error sending request to backend: {e}")
            return False
        except Exception as e:
            logger.error(f"Unexpected error sending to backend: {e}")
            return False
    
    def _cleanup_session(self, session_id: str):
        """
        Clean up session resources.
        
        Args:
            session_id: Session identifier to clean up
        """
        logger.info(f"Cleaning up session: {session_id}")
        
        try:
            with self.lock:
                session_info = self.active_sessions.get(session_id)
                if session_info:
                    # Release audio lock if held
                    if session_info.get("audio_locked", False):
                        if hasattr(self.main_controller, 'release_audio_lock'):
                            self.main_controller.release_audio_lock("VoiceAssistantCoordinator")
                        session_info["audio_locked"] = False
                    
                    # Remove session from active sessions
                    del self.active_sessions[session_id]
                    logger.info(f"Session {session_id} cleaned up successfully")
                else:
                    logger.warning(f"Session {session_id} not found during cleanup")
                    
        except Exception as e:
            logger.error(f"Error cleaning up session {session_id}: {e}")
    
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
    
    def get_active_sessions(self) -> Dict[str, Any]:
        """
        Get information about active sessions.
        
        Returns:
            Dictionary of active sessions
        """
        with self.lock:
            return self.active_sessions.copy()