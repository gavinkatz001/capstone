"""
OpenAI Adapter for Fallyx Gateway

This module encapsulates all interactions with the OpenAI API, including
audio transcription and intent interpretation.
"""

import logging
import io
import json
import os
import openai
import numpy as np
from scipy.io.wavfile import write
from scipy import signal
from typing import Optional, Tuple

logger = logging.getLogger("OpenAIAdapter")

class OpenAIAdapter:
    """
    OpenAI Adapter for handling all OpenAI API interactions.
    
    This adapter is responsible for:
    - Audio transcription using OpenAI Whisper
    - Intent interpretation using OpenAI GPT
    """
    
    def __init__(self, api_key: Optional[str] = None):
        """
        Initialize the OpenAI Adapter.
        
        Args:
            api_key: OpenAI API key. If None, must be set later using set_api_key()
        """
        logger.info("Initializing OpenAI Adapter")
        self.api_key = None
        
        if api_key:
            self.set_api_key(api_key)
    
    def set_api_key(self, api_key: str):
        """
        Set the OpenAI API key.
        
        Args:
            api_key: OpenAI API key
        """
        self.api_key = api_key
        openai.api_key = api_key
        logger.info("OpenAI API key set")
    
    def transcribe_audio(self, audio_data_bytes_io: io.BytesIO, sample_rate: int) -> Optional[str]:
        """
        Transcribe audio data using OpenAI Whisper.
        
        Args:
            audio_data_bytes_io: Audio data as BytesIO object in WAV format
            sample_rate: Sample rate of the audio (for logging purposes)
            
        Returns:
            Transcribed text or None if transcription failed
        """
        if self.api_key is None:
            logger.error("OpenAI API key not set")
            return None
        
        logger.info("Transcribing audio using OpenAI Whisper")
        
        try:
            # Reset buffer position to start
            audio_data_bytes_io.seek(0)
                        
            # Ensure BytesIO has a name attribute with .wav extension for format detection
            if not hasattr(audio_data_bytes_io, 'name'):
                audio_data_bytes_io.name = "audio_recording.wav"
                logger.debug("Added .wav name attribute to BytesIO object")
            
            # Transcribe using OpenAI Whisper with timeout
            transcript = openai.audio.transcriptions.create(
                model="whisper-1",
                file=audio_data_bytes_io,
                timeout=5  # 30 second timeout to prevent hanging
            )
            
            transcribed_text = transcript.text
            logger.info(f"Transcription: {transcribed_text}")
            return transcribed_text
        except Exception as e:
            logger.error(f"Error transcribing audio: {e}")
            return None
    
    def interpret_user_response(self, text: str, context: str = None) -> str:
        """
        Interpret the intent of the text using OpenAI GPT with structured output.
        
        Args:
            text: The text to interpret
            context: Optional context for interpretation (e.g., "fall_check")
            
        Returns:
            One of: 'ok', 'not_ok', 'unclear'
        """
        if self.api_key is None:
            logger.error("OpenAI API key not set")
            return "unclear"
        
        logger.info(f"Interpreting intent for text: '{text}' with context: '{context}'")
        
        try:
            # Create context-specific prompt
            if context == "fall_check":
                prompt = f"""
                An elderly person was asked: "Did you fall?".
                They responded: "{text}"
                
                Classify the intent into one of:
                - 'ok' (they explicitly state they did NOT fall or are fine, regardless of tone or additional words)
                - 'not_ok' (they explicitly state they DID fall or need help)
                - 'unclear' (ambiguous or unrelated response)
                
                Examples:
                - "No, I'm fine" -> ok
                - "No, I didn't fall" -> ok
                - "No, go away" -> ok
                - "Yes, I fell" -> not_ok
                - "Help me" -> not_ok
                - "What?" -> unclear
                - "I'm hungry" -> unclear
                
                Respond with a JSON object that has a single field 'intent' with a value of either 'ok', 'not_ok', or 'unclear'.
                Example response: {{"intent": "ok"}}
                """
            else:
                # Generic intent interpretation
                prompt = f"""
                Interpret the intent of the following text: "{text}"
                
                Classify the intent into one of:
                - 'ok' (positive, affirmative, or indicating everything is fine)
                - 'not_ok' (negative, indicating a problem or need for help)
                - 'unclear' (ambiguous, unrelated, or unclear response)
                
                Respond with a JSON object that has a single field 'intent' with a value of either 'ok', 'not_ok', or 'unclear'.
                Example response: {{"intent": "ok"}}
                """
            
            # Get the response with structured format and timeout
            response = openai.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[{"role": "user", "content": prompt}],
                temperature=0,
                response_format={"type": "json_object"},
                timeout=15  # 15 second timeout to prevent hanging
            )
            
            # Parse the response
            result = response.choices[0].message.content
            parsed_result = json.loads(result)
            intent = parsed_result.get("intent", "unclear")
            
            # Validate intent value
            if intent not in ["ok", "not_ok", "unclear"]:
                logger.warning(f"Invalid intent returned: {intent}, defaulting to 'unclear'")
                intent = "unclear"
            
            logger.info(f"Detected intent: {intent}")
            return intent
        except Exception as e:
            logger.error(f"Error interpreting intent: {e}")
            return "unclear"
    
    def synthesize_speech(self, text: str, voice: str = "alloy", model: str = "tts-1") -> Optional[Tuple[np.ndarray, int]]:
        """
        Synthesize speech from text using OpenAI TTS API.
        
        Args:
            text: Text to convert to speech
            voice: Voice to use for TTS (default: "alloy")
            model: TTS model to use (default: "tts-1")
            
        Returns:
            Tuple of (audio_array, sample_rate) or None if synthesis failed
        """
        if self.api_key is None:
            logger.error("OpenAI API key not set")
            return None
        
        logger.info(f"Synthesizing speech for text: '{text}' using voice: '{voice}', model: '{model}'")
        
        # Ensure OpenAI API key is set
        if openai.api_key is None and self.api_key:
            openai.api_key = self.api_key
        
        try:
            # Call OpenAI TTS API with timeout
            response = openai.audio.speech.create(
                model=model,
                voice=voice,
                input=text,
                response_format="pcm",
                timeout=15  # 15 second timeout to prevent hanging
            )
            
            # Extract PCM audio bytes
            pcm_audio_bytes = response.content
            
            # OpenAI PCM Output Details: 24kHz, 1 channel, 16-bit depth, signed, little-endian
            openai_pcm_sample_rate = 24000
            openai_pcm_dtype = np.int16
            
            # Convert PCM bytes to NumPy array
            audio_np_array = np.frombuffer(pcm_audio_bytes, dtype=openai_pcm_dtype)
            
            logger.info(f"Speech synthesis successful: sample_rate={openai_pcm_sample_rate}, samples={len(audio_np_array)}")
            return (audio_np_array, openai_pcm_sample_rate)
        
        except Exception as e:
            logger.error(f"Error synthesizing speech: {e}")
            return None
