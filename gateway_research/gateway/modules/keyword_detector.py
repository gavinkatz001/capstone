"""
Keyword Detector Module for Fallyx Gateway

This module handles on-device wake word detection using pvporcupine,
running continuously in its own thread with pause/resume capabilities
for audio resource coordination. This version is refactored to use
the 'sounddevice' library exclusively to prevent driver-level conflicts.
"""

import logging
import time
import threading
import struct
import sounddevice as sd
import pvporcupine
import os
import numpy as np
from typing import Optional, Callable

logger = logging.getLogger("KeywordDetector")

class KeywordDetector:
    """
    Keyword Detector for wake word detection using pvporcupine.
    This implementation uses the 'sounddevice' library for audio streaming
    to ensure compatibility with other audio components in the application.
    """
    
    def __init__(self, access_key: str, wake_word: str, callback: Callable[[], None]):
        """
        Initialize the Keyword Detector.
        
        Args:
            access_key: Picovoice access key for pvporcupine
            wake_word: Wake word to detect (e.g., "hey fallyx")
            callback: Function to call when wake word is detected
        """
        logger.info("Initializing Keyword Detector with sounddevice.")
        
        self.access_key = access_key
        self.wake_word = wake_word
        self.callback = callback
        
        self.running = False
        self.paused = False
        self.detection_thread = None
        self.lock = threading.Lock()
        
        self.porcupine = None
        self.audio_stream = None
        
        self.sample_rate = None
        self.frame_length = None
        
        self._initialize_porcupine()
        logger.info(f"Keyword Detector initialized for wake word: '{self.wake_word}'")
    
    def _initialize_porcupine(self):
        """Initialize pvporcupine and get audio parameters."""
        try:
            custom_keyword_path = self._get_custom_keyword_path()
            
            if custom_keyword_path and os.path.exists(custom_keyword_path):
                logger.info(f"Using custom keyword file: {custom_keyword_path}")
                keyword_paths = [custom_keyword_path]
                keywords = None
            else:
                logger.info(f"Using built-in keyword: {self.wake_word}")
                keyword_paths = None
                keywords = [self.wake_word]

            self.porcupine = pvporcupine.create(
                access_key=self.access_key,
                keyword_paths=keyword_paths,
                keywords=keywords
            )
            
            self.sample_rate = self.porcupine.sample_rate
            self.frame_length = self.porcupine.frame_length
            logger.info(f"Porcupine initialized - Sample rate: {self.sample_rate}Hz, Frame length: {self.frame_length}")
            
        except Exception as e:
            logger.error(f"Failed to initialize pvporcupine: {e}")
            raise

    def _get_custom_keyword_path(self):
        """Get path to custom keyword file if it exists."""
        filename = self.wake_word.lower().replace(" ", "-") + ".ppn"
        keywords_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "keywords")
        return os.path.join(keywords_dir, filename)

    def _audio_callback(self, indata: np.ndarray, frames: int, time, status):
        """This callback is called by sounddevice with new audio frames."""
        if not self.running or self.paused:
            return

        if status:
            logger.warning(f"Audio stream status: {status}")

        try:
            # Porcupine process expects a list or tuple of int16 samples.
            pcm = struct.unpack_from("h" * self.frame_length, indata.tobytes())
            
            keyword_index = self.porcupine.process(pcm)
            if keyword_index >= 0:
                logger.info(f"Wake word '{self.wake_word}' detected!")
                if self.callback:
                    # Run the main callback in a new thread to avoid blocking the audio stream.
                    threading.Thread(target=self.callback, daemon=True).start()
        except Exception as e:
            logger.error(f"Error in audio callback: {e}")

    def _start_audio_stream(self):
        """Initialize and start the sounddevice InputStream."""
        if self.audio_stream and self.audio_stream.active:
            logger.debug("Audio stream is already active.")
            return

        try:
            logger.info("Starting sounddevice InputStream for keyword detection.")
            self.audio_stream = sd.InputStream(
                samplerate=self.sample_rate,
                channels=1,
                dtype='int16',
                blocksize=self.frame_length,
                callback=self._audio_callback
            )
            self.audio_stream.start()
            logger.info("Keyword detection audio stream started.")
        except Exception as e:
            logger.error(f"Failed to initialize/start sounddevice InputStream: {e}")
            self.audio_stream = None

    def _stop_audio_stream(self):
        """Stop and close the sounddevice InputStream."""
        if self.audio_stream:
            logger.info("Stopping keyword detection audio stream.")
            try:
                if self.audio_stream.active:
                    self.audio_stream.stop()
                self.audio_stream.close()
            except Exception as e:
                logger.error(f"Error while stopping audio stream: {e}")
            finally:
                self.audio_stream = None
                logger.info("Keyword detection audio stream stopped and closed.")

    def start(self):
        """Start the keyword detection service."""
        with self.lock:
            if self.running:
                logger.warning("Keyword detection is already running.")
                return
            logger.info("Starting keyword detection service.")
            self.running = True
            self.paused = False
            self._start_audio_stream()
    
    def stop(self):
        """Stop keyword detection and cleanup all resources."""
        with self.lock:
            if not self.running:
                return
            logger.info("Stopping keyword detection service.")
            self.running = False
            self.paused = True
        
        self._stop_audio_stream()

        if self.porcupine:
            self.porcupine.delete()
            self.porcupine = None
        
        logger.info("Keyword detection service stopped successfully.")

    def pause(self):
        """Pause keyword detection by stopping the audio stream."""
        with self.lock:
            if not self.running or self.paused:
                return
            logger.info("Pausing keyword detection.")
            self.paused = True
        self._stop_audio_stream()

    def resume(self):
        """Resume keyword detection by starting the audio stream."""
        with self.lock:
            if not self.running or not self.paused:
                return
            logger.info("Resuming keyword detection.")
            self.paused = False
        self._start_audio_stream()

    def is_running(self) -> bool:
        """Check if keyword detection is currently running."""
        with self.lock:
            return self.running

    def is_paused(self) -> bool:
        """Check if keyword detection is currently paused."""
        with self.lock:
            return self.paused