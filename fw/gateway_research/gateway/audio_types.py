"""
Common types and enums for Fallyx Gateway
"""

import enum

class AudioState(enum.Enum):
    """Audio resource states with priority levels."""
    IDLE = "idle"  # No active audio processing
    FALL_VERIFICATION_ACTIVE = "fall_verification_active"  # Highest priority
    VOICE_ASSISTANT_ACTIVE = "voice_assistant_active"  # Medium priority