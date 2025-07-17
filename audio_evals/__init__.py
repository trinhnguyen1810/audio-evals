"""
Audio Evaluation Pipeline

A comprehensive audio analysis system for evaluating voice call quality
in collections and insurance call centers.
"""

__version__ = "1.0.0"
__author__ = "Audio Evaluation Team"

from .pipeline import AudioEvaluationPipeline
from .evaluators.silence_detector import LongSilenceDetector
from .evaluators.volume_consistency import VolumeConsistencyEvaluator
from .evaluators.speaker_overlap_detector import SpeakerOverlapDetector

__all__ = [
    "AudioEvaluationPipeline",
    "LongSilenceDetector", 
    "VolumeConsistencyEvaluator",
    "SpeakerOverlapDetector",
]