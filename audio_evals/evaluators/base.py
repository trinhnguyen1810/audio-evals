"""Base evaluator class for audio evaluation pipeline."""

from abc import ABC, abstractmethod
from typing import Dict, Any
import numpy as np


class BaseEvaluator(ABC):
    """Abstract base class for all audio evaluators."""
    
    def __init__(self, name: str, description: str):
        """
        Initialize base evaluator.
        
        Args:
            name: Unique identifier for this evaluator
            description: Human-readable description of what this evaluator does
        """
        self.name = name
        self.description = description
    
    @abstractmethod
    def evaluate(self, audio_data: np.ndarray, sample_rate: int, call_start_time: str = None) -> Dict[str, Any]:
        """
        Evaluate audio data and return results.
        
        Args:
            audio_data: Audio samples as numpy array
            sample_rate: Sample rate of audio
            call_start_time: ISO timestamp when call started (optional)
            
        Returns:
            Dictionary with evaluation results in standardized format:
            {
                "evaluator_name": {
                    "results": bool,      # Whether issues were found
                    "message": str,       # Human-readable summary
                    "timestamps": List,   # List of specific issues with timing
                    "metadata": Dict      # Additional evaluator-specific data
                }
            }
        """
        pass
    
    def __str__(self) -> str:
        """String representation of evaluator."""
        return f"{self.__class__.__name__}(name='{self.name}')"
    
    def __repr__(self) -> str:
        """Detailed string representation of evaluator."""
        return f"{self.__class__.__name__}(name='{self.name}', description='{self.description}')"