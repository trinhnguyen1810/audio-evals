# app/evaluators/speaker_overlap_detector.py
import numpy as np
from typing import Dict, List, Any, Tuple
from datetime import datetime, timedelta
from ..utils import perform_speaker_diarization

class SpeakerOverlapDetector:
    def __init__(self, min_overlap_duration: float = 2.0):
        self.name = "speaker_overlap_detection"
        self.description = "Detects significant speaker interruptions indicating poor call flow"
        self.min_overlap_duration = min_overlap_duration  # Only flag medium+ overlaps (2s+)
        
        # Classification thresholds (simplified - only medium and major)
        self.medium_threshold = 1.0   # 2.0-3.0s = Medium  
        # 3.0s+ = Major
    
    def evaluate(self, audio_data: np.ndarray, sample_rate: int, call_start_time: str = None) -> Dict[str, Any]:
        """
        Detect speaker overlaps in audio.
        
        Args:
            audio_data: Audio samples as numpy array
            sample_rate: Sample rate of audio
            call_start_time: ISO timestamp when call started
            
        Returns:
            Dictionary with results in expected pipeline format
        """
        
        # Perform speaker diarization to get speaker segments
        print("🎭 Getting speaker segments for overlap detection...")
        speaker_segments = perform_speaker_diarization(audio_data, sample_rate)
        
        if len(speaker_segments) < 2:
            # No overlaps possible with less than 2 speakers
            return {
                self.name: {
                    "results": False,
                    "message": f"No speaker overlaps detected (found {len(speaker_segments)} speaker segments)",
                    "timestamps": [],
                    "overlap_analysis": {
                        "total_interruptions": 0,
                        "total_interruption_duration": 0.0,
                        "overlap_severity_flag": "none",
                        "requires_attention": False,
                        "severity_breakdown": {"medium": 0, "major": 0},
                        "speakers_detected": len(set(seg['speaker'] for seg in speaker_segments))
                    }
                }
            }
        
        # Detect overlaps
        overlaps = self._detect_overlaps(speaker_segments)
        
        # Filter by minimum duration
        significant_overlaps = [
            overlap for overlap in overlaps 
            if overlap["duration"] >= self.min_overlap_duration
        ]
        
        # Convert to timestamp format and classify severity (only medium and major)
        overlap_timestamps = []
        severity_counts = {"medium": 0, "major": 0}
        highest_severity = "none"
        
        for overlap in significant_overlaps:
            # Classify overlap severity based on duration
            duration = overlap["duration"]
            if duration >= self.medium_threshold:  # 3.0s+
                overlap_severity = "major"
            else:  # 2.0-3.0s
                overlap_severity = "medium"
            
            # Track highest severity level
            severity_order = {"none": 0, "medium": 1, "major": 2}
            if severity_order[overlap_severity] > severity_order[highest_severity]:
                highest_severity = overlap_severity
            
            severity_counts[overlap_severity] += 1
            overlap["severity_classification"] = overlap_severity
            
            timestamp_entry = self._create_overlap_timestamp(overlap, call_start_time)
            overlap_timestamps.append(timestamp_entry)
        
        # Calculate summary statistics
        total_overlap_duration = sum(overlap["duration"] for overlap in significant_overlaps)
        call_duration = len(audio_data) / sample_rate
        overlap_percentage = (total_overlap_duration / call_duration) * 100 if call_duration > 0 else 0
        
        # All significant overlaps require attention (2s+ interruptions)
        requires_attention = len(significant_overlaps) > 0
        
        # Create concise message
        has_issues = len(significant_overlaps) > 0
        if has_issues:
            message = f"Found {len(significant_overlaps)} significant interruptions"
            message += f" ({total_overlap_duration:.1f}s total)"
            if severity_counts["major"] > 0:
                message += f" - {severity_counts['major']} major, {severity_counts['medium']} medium"
            else:
                message += f" - {severity_counts['medium']} medium"
        else:
            message = "No significant speaker interruptions detected"
        
        return {
            self.name: {
                "results": has_issues,
                "message": message,
                "timestamps": overlap_timestamps,
                "overlap_analysis": {
                    "total_interruptions": len(significant_overlaps),
                    "total_interruption_duration": round(total_overlap_duration, 2),
                    "overlap_severity_flag": highest_severity,
                    "requires_attention": requires_attention,
                    "severity_breakdown": severity_counts,
                    "speakers_detected": len(set(seg['speaker'] for seg in speaker_segments))
                }
            }
        }
    
    def _detect_overlaps(self, speaker_segments: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Detect overlapping segments between different speakers.
        
        Args:
            speaker_segments: List of speaker segments from diarization
            
        Returns:
            List of overlap periods with timing and speaker information
        """
        overlaps = []
        
        # Sort segments by start time
        sorted_segments = sorted(speaker_segments, key=lambda x: x["start"])
        
        # Check each segment against all others for overlaps
        for i, segment_a in enumerate(sorted_segments):
            for j, segment_b in enumerate(sorted_segments[i+1:], i+1):
                # Skip if same speaker
                if segment_a["speaker"] == segment_b["speaker"]:
                    continue
                
                # Check for temporal overlap
                overlap_start = max(segment_a["start"], segment_b["start"])
                overlap_end = min(segment_a["end"], segment_b["end"])
                
                if overlap_start < overlap_end:
                    # There is an overlap
                    overlap_duration = overlap_end - overlap_start
                    
                    # Determine who interrupted whom (who started speaking first)
                    if segment_a["start"] < segment_b["start"]:
                        primary_speaker = segment_a["speaker"]
                        interrupting_speaker = segment_b["speaker"]
                        interruption_delay = segment_b["start"] - segment_a["start"]
                    else:
                        primary_speaker = segment_b["speaker"]
                        interrupting_speaker = segment_a["speaker"]
                        interruption_delay = segment_a["start"] - segment_b["start"]
                    
                    # Calculate overlap severity score
                    severity_score = self._calculate_overlap_severity(overlap_duration, interruption_delay)
                    
                    overlaps.append({
                        "start": overlap_start,
                        "end": overlap_end,
                        "duration": overlap_duration,
                        "primary_speaker": primary_speaker,
                        "interrupting_speaker": interrupting_speaker,
                        "interruption_delay": interruption_delay,
                        "severity_score": severity_score
                    })
        
        # Sort overlaps by start time
        overlaps.sort(key=lambda x: x["start"])
        
        # Merge overlaps that are very close together (< 0.5s apart)
        merged_overlaps = self._merge_adjacent_overlaps(overlaps)
        
        return merged_overlaps
    
    def _calculate_overlap_severity(self, duration: float, interruption_delay: float) -> float:
        """
        Calculate severity score for significant overlaps (2s+).
        
        Args:
            duration: How long the overlap lasted (2s+)
            interruption_delay: How long the primary speaker was talking before interruption
            
        Returns:
            Severity score from 0.5 (medium) to 1.0 (major interruption)
        """
        
        # Base severity on duration - simplified for 2s+ overlaps
        if duration >= self.medium_threshold:  # 3.0s+ = Major
            duration_score = 0.8 + (min(duration - self.medium_threshold, 2.0) / 2.0) * 0.2  # 0.8-1.0
        else:  # 2.0-3.0s = Medium
            duration_score = 0.5 + ((duration - self.min_overlap_duration) / (self.medium_threshold - self.min_overlap_duration)) * 0.3  # 0.5-0.8
        
        # Factor in interruption timing (quick interruptions are more severe)
        if interruption_delay < 1.0:  # Interrupted within 1 second
            timing_multiplier = 1.2
        elif interruption_delay < 3.0:  # Interrupted within 3 seconds
            timing_multiplier = 1.1
        else:  # Natural conversation gap
            timing_multiplier = 0.9
        
        severity = duration_score * timing_multiplier
        return min(severity, 1.0)
    
    def _merge_adjacent_overlaps(self, overlaps: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Merge overlaps that are very close together to avoid over-reporting."""
        if len(overlaps) <= 1:
            return overlaps
        
        merged = []
        current_overlap = overlaps[0].copy()
        
        for next_overlap in overlaps[1:]:
            # If overlaps are close and involve same speakers, merge them
            time_gap = next_overlap["start"] - current_overlap["end"]
            same_speakers = (
                {current_overlap["primary_speaker"], current_overlap["interrupting_speaker"]} ==
                {next_overlap["primary_speaker"], next_overlap["interrupting_speaker"]}
            )
            
            if time_gap < 0.5 and same_speakers:
                # Merge overlaps
                current_overlap["end"] = next_overlap["end"]
                current_overlap["duration"] = current_overlap["end"] - current_overlap["start"]
                current_overlap["severity_score"] = max(
                    current_overlap["severity_score"], 
                    next_overlap["severity_score"]
                )
            else:
                # Start new overlap
                merged.append(current_overlap)
                current_overlap = next_overlap.copy()
        
        merged.append(current_overlap)
        return merged
    
    def _create_overlap_timestamp(self, overlap: Dict[str, Any], call_start_time: str) -> Dict[str, Any]:
        """Create timestamp entry for an overlap in standard format."""
        
        start_sec = overlap["start"]
        end_sec = overlap["end"]
        duration = overlap["duration"]
        
        # Get severity classification from overlap data
        severity_classification = overlap.get("severity_classification", "minor")
        severity_score = overlap["severity_score"]
        
        # Create issue description with proper classification
        issue_description = f"{severity_classification.title()} interruption: {overlap['interrupting_speaker']} interrupted {overlap['primary_speaker']}"
        
        # Calculate quality score (inverse of severity)
        quality_score = max(0.1, 1.0 - severity_score)
        
        timestamp_entry = {
            "durationSeconds": round(duration, 2),
            "issue": issue_description,
            "score": round(quality_score, 2),
            "overlap_severity": severity_classification,
            "overlap_type": severity_classification + "_interruption",
            "primary_speaker": overlap["primary_speaker"],
            "interrupting_speaker": overlap["interrupting_speaker"],
            "severity_score": round(severity_score, 2),
            "relativeStartTime": self._format_time(start_sec),
            "relativeEndTime": self._format_time(end_sec)
        }
        
        # Add absolute timestamps if call_start_time is provided
        if call_start_time:
            call_start_dt = datetime.fromisoformat(call_start_time.rstrip('Z'))
            start_dt = call_start_dt + timedelta(seconds=start_sec)
            end_dt = call_start_dt + timedelta(seconds=end_sec)
            
            timestamp_entry.update({
                "startTime": start_dt.isoformat() + "Z",
                "endTime": end_dt.isoformat() + "Z"
            })
        
        return timestamp_entry
    
    def _format_time(self, seconds: float) -> str:
        """Format time in MM:SS.mmm format."""
        minutes = int(seconds // 60)
        remaining_seconds = seconds % 60
        return f"{minutes:02d}:{remaining_seconds:06.3f}"