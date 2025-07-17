# app/evaluators/volume_consistency.py
import numpy as np
import librosa
import requests
import tempfile
import os
from typing import Dict, List, Any, Union, Tuple
from urllib.parse import urlparse
from datetime import datetime, timedelta
from ..utils import perform_speaker_diarization, get_speaker_audio_segments

class VolumeConsistencyEvaluator:
    def __init__(self, frame_duration: float = 1.0, volume_threshold_db: float = 10.0, enable_diarization: bool = True):
        self.name = "volume_consistency"
        self.description = "Detects abnormal volume changes per speaker, avoiding false positives from agent/customer volume differences"
        self.frame_duration = frame_duration
        self.volume_threshold_db = volume_threshold_db
        self.enable_diarization = enable_diarization
    
    def _is_url(self, path: str) -> bool:
        """Check if input is a URL."""
        try:
            result = urlparse(path)
            return all([result.scheme, result.netloc])
        except:
            return False
    
    def _download_audio(self, url: str) -> str:
        """Download audio from URL to temporary file."""
        response = requests.get(url, stream=True)
        response.raise_for_status()
        
        temp_file = tempfile.NamedTemporaryFile(delete=False, suffix='.mp3')
        for chunk in response.iter_content(chunk_size=8192):
            temp_file.write(chunk)
        temp_file.close()
        
        return temp_file.name
    
    def _load_audio(self, audio_input: Union[str, np.ndarray], sample_rate: int = None) -> Tuple[np.ndarray, int]:
        """Load audio from file path, URL, or numpy array."""
        if isinstance(audio_input, np.ndarray):
            if sample_rate is None:
                raise ValueError("sample_rate must be provided when audio_input is numpy array")
            return audio_input, sample_rate
        
        temp_file = None
        try:
            if self._is_url(audio_input):
                temp_file = self._download_audio(audio_input)
                audio_path = temp_file
            else:
                audio_path = audio_input
            
            # Load audio and convert to mono
            audio_data, sr = librosa.load(audio_path, sr=None, mono=True)
            return audio_data, sr
        
        finally:
            if temp_file and os.path.exists(temp_file):
                os.unlink(temp_file)
    
    def _calculate_rms_db(self, audio_segment: np.ndarray) -> float:
        """Calculate RMS in dB for an audio segment."""
        rms = np.sqrt(np.mean(audio_segment ** 2))
        if rms == 0:
            return -np.inf
        return 20 * np.log10(rms)
    
    def _format_time(self, seconds: float) -> str:
        """Format time in MM:SS.mmm format."""
        minutes = int(seconds // 60)
        remaining_seconds = seconds % 60
        return f"{minutes:02d}:{remaining_seconds:06.3f}"
    
    def evaluate(self, audio_data: np.ndarray, sample_rate: int, call_start_time: str = None) -> Dict[str, Any]:
        """
        Evaluate volume consistency in audio (pipeline interface).
        
        Args:
            audio_data: Audio samples as numpy array
            sample_rate: Sample rate of audio
            call_start_time: ISO timestamp when call started
            
        Returns:
            Dictionary with results in expected pipeline format
        """
        if self.enable_diarization:
            # Perform speaker diarization first
            speaker_segments = perform_speaker_diarization(audio_data, sample_rate)
            speaker_audio_segments = get_speaker_audio_segments(audio_data, sample_rate, speaker_segments)
            
            # Analyze volume consistency per speaker
            all_issues = []
            speaker_analysis = {}
            
            for speaker_id, audio_segments in speaker_audio_segments.items():
                print(f"🔊 Analyzing volume consistency for {speaker_id}...")
                speaker_issues = self._analyze_speaker_volume_consistency(
                    audio_segments, sample_rate, call_start_time, speaker_id
                )
                all_issues.extend(speaker_issues)
                speaker_analysis[speaker_id] = {
                    "issues_count": len(speaker_issues),
                    "segments_count": len(audio_segments),
                    "total_duration": sum(end - start for _, start, end in audio_segments)
                }
            
            # Create summary message
            unique_speakers = len(speaker_audio_segments)
            total_issues = len(all_issues)
            message = f"Found {total_issues} volume consistency issues across {unique_speakers} speakers"
            if total_issues > 0:
                message += f" (threshold: ±{self.volume_threshold_db} dB per speaker)"
            
            # Sort issues by time
            all_issues.sort(key=lambda x: x.get('startTime', x.get('relativeStartTime', '00:00.000')))
            
            return {
                self.name: {
                    "results": len(all_issues) > 0,
                    "message": message,
                    "timestamps": all_issues,
                    "speaker_analysis": speaker_analysis,
                    "diarization_enabled": True
                }
            }
        else:
            # Original behavior - analyze entire audio uniformly
            issues = self._analyze_volume_consistency(audio_data, sample_rate, call_start_time)
            
            # Convert to pipeline format (matching silence_detector format)
            has_issues = len(issues) > 0
            message = f"Found {len(issues)} volume consistency issues (global analysis)"
            if has_issues:
                message += f" (threshold: ±{self.volume_threshold_db} dB)"
            
            return {
                self.name: {
                    "results": has_issues,
                    "message": message,
                    "timestamps": issues,
                    "diarization_enabled": False
                }
            }
    
    def evaluate_from_file(self, audio_input: Union[str, np.ndarray], sample_rate: int = None) -> List[Dict[str, Any]]:
        """
        Evaluate volume consistency from file/URL (standalone interface).
        
        Args:
            audio_input: Path to MP3 file, MP3 URL, or numpy array of audio data
            sample_rate: Sample rate (required if audio_input is numpy array)
            
        Returns:
            List of volume issues with start_time, end_time, issue, and score
        """
        
        # Load audio data
        audio_data, sr = self._load_audio(audio_input, sample_rate)
        return self._analyze_volume_consistency(audio_data, sr, None)
    
    def _analyze_volume_consistency(self, audio_data: np.ndarray, sample_rate: int, call_start_time: str = None) -> List[Dict[str, Any]]:
        """Analyze volume consistency and return list of issues."""
        
        # Calculate frame size in samples
        frame_size = int(self.frame_duration * sample_rate)
        hop_size = frame_size // 2  # 50% overlap
        
        # Calculate RMS values for each frame
        rms_values = []
        frame_times = []
        
        for i in range(0, len(audio_data) - frame_size, hop_size):
            frame = audio_data[i:i + frame_size]
            rms_db = self._calculate_rms_db(frame)
            if not np.isinf(rms_db):  # Skip silent frames
                rms_values.append(rms_db)
                frame_times.append(i / sample_rate)
        
        if len(rms_values) == 0:
            return []
        
        # Calculate average volume
        avg_volume = np.mean(rms_values)
        
        # Find volume deviations
        issues = []
        current_issue = None
        
        for i, (rms_db, time) in enumerate(zip(rms_values, frame_times)):
            deviation = rms_db - avg_volume
            
            # Check if volume is significantly different
            if abs(deviation) > self.volume_threshold_db:
                if current_issue is None:
                    # Start new issue
                    current_issue = {
                        "start_time": time,
                        "deviation": deviation,
                        "frames": 1
                    }
                else:
                    # Continue current issue if same type
                    if (deviation > 0) == (current_issue["deviation"] > 0):
                        current_issue["frames"] += 1
                        current_issue["deviation"] = (current_issue["deviation"] + deviation) / 2
                    else:
                        # Different type of issue, finish current and start new
                        if current_issue["frames"] * self.frame_duration >= 1.0:  # At least 1 second
                            self._add_issue(issues, current_issue, frame_times, i-1, call_start_time)
                        
                        current_issue = {
                            "start_time": time,
                            "deviation": deviation,
                            "frames": 1
                        }
            else:
                # Volume is normal, finish current issue if exists
                if current_issue is not None:
                    if current_issue["frames"] * self.frame_duration >= 1.0:  # At least 1 second
                        self._add_issue(issues, current_issue, frame_times, i-1, call_start_time)
                    current_issue = None
        
        # Handle issue that extends to end of audio
        if current_issue is not None and current_issue["frames"] * self.frame_duration >= 1.0:
            end_time = len(audio_data) / sample_rate
            self._add_issue_with_end_time(issues, current_issue, end_time, call_start_time)
        
        return issues
    
    def _add_issue(self, issues: List[Dict], issue_data: Dict, frame_times: List[float], 
                   end_frame_idx: int, call_start_time: str):
        """Add issue to issues list."""
        start_time = issue_data["start_time"]
        end_time = frame_times[end_frame_idx] + self.frame_duration
        self._add_issue_with_end_time(issues, issue_data, end_time, call_start_time)
    
    def _add_issue_with_end_time(self, issues: List[Dict], issue_data: Dict, 
                                end_time: float, call_start_time: str):
        """Add issue with specified end time."""
        deviation = issue_data["deviation"]
        start_sec = issue_data["start_time"]
        duration = end_time - start_sec
        
        if deviation > 0:
            issue_type = f"Volume spike (+{deviation:.1f} dB above avg)"
            score = max(0.1, 1.0 - (deviation / 20.0))  # Lower score for louder spikes
        else:
            issue_type = f"Volume drop ({deviation:.1f} dB below avg)"
            score = max(0.1, 1.0 - (abs(deviation) / 20.0))  # Lower score for quieter drops
        
        # Create timestamp in same format as silence_detector
        timestamp_entry = {
            "durationSeconds": round(duration, 2),
            "issue": issue_type,
            "score": round(score, 2),
            "relativeStartTime": self._format_time(start_sec),
            "relativeEndTime": self._format_time(end_time)
        }
        
        # Add absolute timestamps if call_start_time is provided
        if call_start_time:
            call_start_dt = datetime.fromisoformat(call_start_time.rstrip('Z'))
            start_dt = call_start_dt + timedelta(seconds=start_sec)
            end_dt = call_start_dt + timedelta(seconds=end_time)
            
            timestamp_entry.update({
                "startTime": start_dt.isoformat() + "Z",
                "endTime": end_dt.isoformat() + "Z"
            })
        
        issues.append(timestamp_entry)
    
    def _analyze_speaker_volume_consistency(self, speaker_audio_segments: List[Tuple[np.ndarray, float, float]], 
                                          sample_rate: int, call_start_time: str, speaker_id: str) -> List[Dict[str, Any]]:
        """Analyze volume consistency for a specific speaker across their audio segments."""
        
        # Concatenate all speaker segments and track their original timestamps
        all_speaker_audio = []
        segment_timestamps = []
        
        for audio_segment, start_time, end_time in speaker_audio_segments:
            if len(audio_segment) > 0:  # Skip empty segments
                all_speaker_audio.append(audio_segment)
                segment_timestamps.append((start_time, end_time, len(audio_segment)))
        
        if len(all_speaker_audio) == 0:
            return []
        
        # Concatenate all audio for this speaker
        concatenated_audio = np.concatenate(all_speaker_audio)
        
        # Analyze concatenated speaker audio for volume consistency
        speaker_issues = []
        
        # Calculate frame size in samples
        frame_size = int(self.frame_duration * sample_rate)
        hop_size = frame_size // 2  # 50% overlap
        
        # Calculate RMS values for each frame
        rms_values = []
        frame_start_times = []  # Track original timestamps
        
        audio_offset = 0
        for segment_idx, (audio_segment, start_time, end_time) in enumerate(zip(all_speaker_audio, [ts[0] for ts in segment_timestamps], [ts[1] for ts in segment_timestamps])):
            segment_duration = end_time - start_time
            
            # Process frames within this segment
            for i in range(0, len(audio_segment) - frame_size, hop_size):
                frame = audio_segment[i:i + frame_size]
                rms_db = self._calculate_rms_db(frame)
                
                if not np.isinf(rms_db):  # Skip silent frames
                    rms_values.append(rms_db)
                    # Calculate original timestamp for this frame
                    frame_time_in_segment = i / sample_rate
                    original_timestamp = start_time + frame_time_in_segment
                    frame_start_times.append(original_timestamp)
            
            audio_offset += len(audio_segment)
        
        if len(rms_values) == 0:
            return []
        
        # Calculate average volume for this speaker
        avg_volume = np.mean(rms_values)
        
        # Find volume deviations for this speaker
        current_issue = None
        
        for i, (rms_db, original_time) in enumerate(zip(rms_values, frame_start_times)):
            deviation = rms_db - avg_volume
            
            # Check if volume is significantly different
            if abs(deviation) > self.volume_threshold_db:
                if current_issue is None:
                    # Start new issue
                    current_issue = {
                        "start_time": original_time,
                        "deviation": deviation,
                        "frames": 1
                    }
                else:
                    # Continue current issue if same type and close in time
                    time_gap = original_time - current_issue.get("last_time", current_issue["start_time"])
                    if (deviation > 0) == (current_issue["deviation"] > 0) and time_gap < 5.0:  # Allow 5s gap
                        current_issue["frames"] += 1
                        current_issue["deviation"] = (current_issue["deviation"] + deviation) / 2
                        current_issue["last_time"] = original_time
                    else:
                        # Different type of issue or too much gap, finish current and start new
                        if current_issue["frames"] * self.frame_duration >= 1.0:  # At least 1 second
                            self._add_speaker_issue(speaker_issues, current_issue, call_start_time, speaker_id)
                        
                        current_issue = {
                            "start_time": original_time,
                            "deviation": deviation,
                            "frames": 1,
                            "last_time": original_time
                        }
            else:
                # Volume is normal, finish current issue if exists
                if current_issue is not None:
                    if current_issue["frames"] * self.frame_duration >= 1.0:  # At least 1 second
                        self._add_speaker_issue(speaker_issues, current_issue, call_start_time, speaker_id)
                    current_issue = None
        
        # Handle issue that extends to end of speaker audio
        if current_issue is not None and current_issue["frames"] * self.frame_duration >= 1.0:
            self._add_speaker_issue(speaker_issues, current_issue, call_start_time, speaker_id)
        
        return speaker_issues
    
    def _add_speaker_issue(self, issues: List[Dict], issue_data: Dict, call_start_time: str, speaker_id: str):
        """Add speaker-specific issue to issues list."""
        deviation = issue_data["deviation"]
        start_sec = issue_data["start_time"]
        
        # Estimate end time
        if "last_time" in issue_data:
            end_time = issue_data["last_time"] + self.frame_duration
        else:
            end_time = start_sec + (issue_data["frames"] * self.frame_duration)
        
        duration = end_time - start_sec
        
        if deviation > 0:
            issue_type = f"{speaker_id}: Volume spike (+{deviation:.1f} dB above avg)"
            score = max(0.1, 1.0 - (deviation / 20.0))  # Lower score for louder spikes
        else:
            issue_type = f"{speaker_id}: Volume drop ({deviation:.1f} dB below avg)"
            score = max(0.1, 1.0 - (abs(deviation) / 20.0))  # Lower score for quieter drops
        
        # Create timestamp in same format as other evaluators
        timestamp_entry = {
            "durationSeconds": round(duration, 2),
            "issue": issue_type,
            "score": round(score, 2),
            "speaker": speaker_id,
            "relativeStartTime": self._format_time(start_sec),
            "relativeEndTime": self._format_time(end_time)
        }
        
        # Add absolute timestamps if call_start_time is provided
        if call_start_time:
            call_start_dt = datetime.fromisoformat(call_start_time.rstrip('Z'))
            start_dt = call_start_dt + timedelta(seconds=start_sec)
            end_dt = call_start_dt + timedelta(seconds=end_time)
            
            timestamp_entry.update({
                "startTime": start_dt.isoformat() + "Z",
                "endTime": end_dt.isoformat() + "Z"
            })
        
        issues.append(timestamp_entry)