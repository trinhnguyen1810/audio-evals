import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Any


class LongSilenceDetector:
    def __init__(self, silence_threshold_seconds: float = 5.0, energy_threshold: float = 0.01):
        self.name = "long_silence_detection"
        self.description = "Detects gaps longer than 3 seconds with no speaker activity"
        self.silence_threshold_seconds = silence_threshold_seconds
        self.energy_threshold = energy_threshold
    
    def evaluate(self, audio_data: np.ndarray, sample_rate: int, call_start_time: str) -> Dict[str, Any]:
        window_size = int(0.1 * sample_rate)
        hop_size = window_size // 2
        energy_values = []
        for i in range(0, len(audio_data) - window_size, hop_size):
            window = audio_data[i:i + window_size]
            rms_energy = np.sqrt(np.mean(window ** 2))
            energy_values.append(rms_energy)
        
        silent_windows = np.array(energy_values) < self.energy_threshold
        silent_segments = []
        current_start = None
        
        for i, is_silent in enumerate(silent_windows):
            window_time = i * hop_size / sample_rate
            
            if is_silent and current_start is None:
                current_start = window_time
            elif not is_silent and current_start is not None:
                duration = window_time - current_start
                if duration >= self.silence_threshold_seconds:
                    silent_segments.append((current_start, window_time, duration))
                current_start = None
        
        if current_start is not None:
            end_time = len(audio_data) / sample_rate
            duration = end_time - current_start
            if duration >= self.silence_threshold_seconds:
                silent_segments.append((current_start, end_time, duration))
        
        call_start_dt = datetime.fromisoformat(call_start_time.rstrip('Z'))
        timestamps = []
        
        for start_sec, end_sec, duration in silent_segments:
            start_dt = call_start_dt + timedelta(seconds=start_sec)
            end_dt = call_start_dt + timedelta(seconds=end_sec)
            
            start_minutes = int(start_sec // 60)
            start_seconds = int(start_sec % 60)
            end_minutes = int(end_sec // 60)
            end_seconds = int(end_sec % 60)
            
            timestamps.append({
                "startTime": start_dt.isoformat() + "Z",
                "endTime": end_dt.isoformat() + "Z",
                "durationSeconds": round(duration, 2),
                "relativeStartTime": f"{start_minutes:02d}:{start_seconds:02d}",
                "relativeEndTime": f"{end_minutes:02d}:{end_seconds:02d}"
            })
        
        has_long_silences = len(timestamps) > 0
        message = f"Found {len(timestamps)} silence periods longer than {self.silence_threshold_seconds}s"
        
        if has_long_silences:
            total_silence_time = sum(ts["durationSeconds"] for ts in timestamps)
            message += f" (total: {total_silence_time:.1f}s)"
        
        return {
            self.name: {
                "results": has_long_silences,
                "message": message,
                "timestamps": timestamps
            }
        }