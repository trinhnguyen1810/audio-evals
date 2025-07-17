# app/utils.py
import librosa
import numpy as np
import requests
import tempfile
import os
import time
from typing import Tuple, List, Dict, Any

def download_audio(url: str) -> str:
    """
    Download audio file from URL and save to temp file.
    
    Args:
        url: Public audio file URL
        
    Returns:
        Path to temporary file
    """
    try:
        print(f"📥 Downloading from: {url}")
        
        # Download with timeout
        response = requests.get(url, timeout=30, stream=True)
        response.raise_for_status()
        
        # Save to temp file
        temp_file = tempfile.NamedTemporaryFile(delete=False, suffix='.mp3')
        
        for chunk in response.iter_content(chunk_size=8192):
            temp_file.write(chunk)
        
        temp_file.close()
        
        file_size = os.path.getsize(temp_file.name)
        print(f"✅ Downloaded {file_size} bytes to {temp_file.name}")
        
        return temp_file.name
        
    except Exception as e:
        raise Exception(f"Failed to download audio: {e}")

def load_audio(file_path: str, target_sr: int = 16000) -> Tuple[np.ndarray, int]:
    """
    Load audio file and convert to standard format.
    
    Args:
        file_path: Path to audio file
        target_sr: Target sample rate
        
    Returns:
        (audio_data, sample_rate) tuple
    """
    try:
        print(f"🔄 Loading audio from: {file_path}")
        
        # Load and convert to mono at target sample rate
        audio_data, sample_rate = librosa.load(
            file_path, 
            sr=target_sr, 
            mono=True,
            dtype=np.float32
        )
        
        # Normalize to prevent clipping
        if np.max(np.abs(audio_data)) > 0:
            audio_data = audio_data / np.max(np.abs(audio_data)) * 0.95
        
        duration = len(audio_data) / sample_rate
        print(f"✅ Loaded {duration:.2f}s audio at {sample_rate}Hz")
        
        return audio_data, sample_rate
        
    except Exception as e:
        raise Exception(f"Failed to load audio: {e}")

def cleanup_temp_file(file_path: str):
    """Remove temporary file."""
    try:
        if os.path.exists(file_path):
            os.unlink(file_path)
            print(f"🗑️ Cleaned up: {file_path}")
    except Exception as e:
        print(f"⚠️ Failed to cleanup {file_path}: {e}")

def start_pyannote_diarization(audio_url: str, pyannote_token: str, num_speakers: int = 2) -> str:
    headers = {
        "Authorization": f"Bearer {pyannote_token}",
        "Content-Type": "application/json"
    }
    data = {"url": audio_url, "numSpeakers": num_speakers}
    response = requests.post("https://api.pyannote.ai/v1/diarize", json=data, headers=headers)
    if response.status_code != 200:
        raise Exception(f"Pyannote diarization failed: {response.text}")
    job_id = response.json().get("jobId")
    if not job_id:
        raise Exception("No jobId received from Pyannote")
    return job_id

def poll_pyannote_job(job_id: str, pyannote_token: str, max_attempts: int = 60) -> Dict:
    headers = {"Authorization": f"Bearer {pyannote_token}"}
    for i in range(max_attempts):
        time.sleep(5)
        response = requests.get(f"https://api.pyannote.ai/v1/jobs/{job_id}", headers=headers)
        if response.status_code != 200:
            raise Exception(f"Error getting job status: {response.text}")
        status_data = response.json()
        if status_data.get("status") == "succeeded":
            return status_data.get("output", {})
        if status_data.get("status") == "failed":
            raise Exception(f"Pyannote job failed: {status_data.get('error')}")
    raise Exception("Pyannote diarization timeout")

def get_pyannote_diarization(audio_url: str, pyannote_token: str) -> Dict:
    try:
        job_id = start_pyannote_diarization(audio_url, pyannote_token)
        return poll_pyannote_job(job_id, pyannote_token)
    except Exception as e:
        print(f"[WARN] Pyannote diarization failed: {e}")
        return None

def perform_speaker_diarization(audio_data: np.ndarray, sample_rate: int, audio_url: str = None, pyannote_token: str = None) -> List[Dict[str, Any]]:
    """
    Perform speaker diarization on audio to identify speaker segments.
    
    Args:
        audio_data: Audio samples as numpy array (mono)
        sample_rate: Sample rate of audio
        audio_url: Public URL to audio file (for API-based diarization)
        pyannote_token: API token for pyannote.ai service
        
    Returns:
        List of speaker segments with format:
        [
          {
            "start": 0.0,           # Start time in seconds
            "end": 5.2,             # End time in seconds  
            "speaker": "SPEAKER_00", # Speaker ID
            "duration": 5.2         # Duration in seconds
          }
        ]
    """
    # Get API token from environment if not provided
    if pyannote_token is None:
        pyannote_token = os.environ.get('PYANOTE_API_KEY')
    
    # Try API-based diarization first if token and URL are provided
    if audio_url and pyannote_token:
        try:
            print("🎯 Using Pyannote API for diarization...")
            diarization_result = get_pyannote_diarization(audio_url, pyannote_token)
            
            if diarization_result:
                segments = []
                # Convert API response to our format
                for segment in diarization_result.get("diarization", []):
                    segments.append({
                        "start": segment.get("start", 0.0),
                        "end": segment.get("end", 0.0),
                        "speaker": segment.get("speaker", "SPEAKER_00"),
                        "duration": segment.get("end", 0.0) - segment.get("start", 0.0)
                    })
                
                print(f"✅ API Diarization complete: Found {len(set(seg['speaker'] for seg in segments))} speakers in {len(segments)} segments")
                return segments
            else:
                print("⚠️ API diarization failed, falling back to simple segmentation")
        except Exception as e:
            print(f"⚠️ API diarization failed: {e}, using fallback segmentation")
    
    # Fallback to energy-based segmentation
    print("⚠️ Using simple energy-based segmentation")
    return _fallback_speaker_segmentation(audio_data, sample_rate)

def _fallback_speaker_segmentation(audio_data: np.ndarray, sample_rate: int) -> List[Dict[str, Any]]:
    """
    Simple fallback speaker segmentation based on audio energy and voice activity.
    Uses VAD (Voice Activity Detection) and energy clustering to estimate speakers.
    """
    print("📊 Running simple energy-based speaker clustering...")
    
    # Frame parameters
    frame_duration = 1.0  # 1 second frames
    frame_size = int(frame_duration * sample_rate)
    hop_size = frame_size // 2
    
    # Calculate energy for each frame
    frame_energies = []
    frame_times = []
    
    for i in range(0, len(audio_data) - frame_size, hop_size):
        frame = audio_data[i:i + frame_size]
        rms_energy = np.sqrt(np.mean(frame ** 2))
        if rms_energy > 0:
            energy_db = 20 * np.log10(rms_energy)
        else:
            energy_db = -80  # Very quiet
        
        frame_energies.append(energy_db)
        frame_times.append(i / sample_rate)
    
    if len(frame_energies) == 0:
        return []
    
    # Simple voice activity detection (energy threshold)
    energy_threshold = np.percentile(frame_energies, 30)  # Bottom 30% is likely silence
    active_frames = np.array(frame_energies) > energy_threshold
    
    # Cluster active frames by energy level (simple 2-speaker assumption)
    active_energies = np.array(frame_energies)[active_frames]
    if len(active_energies) > 0:
        energy_median = np.median(active_energies)
        
        # Assign speakers based on energy level (rough heuristic)
        speaker_assignments = []
        for i, (energy, is_active) in enumerate(zip(frame_energies, active_frames)):
            if not is_active:
                speaker_assignments.append(None)  # Silence
            elif energy > energy_median:
                speaker_assignments.append("SPEAKER_00")  # Higher energy speaker
            else:
                speaker_assignments.append("SPEAKER_01")  # Lower energy speaker
    else:
        speaker_assignments = [None] * len(frame_energies)
    
    # Group consecutive frames by speaker
    segments = []
    current_speaker = None
    current_start = None
    
    for i, speaker in enumerate(speaker_assignments):
        if speaker != current_speaker:
            # End previous segment
            if current_speaker is not None and current_start is not None:
                end_time = frame_times[i-1] + frame_duration
                segments.append({
                    "start": current_start,
                    "end": end_time,
                    "speaker": current_speaker,
                    "duration": end_time - current_start
                })
            
            # Start new segment
            if speaker is not None:
                current_start = frame_times[i]
                current_speaker = speaker
            else:
                current_start = None
                current_speaker = None
    
    # Handle final segment
    if current_speaker is not None and current_start is not None:
        end_time = len(audio_data) / sample_rate
        segments.append({
            "start": current_start,
            "end": end_time,
            "speaker": current_speaker,
            "duration": end_time - current_start
        })
    
    # Filter out very short segments (< 1 second)
    segments = [seg for seg in segments if seg["duration"] >= 1.0]
    
    unique_speakers = len(set(seg['speaker'] for seg in segments))
    print(f"✅ Simple segmentation complete: Found {unique_speakers} speakers in {len(segments)} segments")
    
    return segments

def get_speaker_audio_segments(audio_data: np.ndarray, sample_rate: int, 
                             speaker_segments: List[Dict[str, Any]]) -> Dict[str, List[Tuple[np.ndarray, float, float]]]:
    """
    Extract audio segments for each speaker.
    
    Args:
        audio_data: Full audio as numpy array
        sample_rate: Sample rate
        speaker_segments: Output from perform_speaker_diarization
        
    Returns:
        Dictionary mapping speaker ID to list of (audio_segment, start_time, end_time) tuples
    """
    speaker_audio = {}
    
    for segment in speaker_segments:
        speaker_id = segment["speaker"]
        start_time = segment["start"]
        end_time = segment["end"]
        
        # Convert to sample indices
        start_sample = int(start_time * sample_rate)
        end_sample = int(end_time * sample_rate)
        
        # Extract audio segment
        audio_segment = audio_data[start_sample:end_sample]
        
        if speaker_id not in speaker_audio:
            speaker_audio[speaker_id] = []
        
        speaker_audio[speaker_id].append((audio_segment, start_time, end_time))
    
    return speaker_audio