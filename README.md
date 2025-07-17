# Audio Evaluation Pipeline

Analyzes audio recordings from URLs for call quality issues in collections and insurance call centers.

## Features

- **Long Silence Detection** - Identifies awkward gaps that disrupt conversation flow
- **Volume Consistency** - Detects microphone issues, mumbling, or equipment problems  
- **Speaker Interruptions** - Finds overlapping speech indicating poor call management
- **AI Agent Detection** - Identifies potential AI agents in conversations

## Installation

```bash
git clone <repository-url>
cd audio_evals
pip install -r requirements.txt
```

## Environment Setup

Create a `.env` file in the root directory with the following configuration:

```env
# OpenAI API Configuration (required for AI agent detection)
OPENAI_API_URL=https://api.openai.com/v1/chat/completions
OPENAI_API_KEY=your_openai_api_key_here

# AI Agent Detection Settings
MIN_SPEAKERS_FOR_AI_DETECTION=3
AI_DETECTION_CONFIDENCE_THRESHOLD=0.7

# Pyannote API Key (required for advanced speaker diarization)
PYANOTE_API_KEY=your_pyannote_api_key_here
```

**Note:** The pipeline will work without these keys but with reduced functionality:
- Without OpenAI API key: AI agent detection will be disabled
- Without Pyannote API key: Falls back to energy-based speaker clustering

## Usage

### Command Line
```bash
# Option 1: Using main.py
python main.py "https://example.com/call-recording.mp3"

# Option 2: Using module
python -m audio_evals.cli "https://example.com/call-recording.mp3"
```

#### CLI Options
- `--output-dir DIR` - Directory to save results (default: results)
- `--quiet` - Suppress progress output
- `--help` - Show help message

#### Examples
```bash
# Basic usage
python main.py "https://example.com/call-recording.mp3"

# Custom output directory
python main.py "https://example.com/call.mp3" --output-dir /path/to/results

# Quiet mode (minimal output)
python main.py "https://example.com/call.mp3" --quiet
```

### Python API
```python
from audio_evals.pipeline import AudioEvaluationPipeline
from audio_evals.evaluators.silence_detector import LongSilenceDetector
from audio_evals.evaluators.volume_consistency import VolumeConsistencyEvaluator
from audio_evals.evaluators.speaker_overlap_detector import SpeakerOverlapDetector

pipeline = AudioEvaluationPipeline()
pipeline.register_evaluator(LongSilenceDetector(silence_threshold_seconds=3.0))
pipeline.register_evaluator(VolumeConsistencyEvaluator(volume_threshold_db=10.0))
pipeline.register_evaluator(SpeakerOverlapDetector(min_overlap_duration=1.0))

result = pipeline.evaluate_audio_url("https://example.com/audio.mp3")
```

## Configuration

### LongSilenceDetector
- `silence_threshold_seconds` (default: 3.0) - Minimum silence duration to flag

### VolumeConsistencyEvaluator  
- `volume_threshold_db` (default: 10.0) - Volume deviation threshold
- `enable_diarization` (default: True) - Use speaker separation

### SpeakerOverlapDetector
- `min_overlap_duration` (default: 1.0) - Minimum overlap duration to flag

## Output Format

Results are saved as JSON with this structure:

```json
{
  "success": true,
  "evaluation_id": "eval_1752722711",
  "audio_metadata": {
    "source_url": "https://example.com/audio.mp3",
    "duration_seconds": 237.5,
    "sample_rate": 16000,
    "channels": 1
  },
  "evaluation_summary": {
    "total_evaluators_run": 4,
    "successful_evaluators": 4,
    "total_issues_found": 19,
    "processing_time_seconds": 20.8
  },
  "results": {
    "long_silence_detection": {
      "results": true,
      "message": "Found 4 silence periods longer than 3.0s (total: 61.8s)",
      "timestamps": [
        {
          "startTime": "2025-07-17T00:44:34.426282Z",
          "endTime": "2025-07-17T00:44:51.876282Z",
          "durationSeconds": 17.45,
          "relativeStartTime": "01:58",
          "relativeEndTime": "02:16"
        }
      ]
    }
  }
}
```

## Dependencies

**Core:**
- `librosa>=0.9.0` - Audio processing
- `numpy>=1.21.0` - Numerical computations  
- `requests>=2.25.0` - HTTP requests
- `scipy>=1.7.0` - Scientific computing

**Optional (Enhanced Features):**
- `torch>=1.9.0` - PyTorch for neural networks
- `pyannote.audio==3.1.1` - Advanced speaker diarization
- `torchaudio>=0.9.0` - Audio processing for PyTorch

## Performance

- **Processing Time**: ~4-6 seconds per minute of audio
- **Memory Usage**: ~100-500MB (base), ~500MB-1GB (with PyTorch)
- **Supported Formats**: MP3, WAV, FLAC, OGG, M4A

## Limitations

- Optimized for 2-speaker conversations (agent/customer)
- Requires HTTP/HTTPS URLs for audio files
- May produce false positives/negatives - use for screening, not final assessment