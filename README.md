# Audio Evaluation Pipeline

A comprehensive audio analysis system for evaluating voice call quality in collections and insurance call centers.

## Overview

Analyzes audio recordings from URLs for three critical quality issues:
- **Long Silence Periods** - Awkward gaps that disrupt conversation flow
- **Volume Consistency Problems** - Microphone issues, mumbling, or equipment problems  
- **Speaker Interruptions** - Overlapping speech that indicates poor call management

## Available Evaluators

### 1. Long Silence Detector

**Purpose**: Identifies extended periods of silence that may indicate technical issues, uncomfortable pauses, or call flow problems.

**Algorithm**:
1. Convert audio to mono and analyze energy levels across overlapping frames
2. Apply energy threshold to distinguish speech from silence
3. Identify continuous silence periods longer than the configured threshold (default: 3.0 seconds)
4. Filter out normal conversation gaps and focus on problematic silences
5. Generate timestamps and duration measurements for each silence period

**Key Parameters**:
- `silence_threshold_seconds`: Minimum duration to flag as problematic (default: 3.0s)
- Energy threshold automatically calculated from audio characteristics

**Output Format**:
```json
{
  "results": true,
  "message": "Found 4 silence periods longer than 3.0s (total: 61.8s)",
  "timestamps": [
    {
      "startTime": "2025-07-16T20:21:44.795586Z",
      "endTime": "2025-07-16T20:22:02.245586Z", 
      "durationSeconds": 17.45
    }
  ]
}
```

### 2. Volume Consistency Evaluator

**Purpose**: Detects significant volume fluctuations that indicate microphone positioning issues, agent mumbling, customer frustration, or technical problems.

**Algorithm**:
1. **Speaker Diarization**: Segment audio by speaker using pyannote.audio (with energy-based fallback)
2. **Per-Speaker Analysis**: Analyze volume consistency separately for each speaker to avoid false positives from natural volume differences between participants
3. **RMS Volume Calculation**: Compute Root Mean Square volume for overlapping 1-second frames
4. **Statistical Analysis**: Calculate rolling mean and standard deviation for each speaker
5. **Anomaly Detection**: Flag frames where volume deviates beyond threshold (±10dB) from speaker's average
6. **Quality Scoring**: Assign scores from 0.1 (severe issue) to 1.0 (no issue) based on deviation magnitude

**Key Parameters**:
- `frame_duration`: Analysis window size (default: 1.0 second)
- `volume_threshold_db`: Deviation threshold per speaker (default: ±10.0 dB)
- `enable_diarization`: Use speaker separation (default: True)

**Speaker Diarization Process**:
- **Primary**: Uses pyannote.audio neural network for accurate speaker separation
- **Fallback**: Energy-based clustering when PyTorch/pyannote unavailable
- Segments audio into speaker-specific regions to prevent cross-speaker false positives

**Output Format**:
```json
{
  "results": true,
  "message": "Found 3 volume consistency issues across 2 speakers (threshold: ±10.0 dB per speaker)",
  "timestamps": [
    {
      "durationSeconds": 1.0,
      "issue": "SPEAKER_01: Volume drop (-19.1 dB below avg)",
      "score": 0.1,
      "speaker": "SPEAKER_01",
      "relativeStartTime": "00:06.500",
      "relativeEndTime": "00:07.500"
    }
  ],
  "speaker_analysis": {
    "SPEAKER_00": {"issues_count": 0, "segments_count": 34, "total_duration": 99.5},
    "SPEAKER_01": {"issues_count": 3, "segments_count": 50, "total_duration": 108.0}
  },
  "diarization_enabled": true
}
```

### 3. Speaker Overlap Detector

**Purpose**: Identifies problematic speaker interruptions that indicate poor call flow, impatience, or communication breakdown.

**Algorithm**:
1. **Speaker Diarization**: Use same diarization system as volume evaluator
2. **Temporal Overlap Detection**: Find time periods where multiple speakers are active simultaneously
3. **Duration Filtering**: Only flag significant interruptions (2+ seconds) to avoid normal conversation overlaps
4. **Interruption Analysis**: Determine who interrupted whom based on speech start times
5. **Severity Classification**: 
   - **Medium**: 2.0-3.0 second interruptions
   - **Major**: 3.0+ second interruptions
6. **Merge Adjacent Overlaps**: Combine closely related interruptions to avoid over-reporting
7. **Attention Flagging**: Mark calls requiring review based on interruption patterns

**Severity Calculation**:
- Base score on interruption duration
- Factor in timing (quick interruptions more severe)
- Consider speaker dynamics and conversation flow

**Key Parameters**:
- `min_overlap_duration`: Minimum duration to flag (default: 2.0 seconds)
- Medium threshold: 3.0 seconds
- Merge window: 0.5 seconds for adjacent overlaps

**Output Format**:
```json
{
  "results": false,
  "message": "No significant speaker interruptions detected",
  "timestamps": [],
  "overlap_analysis": {
    "total_interruptions": 0,
    "total_interruption_duration": 0,
    "overlap_severity_flag": "none",
    "requires_attention": false,
    "severity_breakdown": {"medium": 0, "major": 0},
    "speakers_detected": 2
  }
}
```

## Installation

```bash
git clone <repository-url>
cd audio_evals
pip install -r requirements.txt
```

## Usage

**Command Line**:
```bash
python main.py "https://example.com/call-recording.mp3"
```

**Programmatic**:
```python
from audio_evals import AudioEvaluationPipeline
from audio_evals.evaluators import LongSilenceDetector, VolumeConsistencyEvaluator, SpeakerOverlapDetector

pipeline = AudioEvaluationPipeline()
pipeline.register_evaluator(LongSilenceDetector())
pipeline.register_evaluator(VolumeConsistencyEvaluator(enable_diarization=True))
pipeline.register_evaluator(SpeakerOverlapDetector())

result = pipeline.evaluate_audio_url("https://example.com/audio.mp3")
```

**Testing**:
```bash
python tests/test_pipeline.py
```

## Pipeline Processing Flow

1. **Audio Acquisition**:
   - Download audio from provided URL
   - Support for MP3, WAV, and other common formats
   - Temporary file management with automatic cleanup

2. **Audio Preprocessing**:
   - Convert to mono channel for consistent analysis
   - Normalize sample rate to 16kHz
   - Extract metadata (duration, format, channels)

3. **Speaker Diarization** (for volume and overlap analysis):
   - Attempt pyannote.audio neural network diarization
   - Fall back to energy-based clustering if unavailable
   - Generate speaker-segmented timeline

4. **Parallel Evaluation**:
   - Run all registered evaluators simultaneously
   - Each evaluator processes audio independently
   - Collect results and error handling per evaluator

5. **Result Aggregation**:
   - Combine all evaluator outputs
   - Generate evaluation summary statistics
   - Create standardized JSON output format
   - Save results to specified directory

## Output Format

The pipeline generates comprehensive JSON reports with the following structure:

```json
{
  "success": true,
  "evaluation_id": "eval_1752722711",
  "audio_metadata": {
    "source_url": "https://example.com/audio.mp3",
    "duration_seconds": 237.5,
    "sample_rate": 16000,
    "channels": 1,
    "call_start_time": "2025-07-16T20:19:46.145586Z"
  },
  "evaluation_summary": {
    "total_evaluators_run": 3,
    "successful_evaluators": 3, 
    "total_issues_found": 7,
    "processing_time_seconds": 7.74
  },
  "evaluator_status": {
    "long_silence_detection": "success",
    "volume_consistency": "success", 
    "speaker_overlap_detection": "success"
  },
  "results": {
    "long_silence_detection": { /* evaluator results */ },
    "volume_consistency": { /* evaluator results */ },
    "speaker_overlap_detection": { /* evaluator results */ }
  },
  "generated_at": "2025-07-16T20:19:51.157939Z",
  "pipeline_version": "1.0.0"
}
```

## Configuration

### Evaluator Parameters

**LongSilenceDetector**:
- `silence_threshold_seconds` (float): Minimum silence duration to flag (default: 3.0)

**VolumeConsistencyEvaluator**:
- `frame_duration` (float): Analysis window size in seconds (default: 1.0)
- `volume_threshold_db` (float): Volume deviation threshold in dB (default: 10.0) 
- `enable_diarization` (bool): Use speaker separation (default: True)

**SpeakerOverlapDetector**:
- `min_overlap_duration` (float): Minimum overlap to flag in seconds (default: 2.0)

### Environment Variables

- `PYTORCH_AVAILABLE`: Force disable PyTorch (set to "false")
- `PYANNOTE_AVAILABLE`: Force disable pyannote.audio (set to "false")

## Dependencies

**Core Requirements**:
- `librosa>=0.9.0` - Audio processing and analysis
- `numpy>=1.21.0` - Numerical computations
- `requests>=2.25.0` - HTTP requests for audio download
- `scipy>=1.7.0` - Scientific computing utilities

**Optional (Enhanced Features)**:
- `torch>=1.9.0` - PyTorch for neural networks
- `pyannote.audio==3.1.1` - Advanced speaker diarization
- `torchaudio>=0.9.0` - Audio processing for PyTorch

## Performance Characteristics

**Typical Processing Times** (per minute of audio):
- Long Silence Detection: ~0.5 seconds
- Volume Consistency (with diarization): ~2-3 seconds  
- Speaker Overlap Detection: ~1-2 seconds
- **Total**: ~4-6 seconds per minute of audio

**Memory Usage**:
- Base pipeline: ~100-200MB
- With PyTorch/pyannote: ~500MB-1GB
- Audio buffer: ~16MB per minute of audio

**Supported Audio Formats**:
- MP3, WAV, FLAC, OGG, M4A
- Mono/Stereo (converted to mono for analysis)
- Various sample rates (normalized to 16kHz)

## Error Handling

The pipeline includes robust error handling:

- **Network Issues**: Retry logic for audio downloads
- **Format Issues**: Automatic format conversion attempts
- **Missing Dependencies**: Graceful fallbacks (e.g., energy-based diarization)
- **Memory Issues**: Chunked processing for large files
- **Invalid Audio**: Clear error messages and skipping problematic evaluators

## Use Cases

**Collections Call Centers**:
- Identify calls with poor audio quality affecting collection success
- Flag agents with consistent volume/audio issues for training
- Detect problematic customer interactions requiring supervisor review

**Insurance Call Centers**:
- Ensure clear communication during claim discussions
- Monitor call quality for compliance and customer satisfaction
- Identify technical issues affecting call center operations

**Quality Assurance**:
- Automated pre-screening of calls before manual review
- Objective audio quality metrics for agent performance evaluation
- Bulk analysis of call recordings for trend identification

## Assumptions and Limitations

- **Audio Format**: Supports MP3, WAV, FLAC from HTTP/HTTPS URLs. Normalizes to 16kHz mono for analysis.
- **Call Structure**: Optimized for 2-speaker conversations (agent/customer). Performance degrades with 3+ speakers.
- **Speaker Diarization**: Uses PyTorch/pyannote.audio when available, falls back to energy-based clustering.
- **Threshold-Based Detection**: May produce false positives/negatives. Recommended for initial screening, not final assessment.
- **Processing Requirements**: ~4-6 seconds per minute of audio. Network dependent for URL downloads.

## Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/new-evaluator`)
3. Implement your changes with tests
4. Submit a pull request with detailed description

## License

[Specify your license here]