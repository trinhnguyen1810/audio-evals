# test_pipeline.py
import sys
import os

# Add the parent directory to the Python path so we can import audio_evals
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from audio_evals.pipeline import AudioEvaluationPipeline
from audio_evals.evaluators.silence_detector import LongSilenceDetector
from audio_evals.evaluators.volume_consistency import VolumeConsistencyEvaluator
from audio_evals.evaluators.speaker_overlap_detector import SpeakerOverlapDetector
import json

# Create pipeline
pipeline = AudioEvaluationPipeline()

# Register evaluators
silence_detector = LongSilenceDetector(silence_threshold_seconds=3.0)
pipeline.register_evaluator(silence_detector)

volume_evaluator = VolumeConsistencyEvaluator(frame_duration=1.0, volume_threshold_db=10.0, enable_diarization=True)
pipeline.register_evaluator(volume_evaluator)

overlap_detector = SpeakerOverlapDetector(min_overlap_duration=2.0)
pipeline.register_evaluator(overlap_detector)

# Test with a sample URL
url = "https://domu-call-recordings.s3.amazonaws.com/CLT9dee1c0ed39ef4e2686f974b5d574531/CMPfbaa452c913ca2fa013cb9d39d614d17/CLLa5e39ca666d893be77f978eaa989c488"

result = pipeline.evaluate_audio_url(url)

# Save JSON to file in results folder
import os
os.makedirs("results", exist_ok=True)
output_filename = f"results/audio_evaluation_{result['evaluation_id']}.json"
with open(output_filename, 'w') as f:
    json.dump(result, f, indent=2)

print(f"📄 Results saved to: {output_filename}")
print(json.dumps(result, indent=2))

# Also print a summary of results for easier reading
print("\n" + "="*50)
print("EVALUATION SUMMARY")
print("="*50)

if result["success"]:
    # Silence Detection Summary
    if "long_silence_detection" in result["results"]:
        silence_data = result["results"]["long_silence_detection"]
        print(f"🔇 SILENCE DETECTION: {silence_data['message']}")
        if silence_data.get("timestamps"):
            print(f"   Found {len(silence_data['timestamps'])} silence periods")
    
    # Volume Consistency Summary  
    if "volume_consistency" in result["results"]:
        volume_data = result["results"]["volume_consistency"]
        print(f"🔊 VOLUME CONSISTENCY: {volume_data['message']}")
        
        # Show speaker analysis if available
        if volume_data.get("speaker_analysis"):
            print(f"   Speaker Analysis:")
            for speaker, analysis in volume_data["speaker_analysis"].items():
                duration_min = analysis["total_duration"] / 60
                print(f"   - {speaker}: {analysis['issues_count']} issues in {analysis['segments_count']} segments ({duration_min:.1f}min)")
        
        if volume_data.get("timestamps"):
            print(f"   Found {len(volume_data['timestamps'])} volume issues:")
            for i, timestamp in enumerate(volume_data["timestamps"][:3], 1):  # Show first 3
                rel_start = timestamp.get('relativeStartTime', 'N/A')
                rel_end = timestamp.get('relativeEndTime', 'N/A')
                print(f"   {i}. {rel_start}-{rel_end}: {timestamp['issue']} (score: {timestamp['score']})")
            if len(volume_data["timestamps"]) > 3:
                print(f"   ... and {len(volume_data['timestamps']) - 3} more volume issues")
    
    # Speaker Overlap Summary
    if "speaker_overlap_detection" in result["results"]:
        overlap_data = result["results"]["speaker_overlap_detection"]
        print(f"🗣️  SPEAKER OVERLAPS: {overlap_data['message']}")
        
        # Show overlap analysis if available
        if overlap_data.get("overlap_analysis"):
            analysis = overlap_data["overlap_analysis"]
            severity_flag = analysis.get("overlap_severity_flag", "none")
            requires_attention = analysis.get("requires_attention", False)
            
            print(f"   Analysis: {analysis['speakers_detected']} speakers, severity: {severity_flag}")
            
            # Show severity breakdown
            if "severity_breakdown" in analysis:
                breakdown = analysis["severity_breakdown"]
                print(f"   Breakdown: {breakdown['medium']} medium, {breakdown['major']} major interruptions")
            
            # Flag for attention if needed
            if requires_attention:
                print(f"   🚨 REQUIRES ATTENTION: Significant interruptions detected")
        
        if overlap_data.get("timestamps"):
            print(f"   Found {len(overlap_data['timestamps'])} significant interruptions:")
            for i, timestamp in enumerate(overlap_data["timestamps"][:3], 1):  # Show first 3
                rel_start = timestamp.get('relativeStartTime', 'N/A')
                rel_end = timestamp.get('relativeEndTime', 'N/A')
                overlap_severity = timestamp.get('overlap_severity', 'unknown')
                duration = timestamp.get('durationSeconds', 'N/A')
                print(f"   {i}. {rel_start}-{rel_end}: {timestamp['issue']} ({duration}s)")
            if len(overlap_data["timestamps"]) > 3:
                print(f"   ... and {len(overlap_data['timestamps']) - 3} more interruptions")
    
    print(f"\n📊 Total issues found: {result['evaluation_summary']['total_issues_found']}")
    print(f"⏱️  Processing time: {result['evaluation_summary']['processing_time_seconds']}s")