import sys
import os
import json
import argparse
from datetime import datetime
from dotenv import load_dotenv

load_dotenv()

from .pipeline import AudioEvaluationPipeline
from .evaluators.silence_detector import LongSilenceDetector
from .evaluators.volume_consistency import VolumeConsistencyEvaluator
from .evaluators.speaker_overlap_detector import SpeakerOverlapDetector
from .evaluators.ai_agent_detector import AIAgentDetector


def create_pipeline():
    
    print("🔧 Initializing Audio Evaluation Pipeline...")
    pipeline = AudioEvaluationPipeline()
    
    # Register Long Silence Detector
    print("   📢 Adding Long Silence Detector (3.0s threshold)")
    silence_detector = LongSilenceDetector(silence_threshold_seconds=3.0)
    pipeline.register_evaluator(silence_detector)
    
    # Register Volume Consistency Evaluator with speaker diarization
    print("   🔊 Adding Volume Consistency Evaluator (±10dB threshold, speaker-aware)")
    volume_evaluator = VolumeConsistencyEvaluator(
        frame_duration=1.0, 
        volume_threshold_db=10.0, 
        enable_diarization=True
    )
    pipeline.register_evaluator(volume_evaluator)
    
    # Register Speaker Overlap Detector  
    print("   🗣️  Adding Speaker Overlap Detector (1.0s+ interruptions)")
    overlap_detector = SpeakerOverlapDetector(min_overlap_duration=1.0)
    pipeline.register_evaluator(overlap_detector)
    
    # Register AI Agent Detector
    print("   🤖 Adding AI Agent Detector (3+ speakers)")
    ai_detector = AIAgentDetector()
    pipeline.register_evaluator(ai_detector)
    
    print("✅ Pipeline ready with 4 evaluators\n")
    return pipeline


def save_results(result, output_dir="results"):
    os.makedirs(output_dir, exist_ok=True)
    evaluation_id = result.get('evaluation_id', 'unknown')
    output_filename = os.path.join(output_dir, f"audio_evaluation_{evaluation_id}.json")
    with open(output_filename, 'w') as f:
        json.dump(result, f, indent=2)
    
    print(f"💾 Results saved to: {output_filename}")
    return output_filename


def print_evaluation_summary(result):
    
    print("\n" + "="*60)
    print("AUDIO EVALUATION SUMMARY")
    print("="*60)
    
    if not result.get("success"):
        print("❌ EVALUATION FAILED")
        print(f"Error: {result.get('error', 'Unknown error')}")
        return
    
    metadata = result.get("audio_metadata", {})
    duration_min = metadata.get("duration_seconds", 0) / 60
    print(f"📄 Audio Duration: {duration_min:.1f} minutes")
    print(f"🔍 Evaluators Run: {result['evaluation_summary']['successful_evaluators']}/4")
    print(f"⏱️  Processing Time: {result['evaluation_summary']['processing_time_seconds']:.1f}s")
    print(f"🚨 Total Issues: {result['evaluation_summary']['total_issues_found']}")
    
    results = result.get("results", {})
    
    if "long_silence_detection" in results:
        silence_data = results["long_silence_detection"]
        print(f"\n🔇 SILENCE DETECTION:")
        print(f"   {silence_data['message']}")
        if silence_data.get("timestamps"):
            print(f"   Periods: {len(silence_data['timestamps'])} silence gaps found")
    
    if "volume_consistency" in results:
        volume_data = results["volume_consistency"]
        print(f"\n🔊 VOLUME CONSISTENCY:")
        print(f"   {volume_data['message']}")
        
        if volume_data.get("speaker_analysis"):
            print(f"   Speaker Breakdown:")
            for speaker, analysis in volume_data["speaker_analysis"].items():
                duration_min = analysis["total_duration"] / 60
                print(f"   - {speaker}: {analysis['issues_count']} issues in {duration_min:.1f}min")
        
        if volume_data.get("timestamps"):
            print(f"   Issues: {len(volume_data['timestamps'])} volume problems detected")
    
    if "speaker_overlap_detection" in results:
        overlap_data = results["speaker_overlap_detection"]
        print(f"\n🗣️  SPEAKER OVERLAPS:")
        print(f"   {overlap_data['message']}")
        
        if overlap_data.get("overlap_analysis"):
            analysis = overlap_data["overlap_analysis"]
            severity_flag = analysis.get("overlap_severity_flag", "none")
            requires_attention = analysis.get("requires_attention", False)
            
            print(f"   Analysis: {analysis['speakers_detected']} speakers, severity: {severity_flag}")
            
            if "severity_breakdown" in analysis:
                breakdown = analysis["severity_breakdown"]
                print(f"   Breakdown: {breakdown['medium']} medium, {breakdown['major']} major")
            
    
    if "ai_agent_detection" in results:
        ai_data = results["ai_agent_detection"]
        print(f"\n🤖 AI AGENT DETECTION:")
        print(f"   {ai_data['message']}")
        
        if ai_data.get("ai_analysis"):
            analysis = ai_data["ai_analysis"]
            if analysis.get("detection_performed"):
                print(f"   Analysis: {analysis['speakers_detected']} speakers detected")
                print(f"   Results: {analysis['ai_agents_detected']} AI agents, {analysis['human_speakers']} humans")
                
                if analysis.get("speaker_classifications"):
                    print(f"   Classifications:")
                    for speaker_id, classification in analysis["speaker_classifications"].items():
                        conf = classification.get("confidence", 0)
                        cls = classification.get("classification", "unknown")
                        print(f"   - {speaker_id}: {cls} (confidence: {conf:.2f})")
                
                if ai_data.get("timestamps"):
                    print(f"   🚨 AI Agents Found: {len(ai_data['timestamps'])} segments detected")
    
    print("\n" + "="*60)


def main():
    
    parser = argparse.ArgumentParser(
        description="Audio Evaluation Pipeline - Analyze call recordings for quality issues",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python main.py "https://example.com/call-recording.mp3"
  python main.py "https://s3.amazonaws.com/bucket/call.wav"
        """
    )
    
    parser.add_argument(
        "url", 
        help="URL to audio file (MP3, WAV, etc.)"
    )
    
    parser.add_argument(
        "--output-dir", 
        default="results",
        help="Directory to save results (default: results)"
    )
    
    parser.add_argument(
        "--quiet", 
        action="store_true",
        help="Suppress progress output"
    )
    
    args = parser.parse_args()
    
    if not args.quiet:
        print("🎵 Audio Evaluation Pipeline")
        print("=" * 40)
        print(f"📡 Processing URL: {args.url}")
        print(f"📁 Output Directory: {args.output_dir}")
        print()
    
    try:
        # Create pipeline
        pipeline = create_pipeline()
        
        # Run evaluation
        if not args.quiet:
            print("🔄 Starting audio evaluation...")
        
        result = pipeline.evaluate_audio_url(args.url)
        
        # Save results
        output_file = save_results(result, args.output_dir)
        
        # Print summary
        if not args.quiet:
            print_evaluation_summary(result)
            print(f"\n📄 Detailed results: {output_file}")
        
        # Exit code based on success
        if result.get("success"):
            if not args.quiet:
                print("✅ Evaluation completed successfully!")
            sys.exit(0)
        else:
            if not args.quiet:
                print("❌ Evaluation failed!")
            sys.exit(1)
            
    except KeyboardInterrupt:
        print("\n⚠️  Evaluation interrupted by user")
        sys.exit(130)
        
    except Exception as e:
        print(f"💥 Error during evaluation: {str(e)}")
        if not args.quiet:
            import traceback
            traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()