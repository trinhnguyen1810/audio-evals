# app/pipeline.py
import time
import json
from typing import Dict, List, Any, Optional
from datetime import datetime
from .utils import download_audio, load_audio, cleanup_temp_file

class AudioEvaluationPipeline:
    def __init__(self):
        self.evaluators = {}
        self.version = "1.0.0"
    
    def register_evaluator(self, evaluator):
        """Register an evaluator instance."""
        self.evaluators[evaluator.name] = evaluator
        print(f"✅ Registered evaluator: {evaluator.name}")
    
    def evaluate_audio_url(self, 
                          url: str, 
                          enabled_evaluators: Optional[List[str]] = None,
                          call_start_time: Optional[str] = None) -> Dict[str, Any]:
        """
        Main pipeline method - evaluates audio from URL.
        
        Args:
            url: Public MP3/WAV URL
            enabled_evaluators: List of evaluator names to run (None = all)
            call_start_time: ISO timestamp when call started (for accurate timestamps)
        
        Returns:
            Complete evaluation report in JSON format
        """
        
        start_time = time.time()
        temp_path = None
        
        # Set default call start time if not provided
        if call_start_time is None:
            call_start_time = datetime.now().isoformat() + "Z"
        
        try:
            print(f"🎵 Starting audio evaluation for: {url}")
            
            # Step 1: Download audio
            print("📥 Downloading audio...")
            temp_path = download_audio(url)
            
            # Step 2: Load and preprocess
            print("🔄 Loading audio...")
            audio_data, sample_rate = load_audio(temp_path)
            audio_duration = len(audio_data) / sample_rate
            
            print(f"✅ Audio loaded: {audio_duration:.2f}s at {sample_rate}Hz")
            
            # Step 3: Determine which evaluators to run
            evaluators_to_run = enabled_evaluators or list(self.evaluators.keys())
            print(f"🔍 Running {len(evaluators_to_run)} evaluators: {evaluators_to_run}")
            
            # Step 4: Run evaluators
            evaluation_results = {}
            evaluator_statuses = {}
            
            for evaluator_name in evaluators_to_run:
                if evaluator_name not in self.evaluators:
                    print(f"⚠️ Unknown evaluator: {evaluator_name}")
                    continue
                
                try:
                    print(f"  Running {evaluator_name}...")
                    evaluator = self.evaluators[evaluator_name]
                    
                    # Run evaluator with call start time for accurate timestamps
                    # Pass audio URL to evaluators that support it (like AI Agent Detector)
                    if evaluator_name == "ai_agent_detection":
                        result = evaluator.evaluate(audio_data, sample_rate, call_start_time, audio_url=url)
                    else:
                        result = evaluator.evaluate(audio_data, sample_rate, call_start_time)
                    
                    # Merge result into main results (each evaluator returns its own key)
                    evaluation_results.update(result)
                    evaluator_statuses[evaluator_name] = "success"
                    
                    print(f"  ✅ {evaluator_name} completed")
                    
                except Exception as e:
                    print(f"  ❌ {evaluator_name} failed: {e}")
                    evaluator_statuses[evaluator_name] = f"error: {str(e)}"
                    
                    # Add error result in expected format
                    evaluation_results[evaluator_name] = {
                        "results": True,  # True indicates there WAS an issue (the error)
                        "message": f"Evaluator failed: {str(e)}",
                        "timestamps": []
                    }
            
            # Step 5: Calculate overall metrics
            processing_time = time.time() - start_time
            
            # Count total issues across all evaluators
            total_issues = 0
            for eval_key, eval_result in evaluation_results.items():
                if isinstance(eval_result, dict) and eval_result.get("results") == True:
                    # Count timestamps as individual issues
                    total_issues += len(eval_result.get("timestamps", []))
            
            # Step 6: Build final response
            response = {
                "success": True,
                "evaluation_id": f"eval_{int(time.time())}",
                "audio_metadata": {
                    "source_url": url,
                    "duration_seconds": round(audio_duration, 2),
                    "sample_rate": sample_rate,
                    "channels": 1,  # We convert to mono
                    "call_start_time": call_start_time
                },
                "evaluation_summary": {
                    "total_evaluators_run": len(evaluators_to_run),
                    "successful_evaluators": len([s for s in evaluator_statuses.values() if s == "success"]),
                    "total_issues_found": total_issues,
                    "processing_time_seconds": round(processing_time, 2)
                },
                "evaluator_status": evaluator_statuses,
                "results": evaluation_results,
                "generated_at": datetime.now().isoformat() + "Z",
                "pipeline_version": self.version
            }
            
            print(f"🎉 Evaluation complete! Found {total_issues} issues in {processing_time:.2f}s")
            return response
            
        except Exception as e:
            error_response = {
                "success": False,
                "error": str(e),
                "error_type": type(e).__name__,
                "audio_metadata": {
                    "source_url": url
                },
                "generated_at": datetime.now().isoformat() + "Z",
                "pipeline_version": self.version
            }
            
            print(f"❌ Pipeline failed: {e}")
            return error_response
            
        finally:
            # Always cleanup temp files
            if temp_path:
                cleanup_temp_file(temp_path)
    
    def get_available_evaluators(self) -> List[Dict[str, str]]:
        """Get list of registered evaluators."""
        return [
            {
                "name": evaluator.name,
                "description": getattr(evaluator, "description", "No description available")
            }
            for evaluator in self.evaluators.values()
        ]
    
    def health_check(self) -> Dict[str, Any]:
        """Pipeline health check."""
        return {
            "status": "healthy",
            "pipeline_version": self.version,
            "registered_evaluators": len(self.evaluators),
            "evaluator_names": list(self.evaluators.keys()),
            "timestamp": datetime.now().isoformat() + "Z"
        }