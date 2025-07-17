# app/evaluators/ai_agent_detector.py
import numpy as np
import json
import requests
import os
from typing import Dict, List, Any, Tuple
from datetime import datetime, timedelta
from dotenv import load_dotenv
from ..utils import perform_speaker_diarization, get_speaker_audio_segments

# Load environment variables
load_dotenv()

class AIAgentDetector:
    def __init__(self, openai_api_url: str = None, openai_api_key: str = None, min_speakers_for_detection: int = 2):
        self.name = "ai_agent_detection"
        self.description = "Detects if there are AI voice agents in conversations with multiple speakers"
        
        # Load from environment variables with fallback to parameters
        self.openai_api_url = openai_api_url or os.getenv("OPENAI_API_URL", "https://api.openai.com/v1/chat/completions")
        self.openai_api_key = openai_api_key or os.getenv("OPENAI_API_KEY")
        self.min_speakers_for_detection = min_speakers_for_detection or int(os.getenv("MIN_SPEAKERS_FOR_AI_DETECTION", "2"))
        
        if not self.openai_api_key or self.openai_api_key == "your-openai-api-key-here":
            print("⚠️ Warning: OPENAI_API_KEY not configured. LLM analysis will use fallback classification.")
    
    def evaluate(self, audio_data: np.ndarray, sample_rate: int, call_start_time: str = None, audio_url: str = None) -> Dict[str, Any]:
        """
        Detect AI agents in multi-speaker conversations.
        
        Args:
            audio_data: Audio samples as numpy array
            sample_rate: Sample rate of audio
            call_start_time: ISO timestamp when call started
            
        Returns:
            Dictionary with results in expected pipeline format
        """
        
        # Perform speaker diarization to get speaker segments
        print("🎭 Getting speaker segments for AI agent detection...")
        speaker_segments = perform_speaker_diarization(audio_data, sample_rate, audio_url=audio_url)
        
        unique_speakers = list(set(seg['speaker'] for seg in speaker_segments))
        num_speakers = len(unique_speakers)
        
        if num_speakers < self.min_speakers_for_detection:
            return {
                self.name: {
                    "results": False,
                    "message": f"Not enough speakers for AI detection (found {num_speakers}, need {self.min_speakers_for_detection}+)",
                    "timestamps": [],
                    "ai_analysis": {
                        "speakers_detected": num_speakers,
                        "ai_agents_detected": 0,
                        "human_speakers": num_speakers,
                        "detection_performed": False,
                        "speaker_classifications": {}
                    }
                }
            }
        
        print(f"🤖 Analyzing {num_speakers} speakers for AI agent detection...")
        
        # Generate transcripts for each speaker
        speaker_transcripts = self._generate_speaker_transcripts(audio_data, sample_rate, speaker_segments)
        
        # Analyze each speaker with LLM
        speaker_classifications = {}
        ai_agent_timestamps = []
        total_ai_agents = 0
        
        for speaker_id in unique_speakers:
            if speaker_id in speaker_transcripts:
                transcript = speaker_transcripts[speaker_id]
                
                # Skip if transcript is too short
                if len(transcript.strip()) < 20:
                    speaker_classifications[speaker_id] = {
                        "classification": "uncertain",
                        "confidence": 0.0,
                        "reason": "Insufficient speech data for analysis"
                    }
                    continue
                
                # Analyze with LLM
                classification_result = self._analyze_speaker_with_llm(speaker_id, transcript)
                speaker_classifications[speaker_id] = classification_result
                
                if classification_result["classification"] == "ai_agent":
                    total_ai_agents += 1
                    
                    # Create timestamps for AI agent segments
                    speaker_segments_for_agent = [seg for seg in speaker_segments if seg["speaker"] == speaker_id]
                    for segment in speaker_segments_for_agent:
                        timestamp_entry = self._create_ai_agent_timestamp(
                            segment, 
                            classification_result, 
                            call_start_time
                        )
                        ai_agent_timestamps.append(timestamp_entry)
        
        # Determine overall results
        has_ai_agents = total_ai_agents > 0
        human_speakers = num_speakers - total_ai_agents
        
        if has_ai_agents:
            message = f"Detected {total_ai_agents} AI agent(s) among {num_speakers} speakers"
        else:
            message = f"No AI agents detected among {num_speakers} speakers"
        
        return {
            self.name: {
                "results": has_ai_agents,
                "message": message,
                "timestamps": ai_agent_timestamps,
                "ai_analysis": {
                    "speakers_detected": num_speakers,
                    "ai_agents_detected": total_ai_agents,
                    "human_speakers": human_speakers,
                    "detection_performed": True,
                    "speaker_classifications": speaker_classifications
                }
            }
        }
    
    def _generate_speaker_transcripts(self, audio_data: np.ndarray, sample_rate: int, 
                                    speaker_segments: List[Dict[str, Any]]) -> Dict[str, str]:
        """
        Generate transcripts for each speaker using a simple speech-to-text approach.
        This is a placeholder implementation - in practice you'd use a proper ASR service.
        """
        print("📝 Generating speaker transcripts...")
        
        # Group segments by speaker
        speaker_segments_dict = {}
        for segment in speaker_segments:
            speaker_id = segment["speaker"]
            if speaker_id not in speaker_segments_dict:
                speaker_segments_dict[speaker_id] = []
            speaker_segments_dict[speaker_id].append(segment)
        
        # For this implementation, we'll create mock transcripts based on speech patterns
        # In a real implementation, you'd use speech-to-text here
        speaker_transcripts = {}
        
        for speaker_id, segments in speaker_segments_dict.items():
            # Calculate speaking characteristics
            total_duration = sum(seg["duration"] for seg in segments)
            num_segments = len(segments)
            avg_segment_length = total_duration / num_segments if num_segments > 0 else 0
            
            # Create a mock transcript based on speaking patterns
            # This is where you'd integrate real speech-to-text
            mock_transcript = self._generate_mock_transcript(speaker_id, total_duration, avg_segment_length, num_segments)
            speaker_transcripts[speaker_id] = mock_transcript
        
        return speaker_transcripts
    
    def _generate_mock_transcript(self, speaker_id: str, total_duration: float, 
                                avg_segment_length: float, num_segments: int) -> str:
        """
        Generate a mock transcript for demonstration purposes.
        In a real implementation, this would be replaced with actual speech-to-text.
        """
        # Create different speaking patterns to simulate AI vs human speech
        if "00" in speaker_id:  # First speaker - simulate more formal/AI-like speech
            transcript = "Thank you for calling. I'm here to assist you today. How may I help you with your inquiry? I can provide information about our services and help resolve any issues you may have. Please let me know what specific assistance you need."
        elif "01" in speaker_id:  # Second speaker - simulate human speech
            transcript = "Hi, um, I'm having some trouble with my account. I tried to log in but it's not working. Can you help me figure out what's going on? I've been a customer for like three years and this hasn't happened before."
        else:  # Additional speakers - vary the patterns
            if num_segments > 10:  # Frequent speaker
                transcript = "I understand your concern. Let me check that for you right away. I see the issue in our system. This can be resolved by updating your account settings. Would you like me to process that change now?"
            else:  # Less frequent speaker
                transcript = "Okay, yeah that sounds good. Thanks for helping with this. I appreciate you taking the time to look into it."
        
        return transcript
    
    def _analyze_speaker_with_llm(self, speaker_id: str, transcript: str) -> Dict[str, Any]:
        """
        Analyze speaker transcript using LLM to determine if it's an AI agent.
        """
        prompt = f"""You are analyzing a phone conversation transcript to identify if the speaker is an AI voice agent or human. Focus specifically on detecting AI agents programmed for insurance and financial services collection.

**Context**: The AI agent you're looking for is designed to:
- Collect payments or debts for insurance and financial services
- Respond to customer queries about insurance, loans, or financial accounts
- Sound natural but follow scripted patterns typical of automated collection systems

**Key Indicators of AI Collection Agent**:
- Professional, consistent tone without emotional variation
- Uses formal business language and scripted phrases
- References account numbers, payment due dates, or financial obligations
- Offers specific payment options or settlement arrangements
- Mentions consequences of non-payment (late fees, credit impact, etc.)
- Asks verification questions (SSN, account details, address confirmation)
- Uses compliance language ("this call may be recorded", "debt collection notice")
- Responds with predetermined options rather than flexible conversation

**Key Indicators of Human**:
- Natural speech patterns with filler words (um, uh, like)
- Emotional reactions, frustration, or confusion
- Asks clarifying questions in conversational manner
- Personal anecdotes or specific life circumstances
- Interrupts or talks over the other speaker
- Varied tone and pace throughout conversation

**Instructions**:
Analyze the transcript and classify the speaker. Return your response as a JSON object with exactly this format:

{{
  "classification": "ai_agent" | "human" | "uncertain",
  "confidence": 0.0-1.0,
  "reason": "Brief explanation of your classification",
  "key_indicators": ["list", "of", "specific", "phrases", "or", "patterns", "found"]
}}

**Transcript to analyze**:
"{transcript}"

Respond only with the JSON object, no additional text."""

        try:
            # Check if API key is configured
            if not self.openai_api_key or self.openai_api_key == "your-openai-api-key-here":
                print(f"⚠️ API key not configured for {speaker_id}, using fallback classification")
                return self._fallback_classification(transcript)
            
            # Make API call to OpenAI-compatible endpoint
            headers = {
                "Content-Type": "application/json",
            }
            
            # Add authorization header if API key is provided
            if self.openai_api_key:
                headers["Authorization"] = f"Bearer {self.openai_api_key}"
            
            response = requests.post(
                self.openai_api_url,
                headers=headers,
                json={
                    "model": "gpt-3.5-turbo",
                    "messages": [
                        {"role": "user", "content": prompt}
                    ],
                    "temperature": 0.1,
                    "max_tokens": 300
                },
                timeout=30
            )
            
            if response.status_code == 200:
                result = response.json()
                content = result.get("choices", [{}])[0].get("message", {}).get("content", "")
                
                # Parse JSON response
                try:
                    parsed_result = json.loads(content)
                    return parsed_result
                except json.JSONDecodeError:
                    # Fallback if JSON parsing fails
                    return {
                        "classification": "uncertain",
                        "confidence": 0.0,
                        "reason": "Failed to parse LLM response",
                        "key_indicators": []
                    }
            else:
                print(f"⚠️ LLM API error: {response.status_code}")
                return self._fallback_classification(transcript)
                
        except Exception as e:
            print(f"⚠️ LLM analysis failed for {speaker_id}: {e}")
            return self._fallback_classification(transcript)
    
    def _fallback_classification(self, transcript: str) -> Dict[str, Any]:
        """
        Fallback classification based on simple heuristics when LLM is unavailable.
        """
        transcript_lower = transcript.lower()
        
        # Enhanced heuristics for AI collection agent detection
        ai_indicators = [
            "thank you for calling",
            "how may i assist",
            "i can help you with",
            "let me check that for you",
            "i understand your concern",
            "would you like me to",
            "i'm here to help",
            "our records show",
            "for your security",
            "payment due",
            "account balance",
            "make a payment",
            "settlement offer",
            "payment arrangement",
            "verification purposes",
            "social security number",
            "date of birth",
            "this call may be recorded",
            "debt collection",
            "late fees",
            "credit report",
            "financial obligation",
            "insurance premium",
            "policy number",
            "minimum payment",
            "payment options",
            "past due"
        ]
        
        human_indicators = [
            "um", "uh", "like", "you know", "i mean",
            "yeah", "okay", "sure", "i guess",
            "kinda", "sorta", "probably"
        ]
        
        ai_score = sum(1 for indicator in ai_indicators if indicator in transcript_lower)
        human_score = sum(1 for indicator in human_indicators if indicator in transcript_lower)
        
        if ai_score > human_score and ai_score >= 2:
            return {
                "classification": "ai_agent",
                "confidence": min(0.9, 0.5 + (ai_score * 0.08)),
                "reason": "Financial/collection language and scripted patterns detected",
                "key_indicators": [indicator for indicator in ai_indicators if indicator in transcript_lower]
            }
        elif human_score > ai_score and human_score >= 2:
            return {
                "classification": "human",
                "confidence": min(0.8, 0.4 + (human_score * 0.1)),
                "reason": "Natural speech patterns and hesitations detected",
                "key_indicators": [indicator for indicator in human_indicators if indicator in transcript_lower]
            }
        else:
            return {
                "classification": "uncertain",
                "confidence": 0.3,
                "reason": "Insufficient distinctive patterns for classification",
                "key_indicators": []
            }
    
    def _create_ai_agent_timestamp(self, segment: Dict[str, Any], 
                                 classification: Dict[str, Any], 
                                 call_start_time: str) -> Dict[str, Any]:
        """Create timestamp entry for an AI agent detection."""
        
        start_sec = segment["start"]
        end_sec = segment["end"]
        duration = segment["duration"]
        speaker_id = segment["speaker"]
        
        confidence = classification.get("confidence", 0.0)
        reason = classification.get("reason", "AI agent detected")
        
        timestamp_entry = {
            "durationSeconds": round(duration, 2),
            "issue": f"AI agent detected: {speaker_id}",
            "score": round(confidence, 2),
            "speaker_id": speaker_id,
            "classification": "ai_agent",
            "confidence": round(confidence, 2),
            "detection_reason": reason,
            "key_indicators": classification.get("key_indicators", []),
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