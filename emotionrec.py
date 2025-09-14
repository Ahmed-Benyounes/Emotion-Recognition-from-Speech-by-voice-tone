# Improved Real-Time Emotion Detection Pipeline
# Run this in VSCode locally

import torch
import torchaudio
import librosa
import numpy as np
import pyaudio
import wave
import threading
import time
import random
import json
import os
import pygame
from transformers import Wav2Vec2FeatureExtractor, Wav2Vec2ForSequenceClassification
import openai
from datetime import datetime
from scipy.signal import butter, filtfilt
import warnings
warnings.filterwarnings("ignore")

class ImprovedEmotionDetectionPipeline:
    def __init__(self, model_path, recordings_path, openai_api_key):
        """Initialize the improved emotion detection pipeline"""
        
        # Paths
        self.model_path = model_path
        self.recordings_path = recordings_path
        
        # OpenAI setup
        openai.api_key = openai_api_key
        
        # Audio settings - optimized for better emotion detection
        self.CHUNK = 4096  # Increased for better quality
        self.FORMAT = pyaudio.paFloat32  # Changed to float32 for better precision
        self.CHANNELS = 1
        self.RATE = 16000
        self.RECORD_SECONDS = 4  # Increased to 4 seconds for more context
        
        # Updated emotion settings - MUST match your model's training!
        # Your model was trained on 6 emotions, so we need to use all 6
        self.emotions = ['happy', 'sad', 'angry', 'excited', 'fear', 'neutral']
        
        # Confidence threshold for prediction (lowered for debugging)
        self.confidence_threshold = 0.2
        
        # Initialize components
        self.load_model()
        self.setup_audio()
        
        print("🎤 Improved Emotion Detection Pipeline Ready!")
        print(f"📂 Recordings path: {recordings_path}")
        print(f"🎭 Emotions supported: {', '.join(self.emotions)}")
        print("🤖 Model loaded successfully")
    
    def load_model(self):
        """Load the fine-tuned emotion detection model with better error handling"""
        try:
            # Load emotion mapping
            mapping_path = os.path.join(os.path.dirname(self.model_path), 'emotion_mapping.json')
            
            if os.path.exists(mapping_path):
                with open(mapping_path, 'r') as f:
                    self.emotion_mapping = json.load(f)
                print("✅ Emotion mapping loaded")
            else:
                print("⚠️ No emotion mapping found, using model's built-in mapping")
            
            # Use checkpoint-400 since it's the only one with model weights
            checkpoint_path = os.path.join(self.model_path, 'checkpoint-400')
            
            if os.path.exists(checkpoint_path):
                self.model = Wav2Vec2ForSequenceClassification.from_pretrained(checkpoint_path)
                print(f"✅ Model loaded from: checkpoint-400")
                
                # Get the actual emotion mapping from model config
                if hasattr(self.model.config, 'id2label'):
                    model_emotions = [self.model.config.id2label[i] for i in range(self.model.config.num_labels)]
                    print(f"🎭 Model's emotion order: {model_emotions}")
                    
                    # Update our emotions list to match the model
                    self.emotions = model_emotions
                    print(f"✅ Updated emotions to match model: {self.emotions}")
                
            else:
                raise Exception("checkpoint-400 not found")
            
            # Load feature extractor
            self.feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained(self.model_path)
            
            # Set to evaluation mode
            self.model.eval()
            
            print("✅ Feature extractor loaded")
            print(f"🎯 Final emotion classes: {self.emotions}")
            
        except Exception as e:
            print(f"❌ Error loading model: {e}")
            print("💡 Make sure your model path contains checkpoint-400 with model weights")
            raise
    
    def setup_audio(self):
        """Setup audio recording and playback with better configuration"""
        try:
            self.audio = pyaudio.PyAudio()
            
            # List available audio devices for debugging
            print("\n🎤 Available audio devices:")
            for i in range(self.audio.get_device_count()):
                info = self.audio.get_device_info_by_index(i)
                if info['maxInputChannels'] > 0:
                    print(f"  Device {i}: {info['name']} (Max channels: {info['maxInputChannels']})")
            
            pygame.mixer.init(frequency=22050, size=-16, channels=2, buffer=512)
            print("✅ Audio systems initialized")
            
        except Exception as e:
            print(f"❌ Audio setup error: {e}")
            print("💡 Make sure your microphone is connected and accessible")
            raise
    
    def apply_audio_preprocessing(self, audio_data):
        """Apply audio preprocessing to improve emotion detection"""
        try:
            # 1. Normalize audio
            audio_data = audio_data / (np.max(np.abs(audio_data)) + 1e-8)
            
            # 2. Apply bandpass filter to focus on speech frequencies (80Hz - 8kHz)
            nyquist = self.RATE * 0.5
            low = 80 / nyquist
            high = min(8000 / nyquist, 0.95)  # Ensure high < 1
            
            b, a = butter(4, [low, high], btype='band')
            audio_data = filtfilt(b, a, audio_data)
            
            # 3. Remove silence at the beginning and end
            # Find first and last non-silent samples
            threshold = 0.01 * np.max(np.abs(audio_data))
            non_silent = np.where(np.abs(audio_data) > threshold)[0]
            
            if len(non_silent) > 0:
                start_idx = max(0, non_silent[0] - int(0.1 * self.RATE))  # 0.1s padding
                end_idx = min(len(audio_data), non_silent[-1] + int(0.1 * self.RATE))
                audio_data = audio_data[start_idx:end_idx]
            
            return audio_data
            
        except Exception as e:
            print(f"⚠️ Preprocessing error: {e}, using original audio")
            return audio_data
    
    def record_audio(self):
        """Record audio from microphone with improved quality"""
        try:
            stream = self.audio.open(
                format=self.FORMAT,
                channels=self.CHANNELS,
                rate=self.RATE,
                input=True,
                frames_per_buffer=self.CHUNK,
                input_device_index=None  # Use default device
            )
            
            print("🎤 Recording... Speak now!")
            frames = []
            
            for i in range(0, int(self.RATE / self.CHUNK * self.RECORD_SECONDS)):
                try:
                    data = stream.read(self.CHUNK, exception_on_overflow=False)
                    frames.append(data)
                except Exception as e:
                    print(f"⚠️ Frame read error: {e}")
                    continue
            
            stream.stop_stream()
            stream.close()
            
            # Convert to numpy array
            if self.FORMAT == pyaudio.paFloat32:
                audio_data = np.frombuffer(b''.join(frames), dtype=np.float32)
            else:
                audio_data = np.frombuffer(b''.join(frames), dtype=np.int16).astype(np.float32)
                audio_data = audio_data / 32768.0  # Normalize
            
            # Apply preprocessing
            audio_data = self.apply_audio_preprocessing(audio_data)
            
            # Check if audio was actually recorded
            if np.max(np.abs(audio_data)) < 0.001:
                print("⚠️ Very quiet audio detected. Please speak louder.")
            
            return audio_data
            
        except Exception as e:
            print(f"❌ Recording error: {e}")
            return np.zeros(self.RATE * self.RECORD_SECONDS, dtype=np.float32)
    
    def preprocess_audio_for_model(self, audio_data):
        """Preprocess audio specifically for the model"""
        try:
            # Ensure 4 seconds length (updated for longer context)
            target_length = self.RATE * 4
            
            if len(audio_data) > target_length:
                # Take the middle portion to avoid cutting off speech
                start_idx = (len(audio_data) - target_length) // 2
                audio_data = audio_data[start_idx:start_idx + target_length]
            else:
                # Pad with silence
                padding = target_length - len(audio_data)
                pad_before = padding // 2
                pad_after = padding - pad_before
                audio_data = np.pad(audio_data, (pad_before, pad_after), mode='constant')
            
            return audio_data
            
        except Exception as e:
            print(f"⚠️ Model preprocessing error: {e}")
            target_length = self.RATE * 4
            return np.zeros(target_length, dtype=np.float32)
    
    def detect_emotion(self, audio_data):
        """Detect emotion from audio data with improved accuracy"""
        try:
            # Preprocess audio for model
            audio_data = self.preprocess_audio_for_model(audio_data)
            
            # Get model input
            inputs = self.feature_extractor(
                audio_data, 
                sampling_rate=self.RATE, 
                return_tensors="pt",
                padding=True
            )
            
            # Get prediction
            with torch.no_grad():
                outputs = self.model(**inputs)
                logits = outputs.logits
                predictions = torch.nn.functional.softmax(logits, dim=-1)
                
                # Debug: Print raw logits and predictions
                print(f"🔬 Raw logits: {logits[0].tolist()}")
                print(f"🔬 Softmax probs: {predictions[0].tolist()}")
                
                # Get all predictions for debugging
                all_probs = predictions[0].tolist()
                
                predicted_id = torch.argmax(predictions, dim=-1).item()
                confidence = predictions[0][predicted_id].item()
            
            # Print all emotion predictions
            print(f"🔍 All emotion predictions:")
            for i, (emotion, prob) in enumerate(zip(self.emotions, all_probs)):
                marker = "👉" if i == predicted_id else "  "
                print(f"   {marker} {emotion}: {prob:.3f}")
            
            # Map prediction to emotion (with bounds checking)
            if predicted_id < len(self.emotions):
                emotion = self.emotions[predicted_id]
            else:
                print(f"⚠️ Prediction ID {predicted_id} out of bounds, using neutral")
                emotion = "neutral"
                confidence = 0.5
            
            # Apply confidence threshold with more flexible logic
            if confidence < self.confidence_threshold:
                print(f"⚠️ Low confidence ({confidence:.3f} < {self.confidence_threshold})")
                # Instead of defaulting to neutral, use the highest confidence prediction
                print(f"🎯 Using highest confidence prediction: {emotion}")
            
            return emotion, confidence
            
        except Exception as e:
            print(f"❌ Emotion detection error: {e}")
            import traceback
            traceback.print_exc()
            return "neutral", 0.5
    
    def play_audio_file(self, file_path):
        """Play an audio file with better error handling"""
        try:
            if os.path.exists(file_path):
                pygame.mixer.music.load(file_path)
                pygame.mixer.music.play()
                
                # Wait for audio to finish
                while pygame.mixer.music.get_busy():
                    time.sleep(0.1)
                
                print(f"🔊 Played: {os.path.basename(file_path)}")
                return True
            else:
                print(f"❌ Audio file not found: {file_path}")
                # List available files for debugging
                folder = os.path.dirname(file_path)
                if os.path.exists(folder):
                    files = os.listdir(folder)
                    print(f"   Available files in {folder}: {files}")
                return False
        except Exception as e:
            print(f"❌ Audio playback error: {e}")
            return False
    
    def get_emotional_response(self, emotion):
        """Get random emotional response with fallback"""
        emotion_folder = os.path.join(self.recordings_path, emotion)
        
        if not os.path.exists(emotion_folder):
            print(f"⚠️ Emotion folder not found: {emotion_folder}")
            # Try neutral as fallback
            emotion_folder = os.path.join(self.recordings_path, "neutral")
        
        # Look for available clips
        available_clips = []
        for clip_num in range(1, 4):  # clips 1, 2, 3
            clip_path = os.path.join(emotion_folder, f'clip{clip_num}.wav')
            if os.path.exists(clip_path):
                available_clips.append(clip_path)
        
        if available_clips:
            return random.choice(available_clips)
        else:
            print(f"⚠️ No audio clips found for {emotion}")
            return None
    
    def get_bridge_response(self, emotion):
        """Get bridge audio for emotion with fallback"""
        bridge_path = os.path.join(self.recordings_path, emotion, 'bridge.wav')
        
        if not os.path.exists(bridge_path):
            # Try neutral bridge as fallback
            bridge_path = os.path.join(self.recordings_path, "neutral", 'bridge.wav')
        
        return bridge_path if os.path.exists(bridge_path) else None
    
    def get_bridge_text(self, emotion):
        """Get text content of bridge for ChatGPT context"""
        bridge_texts = {
            'happy': "I will link you with someone to share the joy with you",
            'sad': "Let me connect you with someone who can comfort you", 
            'angry': "Let me find someone to help you work through this",
            'excited': "I'll connect you with someone to share your excitement",
            'fear': "Let me find someone who can help you feel safe",
            'neutral': "I'll connect you with someone to chat with"
        }
        return bridge_texts.get(emotion, "Let me connect you with someone to talk")
    
    def chat_with_gpt(self, emotion, bridge_text, user_context=""):
        """Start conversation with ChatGPT based on detected emotion"""
        try:
            # Create more sophisticated context-aware prompt
            emotion_prompts = {
                'happy': "You are a joyful and enthusiastic companion. The user is feeling happy and excited. Share in their joy and keep the positive energy flowing.",
                'sad': "You are a gentle, empathetic companion. The user is feeling sad and needs comfort. Be understanding, supportive, and offer gentle encouragement.",
                'angry': "You are a calm, understanding companion. The user is feeling angry or frustrated. Help them process their emotions and find constructive ways forward.",
                'excited': "You are an energetic and enthusiastic companion. The user is feeling excited and full of energy. Match their enthusiasm and celebrate with them.",
                'fear': "You are a reassuring and calming companion. The user is feeling scared or anxious. Help them feel safe and provide gentle reassurance.",
                'neutral': "You are a friendly, engaging companion. The user seems neutral or calm. Have a pleasant conversation and be ready to match their energy."
            }
            
            system_prompt = emotion_prompts.get(emotion, emotion_prompts['neutral'])
            system_prompt += f"\n\nContext: {bridge_text}\n\nRespond naturally and conversationally. Keep your response to 1-2 sentences initially."
            
            response = openai.ChatCompletion.create(
                model="gpt-3.5-turbo",
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": f"I just expressed myself and you detected I'm feeling {emotion}. {user_context}".strip()}
                ],
                max_tokens=150,
                temperature=0.7
            )
            
            gpt_response = response.choices[0].message.content.strip()
            print(f"🤖 ChatGPT ({emotion}): {gpt_response}")
            return gpt_response
            
        except Exception as e:
            print(f"❌ ChatGPT error: {e}")
            fallback_responses = {
                'happy': "I can sense your happiness! That's wonderful. What's making you feel so good today?",
                'sad': "I can hear that you're going through something difficult. I'm here to listen and support you.",
                'angry': "I understand you're feeling frustrated. Sometimes it helps to talk through what's bothering you.",
                'excited': "I can feel your excitement! That energy is contagious. Tell me what's got you so pumped up!",
                'fear': "I can sense you're feeling uneasy. That's completely natural. I'm here to help you feel more secure.",
                'neutral': "I'm here and ready to chat with you. What's on your mind today?"
            }
            return fallback_responses.get(emotion, "I'm here to listen and chat with you.")
    
    def run_detection_cycle(self):
        """Run one complete detection cycle with better feedback"""
        print("\n" + "="*60)
        print(f"🎯 Emotion Detection Cycle - {datetime.now().strftime('%H:%M:%S')}")
        print("="*60)
        
        # Step 1: Record audio
        print("Step 1: Recording audio...")
        audio_data = self.record_audio()
        
        # Step 2: Detect emotion
        print("Step 2: Analyzing emotion...")
        emotion, confidence = self.detect_emotion(audio_data)
        print(f"😊 Detected Emotion: {emotion.upper()} (confidence: {confidence:.3f})")
        
        # Step 3: Play emotional response
        print("Step 3: Playing emotional response...")
        emotional_response_path = self.get_emotional_response(emotion)
        if emotional_response_path:
            self.play_audio_file(emotional_response_path)
        else:
            print("⚠️ No emotional response audio available")
        
        # Step 4: Play bridge audio
        print("Step 4: Playing bridge audio...")
        bridge_path = self.get_bridge_response(emotion)
        if bridge_path:
            self.play_audio_file(bridge_path)
        else:
            print("⚠️ No bridge audio available")
        
        # Step 5: Start ChatGPT conversation
        print("Step 5: Starting ChatGPT conversation...")
        bridge_text = self.get_bridge_text(emotion)
        gpt_response = self.chat_with_gpt(emotion, bridge_text)
        
        return emotion, confidence, gpt_response
    
    def test_my_recordings(self):
        """Test emotion detection on pre-recorded clips with detailed analysis"""
        print("\n🧪 Testing with your pre-recorded clips...")
        print("="*60)
        
        total_tests = 0
        correct_predictions = 0
        
        for emotion in self.emotions:
            emotion_folder = os.path.join(self.recordings_path, emotion)
            if os.path.exists(emotion_folder):
                print(f"\n--- Testing {emotion.upper()} ---")
                
                emotion_correct = 0
                emotion_total = 0
                
                # Test each clip
                for clip_num in range(1, 4):  # clips 1, 2, 3
                    clip_path = os.path.join(emotion_folder, f'clip{clip_num}.wav')
                    if os.path.exists(clip_path):
                        try:
                            # Load and test the clip
                            audio_data, _ = librosa.load(clip_path, sr=self.RATE)
                            detected_emotion, confidence = self.detect_emotion(audio_data)
                            
                            # Show result
                            correct = detected_emotion == emotion
                            if correct:
                                emotion_correct += 1
                                correct_predictions += 1
                            
                            emotion_total += 1
                            total_tests += 1
                            
                            status = "✅" if correct else "❌"
                            print(f"  clip{clip_num}.wav: {detected_emotion} ({confidence:.3f}) {status}")
                            
                        except Exception as e:
                            print(f"  clip{clip_num}.wav: Error loading - {e}")
                    else:
                        print(f"  clip{clip_num}.wav: Not found")
                
                if emotion_total > 0:
                    accuracy = (emotion_correct / emotion_total) * 100
                    print(f"  📊 {emotion} accuracy: {emotion_correct}/{emotion_total} ({accuracy:.1f}%)")
        
        if total_tests > 0:
            overall_accuracy = (correct_predictions / total_tests) * 100
            print(f"\n🎯 Overall Accuracy: {correct_predictions}/{total_tests} ({overall_accuracy:.1f}%)")
            
            if overall_accuracy < 70:
                print("💡 Suggestions to improve accuracy:")
                print("   • Record more diverse samples for each emotion")
                print("   • Ensure clear, expressive speech in recordings")
                print("   • Check if model was trained on similar data")
                print("   • Consider re-training with more data")
        
        print("\n" + "="*60)
    
    def start_interactive_mode(self):
        """Start interactive emotion detection with improved interface"""
        print("\n🚀 Starting Interactive Emotion Detection!")
        print("="*60)
        print("Commands:")
        print("  Enter - Record audio and detect emotion")
        print("  'test' - Test with your pre-recorded clips")
        print("  'info' - Show model information")
        print("  'quit' - Exit")
        print("="*60)
        
        while True:
            try:
                user_input = input("\n⏯️  Command (Enter/test/info/quit): ").strip().lower()
                
                if user_input == 'quit':
                    print("👋 Goodbye!")
                    break
                    
                elif user_input == 'test':
                    self.test_my_recordings()
                    continue
                    
                elif user_input == 'info':
                    self.show_model_info()
                    continue
                
                # Default: run detection cycle
                emotion, confidence, gpt_response = self.run_detection_cycle()
                
                # Continue conversation option
                print(f"\n💬 Continue chatting about {emotion}?")
                continue_chat = input("Your response (or Enter to record again): ").strip()
                
                if continue_chat:
                    # Continue ChatGPT conversation
                    follow_up = self.chat_with_gpt(emotion, "", continue_chat)
                    
            except KeyboardInterrupt:
                print("\n👋 Interrupted by user")
                break
            except Exception as e:
                print(f"❌ Error in detection cycle: {e}")
                print("💡 Try again or type 'quit' to exit")
    
    def show_model_info(self):
        """Show information about the loaded model"""
        print("\n📊 Model Information")
        print("="*40)
        print(f"Model path: {self.model_path}")
        print(f"Supported emotions: {', '.join(self.emotions)}")
        print(f"Sample rate: {self.RATE} Hz")
        print(f"Recording duration: {self.RECORD_SECONDS} seconds")
        print(f"Confidence threshold: {self.confidence_threshold}")
        
        if hasattr(self.model.config, 'num_labels'):
            print(f"Model output classes: {self.model.config.num_labels}")
            
            # Check if there's a mismatch
            if self.model.config.num_labels != len(self.emotions):
                print(f"⚠️  MISMATCH: Model expects {self.model.config.num_labels} classes but you have {len(self.emotions)} emotions!")
                print("💡 This could explain the low confidence scores")
        
        # Check if emotion mapping exists and matches
        if hasattr(self, 'emotion_mapping'):
            print(f"Emotion mapping: {self.emotion_mapping}")
        
        # Check recordings folder structure
        print(f"\n📁 Recordings folder structure:")
        for emotion in self.emotions:
            emotion_folder = os.path.join(self.recordings_path, emotion)
            if os.path.exists(emotion_folder):
                files = [f for f in os.listdir(emotion_folder) if f.endswith('.wav')]
                print(f"  {emotion}: {len(files)} files")
            else:
                print(f"  {emotion}: ❌ Missing")
    
    def cleanup(self):
        """Clean up resources"""
        try:
            if hasattr(self, 'audio'):
                self.audio.terminate()
            pygame.mixer.quit()
            print("🧹 Cleanup completed")
        except Exception as e:
            print(f"⚠️ Cleanup warning: {e}")

# Main execution
if __name__ == "__main__":
    # Configuration - UPDATE THESE PATHS
    MODEL_PATH = "C:/Users/bnahm/Desktop/data/emotion_xlsr_model"
    RECORDINGS_PATH = "C:/Users/bnahm/Desktop/data/my_recording"   
    OPENAI_API_KEY = "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"
    
    # Validate paths
    if not os.path.exists(MODEL_PATH):
        print(f"❌ Model path not found: {MODEL_PATH}")
        print("Please update MODEL_PATH in the script")
        exit(1)
    
    if not os.path.exists(RECORDINGS_PATH):
        print(f"❌ Recordings path not found: {RECORDINGS_PATH}")
        print("Please update RECORDINGS_PATH in the script")
        exit(1)
    
    try:
        # Initialize improved pipeline
        pipeline = ImprovedEmotionDetectionPipeline(MODEL_PATH, RECORDINGS_PATH, OPENAI_API_KEY)
        
        # Start interactive mode
        pipeline.start_interactive_mode()
        
    except KeyboardInterrupt:
        print("\n👋 Goodbye!")
    except Exception as e:
        print(f"❌ Fatal error: {e}")
        import traceback
        traceback.print_exc()
    finally:
        if 'pipeline' in locals():
            pipeline.cleanup()