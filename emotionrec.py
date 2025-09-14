import os
import numpy as np
import pyaudio
import wave
import torch
import openai
import soundfile as sf
import simpleaudio as sa
from transformers import Wav2Vec2ForSequenceClassification, Wav2Vec2FeatureExtractor


class EmotionDetectionPipeline:
    def __init__(self, model_path, recordings_path, openai_api_key=None):
        # Paths
        self.model_path = model_path
        self.recordings_path = recordings_path

        # Emotions (must match your trained model)
        self.emotions = ['happy', 'sad', 'angry', 'excited', 'fear', 'neutral']
        self.confidence_threshold = 0.2

        # Audio settings
        self.RATE = 16000
        self.RECORD_SECONDS = 4
        self.CHUNK = 4096
        self.FORMAT = pyaudio.paFloat32
        self.CHANNELS = 1

        # Setup
        self._load_model()
        self._setup_audio()

        # Optional GPT
        if openai_api_key:
            openai.api_key = openai_api_key
            self.use_gpt = True
        else:
            self.use_gpt = False

        print("Pipeline initialized.")

    # ---------------- Core Setup ----------------
    def _load_model(self):
        """Load fine-tuned Wav2Vec2 model and feature extractor."""
        checkpoint = os.path.join(self.model_path, "checkpoint-400")
        self.model = Wav2Vec2ForSequenceClassification.from_pretrained(checkpoint)
        self.feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained(self.model_path)
        self.model.eval()
        print("Model loaded.")

    def _setup_audio(self):
        """Prepare audio interface."""
        self.audio = pyaudio.PyAudio()

    # ---------------- Recording ----------------
    def record_audio(self):
        """Record audio from microphone."""
        stream = self.audio.open(
            format=self.FORMAT,
            channels=self.CHANNELS,
            rate=self.RATE,
            input=True,
            frames_per_buffer=self.CHUNK
        )
        print("Recording...")
        frames = []
        for _ in range(int(self.RATE / self.CHUNK * self.RECORD_SECONDS)):
            data = stream.read(self.CHUNK, exception_on_overflow=False)
            frames.append(data)
        stream.stop_stream()
        stream.close()

        audio_data = np.frombuffer(b''.join(frames), dtype=np.float32)
        print("Recording finished.")
        return audio_data

    def save_audio(self, audio_data, filename="output.wav"):
        """Save recorded audio to a .wav file."""
        filepath = os.path.join(self.recordings_path, filename)
        wf = wave.open(filepath, 'wb')
        wf.setnchannels(self.CHANNELS)
        wf.setsampwidth(self.audio.get_sample_size(self.FORMAT))
        wf.setframerate(self.RATE)
        wf.writeframes(audio_data.tobytes())
        wf.close()
        return filepath

    def play_audio(self, filepath):
        """Play an audio file."""
        print(f"Playing: {filepath}")
        wave_obj = sa.WaveObject.from_wave_file(filepath)
        play_obj = wave_obj.play()
        play_obj.wait_done()

    # ---------------- Detection ----------------
    def detect_emotion(self, audio_data):
        """Predict emotion and confidence from audio."""
        inputs = self.feature_extractor(
            audio_data, sampling_rate=self.RATE,
            return_tensors="pt", padding=True
        )
        with torch.no_grad():
            logits = self.model(**inputs).logits
            probs = torch.nn.functional.softmax(logits, dim=-1).cpu().numpy()[0]

        pred_id = int(np.argmax(probs))
        confidence = float(probs[pred_id])
        emotion = self.emotions[pred_id]

        # Print all probabilities
        print("\nEmotion probabilities:")
        for e, p in zip(self.emotions, probs):
            print(f"  {e:8s}: {p:.3f}")

        return emotion, confidence

    def analyze_recording(self):
        """Record, save, detect emotion, and play back."""
        audio_data = self.record_audio()
        filepath = self.save_audio(audio_data)

        emotion, confidence = self.detect_emotion(audio_data)
        if confidence < self.confidence_threshold:
            print("No strong emotion detected.")
        else:
            print(f"Detected emotion: {emotion} (confidence: {confidence:.2f})")

        self.play_audio(filepath)

        if self.use_gpt:
            reply = self.chat_with_gpt(emotion)
            print("\nGPT Response:")
            print(reply)

    # ---------------- Pre-recorded Audio ----------------
    def analyze_file(self, filepath):
        """Analyze a pre-recorded audio file."""
        audio_data, sr = sf.read(filepath)
        if sr != self.RATE:
            raise ValueError(f"Expected {self.RATE} Hz, but got {sr} Hz")

        emotion, confidence = self.detect_emotion(audio_data)
        print(f"\nFile Analysis: {emotion} (confidence: {confidence:.2f})")
        self.play_audio(filepath)

        if self.use_gpt:
            reply = self.chat_with_gpt(emotion)
            print("\nGPT Response:")
            print(reply)

    # ---------------- GPT Integration ----------------
    def chat_with_gpt(self, emotion):
        """Generate a GPT response based on detected emotion."""
        prompt = f"The detected emotion is '{emotion}'. Respond in a natural way."
        response = openai.ChatCompletion.create(
            model="gpt-4o-mini",
            messages=[{"role": "system", "content": prompt}]
        )
        return response['choices'][0]['message']['content']


# ---------------- Example Usage ----------------
if __name__ == "__main__":
    model_path = "path_to_model"
    recordings_path = "path_to_recordings"
    openai_api_key = "***********"  # the key must be updated each time

    pipeline = EmotionDetectionPipeline(model_path, recordings_path, openai_api_key)

    
    pipeline.analyze_recording()

   
