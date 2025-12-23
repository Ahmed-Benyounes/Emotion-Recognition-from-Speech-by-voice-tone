# Emotion Recognition from Voice Tone

## Overview
This project focuses on designing and implementing a deep learning model to recognize **human emotions from speech signals** regardless of the linguistic context. The goal is to create a robust system that can capture emotional cues purely from vocal tone, making it language-independent and more generalizable.

## Features
- 🎙️ **Speech Emotion Recognition**: Detects emotions such as happiness, sadness, anger, and neutrality from voice input.  
- 🌐 **Language-Independent**: Works on vocal tone rather than linguistic meaning, ensuring cross-lingual applicability.  
- 🎭 **Voice-Over Integration**: Includes pre-recorded voice-over samples for training and testing.  
- 🤖 **Generative AI Experimentation**: Explored integration with Gemini to enable context-aware emotional responses.  

## Technical Details
- **Datasets**:  
  - RAVDESS (Ryerson Audio-Visual Database of Emotional Speech and Song)  
  - Custom voice-over recordings  

- **Modeling**:  
  - Fine-tuned **XLSR / Wav2Vec2.0** for feature extraction  
  - Deep learning classifiers for emotion detection  

- **Tools & Libraries**:  
  - Python, PyTorch, Torchaudio  
  - Hugging Face Transformers  
  - Librosa for signal processing  

## Future Work
- Expand dataset to include more diverse emotional samples.  
- Improve model generalization for real-time applications.  
- Develop an interactive interface that combines emotion recognition with generative models for **emotion-driven dialogue systems**.  

## Author
👤 **Ahmed Ben Youness**  
AI & Operational Research Student | Passionate about Machine Learning, Voice AI, and Creative AI Applications

