import streamlit as st
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import librosa
import numpy as np
import soundfile as sf
import random
from transformers import Wav2Vec2Config
import plotly.graph_objects as go
from audio_recorder_streamlit import audio_recorder
import time
import io
import google.generativeai as genai
import speech_recognition as sr
import base64
from PIL import Image

# ----------------------------
# Page Configuration
# ----------------------------
st.set_page_config(
    page_title="Emotion Voice AI | Deep Learning Project",
    page_icon="🎤",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Enhanced Custom CSS
st.markdown("""
<style>
    /* Main gradient background */
    .stApp {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    }
    
    /* Header styling */
    .main-header {
        font-size: 3.5rem;
        font-weight: 900;
        text-align: center;
        color: white;
        text-shadow: 2px 2px 4px rgba(0,0,0,0.3);
        margin-bottom: 0.5rem;
        animation: fadeIn 1s;
    }
    
    .sub-header {
        text-align: center;
        color: rgba(255,255,255,0.9);
        font-size: 1.3rem;
        margin-bottom: 2rem;
        font-weight: 300;
    }
    
    /* Logo container */
    .logo-container {
        text-align: center;
        margin-bottom: 2rem;
    }
    
    /* Emotion result cards */
    .emotion-winner {
        font-size: 2.8rem;
        font-weight: bold;
        text-align: center;
        padding: 1.5rem;
        border-radius: 20px;
        margin: 1.5rem 0;
        box-shadow: 0 10px 30px rgba(0,0,0,0.3);
        animation: scaleIn 0.5s;
    }
    
    @keyframes scaleIn {
        from { transform: scale(0.8); opacity: 0; }
        to { transform: scale(1); opacity: 1; }
    }
    
    .angry-bg { 
        background: linear-gradient(135deg, #eb3349 0%, #f45c43 100%); 
        color: white; 
    }
    .happy-bg { 
        background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%); 
        color: white; 
    }
    .sad-bg { 
        background: linear-gradient(135deg, #4facfe 0%, #00f2fe 100%); 
        color: white; 
    }
    .neutral-bg { 
        background: linear-gradient(135deg, #43e97b 0%, #38f9d7 100%); 
        color: white; 
    }
    
    /* Chat messages */
    .chat-message {
        padding: 1.2rem;
        border-radius: 15px;
        margin: 0.8rem 0;
        animation: slideIn 0.5s;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
    }
    
    @keyframes slideIn {
        from { opacity: 0; transform: translateX(-20px); }
        to { opacity: 1; transform: translateX(0); }
    }
    
    .user-message {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        margin-left: 15%;
        border-bottom-right-radius: 5px;
    }
    
    .bot-message {
        background: white;
        color: #333;
        margin-right: 15%;
        border-bottom-left-radius: 5px;
    }
    
    /* Info boxes */
    .info-box {
        background: rgba(255,255,255,0.95);
        padding: 1.5rem;
        border-radius: 15px;
        border-left: 5px solid #667eea;
        margin: 1rem 0;
        box-shadow: 0 4px 15px rgba(0,0,0,0.1);
    }
    
    /* Buttons */
    .stButton>button {
        width: 100%;
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        color: white;
        font-size: 1.3rem;
        font-weight: bold;
        padding: 1rem;
        border: none;
        border-radius: 15px;
        box-shadow: 0 5px 15px rgba(102, 126, 234, 0.4);
        transition: all 0.3s;
    }
    
    .stButton>button:hover {
        transform: translateY(-2px);
        box-shadow: 0 7px 20px rgba(102, 126, 234, 0.6);
    }
    
    /* Sidebar styling */
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #667eea 0%, #764ba2 100%);
    }
    
    [data-testid="stSidebar"] * {
        color: white !important;
    }
    
    /* Metrics */
    [data-testid="stMetricValue"] {
        font-size: 2rem;
        font-weight: bold;
        color: white;
    }
    
    /* Tabs */
    .stTabs [data-baseweb="tab-list"] {
        gap: 24px;
        background: rgba(255,255,255,0.1);
        border-radius: 15px;
        padding: 0.5rem;
    }
    
    .stTabs [data-baseweb="tab"] {
        background: transparent;
        color: white;
        font-weight: 600;
        font-size: 1.1rem;
        border-radius: 10px;
        padding: 0.8rem 1.5rem;
    }
    
    .stTabs [aria-selected="true"] {
        background: rgba(255,255,255,0.3);
    }
    
    @keyframes fadeIn {
        from { opacity: 0; }
        to { opacity: 1; }
    }
</style>
""", unsafe_allow_html=True)

# ----------------------------
# Configuration (FIXED ANGRY BIAS)
# ----------------------------
EMOTION_LABELS = {'angry': 0, 'happy': 1, 'sad': 2, 'neutral': 3}
LABEL_TO_EMOTION = {v: k for k, v in EMOTION_LABELS.items()}
EMOTION_EMOJIS = {'angry': '😠', 'happy': '😊', 'sad': '😢', 'neutral': '😐'}
EMOTION_COLORS = {
    'angry': '#eb3349',
    'happy': '#f093fb',
    'sad': '#4facfe',
    'neutral': '#43e97b'
}
SAMPLE_RATE = 16000
MAX_LENGTH = 48000
SILENCE_THRESHOLD = 0.015
MIN_CONFIDENCE = 0.25
TEMPERATURE = 2.5  # ⭐ INCREASED to reduce overconfidence
ANGRY_PENALTY = 0.65  # ⭐ STRONG penalty on angry (35% reduction)
SAD_BOOST = 1.12  # ⭐ Boost sad slightly
NEUTRAL_BOOST = 1.08  # ⭐ Boost neutral slightly
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

BASE_DIR = r"C:\Users\bnahm\Desktop\emotion_voice_ai"
MODEL_PATH = os.path.join(BASE_DIR, "models", "best_model_600.pth")
OUTPUT_VOICES_DIR = os.path.join(BASE_DIR, "output_voices")
LOGO_PATH = os.path.join(BASE_DIR, "logo.png")

# ----------------------------
# Model Definition
# ----------------------------
class EmotionClassifier(nn.Module):
    def __init__(self, num_classes=4, unfreeze_layers=12):
        super().__init__()
        from transformers import Wav2Vec2Model
        config = Wav2Vec2Config.from_pretrained("facebook/wav2vec2-large-xlsr-53")
        self.wav2vec2 = Wav2Vec2Model(config)
        
        for param in self.wav2vec2.feature_extractor.parameters():
            param.requires_grad = False
        
        total_layers = len(self.wav2vec2.encoder.layers)
        freeze_until = total_layers - unfreeze_layers
        for i, layer in enumerate(self.wav2vec2.encoder.layers):
            for param in layer.parameters():
                param.requires_grad = (i >= freeze_until)
        
        self.classifier = nn.Sequential(
            nn.Linear(1024, 768),
            nn.BatchNorm1d(768),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(768, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(0.45),
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.35),
            nn.Linear(256, num_classes)
        )
    
    def forward(self, x):
        features = self.wav2vec2(x).last_hidden_state
        pooled = features.mean(dim=1)
        return self.classifier(pooled)

@st.cache_resource
def load_model():
    model = EmotionClassifier(unfreeze_layers=12).to(DEVICE)
    checkpoint = torch.load(MODEL_PATH, map_location=DEVICE, weights_only=False)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    return model, checkpoint

# ----------------------------
# Audio Processing (FIXED BIAS)
# ----------------------------
def is_silence(audio, threshold=SILENCE_THRESHOLD):
    energy = np.sqrt(np.mean(audio**2))
    max_amplitude = np.max(np.abs(audio))
    return energy < threshold or max_amplitude < threshold

def preprocess_audio(audio, target_sr=SAMPLE_RATE, max_len=MAX_LENGTH):
    audio, _ = librosa.effects.trim(audio, top_db=20)
    if len(audio) > max_len:
        audio = audio[:max_len]
    else:
        padding = max_len - len(audio)
        audio = np.pad(audio, (0, padding), mode='constant')
    audio = audio / (np.max(np.abs(audio)) + 1e-6)
    return torch.tensor(audio, dtype=torch.float32).unsqueeze(0)

def apply_test_time_augmentation(audio, model, num_augments=3, temperature=TEMPERATURE):
    """FIXED: Apply calibration to reduce angry bias"""
    predictions = []
    input_tensor = preprocess_audio(audio).to(DEVICE)
    with torch.no_grad():
        logits = model(input_tensor)
        predictions.append(F.softmax(logits / temperature, dim=1))
    
    for _ in range(num_augments - 1):
        noisy_audio = audio + np.random.randn(len(audio)) * 0.002
        input_tensor = preprocess_audio(noisy_audio).to(DEVICE)
        with torch.no_grad():
            logits = model(input_tensor)
            predictions.append(F.softmax(logits / temperature, dim=1))
    
    avg_probs = torch.stack(predictions).mean(dim=0)
    avg_probs = avg_probs.cpu().numpy()[0]
    
    # ⭐ CRITICAL CALIBRATION
    avg_probs[0] *= ANGRY_PENALTY  # Reduce angry
    avg_probs[2] *= SAD_BOOST      # Boost sad
    avg_probs[3] *= NEUTRAL_BOOST  # Boost neutral
    
    # Re-normalize
    avg_probs = avg_probs / avg_probs.sum()
    return avg_probs

def predict_emotion(audio_data, model):
    if is_silence(audio_data):
        return None, None
    probs = apply_test_time_augmentation(audio_data, model)
    pred_idx = np.argmax(probs)
    emotion = LABEL_TO_EMOTION[pred_idx]
    confidence = probs[pred_idx]
    return emotion, probs

def get_random_response_audio(emotion):
    try:
        emotion_dir = os.path.join(OUTPUT_VOICES_DIR, emotion)
        if not os.path.isdir(emotion_dir):
            return None, f"⚠️ Folder not found: {emotion_dir}"
        wav_files = [f for f in os.listdir(emotion_dir) if f.endswith('.wav')]
        if not wav_files:
            return None, f"⚠️ No .wav files in {emotion_dir}"
        selected_file = random.choice(wav_files)
        return os.path.join(emotion_dir, selected_file), selected_file
    except Exception as e:
        return None, f"Error: {str(e)}"

def transcribe_audio(audio_data, sample_rate, language='en-US'):
    try:
        recognizer = sr.Recognizer()
        audio_io = io.BytesIO()
        sf.write(audio_io, audio_data, sample_rate, format='WAV')
        audio_io.seek(0)
        with sr.AudioFile(audio_io) as source:
            audio = recognizer.record(source)
            text = recognizer.recognize_google(audio, language=language) if language != 'auto' else recognizer.recognize_google(audio)
            return text
    except sr.UnknownValueError:
        return "[Could not understand - speak more clearly]"
    except sr.RequestError:
        return "[Service unavailable]"
    except Exception as e:
        return f"[Error: {str(e)}]"

def get_emotion_aware_response(user_text, detected_emotion, confidence, api_key, language="english"):
    if not api_key:
        return "⚠️ Please add your Gemini API key in the sidebar."
    
    contexts = {
        'english': {'angry': "Be empathetic and calming.", 'happy': "Match their enthusiasm.", 'sad': "Be compassionate.", 'neutral': "Be helpful."},
        'arabic': {'angry': "كن متعاطفًا ومهدئًا.", 'happy': "اجعل ردك متحمسًا.", 'sad': "كن متعاطفًا وداعمًا.", 'neutral': "كن مفيدًا."},
        'french': {'angry': "Soyez empathique.", 'happy': "Soyez enthousiaste.", 'sad': "Soyez compatissant.", 'neutral': "Soyez utile."},
        'japanese': {'angry': "共感的に対応してください。", 'happy': "熱意を持って対応してください。", 'sad': "思いやりを持って対応してください。", 'neutral': "役立つ情報を提供してください。"}
    }
    
    lang_instr = {'english': "Respond in English.", 'arabic': "الرد بالعربية.", 'french': "Répondez en français.", 'japanese': "日本語で返信。"}
    
    prompt = f"""Empathetic AI assistant.
Emotion: {detected_emotion.upper()} ({confidence*100:.1f}%)
Context: {contexts.get(language, contexts['english'])[detected_emotion]}
User: "{user_text}"
{lang_instr.get(language, lang_instr['english'])}
Keep it 2-3 sentences."""
    
    try:
        genai.configure(api_key=api_key)
        model = genai.GenerativeModel('models/gemini-2.5-flash')
        return model.generate_content(prompt).text
    except Exception as e:
        return f"Error: {str(e)}"

def create_probability_chart(probs):
    emotions = ['Angry', 'Happy', 'Sad', 'Neutral']
    percentages = [p * 100 for p in probs]
    colors = [EMOTION_COLORS[e.lower()] for e in emotions]
    sorted_data = sorted(zip(emotions, percentages, colors), key=lambda x: x[1], reverse=True)
    emotions_sorted, percentages_sorted, colors_sorted = zip(*sorted_data)
    
    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=percentages_sorted, y=emotions_sorted, orientation='h',
        marker=dict(color=colors_sorted, line=dict(color='white', width=3)),
        text=[f'{p:.1f}%' for p in percentages_sorted],
        textposition='outside',
        textfont=dict(size=16, color='white', family='Arial Black')
    ))
    
    # ✅ FIX: Updated to use title_font instead of titlefont
    fig.update_layout(
        title=dict(text='🎭 Emotion Probabilities', font=dict(size=22, color='white', family='Arial Black'), x=0.5),
        xaxis=dict(
            title='Probability (%)',
            gridcolor='rgba(255,255,255,0.2)', 
            range=[0, 105],
            title_font=dict(color='white'),  # ✅ CORRECTED HERE
            tickfont=dict(color='white')     # Optional but recommended
        ),
        yaxis=dict(
            tickfont=dict(size=15, color='white', family='Arial Black')
        ),
        height=350, 
        plot_bgcolor='rgba(0,0,0,0)', 
        paper_bgcolor='rgba(0,0,0,0)',
        margin=dict(l=20, r=100, t=80, b=40), 
        showlegend=False
    )
    return fig

# ----------------------------
# Main App
# ----------------------------
def main():
    # Initialize session state
    for key in ['chat_history', 'prediction_history', 'api_key', 'selected_language', 'enable_tts']:
        if key not in st.session_state:
            st.session_state[key] = [] if 'history' in key else ("" if key == 'api_key' else ('english' if key == 'selected_language' else False))
    
    # Logo and Header
    col_logo1, col_logo2, col_logo3 = st.columns([1, 2, 1])
    with col_logo2:
        if os.path.exists(LOGO_PATH):
            logo = Image.open(LOGO_PATH)
            st.image(logo, use_container_width=True)
        else:
            st.markdown('<div class="logo-container">🎤</div>', unsafe_allow_html=True)
    
    st.markdown('<h1 class="main-header">Emotion Voice AI</h1>', unsafe_allow_html=True)
    st.markdown('<p class="sub-header">🧠 Deep Learning | 🗣️ Multilingual Speech Recognition | 🤖 AI-Powered Chatbot</p>', unsafe_allow_html=True)
    
    # Sidebar
    with st.sidebar:
        st.markdown("## 🔑 API Key")
        api_key = st.text_input("Gemini API Key", value=st.session_state.api_key, type="password", help="Get at: aistudio.google.com")
        if api_key != st.session_state.api_key:
            st.session_state.api_key = api_key
            if api_key:
                try:
                    genai.configure(api_key=api_key)
                    st.success("✅ Configured!")
                except Exception as e:
                    st.error(f"❌ Invalid: {e}")
        
        st.markdown("---")
        st.markdown("## 🌍 Language")
        lang_opts = {'English': 'english', 'العربية': 'arabic', 'Français': 'french', '日本語': 'japanese'}
        speech_codes = {'english': 'en-US', 'arabic': 'ar-SA', 'french': 'fr-FR', 'japanese': 'ja-JP'}
        sel_lang = st.selectbox("Select Language", list(lang_opts.keys()), index=0)
        st.session_state.selected_language = lang_opts[sel_lang]
        st.session_state.speech_lang_code = speech_codes[st.session_state.selected_language]
        
        st.session_state.enable_tts = st.checkbox("🔊 Text-to-Speech", value=st.session_state.enable_tts)
        
        st.markdown("---")
        st.markdown("## 📊 Model Stats")
        try:
            _, checkpoint = load_model()
            st.metric("Accuracy", f"{checkpoint.get('val_acc', 0)*100:.1f}%")
            st.metric("Device", "🚀 GPU" if DEVICE.type == "cuda" else "💻 CPU")
        except:
            pass
        
        st.markdown("---")
        st.info("""
        **Features:**
        • Emotion Detection (4 classes)
        • Multilingual Support
        • Voice Responses
        • AI Chatbot
        """)
    
    # Main Tabs
    tab1, tab2 = st.tabs(["🎙️ Voice Chat", "📊 History"])
    
    with tab1:
        col1, col2 = st.columns([1, 1])
        
        with col1:
            st.markdown("### 🎤 Record")
            st.markdown("""
            <div class="info-box">
                <b>📝 Instructions:</b><br>
                1️⃣ Click microphone to record<br>
                2️⃣ Speak clearly (3 seconds)<br>
                3️⃣ AI analyzes emotion<br>
                4️⃣ Get personalized response
            </div>
            """, unsafe_allow_html=True)
            
            audio_bytes = audio_recorder(text="🎤 Record", recording_color="#667eea", neutral_color="#6e7079", icon_name="microphone", icon_size="3x")
            
            if audio_bytes:
                st.audio(audio_bytes, format="audio/wav")
                
                if st.button("🧠 Analyze Emotion", type="primary"):
                    try:
                        model, _ = load_model()
                        with st.spinner("🔮 Analyzing..."):
                            audio_data, sr = sf.read(io.BytesIO(audio_bytes))
                            if len(audio_data.shape) > 1: audio_data = audio_data[:, 0]
                            if sr != SAMPLE_RATE: audio_data = librosa.resample(audio_data, orig_sr=sr, target_sr=SAMPLE_RATE)
                            
                            emotion, probs = predict_emotion(audio_data, model)
                            
                            if emotion is None:
                                st.warning("🔇 Silence detected!")
                                silence_dir = os.path.join(OUTPUT_VOICES_DIR, "silence")
                                if os.path.isdir(silence_dir):
                                    files = [f for f in os.listdir(silence_dir) if f.endswith('.wav')]
                                    if files:
                                        path = os.path.join(silence_dir, random.choice(files))
                                        with open(path, 'rb') as f:
                                            sb = f.read()
                                        st.markdown(f'<audio autoplay><source src="data:audio/wav;base64,{base64.b64encode(sb).decode()}"></audio>', unsafe_allow_html=True)
                                        st.audio(sb, format='audio/wav')
                                return
                            
                            confidence = probs[EMOTION_LABELS[emotion]]
                            user_text = transcribe_audio(audio_data, SAMPLE_RATE, st.session_state.speech_lang_code)
                            
                            st.session_state.prediction_history.append({
                                'emotion': emotion, 'confidence': confidence,
                                'text': user_text, 'timestamp': time.strftime("%H:%M:%S")
                            })
                            
                            emoji = EMOTION_EMOJIS[emotion]
                            st.markdown(f'<div class="emotion-winner {emotion}-bg">{emoji} {emotion.upper()}<br><span style="font-size:1.5rem">{confidence*100:.1f}%</span></div>', unsafe_allow_html=True)
                            
                            st.plotly_chart(create_probability_chart(probs), use_container_width=True)
                            st.markdown(f"**📝 You said:** {user_text}")
                            
                            st.markdown("---")
                            st.markdown("### 🔊 Voice Response")
                            path, name = get_random_response_audio(emotion)
                            if path:
                                st.success(f"🎵 {name}")
                                with open(path, 'rb') as f: ab = f.read()
                                st.markdown(f'<audio autoplay><source src="data:audio/wav;base64,{base64.b64encode(ab).decode()}"></audio>', unsafe_allow_html=True)
                                st.audio(ab, format='audio/wav')
                            
                            bot_response = get_emotion_aware_response(user_text, emotion, confidence, st.session_state.api_key, st.session_state.selected_language)
                            
                            if st.session_state.enable_tts and bot_response and not bot_response.startswith("⚠"):
                                st.markdown("### 🗣️ AI Voice")
                                try:
                                    from gtts import gTTS
                                    import tempfile
                                    tts_lang_map = {'english': 'en', 'arabic': 'ar', 'french': 'fr', 'japanese': 'ja'}
                                    tts_lang = tts_lang_map.get(st.session_state.selected_language, 'en')
                                    tts = gTTS(text=bot_response, lang=tts_lang, slow=False)
                                    with tempfile.NamedTemporaryFile(delete=False, suffix='.mp3') as fp:
                                        tts.save(fp.name)
                                        with open(fp.name, 'rb') as f: tb = f.read()
                                        st.markdown(f'<audio autoplay><source src="data:audio/mp3;base64,{base64.b64encode(tb).decode()}"></audio>', unsafe_allow_html=True)
                                        st.audio(tb, format='audio/mp3')
                                        try: os.unlink(fp.name)
                                        except: pass
                                except Exception as e:
                                    st.warning(f"TTS error: {e}")
                            
                            st.session_state.chat_history.append({'user': user_text, 'emotion': emotion, 'bot': bot_response, 'timestamp': time.strftime("%H:%M:%S")})
                    except Exception as e:
                        st.error(f"Error: {e}")
        
        with col2:
            st.markdown("### 💬 Conversation")
            if st.session_state.chat_history:
                for chat in st.session_state.chat_history[-5:]:
                    e = EMOTION_EMOJIS[chat['emotion']]
                    st.markdown(f'<div class="chat-message user-message"><small>{chat["timestamp"]}</small><br><b>{e} You ({chat["emotion"]}):</b><br>{chat["user"]}</div>', unsafe_allow_html=True)
                    st.markdown(f'<div class="chat-message bot-message"><b>🤖 AI:</b><br>{chat["bot"]}</div>', unsafe_allow_html=True)
                if st.button("🗑️ Clear"): st.session_state.chat_history = []; st.rerun()
            else:
                st.info("💡 Record to start!")
    
    with tab2:
        st.markdown("### 📊 History")
        if st.session_state.prediction_history:
            for p in reversed(st.session_state.prediction_history[-10:]):
                with st.expander(f"{p['timestamp']} - {EMOTION_EMOJIS[p['emotion']]} {p['emotion'].upper()} ({p['confidence']*100:.1f}%)"):
                    st.write(f"**Text:** {p['text']}")
            if st.button("🗑️ Clear"): st.session_state.prediction_history = []; st.rerun()
        else:
            st.info("No history yet!")

if __name__ == "__main__":
    main()
