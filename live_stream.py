"""
Real-Time Continuous Live Microphone Stream for Wake Word Engine
===============================================================
- Enrolls custom wake phrase with 3 quick voice recordings (saves to profile.json)
- Runs a non-stop continuous live mic stream
- Displays live probability meters and instant trigger notifications in the terminal
"""

import os
import sys
import time
import json
import numpy as np
import torch
import torchaudio.transforms as T

try:
    import sounddevice as sd
except ImportError:
    print("\n⚠️  'sounddevice' is required for live mic feed. Install with:")
    print("   pip install sounddevice soundfile\n")
    sys.exit(1)

try:
    import onnxruntime as ort
except ImportError:
    print("\n⚠️  'onnxruntime' is required. Install with:")
    print("   pip install onnxruntime\n")

from config import SR, N_MELS, N_FFT, HOP_LENGTH, EMBED_DIM, STAGE2_GLOBAL_THRESHOLD, SUFFIX_REJECTION_THRESHOLD
from audio_utils import endpoint_utterance, time_stretch_audio, generate_phonetic_minimal_pairs
from inference import WakeWordCascade

PROFILE_PATH = "wakeword_profile.json"
DEFAULT_MODEL_PATH = "wakeword_student.onnx"

class LiveWakeWordEngine:
    def __init__(self, model_path=DEFAULT_MODEL_PATH, phrase="Hey Karthika"):
        self.model_path = model_path
        self.phrase = phrase
        self.sr = SR
        self.cascade = WakeWordCascade(
            stage1_path="", 
            stage2_path=model_path if os.path.exists(model_path) else "",
            threshold1=0.55, 
            threshold2=STAGE2_GLOBAL_THRESHOLD, 
            suffix_threshold=SUFFIX_REJECTION_THRESHOLD
        )
        self.mel_transform = T.MelSpectrogram(
            sample_rate=SR, n_fft=N_FFT, hop_length=HOP_LENGTH, n_mels=N_MELS, power=2.0
        )
        
    def record_clip(self, duration_sec=1.5, prompt="Say wake word"):
        print(f"\n🎤 {prompt}... (Listening for {duration_sec}s)")
        audio = sd.rec(int(duration_sec * self.sr), samplerate=self.sr, channels=1, dtype='float32')
        sd.wait()
        wav = audio.squeeze()
        print("  ✓ Audio captured.")
        return wav

    def enroll_interactive(self, num_clips=3):
        """Interactively records 3 user voice samples to establish the acoustic profile."""
        print(f"\n=======================================================")
        print(f"🎙️  ENROLLMENT: Setting up Wake Word '{self.phrase}'")
        print(f"=======================================================")
        print(f"Please say '{self.phrase}' clearly into your microphone 3 times.")
        
        clips = []
        for i in range(num_clips):
            input(f"\nPress [ENTER] to record sample {i+1}/{num_clips}...")
            wav = self.record_clip(duration_sec=1.5, prompt=f"Sample {i+1}: Say '{self.phrase}'")
            # Endpoint to clean boundaries
            clean_wav = endpoint_utterance(wav, self.sr)
            if clean_wav is not None:
                clips.append(clean_wav)
            else:
                clips.append(wav)
                
        print("\n⏳ Processing multi-tempo variations & minimal pairs...")
        self.cascade.enroll(clips, phrase=self.phrase, sr=self.sr)
        
        # Save profile locally so enrollment is only done once
        profile_data = {
            "phrase": self.phrase,
            "global_target_profile": self.cascade.global_target_profile.tolist(),
            "segment_target_profiles": self.cascade.segment_target_profiles.tolist(),
            "mean_duration": self.cascade.mean_duration,
            "std_duration": self.cascade.std_duration,
            "minimal_pairs": self.cascade.minimal_pair_negatives
        }
        with open(PROFILE_PATH, "w") as f:
            json.dump(profile_data, f, indent=2)
            
        print(f"✅ Enrollment saved to '{PROFILE_PATH}'!")

    def load_profile(self):
        if not os.path.exists(PROFILE_PATH):
            return False
        with open(PROFILE_PATH, "r") as f:
            data = json.load(f)
        self.phrase = data.get("phrase", self.phrase)
        self.cascade.global_target_profile = np.array(data["global_target_profile"], dtype=np.float32)
        self.cascade.segment_target_profiles = np.array(data["segment_target_profiles"], dtype=np.float32)
        self.cascade.mean_duration = data["mean_duration"]
        self.cascade.std_duration = data["std_duration"]
        self.cascade.minimal_pair_negatives = data.get("minimal_pairs", [])
        print(f"✅ Loaded saved profile for '{self.phrase}' from {PROFILE_PATH}")
        return True

    def run_live_stream(self):
        """Non-stop continuous streaming loop displaying live probability meter."""
        if not self.load_profile():
            self.enroll_interactive(num_clips=3)

        print(f"\n=======================================================")
        print(f"👂 LIVE STREAM ACTIVE: Listening for '{self.phrase}'")
        print(f"=======================================================")
        print("Press Ctrl+C to stop listening.\n")

        # Sliding window buffer: 1.2 seconds of audio with 0.1s hop
        buffer_len = int(1.2 * self.sr)
        hop_len = int(0.15 * self.sr)
        audio_buffer = np.zeros(buffer_len, dtype=np.float32)

        try:
            with sd.InputStream(samplerate=self.sr, channels=1, dtype='float32', blocksize=hop_len) as stream:
                last_trigger_time = 0.0
                
                while True:
                    # Read new audio hop
                    chunk, _ = stream.read(hop_len)
                    chunk = chunk.squeeze()
                    
                    # Slide buffer
                    audio_buffer = np.roll(audio_buffer, -hop_len)
                    audio_buffer[-hop_len:] = chunk
                    
                    # Quick RMS energy check
                    rms = np.sqrt(np.mean(audio_buffer**2) + 1e-10)
                    db = 20 * np.log10(rms)
                    
                    if db < -45.0:
                        # Ambient silence
                        bar = "░" * 20
                        sys.stdout.write(f"\r[ {bar} ]  0.0%  (Ambient Silence: {db:.1f} dB)    ")
                        sys.stdout.flush()
                        continue

                    # Extract Log-Mel Spectrogram
                    wav_t = torch.tensor(audio_buffer, dtype=torch.float32).unsqueeze(0)
                    mel = self.mel_transform(wav_t)
                    log_mel = torch.log(torch.clamp(mel, min=1e-5))
                    feat = log_mel.transpose(1, 2).unsqueeze(0).numpy() # (1, 1, time, n_mels)
                    
                    # Run Verification
                    triggered, metrics = self.cascade.verify_utterance(
                        feat, duration_sec=1.2, ctc_suffix_prob=0.90
                    )
                    
                    similarity = metrics.get("stage2_global_similarity", 0.0)
                    # Convert cosine similarity (-1 to 1) into a probability percentage (0 to 100)
                    prob = max(0.0, min(100.0, ((similarity + 1.0) / 2.0) * 100.0))
                    
                    # Visual Progress Bar
                    filled = int(prob / 5)
                    bar = "█" * filled + "░" * (20 - filled)
                    
                    current_time = time.time()
                    if triggered and (current_time - last_trigger_time > 2.0):
                        last_trigger_time = current_time
                        sys.stdout.write(f"\r\n\n🚨 [{bar}] {prob:.1f}% -> WAKE WORD DETECTED: '{self.phrase}'! 🚀\n\n")
                        sys.stdout.flush()
                    else:
                        status = "Listening..." if prob < 60 else "Matching..."
                        if "rejection_reason" in metrics:
                            status = metrics["rejection_reason"]
                        sys.stdout.write(f"\r[ {bar} ] {prob:5.1f}%  ({status})    ")
                        sys.stdout.flush()

        except KeyboardInterrupt:
            print("\n\n🛑 Live streaming stopped by user.")

if __name__ == "__main__":
    wake_phrase = sys.argv[1] if len(sys.argv) > 1 else "Hey Karthika"
    engine = LiveWakeWordEngine(phrase=wake_phrase)
    engine.run_live_stream()
