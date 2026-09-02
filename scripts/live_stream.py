import os
import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import time
import json
import numpy as np
import torch
import torchaudio.transforms as T
import hashlib

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

from core.config import SR, N_MELS, N_FFT, HOP_LENGTH, EMBED_DIM, STAGE2_GLOBAL_THRESHOLD, SUFFIX_REJECTION_THRESHOLD
from core.audio_utils import endpoint_utterance, time_stretch_audio, generate_phonetic_minimal_pairs, word_to_phoneme_tokens
from core.inference import WakeWordCascade

PROFILE_PATH = "wakeword_profile.json"
DEFAULT_MODEL_PATH = "wakeword_student.onnx"

def get_file_md5(fname):
    if not os.path.exists(fname): return ""
    hash_md5 = hashlib.md5()
    with open(fname, "rb") as f:
        for chunk in iter(lambda: f.read(4096), b""):
            hash_md5.update(chunk)
    return hash_md5.hexdigest()

def compute_ctc_posterior(ctc_logits, target_tokens):
    if not target_tokens:
        return 1.0
    probs = np.exp(ctc_logits[0]) # [T, 42]
    token_probs = []
    for token in target_tokens:
        if token != 41: # ignore <UNK>
            token_probs.append(np.max(probs[:, token]))
    if not token_probs: return 1.0
    return float(np.mean(token_probs))

class LiveWakeWordEngine:
    def __init__(self, model_path=DEFAULT_MODEL_PATH, phrase="Hey Karthika", debug=False, threshold=None, suffix_threshold=None):
        self.model_path = model_path
        self.phrase = phrase
        self.debug = debug
        self.sr = SR
        self.model_hash = get_file_md5(model_path)
        g_thresh = threshold if threshold is not None else STAGE2_GLOBAL_THRESHOLD
        s_thresh = suffix_threshold if suffix_threshold is not None else SUFFIX_REJECTION_THRESHOLD
        self.cascade = WakeWordCascade(
            stage1_path="", 
            stage2_path=model_path if os.path.exists(model_path) else "",
            threshold1=0.55, 
            threshold2=g_thresh, 
            suffix_threshold=s_thresh
        )
        self.mel_transform = T.MelSpectrogram(
            sample_rate=SR, n_fft=N_FFT, hop_length=HOP_LENGTH, n_mels=N_MELS, power=2.0
        )
        words = phrase.split()
        self.target_tokens = word_to_phoneme_tokens(words[-1])
        
    def record_clip(self, duration_sec=1.5, prompt="Say wake word"):
        print(f"\n🎤 {prompt}... (Listening for {duration_sec}s)")
        audio = sd.rec(int(duration_sec * self.sr), samplerate=self.sr, channels=1, dtype='float32')
        sd.wait()
        wav = audio.squeeze()
        print("  ✓ Audio captured.")
        return wav

    def enroll_interactive(self, num_clips=3):
        print(f"\n=======================================================")
        print(f"🎙️  ENROLLMENT: Setting up Wake Word '{self.phrase}'")
        print(f"=======================================================")
        print(f"Please say '{self.phrase}' clearly into your microphone 3 times.")
        
        clips = []
        for i in range(num_clips):
            while True:
                input(f"\nPress [ENTER] to record sample {i+1}/{num_clips}...")
                wav = self.record_clip(duration_sec=1.5, prompt=f"Sample {i+1}: Say '{self.phrase}'")
                clean_wav = endpoint_utterance(wav, self.sr)
                if clean_wav is not None:
                    clips.append(clean_wav)
                    break
                else:
                    print("⚠️  Could not detect speech in that recording. Please speak louder and retry.")
                
        print("\n⏳ Processing multi-tempo variations & minimal pairs...")
        self.cascade.enroll(clips, phrase=self.phrase, sr=self.sr)
        
        profile_data = {
            "phrase": self.phrase,
            "model_hash": self.model_hash,
            "global_target_profile": self.cascade.global_target_profile.tolist(),
            "suffix_target_profile": self.cascade.suffix_target_profile.tolist(),
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
            
        if data.get("model_hash") != self.model_hash:
            print("⚠️  Model version changed since last enrollment. Re-enrolling is required.")
            return False
            
        self.phrase = data.get("phrase", self.phrase)
        self.cascade.global_target_profile = np.array(data["global_target_profile"], dtype=np.float32)
        self.cascade.suffix_target_profile = np.array(data["suffix_target_profile"], dtype=np.float32)
        self.cascade.mean_duration = data["mean_duration"]
        self.cascade.std_duration = data["std_duration"]
        self.cascade.minimal_pair_negatives = data.get("minimal_pairs", [])
        print(f"✅ Loaded saved profile for '{self.phrase}' from {PROFILE_PATH}")
        return True

    def run_live_stream(self):
        if not self.load_profile():
            self.enroll_interactive(num_clips=3)

        print(f"\n=======================================================")
        print(f"👂 LIVE STREAM ACTIVE: Listening for '{self.phrase}'")
        print(f"=======================================================")
        print("Press Ctrl+C to stop listening.\n")

        buffer_len = int(1.5 * self.sr)
        hop_len = int(0.15 * self.sr)
        audio_buffer = np.zeros(buffer_len, dtype=np.float32)
        
        session = self.cascade.stage2
        input_name = session.get_inputs()[0].name if session else ""
        outputs = session.get_outputs() if session else []
        has_ctc = any(o.name == 'ctc_logits' for o in outputs)

        try:
            with sd.InputStream(samplerate=self.sr, channels=1, dtype='float32', blocksize=hop_len) as stream:
                last_trigger_time = 0.0
                
                while True:
                    chunk, _ = stream.read(hop_len)
                    chunk = chunk.squeeze()
                    
                    audio_buffer = np.roll(audio_buffer, -hop_len)
                    audio_buffer[-hop_len:] = chunk
                    
                    rms = np.sqrt(np.mean(audio_buffer**2) + 1e-10)
                    db = 20 * np.log10(rms)
                    
                    if db < -45.0:
                        bar = "░" * 20
                        sys.stdout.write(f"\r[ {bar} ]  0.0%  (Ambient Silence: {db:.1f} dB)    ")
                        sys.stdout.flush()
                        continue

                    current_time = time.time()
                    if current_time - last_trigger_time < 2.0:
                        audio_buffer.fill(0) # Flush buffer after trigger
                        continue

                    # Endpoint to match enrollment conditions
                    clean_wav = endpoint_utterance(audio_buffer, self.sr)
                    if clean_wav is None:
                        continue # wait for a clean endpointed utterance
                        
                    wav_t = torch.tensor(clean_wav, dtype=torch.float32).unsqueeze(0)
                    mel = self.mel_transform(wav_t)
                    log_mel = torch.log(torch.clamp(mel, min=1e-5))
                    # CMVN
                    log_mel = (log_mel - log_mel.mean(dim=-1, keepdim=True)) / (log_mel.std(dim=-1, keepdim=True) + 1e-5)
                    feat = log_mel.transpose(1, 2).unsqueeze(0).numpy()
                    
                    ctc_suffix_prob = 1.0
                    if session and has_ctc:
                        outs = session.run(None, {input_name: feat})
                        # Typically embedding, suffix_embed, ctc_logits
                        # Find ctc_logits by matching name
                        ctc_idx = next(i for i, o in enumerate(outputs) if o.name == 'ctc_logits')
                        ctc_logits = outs[ctc_idx]
                        ctc_suffix_prob = compute_ctc_posterior(ctc_logits, self.target_tokens[-3:])
                    
                    triggered, metrics = self.cascade.verify_utterance(
                        feat, duration_sec=len(clean_wav)/self.sr,
                        audio_buffer=None, # Already endpointed, so passed duration is exact
                        sr=self.sr,
                        ctc_suffix_prob=ctc_suffix_prob
                    )
                    
                    similarity = metrics.get("stage2_global_similarity", 0.0)
                    suffix_sim = metrics.get("suffix_phase_similarity", 0.0)
                    
                    prob = max(0.0, min(100.0, ((similarity - 0.80) / 0.18) * 100.0))
                    
                    filled = int(prob / 5)
                    bar = "█" * filled + "░" * (20 - filled)
                    
                    if triggered:
                        last_trigger_time = current_time
                        sys.stdout.write(f"\r\n\n🚨 [{bar}] {prob:.1f}% -> WAKE WORD DETECTED: '{self.phrase}'! 🚀\n   [Telemetry: CosSim={similarity:.4f} (Thresh={self.cascade.threshold2}), SuffixSim={suffix_sim:.4f}, Energy={db:.1f} dB]\n\n")
                        sys.stdout.flush()
                        audio_buffer.fill(0)
                    else:
                        status = "Listening..." if prob < 60 else "Matching..."
                        if metrics.get("near_miss"):
                            status = f"⚡ Near-miss ({metrics.get('near_miss_stage', '?')})"
                        if "rejection_reason" in metrics:
                            status = metrics["rejection_reason"]
                        dur_str = f"{metrics.get('duration_sec', 0.0):.2f}s"
                        
                        log_str = f"[ {bar} ] {prob:5.1f}% | Sim: {similarity:.4f} | Suf: {suffix_sim:.4f} | dB: {db:5.1f} | Dur: {dur_str} | ({status})"
                        if hasattr(self, 'debug') and self.debug:
                            print(log_str)
                        else:
                            sys.stdout.write(f"\r{log_str}    ")
                            sys.stdout.flush()

        except KeyboardInterrupt:
            print("\n\n🛑 Live streaming stopped by user.")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Live Wake Word Stream Verification")
    parser.add_argument("phrase", type=str, nargs="?", default="Hey Karthika", help="Target wake word phrase (default: 'Hey Karthika')")
    parser.add_argument("--threshold", "-t", type=float, default=None, help="Global cosine similarity threshold (e.g. 0.85)")
    parser.add_argument("--suffix_threshold", "-st", type=float, default=None, help="Suffix embedding rejection threshold (e.g. 0.82)")
    parser.add_argument("--model_path", "-m", type=str, default=DEFAULT_MODEL_PATH, help="Path to exported ONNX model")
    parser.add_argument("--debug", action="store_true", help="Print verbose step-by-step telemetry")
    parser.add_argument("--re_enroll", action="store_true", help="Force new interactive voice enrollment")
    
    args = parser.parse_args()
    
    if args.re_enroll and os.path.exists(PROFILE_PATH):
        os.remove(PROFILE_PATH)
        print("🗑️ Removed existing profile for fresh re-enrollment.")

    engine = LiveWakeWordEngine(
        model_path=args.model_path,
        phrase=args.phrase,
        debug=args.debug,
        threshold=args.threshold,
        suffix_threshold=args.suffix_threshold
    )
    
    g_thresh = engine.cascade.threshold2
    s_thresh = engine.cascade.suffix_threshold
    print(f"🔧 Live Stream Configured | Phrase: '{args.phrase}' | Global Threshold: {g_thresh} | Suffix Threshold: {s_thresh} | Debug: {args.debug}")
    engine.run_live_stream()
