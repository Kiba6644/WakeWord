import os
import argparse
import asyncio
import edge_tts
import random

# A subset of available English voices in edge-tts
VOICES = [
    "en-US-AriaNeural", "en-US-GuyNeural", "en-US-JennyNeural", 
    "en-GB-SoniaNeural", "en-GB-RyanNeural", "en-AU-NatashaNeural",
    "en-AU-WilliamNeural", "en-CA-ClaraNeural", "en-CA-LiamNeural",
    "en-IE-EmilyNeural", "en-IN-NeerjaNeural", "en-IN-PrabhatNeural",
    "en-NZ-MitchellNeural", "en-ZA-LukeNeural"
]

async def generate_clip(text, voice, rate, pitch, output_path):
    rate_str = f"{rate:+}%"
    pitch_str = f"{pitch:+}Hz"
    communicate = edge_tts.Communicate(text, voice, rate=rate_str, pitch=pitch_str)
    await communicate.save(output_path)
    print(f"Generated {output_path}")

async def main(args):
    os.makedirs(args.output_dir, exist_ok=True)
    
    texts = ["Hey", "Hey.", "Hey!", "Hi", "Hello"]
    
    tasks = []
    for i in range(args.count):
        text = random.choice(texts)
        voice = random.choice(VOICES)
        rate = random.randint(-20, 20)
        pitch = random.randint(-10, 10)
        
        output_path = os.path.join(args.output_dir, f"hey_{i:04d}_{voice}.wav")
        tasks.append(generate_clip(text, voice, rate, pitch, output_path))
    
    # Run concurrently with a reasonable limit to avoid rate limits
    chunk_size = 10
    for i in range(0, len(tasks), chunk_size):
        await asyncio.gather(*tasks[i:i+chunk_size])
    
    print(f"\nSuccessfully generated {args.count} synthetic 'hey' clips in '{args.output_dir}'")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate synthetic 'hey' clips using edge-tts")
    parser.add_argument("--output_dir", type=str, default="./hey_clips", help="Output directory")
    parser.add_argument("--count", type=int, default=120, help="Number of clips to generate")
    args = parser.parse_args()
    
    asyncio.run(main(args))
