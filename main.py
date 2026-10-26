import numpy as np
from src.audio_loader import AudioPipeline

# Initialize pipeline
pipeline = AudioPipeline(sample_rate=22050, duration=3.0)

# Generate a synthetic audio signal (3 seconds)
t = np.linspace(0, 3.0, int(22050 * 3.0))
test_signal = 0.5 * np.sin(2 * np.pi * 440 * t)

# Convert to Log-Mel Spectrogram
spectrogram = pipeline.extract_mel_spectrogram(test_signal)

print("✅ Audio pipeline test successful!")
print(f"Log-Mel Spectrogram matrix shape: {spectrogram.shape}")