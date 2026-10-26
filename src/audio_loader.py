import librosa
import numpy as np

class AudioPipeline:
    def __init__(self, sample_rate: int = 32000, duration: float = 5.0):
        self.sample_rate = sample_rate
        self.duration = duration

    def load_and_pad(self, file_path: str) -> np.ndarray:
        """Loads an audio file and enforces a fixed length."""
        target_length = int(self.sample_rate * self.duration)
        audio, _ = librosa.load(file_path, sr=self.sample_rate, duration=self.duration)
        
        if len(audio) < target_length:
            audio = np.pad(audio, (0, target_length - len(audio)), mode='constant')
        else:
            audio = audio[:target_length]
            
        return audio

    def extract_mel_spectrogram(self, audio: np.ndarray, n_mels: int = 256) -> np.ndarray:
        """Converts raw audio array into a normalized Log-Mel Spectrogram."""
        mel_spec = librosa.feature.melspectrogram(
            y=audio,
            sr=self.sample_rate,
            n_mels=n_mels
        )
        log_mel_spec = librosa.power_to_db(mel_spec, ref=np.max)
        return log_mel_spec

    def segment_audio(self, file_path: str, segment_duration: float = 3.0, overlap: float = 0.5) -> list[np.ndarray]:
        """Slices a long audio recording into fixed-duration segments with overlapping windows."""
        audio, _ = librosa.load(file_path, sr=self.sample_rate)
        
        segment_samples = int(segment_duration * self.sample_rate)
        hop_samples = int(segment_samples * (1.0 - overlap))
        
        segments = []
        for start in range(0, len(audio) - segment_samples + 1, hop_samples):
            chunk = audio[start : start + segment_samples]
            segments.append(chunk)
            
        return segments