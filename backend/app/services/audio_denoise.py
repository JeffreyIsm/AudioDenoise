'''
This is where the audio gets processed by the ML model
'''
import os
import librosa
import soundfile as sf
import numpy as np
from app.models.model_loader import model

n_fft = 256
hop_length_fft = 128
target_shape = (129, 256)

def normalize_db(x):
    return (x + 80.0) / 80.0

def denormalize_db(x):
    return x * 80.0 - 80.0

def pad_crop(x, target_shape=(129,256)):
    if x.shape[1] > target_shape[1]:
        x = x[:, :target_shape[1]]
    elif x.shape[1] < target_shape[1]:
        pad_width = target_shape[1] - x.shape[1]
        x = np.pad(x, ((0,0), (0,pad_width)), mode='constant')
    return x

def audio_to_magnitude_db(n_fft, hop_length_fft, audio, window='hann', center=True):
    stft_audio = librosa.stft(audio, n_fft=n_fft, hop_length=hop_length_fft, window=window, center=center)
    mag, phase = librosa.magphase(stft_audio)
    mag_db = librosa.amplitude_to_db(mag, ref=np.max)
    return mag_db, phase

def process_audio(input_path): #string of file path to uploaded audio file
    window = 'hann'
    center = True
    # Load input audio
    audio, sr = librosa.load(input_path, sr=16000)
    # Convert to dB spectrogram
    spec_db, phase = audio_to_magnitude_db(n_fft, hop_length_fft, audio, window=window, center=center)
    num_frames = spec_db.shape[1]
    denoised_spec = np.zeros_like(spec_db)
    # Process in 256-frame chunks
    for start in range(0, num_frames, target_shape[1]):
        end = start + target_shape[1]
        chunk = spec_db[:, start:end]
        chunk_phase = phase[:, start:end]
        # Pad/crop chunk to target shape
        chunk = pad_crop(chunk, target_shape)
        chunk_phase = pad_crop(chunk_phase, target_shape)
        # Normalize
        norm_chunk = normalize_db(chunk)
        input_tensor = norm_chunk[np.newaxis, ..., np.newaxis]
        # Predict
        pred = model.predict(input_tensor)[0, :, :, 0]
        # Denormalize
        pred = denormalize_db(pred)
        # Place the denoised chunk back (handle last chunk size)
        actual_end = min(end, num_frames)
        denoised_spec[:, start:actual_end] = pred[:, :actual_end-start]
    # Convert back to amplitude and reconstruct denoised audio (normalized)
    predicted_amp = librosa.db_to_amplitude(denoised_spec)
    phase = phase[:, :predicted_amp.shape[1]]
    stft_denoised = predicted_amp * phase
    denoised_audio = librosa.istft(stft_denoised, hop_length=hop_length_fft, window=window, center=center, length=len(audio))
    denoised_audio_norm = denoised_audio / (np.max(np.abs(denoised_audio)) + 1e-8)
    base, ext = os.path.splitext(input_path)
    output_path = f"{base}_denoised{ext}"
    sf.write(output_path, denoised_audio_norm, samplerate=sr)
    return output_path