# %% [markdown]
# source ~/tfvenv/bin/activate
# 

# %% [markdown]
# jupyter notebook --no-browser --port=8888

# %% [markdown]
# ## DATA PREP

# %%
import numpy as np
import librosa
import tensorflow as tf
import warnings
warnings.filterwarnings("ignore", category=FutureWarning)

# %%
# Import from HuggingFace dataset of clean voice and clean voice with bg
from datasets import load_dataset

db = load_dataset("JacobLinCool/VoiceBank-DEMAND-16k")

# %%
print(db.keys())

# one sample
sample = db['train'][1]
print(sample.keys())

# %%
# Print everything inside the 'clean' field
print("CLEAN FIELD:")
print(sample['clean'])

# Check individual components
print("\nWaveform type:", type(sample['clean']['array']))
print("Waveform shape:", sample['clean']['array'].shape)

# %% [markdown]
# we dont need to convert to numpy, since the dataset we are using is already in numpy format

# %% [markdown]
# ## Convert numpy matrix to spectogram

# %% [markdown]
# Spectograms are used as input for the encoder-decoder model, for model to learn diff freqs

# %%
n_fft = 256  # Number of frequency bins (defines resolution of frequency)
hop_length_fft = 128  # Step size for moving the window
dim_square_spec = n_fft // 2  # Typically 128 (since STFT returns n_fft//2 + 1 freq bins)
'''
n_fft controls the size of the FFT window → bigger = more frequency detail, less time resolution.

hop_length_fft controls the shift between windows.

dim_square_spec is used to crop/pad the spectrogram to a square shape the CNN can handle (e.g., 128×128).
'''

# %%
# Takes waveform and returns spectogram (dB) and phase
# Call for EACH clean/ noisy clip
def audio_to_magnitude_db_phase(n_fft, hop_length_fft, audio):
    stft_audio = librosa.stft(audio, n_fft=n_fft, hop_length=hop_length_fft)
    stft_audio_magnitude, stft_audio_phase = librosa.magphase(stft_audio)
    stft_audio_magnitude_db = librosa.amplitude_to_db(stft_audio_magnitude,ref=np.max)
    return stft_audio_magnitude_db, stft_audio_phase

# %% [markdown]
# ## To keep same size

# %%
def pad_or_crop(spec, target_shape=(129, 128)):
    # Pad or crop the time axis
    if spec.shape[1] > target_shape[1]:
        spec = spec[:, :target_shape[1]]
    elif spec.shape[1] < target_shape[1]:
        pad_width = target_shape[1] - spec.shape[1]
        spec = np.pad(spec, ((0, 0), (0, pad_width)), mode='constant')
    return spec


# %% [markdown]
# ## Main training loop

# %%
n_fft = 256 # num of freq bins
hop_length_fft = 128 # stride of window
target_shape = (129, 256)  # Frequency bins from STFT with n_fft=256 → 129 bins

# %%
'''
For each sample:
1. extract noisy and clean waveform
2. convert both to dB spectograms
3. pad/crop both
4. store into input and target
'''

db = load_dataset("JacobLinCool/VoiceBank-DEMAND-16k")
db_train = db['train']
db_val = db['test']

# Generator to yield (noisy, clean) spectrogram pairs
def normalize_db(spec_db):
    # Normalize from [-80, 0] → [0, 1]
    return (spec_db + 80.0) / 80.0

def data_generator(dataset_split):
    for sample in dataset_split:
        noisy = sample['noisy']['array']
        clean = sample['clean']['array']
        noisy_spec, _ = audio_to_magnitude_db_phase(n_fft, hop_length_fft, noisy)
        clean_spec, _ = audio_to_magnitude_db_phase(n_fft, hop_length_fft, clean)
        noisy_spec = pad_or_crop(noisy_spec, target_shape)
        clean_spec = pad_or_crop(clean_spec, target_shape)
        noisy_spec = normalize_db(noisy_spec)
        clean_spec = normalize_db(clean_spec)
        yield noisy_spec[..., np.newaxis], clean_spec[..., np.newaxis]

# Build datasets
train_dataset = tf.data.Dataset.from_generator(
    lambda: data_generator(db_train),
    output_signature=(
        tf.TensorSpec(shape=(129, 256, 1), dtype=tf.float32),
        tf.TensorSpec(shape=(129, 256, 1), dtype=tf.float32),
    )
).batch(8).prefetch(tf.data.AUTOTUNE)

val_dataset = tf.data.Dataset.from_generator(
    lambda: data_generator(db_val),
    output_signature=(
        tf.TensorSpec(shape=(129, 256, 1), dtype=tf.float32),
        tf.TensorSpec(shape=(129, 256, 1), dtype=tf.float32),
    )
).batch(8).prefetch(tf.data.AUTOTUNE)


# %% [markdown]
# ## CHeck GPU
# 

# %%
build_info = tf.sysconfig.get_build_info()
print("TF Version:", tf.__version__)
print("CUDA version:", build_info.get("cuda_version", "Unknown"))
print("cuDNN version:", build_info.get("cudnn_version", "Unknown"))
print("GPUs:", tf.config.list_physical_devices('GPU'))


# %%
physical_devices = tf.config.list_physical_devices('GPU')
if physical_devices:
    try:
        tf.config.experimental.set_memory_growth(physical_devices[0], True)
    except:
        pass


# %% [markdown]
# ## Model Choice

# %%
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, UpSampling2D, concatenate, ZeroPadding2D
from tensorflow.keras.models import Model

def unet(input_shape=(129,256,1)):
    inputs = Input(shape=input_shape)

    #Encoder
    c1 = Conv2D(32, (3,3), activation='relu', padding='same')(inputs)
    p1 = MaxPooling2D((2,2))(c1)

    c2 = Conv2D(64, (3,3), activation='relu', padding='same')(p1)
    p2 = MaxPooling2D((2,2))(c2)

    c3 = Conv2D(128, (3,3), activation='relu', padding='same')(p2)

    #Decoder
    u1 = UpSampling2D((2,2))(c3)
    concat1 = concatenate([u1,c2])
    c4 = Conv2D(64, (3,3), activation='relu', padding='same')(concat1)

    u2 = UpSampling2D((2,2))(c4)
    u2 = ZeroPadding2D(((1,0), (0,0)))(u2)
    
    concat2 = concatenate([u2,c1])
    c5 = Conv2D(32, (3,3), activation='relu', padding='same')(concat2)

    outputs = Conv2D(1, (1,1), activation='linear', padding='same')(c5)

    model = Model(inputs, outputs)
    return model

# %% [markdown]
# ## Training

# %%
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.callbacks import EarlyStopping

checkpoint_cb = ModelCheckpoint(
    'best_model.h5',
    save_best_only=True,
    monitor='val_loss',
    mode='min',
    verbose=1
)

early_stop_cb = EarlyStopping(
    monitor='val_loss',
    patience=5,           # stop if no improvement for 5 epochs
    restore_best_weights=True
)

model = unet(input_shape=(129,256,1))
model.compile(optimizer='adam', loss='mean_squared_error')
model.fit(train_dataset, validation_data=val_dataset, epochs=50, callbacks=[checkpoint_cb, early_stop_cb])

# %% [markdown]
# ## Evaluation

# %%
loss = model.evaluate(val_dataset)
print(f"Final Training Loss: {loss:.4f}")

# %% [markdown]
# ## Visualize

# %%
import matplotlib.pyplot as plt

# Take 1 batch and 1 sample from the generator
for noisy_batch, clean_batch in val_dataset.take(1):
    input_spec = noisy_batch[0].numpy().squeeze()  # shape: (129, 256)
    target_spec = clean_batch[0].numpy().squeeze()
    pred = model.predict(noisy_batch[:1])[0, :, :, 0]
    break

# Plot
plt.figure(figsize=(15, 4))
plt.subplot(1, 3, 1)
plt.imshow(input_spec, origin='lower', aspect='auto')
plt.title("Noisy Input")

plt.subplot(1, 3, 2)
plt.imshow(target_spec, origin='lower', aspect='auto')
plt.title("Clean Target")

plt.subplot(1, 3, 3)
plt.imshow(pred, origin='lower', aspect='auto')
plt.title("Model Output")

plt.tight_layout()
plt.show()


# %% [markdown]
# ## Sampling test

# %%
import soundfile as sf
from IPython.display import Audio

# Pick a raw index (same dataset as original HuggingFace one)
sample = db['test'][4]
noisy_waveform = sample['noisy']['array']
clean_waveform = sample['clean']['array']

# Convert to spectrogram
noisy_spec_db, noisy_phase = audio_to_magnitude_db_phase(n_fft, hop_length_fft, noisy_waveform)
noisy_spec_db = pad_or_crop(noisy_spec_db, target_shape)
noisy_spec_db = normalize_db(noisy_spec_db) 

# Predict
input_tensor = noisy_spec_db[np.newaxis, ..., np.newaxis]
predicted_spec_db = model.predict(input_tensor)[0, :, :, 0]

# Clip
# Convert to amplitude
predicted_spec_db = predicted_spec_db * 80.0 - 80.0
pred_amp = librosa.db_to_amplitude(predicted_spec_db)

# Crop phase and multiply
noisy_phase = noisy_phase[:, :pred_amp.shape[1]]
pred_stft = pred_amp * noisy_phase

# Reconstruct waveform
reconstructed_wave = librosa.istft(pred_stft, hop_length=128)

# Save audio
sf.write("denoised.wav", reconstructed_wave, samplerate=16000)
sf.write("original_noisy.wav", noisy_waveform, samplerate=16000)
sf.write("original_clean.wav", clean_waveform, samplerate=16000)

# Listen
print("Denoised:")
display(Audio(reconstructed_wave, rate=16000))

print("Original Noisy:")
display(Audio(noisy_waveform, rate=16000))

print("Ground Truth Clean:")
display(Audio(clean_waveform, rate=16000))


# %%



