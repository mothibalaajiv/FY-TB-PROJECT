import numpy as np
import librosa
import tensorflow as tf

from config import (
    SR,
    AUDIO_LENGTH,
    N_MELS,
    PROJECT_ROOT
)

# Load trained model
MODEL_PATH = (
    PROJECT_ROOT
    / "backend"
    / "models"
    / "tb_cough_base_model.h5"
)

model = tf.keras.models.load_model(MODEL_PATH)


def predict_audio(audio_path):
    # Load audio
    audio, sr = librosa.load(audio_path, sr=SR)

    # Pad or trim to 3 seconds
    if len(audio) < AUDIO_LENGTH:
        audio = np.pad(
            audio,
            (0, AUDIO_LENGTH - len(audio)),
            mode="constant"
        )
    else:
        audio = audio[:AUDIO_LENGTH]

    # Create Mel Spectrogram
    mel_spec = librosa.feature.melspectrogram(
        y=audio,
        sr=SR,
        n_mels=N_MELS
    )

    # Convert to Log-Mel Spectrogram
    log_mel_spec = librosa.power_to_db(
        mel_spec,
        ref=np.max
    )

    # Add dimensions
    input_data = log_mel_spec[
        np.newaxis,
        ...,
        np.newaxis
    ]

    # Make prediction
    prediction = model.predict(
        input_data,
        verbose=0
    )[0][0]

    # 0 = NOT TB, 1 = TB
    if prediction >= 0.5:
        result = "TB"
        confidence = prediction
    else:
        result = "NOT TB"
        confidence = 1 - prediction

    return result, confidence


if __name__ == "__main__":
    audio_file = input("Enter audio file path: ")

    result, confidence = predict_audio(audio_file)

    print(f"\nPrediction: {result}")
    print(f"Confidence: {confidence * 100:.2f}%")

    #../../data/test_audio.wav