import streamlit as st
import tensorflow as tf
import librosa
import numpy as np
import tempfile
from streamlit_mic_recorder import mic_recorder


# Set page layout and configuration
st.set_page_config(
    page_title="TB Cough Assessment",
    page_icon="🫁",
    layout="centered"
)


# Load model once
@st.cache_resource
def load_my_model():
    return tf.keras.models.load_model("tb_cough_base_model_fixed.h5")


try:
    model = load_my_model()
    model_loaded = True

except Exception as e:
    model_loaded = False
    st.error(f"Could not load the model file. Error: {e}")


# Model configurations
SR = 16000
DURATION = 3
AUDIO_LENGTH = SR * DURATION
N_MELS = 128

CLASSES = [
    "No Tuberculosis (Healthy/Other Cough)",
    "Tuberculosis Detected"
]


# UI Layout

st.title("🫁 Live TB Cough Detection System")

st.write(
    "A Deep Learning project for rapid screening using cough audio analysis."
)

st.markdown("---")


if model_loaded:

    st.subheader("Record Your Cough")

    st.info(
        "Click the button below and cough clearly for 3 seconds."
    )


    # Microphone recorder

    audio = mic_recorder(
        start_prompt="🎤 Click to Record 3 Seconds",
        stop_prompt="⏹️ Stop Recording",
        key="recorder"
    )


    if audio:

        # Play recorded audio

        st.audio(
            audio["bytes"],
            format="audio/wav"
        )


        with st.spinner(
            "Processing audio and running model inference..."
        ):


            # Save audio temporarily

            with tempfile.NamedTemporaryFile(
                delete=False,
                suffix=".wav"
            ) as f:

                f.write(audio["bytes"])
                temp_path = f.name


            # Load audio

            y, sr = librosa.load(
                temp_path,
                sr=SR
            )


            # Padding / trimming

            if len(y) < AUDIO_LENGTH:

                y = np.pad(
                    y,
                    (0, AUDIO_LENGTH - len(y)),
                    mode="constant"
                )

            else:

                y = y[:AUDIO_LENGTH]


            # Mel Spectrogram extraction

            mel = librosa.feature.melspectrogram(
                y=y,
                sr=SR,
                n_mels=N_MELS
            )


            mel = librosa.power_to_db(
                mel,
                ref=np.max
            )


            # Model input shape

            sample = mel[
                np.newaxis,
                ...,
                np.newaxis
            ]


            # Prediction

            prob = model.predict(
                sample,
                verbose=0
            )[0][0]


            # Classification

            if prob >= 0.5:

                result_text = CLASSES[1]
                confidence = prob
                is_tb = True


            else:

                result_text = CLASSES[0]
                confidence = 1 - prob
                is_tb = False



        # Display result

        st.markdown(
            "### 📊 Assessment Summary"
        )


        if is_tb:

            st.error(
                f"⚠️ Prediction: {result_text}"
            )

            st.metric(
                "Model Confidence",
                f"{confidence*100:.2f}%"
            )

            st.warning(
                "This is an AI screening tool project, "
                "not a definitive medical diagnosis."
            )


        else:

            st.success(
                f"✅ Prediction: {result_text}"
            )

            st.metric(
                "Model Confidence",
                f"{confidence*100:.2f}%"
            )