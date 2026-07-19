import zipfile
import numpy as np
import pandas as pd
import librosa

from config import (
    ZIP_FILE_PATH,
    EXTRACT_PATH,
    CLASSES,
    SR,
    AUDIO_LENGTH,
    N_MELS,
    DATA_DIR
)


def prepare_dataset_directory():

    # ==========================================
    # 1. CHECK AND EXTRACT DATASET
    # ==========================================

    if not EXTRACT_PATH.exists():

        if not ZIP_FILE_PATH.exists():

            raise FileNotFoundError(
                f"Could not find the ZIP file at:\n"
                f"{ZIP_FILE_PATH}"
            )

        print(
            f"Extracting dataset from:\n"
            f"{ZIP_FILE_PATH}"
        )

        with zipfile.ZipFile(
            ZIP_FILE_PATH,
            'r'
        ) as zip_ref:

            zip_ref.extractall(
                EXTRACT_PATH
            )

        print(
            "Extraction complete."
        )

    else:

        print(
            f"Extracted folder already exists at:\n"
            f"{EXTRACT_PATH}"
        )

    return EXTRACT_PATH


def extract_features_and_log(
    dataset_path
):

    X = []

    y = []

    manifest_data = []

    print(
        f"\nStarting audio feature extraction from:"
        f"\n{dataset_path}"
    )


    # ==========================================
    # 2. PROCESS EACH CLASS
    # ==========================================

    for label_idx, class_name in enumerate(CLASSES):

        class_folder = None


        # Search for the class folder
        for root, dirs, files in dataset_path.walk():

            if class_name in dirs:

                class_folder = root / class_name

                break


        if class_folder is None:

            print(
                f"Warning: Could not locate directory "
                f"'{class_name}'"
            )

            continue


        print(
            f"Processing folder: {class_name}"
        )


        # ==========================================
        # 3. PROCESS AUDIO FILES
        # ==========================================

        for file_path in class_folder.iterdir():

            if file_path.suffix.lower() in (

                '.wav',
                '.mp3',
                '.m4a',
                '.3gp'

            ):

                try:

                    # Load audio
                    audio, sr = librosa.load(

                        str(file_path),

                        sr=SR

                    )


                    # ==========================================
                    # 4. PAD OR TRIM AUDIO TO 3 SECONDS
                    # ==========================================

                    if len(audio) < AUDIO_LENGTH:

                        audio = np.pad(

                            audio,

                            (
                                0,
                                AUDIO_LENGTH - len(audio)
                            ),

                            'constant'

                        )

                    else:

                        audio = audio[
                            :AUDIO_LENGTH
                        ]


                    # ==========================================
                    # 5. CREATE MEL-SPECTROGRAM
                    # ==========================================

                    mel_spec = librosa.feature.melspectrogram(

                        y=audio,

                        sr=SR,

                        n_mels=N_MELS

                    )


                    # Convert to log scale
                    log_mel_spec = librosa.power_to_db(

                        mel_spec,

                        ref=np.max

                    )


                    # Store features
                    X.append(
                        log_mel_spec
                    )


                    # Store label
                    y.append(
                        label_idx
                    )


                    # ==========================================
                    # 6. CREATE MANIFEST RECORD
                    # ==========================================

                    manifest_data.append({

                        "audio_id":
                        file_path.stem,

                        "file_name":
                        file_path.name,

                        "class_folder":
                        class_name,

                        "tb_status_label":
                        label_idx

                    })


                except Exception as e:

                    print(

                        f"Error processing "
                        f"{file_path.name}: {e}"

                    )


    # ==========================================
    # 7. SAVE MANIFEST CSV
    # ==========================================

    if manifest_data:

        df_manifest = pd.DataFrame(
            manifest_data
        )


        csv_save_path = (

            DATA_DIR
            / "audio_dataset_manifest.csv"

        )


        df_manifest.to_csv(

            csv_save_path,

            index=False

        )


        print(

            f"\nManifest index sheet updated at:"
            f"\n{csv_save_path}"

        )


    return (

        np.array(X),

        np.array(y)

    )