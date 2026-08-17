import os
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

import tempfile
from pathlib import Path

import numpy as np
import nibabel as nib
import streamlit as st
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import load_model
from matplotlib import pyplot as plt
from scipy.ndimage import zoom

st.set_page_config(
    page_title="Brain Tumor Segmentation Using 3D U-Net",
    layout="wide"
)

BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "default_model.keras"

TARGET_SHAPE = (64, 64, 64, 4)
CROP = (slice(56, 184), slice(56, 184), slice(13, 141))
MODALITY_ORDER = ["t1n", "t1c", "t2f", "t2w"]
SLICE_INDICES = [50, 75, 90]


@st.cache_resource
def load_segmentation_model():
    if not MODEL_PATH.exists():
        st.error(f"Model file not found at expected location:\n{MODEL_PATH}")
        return None
    try:
        model = load_model(str(MODEL_PATH), compile=False)
        return model
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None


def identify_modality(file_name):
    name = file_name.lower()
    if "seg" in name:
        return "seg"
    for tag in MODALITY_ORDER:
        if tag in name:
            return tag
    return None


def process_uploaded_files(uploaded_files):
    """Load each uploaded NIfTI file, tag it by modality, and normalize it."""
    volumes = {}
    reference_img = None
    scaler = MinMaxScaler()

    for uploaded_file in uploaded_files:
        tag = identify_modality(uploaded_file.name)
        if tag is None:
            st.warning(f"Could not identify modality for file: {uploaded_file.name}")
            continue

        with tempfile.NamedTemporaryFile(delete=False, suffix=".nii.gz") as tmp:
            tmp.write(uploaded_file.getbuffer())
            tmp_path = tmp.name

        try:
            img = nib.load(tmp_path)
            data = img.get_fdata()

            if tag == "seg":
                volumes["seg"] = data.astype(np.uint8)
            else:
                data = scaler.fit_transform(
                    data.reshape(-1, data.shape[-1])
                ).reshape(data.shape)
                volumes[tag] = data
                if tag == "t1n":
                    reference_img = img
        except Exception as e:
            st.error(f"Error processing {uploaded_file.name}: {e}")
        finally:
            os.unlink(tmp_path)

    return volumes, reference_img


def validate_modalities(volumes):
    missing = [m for m in MODALITY_ORDER if m not in volumes]
    if missing:
        return f"Missing modalities: {', '.join(missing)}. Please upload all four MRI modalities: T1n, T1c, T2f and T2w."

    shapes = {m: volumes[m].shape for m in MODALITY_ORDER}
    if len(set(shapes.values())) > 1:
        return f"All four MRI modalities must have the same dimensions. Got: {shapes}"

    return None


def prepare_input(volumes):
    combined = np.stack([volumes[m] for m in MODALITY_ORDER], axis=-1)
    cropped = combined[CROP[0], CROP[1], CROP[2], :]
    downsampled = cropped[::2, ::2, ::2, :]
    return downsampled, cropped


def run_prediction(model, model_input):
    batched = np.expand_dims(model_input, axis=0)
    prediction = model.predict(batched, verbose=0)
    return np.argmax(prediction, axis=-1)[0]


def upsample_to_target(volume, target_shape):
    zoom_factors = tuple(t / s for t, s in zip(target_shape, volume.shape))
    return zoom(volume, zoom_factors, order=0)


def visualize_results(cropped_input, prediction, ground_truth=None):
    t1c = cropped_input[:, :, :, 1]
    n_cols = 3 if ground_truth is not None else 2
    fig, axes = plt.subplots(len(SLICE_INDICES), n_cols, figsize=(4 * n_cols, 10))

    for i, slice_idx in enumerate(SLICE_INDICES):
        axes[i, 0].imshow(np.rot90(t1c[:, :, slice_idx]), cmap="gray")
        axes[i, 0].set_title(f"T1c - Slice {slice_idx}")
        axes[i, 0].axis("off")

        axes[i, 1].imshow(np.rot90(prediction[:, :, slice_idx]))
        axes[i, 1].set_title(f"Prediction - Slice {slice_idx}")
        axes[i, 1].axis("off")

        if ground_truth is not None:
            axes[i, 2].imshow(np.rot90(ground_truth[:, :, slice_idx]))
            axes[i, 2].set_title(f"Ground Truth - Slice {slice_idx}")
            axes[i, 2].axis("off")

    plt.tight_layout()
    return fig


def main():
    st.title("Brain Tumor Segmentation Using 3D U-Net")
    st.write(
    "A lightweight 3D U-Net for brain tumor segmentation "
    "designed to run on normal CPUs."
    )
    with st.expander("How to use this app"):
        st.markdown(
            """
            1. Upload the four MRI modalities: T1n, T1c, T2f and T2w.
            2. Optionally upload a segmentation mask for comparison.
            3. Click **Process and Predict**.
            4. View and download the predicted segmentation.
            """
        )

    model = load_segmentation_model()
    if model is None:
        return

    st.caption(f"Model input shape: {model.input_shape}")
    st.caption(f"Model output shape: {model.output_shape}")

    uploaded_files = st.file_uploader(
        "Upload MRI scans (NIfTI format)",
        type=["nii", "gz"],
        accept_multiple_files=True,
    )

    if not uploaded_files:
        return

    if len(uploaded_files) < 4:
        st.warning("Please upload all four modalities (T1n, T1c, T2f, T2w).")
        return

    if not st.button("Process and Predict"):
        return

    with st.spinner("Loading and preprocessing files..."):
        volumes, reference_img = process_uploaded_files(uploaded_files)
        error = validate_modalities(volumes)
        if error:
            st.error(error)
            return

        model_input, cropped_input = prepare_input(volumes)

        if model_input.shape != TARGET_SHAPE:
            st.error(
                f"Prepared input shape {model_input.shape} does not match "
                f"expected shape {TARGET_SHAPE}. Aborting to avoid feeding "
                "the model incorrect data."
            )
            return

        ground_truth = volumes.get("seg")
        if ground_truth is not None:
            ground_truth = ground_truth[CROP[0], CROP[1], CROP[2]]
            ground_truth[ground_truth == 4] = 3

    st.success(f"Input ready: {model_input.shape}")

    with st.spinner("Running segmentation..."):
        try:
            prediction = run_prediction(model, model_input)
            prediction = upsample_to_target(prediction, cropped_input.shape[:3])
            prediction = prediction.astype(np.int32)
        except Exception as e:
            st.error(f"Prediction failed: {e}")
            return

    st.success("Segmentation complete.")

    fig = visualize_results(cropped_input, prediction, ground_truth)
    st.pyplot(fig)

    st.subheader("Download Prediction")

    reference_file = next(
        file for file in uploaded_files
        if "t1n" in file.name.lower()
    )

    with tempfile.NamedTemporaryFile(
        suffix=".nii.gz",
        delete=False
    ) as tmp:
        reference_path = tmp.name
        tmp.write(reference_file.getbuffer())

    try:
        reference_img = nib.load(reference_path)

        prediction_img = nib.Nifti1Image(
            prediction.astype(np.int32),
            reference_img.affine,
            reference_img.header
        )

        with tempfile.NamedTemporaryFile(
            suffix=".nii.gz",
            delete=False
        ) as tmp:
            prediction_path = tmp.name

        try:
            nib.save(prediction_img, prediction_path)

            with open(prediction_path, "rb") as f:
                pred_bytes = f.read()

        finally:
            if os.path.exists(prediction_path):
                os.remove(prediction_path)

    finally:
        if os.path.exists(reference_path):
            os.remove(reference_path)

    st.download_button(
        label="Download Segmentation (NIfTI)",
        data=pred_bytes,
        file_name="glioma_segmentation.nii.gz",
        mime="application/octet-stream",
    )


if __name__ == "__main__":
        main()