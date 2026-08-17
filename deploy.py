import os
import numpy as np
import nibabel as nib
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import load_model


MODEL_PATH = "default_model.keras"
PATCH_SIZE = (64, 64, 64)


model = load_model(MODEL_PATH, compile=False)

print("Model loaded successfully!")
print("Input shape:", model.input_shape)
print("Output shape:", model.output_shape)


def load_modality(file_path):
    image = nib.load(file_path).get_fdata()

    scaler = MinMaxScaler()
    image = scaler.fit_transform(
        image.reshape(-1, 1)
    ).reshape(image.shape)

    return image


def preprocess_nifti(t1n_file, t1c_file, t2f_file, t2w_file):
    t1n = load_modality(t1n_file)
    t1c = load_modality(t1c_file)
    t2f = load_modality(t2f_file)
    t2w = load_modality(t2w_file)

    if not (t1n.shape == t1c.shape == t2f.shape == t2w.shape):
        raise ValueError("All four MRI modalities must have the same shape.")

    image = np.stack([t1n, t1c, t2f, t2w], axis=-1)

    x, y, z, _ = image.shape
    px, py, pz = PATCH_SIZE

    if x < px or y < py or z < pz:
        raise ValueError(
            f"Input volume {image.shape[:3]} is smaller than {PATCH_SIZE}."
        )

    start_x = (x - px) // 2
    start_y = (y - py) // 2
    start_z = (z - pz) // 2

    image = image[
        start_x:start_x + px,
        start_y:start_y + py,
        start_z:start_z + pz,
        :
    ]

    return np.expand_dims(image, axis=0)


def run_segmentation(
    model,
    t1n_file,
    t1c_file,
    t2f_file,
    t2w_file,
    output_folder
):
    input_image = preprocess_nifti(
        t1n_file,
        t1c_file,
        t2f_file,
        t2w_file
    )

    prediction = model.predict(input_image, verbose=0)
    prediction = np.argmax(prediction, axis=-1)[0].astype(np.int32)

    os.makedirs(output_folder, exist_ok=True)

    reference = nib.load(t1n_file)

    output_file = os.path.join(
        output_folder,
        "glioma_segmentation.nii.gz"
    )

    output_image = nib.Nifti1Image(
        prediction,
        reference.affine,
        reference.header
    )

    nib.save(output_image, output_file)

    print(f"Segmentation saved to: {output_file}")


if __name__ == "__main__":
    run_segmentation(
        model,
        "t1n.nii.gz",
        "t1c.nii.gz",
        "t2f.nii.gz",
        "t2w.nii.gz",
        "output"
    )