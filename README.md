# Brain Tumor Segmentation Using 3D U-Net

### Lightweight Architecture on Normal CPUs

A lightweight 3D brain tumor segmentation application based on the **BraTS-Africa 2024** dataset and the 3D U-Net workflow provided in the Medical Artificial Intelligence (MAI) Lab's *Lightweight Brain Tumor Segmentation on Low-Resource Systems* tutorial.

This repository contains my implementation of the assigned tutorial workflow, including preprocessing, model inference, evaluation/visualization, and deployment as an interactive Streamlit application.

> **Important:** This project follows and implements the provided MAI Lab tutorial. The underlying tutorial, model architecture, and training workflow are not presented as my original research contribution. My work focused on completing the workflow, adapting the deployment stage into an interactive application, troubleshooting the implementation, and deploying it for practical use.

## Live Application

**Streamlit App:**
[https://lightweight-brain-tumor-segmentation-dqe2lqbhqpkfmdwgkpsekr.streamlit.app/](https://lightweight-brain-tumor-segmentation-hpeqgpkepytefzdrphp58g.streamlit.app/)

The deployed application allows users to upload the four required MRI modalities and generate a predicted brain tumor segmentation.

## Repository

**GitHub:**
https://github.com/Imoleoluwanimi/lightweight-brain-tumor-segmentation

---

## Project Overview

Brain tumor segmentation involves identifying and classifying tumor regions within multimodal MRI scans.

The assigned tutorial demonstrates an end-to-end workflow for developing a lightweight deep learning model for automated brain tumor segmentation using the **BraTS-Africa 2024 dataset**, with an emphasis on making the workflow accessible on systems with limited computational resources.

The workflow covered in this project follows four main stages:

1. Data collection, preparation and preprocessing
2. Building the 3D U-Net model
3. Training and evaluation
4. Deployment and practical usage

The deployment stage was adapted into a Streamlit application that can accept NIfTI MRI files and return a predicted segmentation.

---

## MRI Modalities

The application expects four MRI modalities:

* **T1n** — T1-native
* **T1c** — T1-contrast enhanced
* **T2f** — T2-FLAIR
* **T2w** — T2-weighted

A segmentation mask (`seg`) can also be uploaded optionally when comparison with a ground-truth mask is desired.

The four MRI modalities are combined as the model's input channels.

### Model Input

```text
(None, 64, 64, 64, 4)
```

### Model Output

```text
(None, 64, 64, 64, 4)
```

The trained model used in the application contains approximately **5.65 million parameters**.

---

## Preprocessing

The deployment application performs preprocessing before inference.

For each MRI modality:

1. The uploaded NIfTI file is loaded using NiBabel.
2. The image volume is normalized using `MinMaxScaler`.
3. The four modalities are checked to ensure that their dimensions match.
4. The modalities are stacked in the expected order:

```text
T1n → T1c → T2f → T2w
```

5. A fixed 3D region is extracted from the volume.
6. The cropped volume is downsampled by a factor of 2.
7. The resulting input is checked against the expected model input shape:

```text
64 × 64 × 64 × 4
```

This validation prevents an incorrectly shaped volume from being passed to the model.

---

## Patch Size Adaptation

The original tutorial workflow uses **96 × 96 × 96** 3D patches for model input during the training and prediction workflow.

For this implementation, the deployed model uses a smaller **64 × 64 × 64** spatial input with four MRI modalities:

```text
Original tutorial workflow:
96 × 96 × 96

This deployed implementation:
64 × 64 × 64 × 4
```

The change to a smaller input volume was used for the deployed lightweight application and is consistent with the model available in this implementation, which expects:

```text
(None, 64, 64, 64, 4)
```

The four MRI modalities are treated as the four input channels:

```text
T1n → channel 1
T1c → channel 2
T2f → channel 3
T2w → channel 4
```

This distinction is important because the deployment implementation is not simply a direct copy of the tutorial's original inference code. The deployment workflow was adapted around the trained model's expected input shape while retaining the overall preprocessing → inference → visualization → NIfTI download workflow described in the tutorial.

The smaller input volume also makes the inference workflow more suitable for the project's focus on running segmentation on ordinary CPU-based systems.


## Segmentation Inference

Once preprocessing is complete, the input volume receives a batch dimension:

```text
(64, 64, 64, 4)
        ↓
(1, 64, 64, 64, 4)
```

The trained 3D U-Net then generates a four-channel prediction.

This distinction is important because the deployment implementation is not simply a direct copy of the tutorial's original inference code. The deployment workflow was adapted around the trained model's expected input shape while retaining the overall preprocessing → inference → visualization → NIfTI download workflow described in the tutorial.

The deployment therefore uses the 64 × 64 × 64 input expected by the trained model while maintaining the project's focus on lightweight segmentation on ordinary CPU-based systems.
---

## Streamlit Application

The deployed application provides a simple interface for running inference.

### Workflow

1. Upload the four MRI modalities in NIfTI format.
2. Optionally upload the segmentation mask.
3. Click **Process and Predict**.
4. The application preprocesses the MRI volumes.
5. The trained 3D U-Net generates the segmentation.
6. The prediction is visualized alongside the MRI image.
7. The predicted segmentation can be downloaded as a NIfTI file.

### Supported files

```text
.nii
.nii.gz
```

The application does **not** require users to upload the dataset itself. Only the MRI volumes required for inference need to be uploaded.

---

## Visualization

The application displays selected slices of the processed T1c volume alongside the predicted segmentation.

When a ground-truth segmentation mask is provided, the application can also display the ground truth for comparison.

---

## Deployment

The application was deployed using:

* GitHub
* Streamlit Community Cloud
* Python
* TensorFlow
* Streamlit

The trained model is included in the repository as:

```text
default_model.keras
```

This allows the deployed application to load the trained model directly without requiring the training dataset to be included in the repository.

---

## Repository Structure

```text
lightweight-brain-tumor-segmentation/
│
├── app.py
│   └── Streamlit application for preprocessing,
│       inference, visualization and downloading predictions
│
├── deploy.py
│   └── Local deployment/inference workflow
│
├── bt_processing.ipynb
│   └── Data processing and preprocessing workflow
│
├── bt_segmentation.ipynb
│   └── Model development, training and evaluation workflow
│
├── default_model.keras
│   └── Trained 3D U-Net model used for inference
│
├── requirements.txt
│   └── Python dependencies
│
├── runtime.txt
│   └── Python runtime specification for deployment
│
└── .gitignore
    └── Files and directories excluded from version control
```

The original BraTS-Africa dataset is **not included** in this repository because of its size. The repository instead contains the code and trained model required for the deployed inference application.

---


## Model and Tutorial Attribution

This implementation is based on the tutorial:

**Lightweight Brain Tumor Segmentation on Low-Resource Systems: A Step-by-Step Guide with 3D U-Net**

The tutorial was collaboratively developed by research staff at the **Medical Artificial Intelligence (MAI) Lab, Lagos, Nigeria**, and participants of the **Sprint AI Training for African Medical Imaging Knowledge Translation (SPARK)**.

The tutorial provides an end-to-end workflow for brain tumor segmentation using the BraTS-Africa 2024 dataset and focuses on making the workflow accessible on systems with limited computational resources.

Tutorial DOI:

https://doi.org/10.17504/protocols.io.dm6gpdwmdgzp/v5

This repository should therefore be understood as an implementation and deployment of the assigned tutorial workflow rather than a claim of original authorship of the underlying methodology.

---

## Limitations

* The application performs inference using the trained model provided through the tutorial workflow.
* The deployed application is intended for demonstration and educational purposes.
* The model should not be interpreted as a clinically validated diagnostic system.
* The complete BraTS-Africa dataset is not included in this repository.
* Prediction quality depends on the uploaded MRI data being compatible with the preprocessing assumptions and modality format expected by the model.

---

## Acknowledgements

I acknowledge the **Medical Artificial Intelligence (MAI) Lab, Lagos, Nigeria**, and the contributors to the SPARK training program for the original tutorial, dataset workflow, and learning materials that formed the basis of this implementation.

---

## Disclaimer

This project is for educational and demonstration purposes only. It is **not intended for clinical diagnosis or medical decision-making**.
