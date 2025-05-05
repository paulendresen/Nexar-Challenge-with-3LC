# Nexar Dashcam Kaggle Challenge

This repository contains a full pipeline for video-based event prediction training mvit_v2_s and the 3LC framework. Below is an overview of the workflow and a description of each main file.

This solution requires 3LC to be pip installed, see requirements.txt, and the 3LC service running for being able to use the UI to inspect training runs (and edit training data)

For more information, see https://docs.3lc.ai/

## Workflow Overview

1. **Preprocess Videos**: Extract frames from raw videos for training, validation, and testing.
2. **Create 3LC Tables**: Generate 3LC-compatible tables from the extracted frames.
3. **Train Model**: Train a deep learning model using the generated tables.
4. **Run Inference & Debug**: Run inference to capture metrics, debug training data, and iteratively improve the dataset.
5. **Final Inference & Submission**: Once satisfied, run inference on the test set and generate a `submission.csv` file for competition submission.

---

## File Descriptions for files in src

### 1. `preprocess_videos.py`
Extracts frames from raw video files and organizes them into directories for training, validation, and testing. This is the first step in the pipeline.

### 2. `create3LCTables.py`
Processes the extracted frames and creates 3LC-compatible tables (metadata and frame lists) required for training and evaluation. This script is run after frame extraction.

### 3. `main.py`
The main entry point for training the model. It loads the data, sets up the model, and starts the training process using the 3LC tables generated in the previous step.

### 4. `train.py`
Contains the core training loop, model checkpointing, and metric calculation logic. This file is called by `main.py` and handles the details of model optimization and evaluation.

### 5. `utils.py`
Utility functions and classes used throughout the pipeline, such as data transformations, frame loading, and mapping functions for 3LC tables.

### 6. `inference.py`
Runs inference on a dataset (training, validation or test) using a trained model. Captures embeddings, predictions, and metrics, and can be used to debug and analyze dataset and model performance. This information is used to understand how the model learns from the training data and actions are taken in the UI accordingly, in this case to delete and weight data.This is used iteratively to improve the dataset and model.

### 7. `submissionFromRun.py`
After the best training dataset is created and training is run and best model is selected based on validation dataset, this script runs inference on the test set and generates a `submission.csv` file in the required format for competition submission.

---

## Additional Notes
- Checkpoints and intermediate results are stored in the `checkpoints_newmodel/`
- The `src/` directory contains all main scripts; utility and helper files are also located here.
- The pipeline is designed for iterative improvement: you can repeat the data preparation, training, and inference steps as needed.

---

For more details on each script, see the comments and docstrings within the code files. 
