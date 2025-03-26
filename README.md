# HW1-Vision-Recognition-using-Deep-Learning-

# Introduction 

This project follows a modular structure to separate code, experiments, and outputs for better readability and maintainability.

`Notebooks/`: Contains exploratory analysis (data-exploration.ipynb) and experimental notebooks for training and evaluating models (model-c5.ipynb, model-c6.ipynb). It also includes prediction outputs in CSV format.
`src/`: Main source code directory, organized by functionality:
`block/`: Custom model components such as CBAM, SE, and Inception blocks.
`data/`: Scripts for loading and preparing datasets (make_dataset.py).
`features/`: Code for feature engineering or transformation logic.
`models/`: Training and inference scripts, including train_model.py and predict_model.py. Also includes a saved class-to-index mapping (class_to_idx.json).
`visualization/`: Helper functions for visualizing data, model predictions, and training metrics.

