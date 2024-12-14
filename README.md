# Skin Cancer Classifier Project

## Author
Daniel Kaijzer

## Project Overview
This project is a web application that allows users to upload an image of a skin lesion and receive a prediction of the probability of the lesion being malignant. The application uses a machine learning model that was trained on a dataset of skin lesion images and associated metadata to make the predictions.

## Features
- Upload an image of a skin lesion
- Provide additional information about the lesion, such as the patient's age and the lesion location
- Receive a prediction of the probability of the lesion being malignant, along with a risk level (low, medium, or high)

## Usage
1. Launch the Streamlit application using `streamlit run webapp.py` 
2. In the left column, enter the required information about the skin lesion, including the patient's age, the lesion's diameter, the patient's sex, and the lesion's anatomical location.
3. In the right column, upload an image of the skin lesion. The image must be square and between 90x90 or 2000x2000 pixels. It doesn't need to be square but will work better if more square.
4. Click the "Analyze Lesion" button to initiate the analysis.
5. The application will display the predicted probability of the lesion being malignant, the mask calculated by my `image_feature_extractor.py` file and then a risk level (low, medium, or high). 
6. To use `image_feature_extractor.py` by itself requires downloading train-metadata.csv from the ISIC 2024 dataset: https://www.kaggle.com/competitions/isic-2024-challenge/data . Once downloaded you can run the program using `python image_feature_extractor.py`. Then you will see the differences between the feature values extracted from my program and ground truth data from the CSV.


## Project Structure
- `requirements.txt`: Lists the required Python packages for running the project
- `Image_Feature_Extractor.py`: Uses OpenCV to extract tabular geometric and color features from - image files directly
- `webapp.py`: Streamlit app that uses the tabular-only model for predictions
- `EDA.ipynb`: Jupyter Notebook containing basic exploratory data analysis on the training data CSV file
- `model.ipynb`: Jupyter Notebook documenting the model creation process and comparing the performance of the tabular-only and hybrid models
- `model.pkl`: Serialized tabular-only model
- `feature_columns.json`: Required for the Streamlit app to perform inference properly
- `encoder.pkl`: Required for the Streamlit app to perform inference properly
- `Sample_Images/`: Folder containing sample images from holdout test set for testing the web application. These images have not been seen by my model before inference.
- `Skin Cancer Classifier Presentation.pdf`: My presentation detailing this project.

## Model Development
The `EDA.ipynb` notebook contains basic exploratory data analysis performed on the training data CSV file. The `model.ipynb` notebook documents the process of creating the ML models and compares the performance of the tabular-only model and the hybrid model that combines tabular data with a CNN–the CNN creates an embedding matrix that is combined with tabular data for retraining my LGBM-based model.

## Model Architecture
1. Tabular-only Model:
- This model uses only the tabular data (patient metadata and derived features) for prediction.
- It's an ensemble of LightGBM models (VotingClassifier) with different random states for diversity.
- Each LightGBM model is wrapped in a Pipeline with a RandomUnderSampler to handle class imbalance.
- The models are trained using cross-validation with StratifiedGroupKFold to ensure proper data splitting.
- Feature importances are calculated by averaging importances across all folds and models.


2. Hybrid Model (Tabular + CNN):

- This model combines the tabular data with embeddings generated by a CNN.
- The CNN (EfficientNet-B0) is pre-trained on ImageNet and fine-tuned on a balanced subset of your training data.
- The CNN generates a 1792-dimensional embedding for each image.
- The image embeddings are combined with the tabular features to create an enhanced feature set.
- The same ensemble of LightGBM models is then trained on this enhanced feature set.
- The CNN architecture and training process is defined in the train_embedding_model_with_hdf5 function.
- NOTE: I have decided not use the model in my webapp because it does not perform as well (see slides for comparison). The code to build the model is in model.ipynb.

## Image Feature Extractor program
This program uses OpenCV to extract various geometric and color features directly from the image files. This enables the streamlit web app to function without requiring too much user input. Here's a breakdown of its main components:
1. create_masks function:
- Converts the image to the LAB color space.
- Applies thresholding on the L, A, and B channels to identify potential lesion regions.
- Performs morphological operations (opening and closing) to refine the binary mask.
- Selects the darkest contour as the lesion region.
- Creates lesion and surrounding area masks.


2. calculate_shape_features function:
- Calculates shape-related features like area, perimeter, minor axis length, eccentricity, and area-perimeter ratio.
- Uses OpenCV functions like contourArea, arcLength, minAreaRect, and moments for feature extraction.


3. calculate_color_features function:
- Calculates color-related features in the LAB color space.
- Computes means and differences of L, A, and B values inside and outside the lesion.
- Calculates derived features like hue, chroma, and color differences.


4. visualize_analysis function:
- Creates visualizations to illustrate the analysis process.
- Displays the original image, L and A channels, detected lesion contour, and masks.
- Plots color distributions inside and outside the lesion.


5. analyze_lesion function:
- Main function that orchestrates the entire analysis pipeline.
- Reads the image, creates masks, calculates scale factor, and computes shape and color features.
- Calls the visualize_analysis function to generate visualizations.
The image_feature_extractor.py file provides a comprehensive set of features that capture important characteristics of skin lesions. These features are then used along with patient metadata to train the ML models for malignancy prediction.


## Requirements
* See requirements.txt file

## Acknowledgments
The datasets used for training the machine learning model was obtained from the ISIC Archive.
My tabular model builds on the Kaggle notebook made by Farukcan Saglam: https://www.kaggle.com/code/greysky/isic-2024-only-tabular-data
