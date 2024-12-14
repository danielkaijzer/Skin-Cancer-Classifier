"""

Skin Lesion Malignancy Probability Web App
Author: Daniel Kaijzer

"""

import os
import sys
import streamlit as st
import pandas as pd
import numpy as np
from PIL import Image
from pathlib import Path
import io
import json
import joblib
import cv2

# Get the relative paths using Path
APP_DIR = Path(__file__).resolve().parent
ROOT_DIR = APP_DIR.parent
SRC_DIR = ROOT_DIR / 'src'
MODELS_DIR = ROOT_DIR / 'models'

# Add src directory to path
if str(SRC_DIR) not in sys.path:
    sys.path.append(str(SRC_DIR))

from image_feature_extractor import create_masks, calculate_shape_features, calculate_color_features, analyze_lesion
from sklearn.preprocessing import OneHotEncoder
import lightgbm as lgb

# Page config
st.set_page_config(page_title="Skin Lesion Analysis", layout="wide")

# Debug mode checkbox
# debug_mode = st.sidebar.checkbox("Debug Mode", value=False)
debug_mode = False

# Load model and assets
model = joblib.load(MODELS_DIR / "model.pkl")
encoder = joblib.load(MODELS_DIR / "encoder.pkl")
with open(MODELS_DIR / "feature_columns.json", 'r') as f:
    cols_info = json.load(f)

# These are the original categorical columns for input to encoder
input_cat_cols = ['sex', 'anatom_site_general', 'tbp_lv_location', 'tbp_lv_location_simple']
# These are the one-hot encoded column names from the jsonco
output_cat_cols = cols_info['cat_cols']
new_feature_cols = cols_info['new_feature_cols']

if debug_mode:
    try:
        metadata_df = pd.read_csv('train-metadata.csv')
        st.sidebar.success("Loaded metadata for debugging")
    except FileNotFoundError:
        st.sidebar.error("train-metadata.csv not found!")
        debug_mode = False

def simplify_location(tbp_lv_location):
    parts = tbp_lv_location.split()
    return ' '.join(parts[:2])

# UI Layout
st.title("Skin Lesion Malignancy Probability")

# Create two columns
col1, col2 = st.columns(2)

with col1:
    if debug_mode:
        # Specify the exact test image
        selected_image_id = "ISIC_0082829"  # without .jpg extension
        try:
            sample_metadata = metadata_df[metadata_df['isic_id'] == selected_image_id].iloc[0]
            st.success(f"Using test image: {selected_image_id}")
        except IndexError:
            st.error(f"Test image {selected_image_id} not found in metadata")

        
        age_approx = st.number_input("Age Approximate (years)", 
                                   value=int(sample_metadata['age_approx']))
        clin_size_long_diam_mm = st.number_input("Longest Diameter of Lesion (mm)", 
                               value=float(sample_metadata['clin_size_long_diam_mm']))
        sex = st.selectbox("Sex", ["male", "female"], 
                        index=0 if sample_metadata['sex'] == 'male' else 1)
        anatom_site_general = st.selectbox("Anatomical Site",
                                        ["lower extremity", "head/neck", "posterior torso", 
                                        "anterior torso", "upper extremity"],
                                        index=["lower extremity", "head/neck", "posterior torso", 
                                            "anterior torso", "upper extremity"].index(sample_metadata['anatom_site_general']))
        tbp_lv_location = st.selectbox("Location (detailed)",
                                    metadata_df['tbp_lv_location'].unique().tolist(),
                                    index=metadata_df['tbp_lv_location'].unique().tolist().index(sample_metadata['tbp_lv_location']))
    else: 
        # User inputs
        age_approx = st.number_input("Age Approximate (years)", min_value=0, max_value=120, value=30, step=1)
        clin_size_long_diam_mm = st.number_input("Longest Diameter of Lesion (mm)", min_value=0.0, value=5.0, step=0.01, format="%.2f")
        sex = st.selectbox("Sex", ["male", "female"])
        anatom_site_general = st.selectbox(
            "Anatomical Site",
            ["lower extremity", "head/neck", "posterior torso", "anterior torso", "upper extremity"]
        )
        tbp_lv_location = st.selectbox(
            "Location (detailed)",
            [
                "Right Leg - Upper", "Head & Neck", "Torso Back Top Third", "Torso Front Top Half",
                "Right Arm - Upper", "Left Leg - Upper", "Torso Front Bottom Half", "Left Arm - Upper",
                "Right Leg", "Torso Back Middle Third", "Right Arm - Lower", "Right Leg - Lower",
                "Left Leg - Lower", "Left Arm - Lower", "Unknown", "Left Leg",
                "Torso Back Bottom Third", "Left Arm", "Right Arm", "Torso Front", "Torso Back"
            ]
        )
    
with col2:
    if debug_mode:
        st.write(f"Using metadata for image: {selected_image_id}")
        # Try to load and display the actual image if available
        image_path = Path('Sample_Images') / f"{selected_image_id}.jpg"
        if image_path.exists():
            img = Image.open(image_path)
            st.image(img, caption=f"Image: {selected_image_id}", use_container_width=True)
        else:
            st.warning("Image file not found in Sample_Images folder")
            
        if st.button("Analyze Using Metadata"):
            try:
                with st.spinner('Analyzing using metadata...'):
                    # Create dataframe exactly as in notebook
                    data_point = metadata_df[metadata_df['isic_id'] == selected_image_id].copy()
                    
                    # Calculate derived features
                    data_point['lesion_size_ratio'] = data_point['tbp_lv_minorAxisMM'] / data_point['clin_size_long_diam_mm']
                    data_point['lesion_shape_index'] = data_point['tbp_lv_areaMM2'] / (data_point['tbp_lv_perimeterMM'] ** 2)
                    data_point['hue_contrast'] = abs(data_point['tbp_lv_H'] - data_point['tbp_lv_Hext'])
                    data_point['luminance_contrast'] = abs(data_point['tbp_lv_L'] - data_point['tbp_lv_Lext'])
                    data_point['lesion_color_difference'] = np.sqrt(data_point['tbp_lv_deltaA']**2 + data_point['tbp_lv_deltaB']**2 + data_point['tbp_lv_deltaL']**2)
                    data_point['perimeter_to_area_ratio'] = data_point['tbp_lv_perimeterMM'] / data_point['tbp_lv_areaMM2']
                    data_point['area_to_perimeter_ratio'] = data_point['tbp_lv_areaMM2'] / data_point['tbp_lv_perimeterMM']
                    data_point['size_age_interaction'] = data_point['clin_size_long_diam_mm'] * data_point['age_approx']
                    data_point['color_contrast_index'] = data_point['tbp_lv_deltaA'] + data_point['tbp_lv_deltaB'] + data_point['tbp_lv_deltaL'] + data_point['tbp_lv_deltaLBnorm']
                    data_point['log_lesion_area'] = np.log(data_point['tbp_lv_areaMM2'] + 1)
                    data_point['normalized_lesion_size'] = data_point['clin_size_long_diam_mm'] / data_point['age_approx']
                    data_point['mean_hue_difference'] = (data_point['tbp_lv_H'] + data_point['tbp_lv_Hext']) / 2
                    data_point['std_dev_contrast'] = np.sqrt((data_point['tbp_lv_deltaA']**2 + data_point['tbp_lv_deltaB']**2 + data_point['tbp_lv_deltaL']**2) / 3)
                    data_point['overall_color_difference'] = (data_point['tbp_lv_deltaA'] + data_point['tbp_lv_deltaB'] + data_point['tbp_lv_deltaL']) / 3
                    data_point['size_color_contrast_ratio'] = data_point['clin_size_long_diam_mm'] / data_point['tbp_lv_deltaLBnorm']
                    data_point['color_range'] = abs(data_point['tbp_lv_L'] - data_point['tbp_lv_Lext']) + abs(data_point['tbp_lv_A'] - data_point['tbp_lv_Aext']) + abs(data_point['tbp_lv_B'] - data_point['tbp_lv_Bext'])
                    data_point['border_length_ratio'] = data_point['tbp_lv_perimeterMM'] / (2 * np.pi * np.sqrt(data_point['tbp_lv_areaMM2'] / np.pi))
                    
                    # Process categorical features
                    for col in input_cat_cols:  # Changed from cat_cols
                        data_point[col] = data_point[col].astype('category')

                    # Encode categorical features
                    encoded_cats = encoder.transform(data_point[input_cat_cols])  # Changed from cat_cols
                    encoded_df = pd.DataFrame(encoded_cats, columns=output_cat_cols, index=data_point.index)
                    encoded_df = encoded_df.astype('category')

                    # Drop original categorical columns and add encoded ones
                    data_point = data_point.drop(columns=input_cat_cols)
                    data_point = pd.concat([data_point, encoded_df], axis=1)

                    # Extract features in exact order
                    X_point = data_point[new_feature_cols]

                    # X_point.to_csv('Test_Prediction_Metadata.csv')

                    # Before prediction
                    st.write("Data point columns:", data_point.columns.tolist())
                    st.write("Looking for columns:", new_feature_cols)
                    
                    # Make prediction
                    prediction = model.predict_proba(X_point)[0][1]
                    actual_target = sample_metadata.get('target', 'Unknown')
                    
                    # Display results
                    st.markdown("---")
                    st.markdown("### Analysis Results")
                    
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.markdown("#### Predicted Probability")
                        st.markdown(f"<h2>{prediction*100:.1f}%</h2>", 
                                  unsafe_allow_html=True)
                    with col2:
                        st.markdown("#### Actual Label")
                        st.markdown(f"<h2>{actual_target}</h2>", 
                                  unsafe_allow_html=True)
                    with col3:
                        st.markdown("#### Prediction Error")
                        if actual_target != 'Unknown':
                            error = abs(prediction - float(actual_target))
                            st.markdown(f"<h2>{error*100:.1f}%</h2>", 
                                      unsafe_allow_html=True)
                    
                    # Debug information
                    # if st.checkbox("Show Debug Information"):
                    #     st.write("Input Features:", df)
                    #     st.write("Raw Metadata:", sample_metadata)
                    #     st.write("Feature Columns Used:", new_feature_cols)
                        
            except Exception as e:
                st.error(f"Error during analysis: {str(e)}")
                print(f"Full error traceback: {e.__class__.__name__}: {str(e)}")
    else:
    # Image upload
        st.write("Upload Lesion Image")
        uploaded_file = st.file_uploader("Image must be under 2000 pixels in width or height (JPG/PNG)", type=["jpg","jpeg","png"])
        
        if uploaded_file is not None:
            # Display uploaded image
            image_data = uploaded_file.read()
            img = Image.open(io.BytesIO(image_data)).convert("RGB")
            img_array = np.array(img)
            st.image(img_array, caption="Uploaded Image", use_container_width=True)
            
            # Validate image size
            height, width = img_array.shape[:2]
            # if height != width:
            #     st.error("⚠️ Image must be square (same width and height)")
            if height > 2000 or width > 2000:  
                st.error("⚠️ Image is too large. Please use an image 2000x2000 pixels or smaller")
            elif height < 90 or width < 90:  
                st.error("⚠️ Image is too small. Please use an image at least 127x127 pixels")
            elif age_approx <= 0:
                st.warning("⚠️ Please enter a valid age")
            elif clin_size_long_diam_mm <= 0:
                st.warning("⚠️ Please enter a valid lesion diameter")
            else:
                # Proceed with analysis
                if st.button("Analyze Lesion"):
                    try:
                        with st.spinner('Analyzing image...'):
                            # Derive simple location
                            tbp_lv_location_simple = simplify_location(tbp_lv_location)
                            
                            # Convert RGB to BGR for OpenCV processing
                            bgr_img = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)

                            # Process image - use BGR image for both operations
                            lab_img = cv2.cvtColor(bgr_img, cv2.COLOR_BGR2LAB)
                            contour, mask_bool, outside_mask = create_masks(bgr_img)

                            # Show contour (on RGB image)
                            display_img = img_array.copy()  # Use RGB for display
                            cv2.drawContours(display_img, [contour], -1, (0, 255, 0), 2)
                            st.image(display_img, caption="Detected Lesion Boundary", use_container_width=True)

                            # # segmentation mask
                            # mask_viz = np.zeros_like(img_array)
                            # mask_viz[mask_bool] = [0, 255, 0]  # Green for lesion
                            # mask_viz[outside_mask] = [255, 0, 0]  # Red for outside
                            # st.image(mask_viz, caption="Segmentation Mask (Green=Lesion, Red=Outside)", use_container_width=True)
                            
                            # Calculate features
                            _, _, w, h = cv2.boundingRect(contour)
                            max_diameter_pixels = max(w, h)
                            mm_per_pixel = clin_size_long_diam_mm / max_diameter_pixels
                            
                            shape_features = calculate_shape_features(contour, mm_per_pixel)
                            color_features = calculate_color_features(lab_img, mask_bool, outside_mask)
                            features = {**shape_features, **color_features}
                            
                            # Create DataFrame
                            data_point = pd.DataFrame([{
                                'age_approx': age_approx,
                                'clin_size_long_diam_mm': clin_size_long_diam_mm,
                                'sex': sex,
                                'anatom_site_general': anatom_site_general,
                                'tbp_lv_location': tbp_lv_location,
                                'tbp_lv_location_simple': tbp_lv_location_simple,
                                **features
                            }])
                            
                            # Calculate derived features
                            data_point['lesion_size_ratio'] = data_point['tbp_lv_minorAxisMM'] / data_point['clin_size_long_diam_mm']
                            data_point['lesion_shape_index'] = data_point['tbp_lv_areaMM2'] / (data_point['tbp_lv_perimeterMM'] ** 2)
                            data_point['hue_contrast'] = abs(data_point['tbp_lv_H'] - data_point['tbp_lv_Hext'])
                            data_point['luminance_contrast'] = abs(data_point['tbp_lv_L'] - data_point['tbp_lv_Lext'])
                            data_point['lesion_color_difference'] = np.sqrt(data_point['tbp_lv_deltaA']**2 + data_point['tbp_lv_deltaB']**2 + data_point['tbp_lv_deltaL']**2)
                            data_point['perimeter_to_area_ratio'] = data_point['tbp_lv_perimeterMM'] / data_point['tbp_lv_areaMM2']
                            data_point['area_to_perimeter_ratio'] = data_point['tbp_lv_areaMM2'] / data_point['tbp_lv_perimeterMM']
                            data_point['size_age_interaction'] = data_point['clin_size_long_diam_mm'] * data_point['age_approx']
                            data_point['color_contrast_index'] = data_point['tbp_lv_deltaA'] + data_point['tbp_lv_deltaB'] + data_point['tbp_lv_deltaL'] + data_point['tbp_lv_deltaLBnorm']
                            data_point['log_lesion_area'] = np.log(data_point['tbp_lv_areaMM2'] + 1)
                            data_point['normalized_lesion_size'] = data_point['clin_size_long_diam_mm'] / data_point['age_approx']
                            data_point['mean_hue_difference'] = (data_point['tbp_lv_H'] + data_point['tbp_lv_Hext']) / 2
                            data_point['std_dev_contrast'] = np.sqrt((data_point['tbp_lv_deltaA']**2 + data_point['tbp_lv_deltaB']**2 + data_point['tbp_lv_deltaL']**2) / 3)
                            data_point['overall_color_difference'] = (data_point['tbp_lv_deltaA'] + data_point['tbp_lv_deltaB'] + data_point['tbp_lv_deltaL']) / 3
                            data_point['size_color_contrast_ratio'] = data_point['clin_size_long_diam_mm'] / data_point['tbp_lv_deltaLBnorm']
                            data_point['color_range'] = abs(data_point['tbp_lv_L'] - data_point['tbp_lv_Lext']) + abs(data_point['tbp_lv_A'] - data_point['tbp_lv_Aext']) + abs(data_point['tbp_lv_B'] - data_point['tbp_lv_Bext'])
                            data_point['border_length_ratio'] = data_point['tbp_lv_perimeterMM'] / (2 * np.pi * np.sqrt(data_point['tbp_lv_areaMM2'] / np.pi))
                            
                            # Process categorical features
                            for col in input_cat_cols:  # Changed from cat_cols
                                data_point[col] = data_point[col].astype('category')

                            # Encode categorical features
                            encoded_cats = encoder.transform(data_point[input_cat_cols])  # Changed from cat_cols
                            encoded_df = pd.DataFrame(encoded_cats, columns=output_cat_cols, index=data_point.index)
                            encoded_df = encoded_df.astype('category')

                            # Drop original categorical columns and add encoded ones
                            data_point = data_point.drop(columns=input_cat_cols)
                            data_point = pd.concat([data_point, encoded_df], axis=1)

                            # Extract features in exact order
                            X_point = data_point[new_feature_cols]

                            # X_point.to_csv('X_point_streamlit_nondebug.csv')

                            # Make prediction
                            prediction = model.predict_proba(X_point)[0][1]
                            
                            # Display results with appropriate styling
                            st.markdown("---")
                            st.markdown("### Analysis Results")
                            
                            # Determine risk level and color
                            if prediction < 0.3:
                                risk_level = "Low Risk"
                                color = "green"
                            elif prediction < 0.7:
                                risk_level = "Medium Risk"
                                color = "orange"
                            else:
                                risk_level = "High Risk"
                                color = "red"
                            
                            # Display results
                            col1, col2 = st.columns(2)
                            with col1:
                                st.markdown(f"#### Malignancy Probability")
                                st.markdown(f"<h2 style='color: {color}'>{prediction*100:.1f}%</h2>", unsafe_allow_html=True)
                            with col2:
                                st.markdown(f"#### Risk Level")
                                st.markdown(f"<h2 style='color: {color}'>{risk_level}</h2>", unsafe_allow_html=True)
                            
                            if risk_level == "High Risk":
                                st.warning("⚠️ Please consult a healthcare professional for further evaluation.")
                            
                    except Exception as e:
                        st.error(f"Error during analysis: {str(e)}")
                        print(f"Full error traceback: {e.__class__.__name__}: {str(e)}")

# Add footer
st.markdown("---")
st.markdown("""
    <div style='text-align: center;'>
        <p>This tool is for educational purposes only and should not be used as a substitute for professional medical advice.</p>
    </div>
    """, unsafe_allow_html=True)