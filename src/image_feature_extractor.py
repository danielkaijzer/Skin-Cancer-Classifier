"""

Image Feature Extrator (python file)
Author: Daniel Kaijzer

"""


import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from skimage.filters import gaussian
from skimage.segmentation import active_contour


def create_masks(img):
    """Create lesion and surrounding area masks."""

    # Double check BGR conversion (for web app to work properly)
    # check if image is BGR or RGB and convert to BGR if needed
    if len(img.shape) == 3 and img.shape[2] == 3:
        # Check if image is already BGR
        if isinstance(img, np.ndarray) and img.dtype == np.uint8:
            bgr_img = img
        else:
            # Convert RGB to BGR
            bgr_img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    else:
        raise ValueError("Image must be a 3-channel color image")


    # Convert to LAB
    lab_img = cv2.cvtColor(bgr_img, cv2.COLOR_BGR2LAB)
    L = lab_img[:,:,0]
    A = lab_img[:,:,1]
    B = lab_img[:,:,2]  # Add B channel
    
    # Threshold L channel for dark spots
    thresh_L = np.percentile(L, 20)
    _, binary_L = cv2.threshold(L, thresh_L, 255, cv2.THRESH_BINARY_INV)
    
    # Threshold A channel for reddish areas
    _, binary_A = cv2.threshold(A, 128, 255, cv2.THRESH_BINARY)
    
    # Threshold B channel using mean
    _, binary_B = cv2.threshold(B, np.mean(B), 255, cv2.THRESH_BINARY)
    
    # Combine conditions
    binary = cv2.bitwise_and(binary_L, binary_A)
    
    # Use larger kernel for morphological operations
    kernel = np.ones((7,7), np.uint8)
    binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)
    binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
    
    # Find contours
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    
    # Filter by size and compactness
    min_area = 200
    valid_contours = []
    for c in contours:
        area = cv2.contourArea(c)
        perimeter = cv2.arcLength(c, True)
        if area > min_area:
            compactness = 4 * np.pi * area / (perimeter * perimeter)
            if compactness > 0.15:
                valid_contours.append(c)
    
    if not valid_contours:
        raise ValueError("No valid lesion contours found")
    
    # Get darkest contour
    contour = min(valid_contours, key=lambda c: np.mean(L[cv2.drawContours(
        np.zeros_like(L), [c], -1, 255, cv2.FILLED) > 0]))
    
    # Create masks
    mask = np.zeros_like(L)
    cv2.drawContours(mask, [contour], -1, 255, -1)
    mask_bool = mask > 0
    
    # Create dilated mask for outside region using larger kernel
    kernel = np.ones((5,5), np.uint8)
    dilated_mask = cv2.dilate(mask, kernel, iterations=1)
    outside_mask = (dilated_mask > 0) & (~mask_bool)
    
    return contour, mask_bool, outside_mask

def calculate_shape_features(contour, mm_per_pixel):
    """Calculate shape-related features."""
    # Get rotated rectangle for better axis measurements
    rect = cv2.minAreaRect(contour)
    (x, y), (width, height), angle = rect
    
    # Calculate area and perimeter
    area_pixels = cv2.contourArea(contour)
    area_mm2 = area_pixels * (mm_per_pixel ** 2)
    perimeter_pixels = cv2.arcLength(contour, True)
    perimeter_mm = perimeter_pixels * mm_per_pixel
    
    # Calculate moments for axis and eccentricity
    moments = cv2.moments(contour)
    
    if moments['m00'] == 0:
        return None
        
    # Central moments
    mu20 = moments['mu20'] / moments['m00']
    mu02 = moments['mu02'] / moments['m00']
    mu11 = moments['mu11'] / moments['m00']
    
    # Calculate eigenvalues for axes
    delta = np.sqrt((mu20 - mu02)**2 + 4*mu11**2)
    major_axis = 2 * np.sqrt(2 * (mu20 + mu02 + delta)) * mm_per_pixel
    minor_axis = 2 * np.sqrt(2 * (mu20 + mu02 - delta)) * mm_per_pixel
    
    # Calculate eccentricity
    lambda1 = (mu20 + mu02 + delta) / 2
    lambda2 = (mu20 + mu02 - delta) / 2
    eccentricity = np.sqrt(1 - (lambda2 / lambda1)) if lambda1 != 0 else 0
    
    area_perim_ratio = (perimeter_mm ** 2) / (area_mm2)
    
    return {
        'tbp_lv_areaMM2': area_mm2,
        'tbp_lv_perimeterMM': perimeter_mm,
        'tbp_lv_minorAxisMM': minor_axis, # Axis of least second moment
        'tbp_lv_eccentricity': eccentricity,
        'tbp_lv_area_perim_ratio': area_perim_ratio
    }

def calculate_color_features(lab_img, mask_bool, outside_mask):
    """Calculate color-related features."""
    # Split LAB channels
    L, A, B = cv2.split(lab_img)
    
    # Normalize L to 0-100 range, A and B to -128 to +127
    L_raw = L
    L = L * (100/255)
    A = A - 128
    B = B - 128
    
    # Calculate means for inside lesion
    L_in = np.mean(L[mask_bool]) 
    A_in = np.mean(A[mask_bool])
    B_in = np.mean(B[mask_bool])
    
    # Calculate means for outside lesion
    L_ext = np.mean(L[outside_mask]) 
    A_ext = np.mean(A[outside_mask])
    B_ext = np.mean(B[outside_mask])

    # Calculate deltas
    deltaL = L_in - L_ext
    deltaA = A_in - A_ext
    deltaB = B_in - B_ext

    # Calculate deltaLBnorm
    deltaLBnorm = np.sqrt(deltaL**2 + deltaB**2)
    
    # Calculate standard deviations
    # stdL_in = np.std(L[mask_bool])
    # stdL_ext = np.std(L[outside_mask])
    
    # Calculate Hue (degrees)
    H_in = np.degrees(np.arctan2(B_in, A_in))
    H_in = H_in + 360 if H_in < 0 else H_in
    
    H_ext = np.degrees(np.arctan2(B_ext, A_ext))
    H_ext = H_ext + 360 if H_ext < 0 else H_ext

    # Calculate Chroma
    C_in = np.sqrt(A_in**2 + B_in**2)
    C_ext = np.sqrt(A_ext**2 + B_ext**2)
    
    # Calculate color irregularity with normalization
    # color_std_mean = np.mean([
    #     np.std(L[mask_bool])/100,
    #     np.std(A[mask_bool])/127,
    #     np.std(B[mask_bool])/127
    # ]) * 0.443  # Scale factor to match reference values

    # Modified color irregularity calculation
    # Weight L channel less and increase overall scale
    # color_std_mean = np.mean([
    #     np.std(L[mask_bool])/200,      # Reduced L channel weight
    #     np.std(A[mask_bool])/127 * 2,  # Increased A channel weight
    #     np.std(B[mask_bool])/127 * 2   # Increased B channel weight
    # ]) * 0.443  # Original scale factor
    # Calculate normalized standard deviations for each channel
    # avoid division by zero
    # L_std_norm = np.std(L[mask_bool]) / np.mean(L[mask_bool])
    # A_std_norm = np.std(A[mask_bool]) / (np.std(A[outside_mask]) + 1e-6)
    # B_std_norm = np.std(B[mask_bool]) / (np.std(B[outside_mask]) + 1e-6)
    
    # Combine with empirically determined weights
    # color_std_mean = (L_std_norm * 0.4 + A_std_norm * 0.3 + B_std_norm * 0.3) * 0.443 * 2.0
    
    # Ensure the value is in the expected range
    # color_std_mean = np.clip(color_std_mean, 0, 1) * 0.443
    
    return {
        'tbp_lv_L': L_in,
        'tbp_lv_Lext': L_ext,
        'tbp_lv_A': A_in,
        'tbp_lv_Aext': A_ext,
        'tbp_lv_B': B_in,
        'tbp_lv_Bext': B_ext,
        'tbp_lv_C': C_in,
        'tbp_lv_Cext': C_ext,
        'tbp_lv_H': H_in,
        'tbp_lv_Hext': H_ext,
        'tbp_lv_deltaL': deltaL,
        'tbp_lv_deltaA': deltaA,
        'tbp_lv_deltaB': deltaB,
        # 'tbp_lv_stdL': stdL_in,
        # 'tbp_lv_stdLExt': stdL_ext,
        'tbp_lv_deltaLBnorm': deltaLBnorm,
        # 'tbp_lv_color_std_mean': color_std_mean
    }

def visualize_analysis(img, lab_img, contour, mask_bool, outside_mask):
    """Create visualization plots for the analysis process."""
    L, A, B = cv2.split(lab_img)
    
    # Create figure for all visualizations
    plt.figure(figsize=(20,10))
    
    # Original image
    plt.subplot(231)
    plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    plt.title('Original Image')
    plt.axis('off')
    
    # L and A channels
    plt.subplot(232)
    plt.imshow(L, cmap='gray')
    plt.title('L Channel')
    plt.colorbar()
    plt.axis('off')
    
    plt.subplot(233)
    plt.imshow(A, cmap='gray')
    plt.title('A Channel')
    plt.colorbar()
    plt.axis('off')
    
    # Draw contour on original image
    img_with_contour = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).copy()
    cv2.drawContours(img_with_contour, [contour], -1, (255,0,0), 2)
    plt.subplot(234)
    plt.imshow(img_with_contour)
    plt.title('Detected Lesion Contour')
    plt.axis('off')
    
    # Show masks
    mask_viz = np.zeros_like(img)
    mask_viz[mask_bool] = [0,255,0]  # Green for lesion
    mask_viz[outside_mask] = [255,0,0]  # Red for outside region
    plt.subplot(235)
    plt.imshow(mask_viz)
    plt.title('Masks (Green=Lesion, Red=Outside)')
    plt.axis('off')
    
    plt.tight_layout()
    plt.show()
    
    # Show color distributions
    plt.figure(figsize=(15,5))
    
    plt.subplot(131)
    plt.hist(L[mask_bool].flatten(), bins=50, alpha=0.5, label='Inside')
    plt.hist(L[outside_mask].flatten(), bins=50, alpha=0.5, label='Outside')
    plt.title('L Channel Distribution')
    plt.legend()
    
    plt.subplot(132)
    plt.hist(A[mask_bool].flatten(), bins=50, alpha=0.5, label='Inside')
    plt.hist(A[outside_mask].flatten(), bins=50, alpha=0.5, label='Outside')
    plt.title('A Channel Distribution')
    plt.legend()
    
    plt.subplot(133)
    plt.hist(B[mask_bool].flatten(), bins=50, alpha=0.5, label='Inside')
    plt.hist(B[outside_mask].flatten(), bins=50, alpha=0.5, label='Outside')
    plt.title('B Channel Distribution')
    plt.legend()
    
    plt.tight_layout()
    plt.show()

def analyze_feature_differences(original_values, calculated_values):
    """
    Helper function to debug feature extraction differences.
    Use during development/testing.
    """
    significant_diffs = {}
    for feature in original_values:
        orig = original_values[feature]
        calc = calculated_values[feature]
        rel_diff = abs(orig - calc) / abs(orig) if orig != 0 else abs(calc)
        if rel_diff > 0.05:
            significant_diffs[feature] = {
                'original': orig,
                'calculated': calc,
                'relative_diff': rel_diff * 100
            }
    return significant_diffs

def analyze_lesion(image_path, diameter_mm):
    """Main function to analyze a lesion image."""
    # Read image
    img = cv2.imread(str(image_path))
    if img is None:
        raise ValueError(f"Could not read image: {image_path}")
    
    # Convert to LAB
    lab_img = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    
    # Create masks and get contour
    contour, mask_bool, outside_mask = create_masks(img)
    
    # Calculate scale factor
    _, _, w, h = cv2.boundingRect(contour)
    max_diameter_pixels = max(w, h)
    mm_per_pixel = diameter_mm / max_diameter_pixels
    
    # Calculate features
    shape_features = calculate_shape_features(contour, mm_per_pixel)
    color_features = calculate_color_features(lab_img, mask_bool, outside_mask)
    
    # Visualize results
    visualize_analysis(img, lab_img, contour, mask_bool, outside_mask)
    
    # Combine all features
    features = {**shape_features, **color_features}
    
    return features


def main():
    """
    Main execution function with user input for image selection.
    """
    # Setup paths
    # train_path = Path('train-metadata.csv')
    # image_folder = Path('Test_Images')
    src_dir = Path(__file__).resolve().parent
    root_dir = src_dir.parent
    train_path = root_dir / 'data' / 'test-metadata.csv'
    image_folder = root_dir / 'Test_Images'  

    # Get user input for image file
    print("\nAvailable images in Test_Images folder:")
    try:
        image_files = list(image_folder.glob('*.jpg')) 
        for idx, file in enumerate(image_files, 1):
            print(f"{idx}. {file.name}")
        
        # Get user selection
        while True:
            try:
                selection = input("\nEnter the number of the image you want to analyze (or 'q' to quit): ")
                if selection.lower() == 'q':
                    return
                
                idx = int(selection) - 1
                if 0 <= idx < len(image_files):
                    image_path = image_files[idx]
                    break
                else:
                    print("Invalid selection. Please try again.")
            except ValueError:
                print("Please enter a valid number or 'q' to quit.")
    except Exception as e:
        raise Exception(f"Error accessing image folder: {e}")

    # Extract image ID from filename
    image_id = image_path.stem

    # Read metadata
    try:
        metadata_df = pd.read_csv(train_path)
    except FileNotFoundError:
        raise FileNotFoundError(f"Metadata file {train_path} not found.")

    # Locate metadata for the specific image
    try:
        sample_metadata = metadata_df[metadata_df['isic_id'] == image_id].iloc[0]
    except IndexError:
        raise ValueError(f"Metadata for image ID {image_id} not found in {train_path}.")

    # Get diameter
    diameter_mm = sample_metadata['clin_size_long_diam_mm']
    print(f"\nAnalyzing image: {image_path.name}")
    print(f"Lesion diameter: {diameter_mm}mm")

    # Calculate features
    features = analyze_lesion(image_path, diameter_mm)

    # Print comparison
    print("\nFeature Comparison:")
    print("-" * 50)
    for col in features.keys():
        if col in sample_metadata:
            calc_val = features[col]
            orig_val = sample_metadata[col]
            print(f"\n{col}:")
            print(f"  Calculated: {calc_val:.4f}")
            print(f"  Original:   {orig_val:.4f}")
            print(f"  Difference: {abs(calc_val - orig_val):.4f}")

if __name__ == "__main__":
    main()