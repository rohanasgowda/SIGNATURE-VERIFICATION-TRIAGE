import cv2
import numpy as np
import matplotlib.pyplot as plt
import os
from glob import glob
from tqdm import tqdm
import seaborn as sns
import joblib

# Feature Extraction
from skimage.feature import local_binary_pattern, hog
from skimage import exposure  # Added for HOG visualization

# Preprocessing and Feature Selection
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

# Models
from xgboost import XGBClassifier

# Metrics
from sklearn.metrics import classification_report, confusion_matrix, roc_curve, auc

# --- CONFIGURATION ---
DATA_FOLDER = "my_custom_data"  # Points to your 12-user folder
MODEL_PREFIX = "custom"         # naming for files (custom_plot_1...)

def preprocess_image(image_path, target_size=(1000, 500)):
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        print(f"Warning: Could not read image at {image_path}. Skipping.")
        return None
    img_resized = cv2.resize(img, target_size, interpolation=cv2.INTER_AREA)
    img_filtered = cv2.GaussianBlur(img_resized, (5, 5), 0)
    _, img_binary = cv2.threshold(img_filtered, 0, 255, 
                                 cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    return img_binary

def extract_features(binary_image):
    # LBP
    P = 8
    R = 1
    lbp = local_binary_pattern(binary_image, P, R, method="uniform")
    (lbp_hist, _) = np.histogram(lbp.ravel(), bins=np.arange(0, P + 3), range=(0, P + 2))
    lbp_hist = lbp_hist.astype("float")
    lbp_hist /= (lbp_hist.sum() + 1e-6)
    
    # HOG
    hog_features = hog(binary_image, pixels_per_cell=(32, 32),
                       cells_per_block=(2, 2), visualize=False, block_norm='L2-Hys')
    
    final_feature_vector = np.hstack([lbp_hist, hog_features])
    return final_feature_vector


# --- Main execution block ---
if __name__ == "__main__":
    
    all_features = []
    all_labels = []
    
    example_genuine_path = None
    example_forged_path = None

    # 1. LOAD GENUINE
    print("--- Processing Custom Genuine Signatures ---")
    genuine_files = glob(f"{DATA_FOLDER}/genuine/*.jpg") + glob(f"{DATA_FOLDER}/genuine/*.jpeg") + glob(f"{DATA_FOLDER}/genuine/*.png")

    if not genuine_files:
        print(f"Warning: No genuine images found! Check the '{DATA_FOLDER}/genuine' folder.")
        
    for image_path in tqdm(genuine_files, desc="Genuine"):
        if example_genuine_path is None: example_genuine_path = image_path
        preprocessed_img = preprocess_image(image_path)
        if preprocessed_img is not None:
            features = extract_features(preprocessed_img)
            all_features.append(features)
            all_labels.append(0) # 0 for Genuine

    # 2. LOAD FORGED
    print("\n--- Processing Custom Forged Signatures ---")
    forged_files = glob(f"{DATA_FOLDER}/forged/*.jpg") + glob(f"{DATA_FOLDER}/forged/*.jpeg") + glob(f"{DATA_FOLDER}/forged/*.png")

    if not forged_files:
        print(f"Warning: No forged images found! Check the '{DATA_FOLDER}/forged' folder.")
        
    for image_path in tqdm(forged_files, desc="Forged"):
        if example_forged_path is None: example_forged_path = image_path
        preprocessed_img = preprocess_image(image_path)
        if preprocessed_img is not None:
            features = extract_features(preprocessed_img)
            all_features.append(features)
            all_labels.append(1) # 1 for Forged
    
    if len(all_features) == 0:
        print("\nFATAL ERROR: No images were processed. Cannot train models. Exiting.")
        exit()

    # ==========================================
    # VISUALIZATION SECTION
    # ==========================================
    print("\n--- Generating Visualization Plots ---")
    
    if example_genuine_path:
        # Load raw and binary for plotting
        raw_img = cv2.imread(example_genuine_path, cv2.IMREAD_GRAYSCALE)
        bin_img = preprocess_image(example_genuine_path)
        
        # --- PLOT 1: Preprocessing ---
        plt.figure(figsize=(10, 5))
        plt.subplot(1, 2, 1); plt.imshow(raw_img, cmap='gray'); plt.title('Step 1: Raw Input')
        plt.subplot(1, 2, 2); plt.imshow(bin_img, cmap='gray'); plt.title('Step 2: Preprocessed (Binary)')
        plt.savefig(f'{MODEL_PREFIX}_plot_1_preprocessing.png')
        print(f"Saved {MODEL_PREFIX}_plot_1_preprocessing.png")

        # --- PLOT 2: LBP Texture Map ---
        lbp_viz = local_binary_pattern(bin_img, 8, 1, method="uniform")
        plt.figure(figsize=(6, 6))
        plt.imshow(lbp_viz, cmap='nipy_spectral')
        plt.title('Step 3a: LBP Texture Features')
        plt.axis('off')
        plt.savefig(f'{MODEL_PREFIX}_plot_2_lbp_texture.png')
        print(f"Saved {MODEL_PREFIX}_plot_2_lbp_texture.png")

        # --- PLOT 3: HOG Gradient Map ---
        _, hog_viz = hog(bin_img, pixels_per_cell=(32, 32),
                         cells_per_block=(2, 2), visualize=True, block_norm='L2-Hys')
        hog_viz_rescaled = exposure.rescale_intensity(hog_viz, in_range=(0, 10))
        plt.figure(figsize=(6, 6))
        plt.imshow(hog_viz_rescaled, cmap='gray')
        plt.title('Step 3b: HOG Features')
        plt.axis('off')
        plt.savefig(f'{MODEL_PREFIX}_plot_3_hog_features.png')
        print(f"Saved {MODEL_PREFIX}_plot_3_hog_features.png")

    # ==========================================
    # TRAINING SECTION
    # ==========================================
    X = np.array(all_features)
    y = np.array(all_labels)
    
    print(f"\n--- Custom Dataset Created: {X.shape} samples ---")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    print("Scaling Custom Data...")
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # --- PLOT 4: PCA Variance ---
    print("Running PCA...")
    pca = PCA(n_components=0.95) 
    pca.fit(X_train_scaled)
    X_train_pca = pca.transform(X_train_scaled)
    X_test_pca = pca.transform(X_test_scaled)
    print(f"Reduced features (PCA): {X_train_pca.shape[1]}")
    
    plt.figure(figsize=(8, 5))
    plt.plot(np.cumsum(pca.explained_variance_ratio_))
    plt.xlabel('Number of Components'); plt.ylabel('Variance Explained')
    plt.title('Step 4: PCA Feature Reduction')
    plt.grid(True)
    plt.savefig(f'{MODEL_PREFIX}_plot_4_pca_variance.png')
    print(f"Saved {MODEL_PREFIX}_plot_4_pca_variance.png")

    # --- PLOT 5: 2D Scatter Plot ---
    # Visualizing the 12 users vs forgeries in 2D space
    pca_2d = PCA(n_components=2)
    X_2d = pca_2d.fit_transform(X_train_scaled)
    
    plt.figure(figsize=(8, 6))
    sns.scatterplot(x=X_2d[:,0], y=X_2d[:,1], hue=y_train, palette=['green', 'red'], style=y_train)
    plt.title('Step 5: Custom Group Clusters (Green=Genuine, Red=Forged)')
    plt.xlabel('PCA Component 1'); plt.ylabel('PCA Component 2')
    plt.legend(title='Type', labels=['Genuine', 'Forged'])
    plt.savefig(f'{MODEL_PREFIX}_plot_5_pca_2d_scatter.png')
    print(f"Saved {MODEL_PREFIX}_plot_5_pca_2d_scatter.png")
    
    # Classification
    print("\nTraining XGBoost (PCA)...")
    model = XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=42)
    model.fit(X_train_pca, y_train)
    
    print("Evaluating XGBoost (PCA)...")
    y_pred = model.predict(X_test_pca)
    y_proba = model.predict_proba(X_test_pca)[:, 1]

    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, target_names=['Genuine', 'Forged']))
    
    # --- PLOT 6: Confusion Matrix ---
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=['Genuine', 'Forged'], 
                yticklabels=['Genuine', 'Forged'])
    plt.title('Step 6: Accuracy (Confusion Matrix)')
    plt.ylabel('Actual'); plt.xlabel('Predicted')
    plt.savefig(f'{MODEL_PREFIX}_plot_6_confusion_matrix.png')
    print(f"Saved {MODEL_PREFIX}_plot_6_confusion_matrix.png")
    
    # --- PLOT 7: ROC Curve ---
    fpr, tpr, _ = roc_curve(y_test, y_proba)
    roc_auc = auc(fpr, tpr)
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (area = {roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlabel('False Positive Rate'); plt.ylabel('True Positive Rate')
    plt.title('Step 7: ROC Performance Curve')
    plt.legend(loc="lower right")
    plt.savefig(f'{MODEL_PREFIX}_plot_7_roc_curve.png')
    print(f"Saved {MODEL_PREFIX}_plot_7_roc_curve.png")

    # --- SAVE THE CUSTOM PIPELINE ---
    print("\n" + "="*50)
    print("Saving the trained CUSTOM model and transformers...")
    
    joblib.dump(scaler, f'{MODEL_PREFIX}_scaler.joblib')
    joblib.dump(pca, f'{MODEL_PREFIX}_pca.joblib')
    joblib.dump(model, f'{MODEL_PREFIX}_model.joblib')
    
    print(f"Models saved successfully as '{MODEL_PREFIX}_scaler.joblib', etc.")
    print("Complete! 🎉")