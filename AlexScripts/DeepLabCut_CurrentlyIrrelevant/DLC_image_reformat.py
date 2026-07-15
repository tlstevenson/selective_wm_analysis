import os
import cv2
import shutil

def process_and_mirror_data(input_parent_dir, output_parent_dir):
    """
    Crawls subdirectories, converts .png files to grayscale, and copies over 
    specific .csv and .h5 data files to a mirrored directory structure.
    """
    if not os.path.exists(input_parent_dir):
        print(f"Error: The input directory '{input_parent_dir}' does not exist.")
        return

    for root, dirs, files in os.walk(input_parent_dir):
        for filename in files:
            input_path = os.path.join(root, filename)
            
            # --- Mirror the Folder Structure ---
            rel_path = os.path.relpath(root, input_parent_dir)
            target_dir = os.path.join(output_parent_dir, rel_path)
            os.makedirs(target_dir, exist_ok=True)
            output_path = os.path.join(target_dir, filename)

            # Skip if file has already been processed or copied
            if os.path.exists(output_path):
                print(f"Skipping (already exists): {output_path}")
                continue

            # --- Handle Image Files ---
            if filename.lower().endswith('.png'):
                try:
                    img = cv2.imread(input_path, cv2.IMREAD_GRAYSCALE)
                    if img is not None:
                        cv2.imwrite(output_path, img)
                        print(f"Converted Image: {output_path}")
                    else:
                        print(f"Warning: OpenCV could not read {input_path}")
                except Exception as e:
                    print(f"Error processing {input_path}: {e}")

            # --- Handle Annotation/Data Files ---
            elif filename in ['CollectedData_AITapus.csv', 'CollectedData_AITapus.h5']:
                try:
                    # copy2 preserves the original file metadata (timestamps, etc.)
                    shutil.copy2(input_path, output_path)
                    print(f"Copied Data File: {output_path}")
                except Exception as e:
                    print(f"Error copying {input_path}: {e}")

# ==========================================
# Run the Code
# ==========================================
if __name__ == "__main__":
    # Update these paths to match your actual directories
    INPUT_FOLDER = r"C:\Users\cns-th-lab\DeepLabCut_Projects\AllRatsBulky-AITapus-2026-04-08\labeled-data"
    OUTPUT_FOLDER = r"C:\Users\cns-th-lab\SLEAP_Projects\DLC_reformatted"
    
    print(f"Starting batch process...\nReading from: {INPUT_FOLDER}\nSaving to: {OUTPUT_FOLDER}\n")
    process_and_mirror_data(INPUT_FOLDER, OUTPUT_FOLDER)
    print("\nFinished processing and copying files!")