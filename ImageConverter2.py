import os
import csv
from PIL import Image
import numpy as np

# Path to the root directory containing folders of images
root_dir = 'train'
output_csv = 'card_images.csv'
image_size = (224, 224)  # Resize all images to this size

# Collect all data rows here
data_rows = []

# Walk through each folder in the dataset
for label_folder in os.listdir(root_dir):
    folder_path = os.path.join(root_dir, label_folder)
    if not os.path.isdir(folder_path):
        continue

    for img_file in os.listdir(folder_path):
        img_path = os.path.join(folder_path, img_file)
        try:
            with Image.open(img_path) as img:
                img = img.convert('RGB')  # Ensure it's RGB
                img = img.resize(image_size)
                img_array = np.array(img).flatten()  # Flatten all pixels (R,G,B)
                row = [label_folder] + img_array.tolist()
                data_rows.append(row)
        except Exception as e:
            print(f"Error processing {img_path}: {e}")

# Write to CSV
header = ['label'] + [f'pixel_{i}' for i in range(224 * 224 * 3)]

with open(output_csv, 'w', newline='') as f:
    writer = csv.writer(f)
    writer.writerow(header)
    writer.writerows(data_rows)

print(f"Saved {len(data_rows)} images to {output_csv}")
