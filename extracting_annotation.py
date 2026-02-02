import os
import shutil

# Paths
image_root = "processed"
label_root = "labels"  # Your existing labels folder
output_root = "dataset\labels"  # New structured labels folder

# Ensure the output directory structure exists
for folder in ["train", "test", "val"]:
    os.makedirs(os.path.join(output_root, folder), exist_ok=True)

# Iterate through train, test, val folders
for folder in ["train", "test", "val"]:
    image_folder = os.path.join(image_root, folder)
    label_folder = os.path.join(output_root, folder)

    # Get image names (without extension)
    image_names = {os.path.splitext(img)[0] for img in os.listdir(image_folder) if img.endswith(('.png', '.jpg', '.jpeg'))}

    # Move corresponding label files
    for label_file in os.listdir(label_root):
        label_name, ext = os.path.splitext(label_file)
        if label_name in image_names:
            shutil.move(os.path.join(label_root, label_file), os.path.join(label_folder, label_file))

    print(f"Moved labels for {folder}")