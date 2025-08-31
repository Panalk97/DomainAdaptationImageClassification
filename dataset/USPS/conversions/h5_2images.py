import os
import h5py
import numpy as np
from PIL import Image

# Path to the USPS dataset in HDF5 format
usps_h5_path = "usps.h5"

# Output directory where images will be organized into train/test folders by digit
output_dir = "usps_digits"

def load_usps_dataset(h5_path):
    """Load the USPS dataset from an HDF5 file."""
    with h5py.File(h5_path, "r") as h5_file:
        # Load train data
        train_images = h5_file['train']['data'][:]
        train_labels = h5_file['train']['target'][:]
        
        # Load test data
        test_images = h5_file['test']['data'][:]
        test_labels = h5_file['test']['target'][:]
    
    # Normalize images to 0-255
    train_images = (train_images * 255).astype(np.uint8)
    test_images = (test_images * 255).astype(np.uint8)
    
    return (train_images, train_labels), (test_images, test_labels)

def save_images(images, labels, output_folder, dataset_type):
    """Save images into directories organized by dataset type (train/test) and label."""
    dataset_folder = os.path.join(output_folder, dataset_type)
    if not os.path.exists(dataset_folder):
        os.makedirs(dataset_folder)
    
    # Ensure folders exist for digits 0-9
    for digit in range(10):
        digit_folder = os.path.join(dataset_folder, str(digit))
        os.makedirs(digit_folder, exist_ok=True)
    
    for idx, (image, label) in enumerate(zip(images, labels)):
        digit_folder = os.path.join(dataset_folder, str(label))
        image_path = os.path.join(digit_folder, f"{idx}.png")
        img = Image.fromarray(image.reshape((16, 16)))  # Reshape into 16x16 image
        img = img.convert("L")  # Convert to grayscale
        img.save(image_path)

def main():
    # Load USPS dataset
    print("Loading USPS dataset...")
    (train_images, train_labels), (test_images, test_labels) = load_usps_dataset(usps_h5_path)
    print(f"Loaded {len(train_images)} training images and {len(test_images)} test images.")

    # Save train images
    print(f"Saving training images to {output_dir}/train...")
    save_images(train_images, train_labels, output_dir, "train")

    # Save test images
    print(f"Saving test images to {output_dir}/test...")
    save_images(test_images, test_labels, output_dir, "test")

    print(f"Images saved to {output_dir}.")

if __name__ == "__main__":
    main()
