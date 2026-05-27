import os
import json

def generate_index(directory):
    """
    Generate a JSON index for the given directory.

    Args:
        directory (str): The root directory containing subdirectories representing videos.

    Returns:
        dict: A dictionary representing the structure.
    """
    index = {}
    image_extensions = {".jpg", ".jpeg", ".png", ".bmp", ".tif"}

    # Iterate over subdirectories
    subdirs = sorted([subdir for subdir in os.listdir(directory) if os.path.isdir(os.path.join(directory, subdir))])
    for subdir in subdirs:
        if "_gt" in subdir:
            continue
        subdir_path = os.path.join(directory, subdir)
        if os.path.isdir(subdir_path):
            # List images in the subdirectory with valid extensions
            images = [os.path.splitext(img)[0] for img in os.listdir(subdir_path) 
                      if os.path.isfile(os.path.join(subdir_path, img)) and 
                      os.path.splitext(img)[1].lower() in image_extensions]
            images.sort()  # Optional: Sort the images alphabetically
            index[subdir] = images

    return index

def save_to_json(data, output_file):
    """
    Save data to a JSON file.

    Args:
        data (dict): The data to save.
        output_file (str): Path to the output JSON file.
    """
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=4, ensure_ascii=False)

if __name__ == "__main__":
    # Define the root directory and output file
    root_directory = '/home/lijuntong/dataset/UCSD_Anomaly_Dataset.v1p2/UCSDped2/Train'
    output_json = 'ped2_train.json'
    # Generate the index and save it
    index_data = generate_index(root_directory)
    save_to_json(index_data, output_json)

    print(f"Index successfully saved to {output_json}")