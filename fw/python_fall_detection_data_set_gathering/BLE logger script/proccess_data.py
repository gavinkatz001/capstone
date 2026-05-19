import json
import os

def reformat_json(input_file, output_file=None):
    # Read the input JSON file
    with open(input_file, 'r') as f:
        data = json.load(f)
    
    # Extract metadata
    metadata = data['metadata']
    
    # Initialize the new format
    new_format = {
        "Location": "Rishi's Suite",  # You might want to make this configurable
        "Timestamp": [],
        "Ax": [],
        "Ay": [],
        "Az": [],
        "Gx": [],
        "Gy": [],
        "Gz": [],
        "Pressure": [],
        "FSR": []
    }
    
    # Extract data from each sample
    for sample in data['data']:
        new_format["Timestamp"].append(sample["ts"])
        new_format["Ax"].append(sample["ax"])
        new_format["Ay"].append(sample["ay"])
        new_format["Az"].append(sample["az"])
        new_format["Gx"].append(sample["gx"])
        new_format["Gy"].append(sample["gy"])
        new_format["Gz"].append(sample["gz"])
        new_format["Pressure"].append(sample["p"])
        new_format["FSR"].append(sample["fsr"])
    
    # If output_file is not specified, generate it from input_file
    if output_file is None:
        base = os.path.basename(input_file)
        name, _ = os.path.splitext(base)
        output_file = f"reformatted_{name}.json"

    # Write the reformatted data to output file
    with open(output_file, 'w') as f:
        json.dump(new_format, f, indent=4)

    return output_file

def fix_timestamps(input_file, output_file=None, threshold=25, step=20):
    with open(input_file, 'r') as f:
        data = json.load(f)
    
    timestamps = data["Timestamp"]
    fixed = False
    for i in range(1, len(timestamps)):
        if not fixed and (timestamps[i] - timestamps[i-1]) > threshold:
            # Start fixing from here
            fixed = True
            for j in range(i, len(timestamps)):
                timestamps[j] = timestamps[j-1] + step
            break  # Only fix the first jump
    
    data["Timestamp"] = timestamps
    
    if output_file is None:
        base = os.path.basename(input_file)
        name, ext = os.path.splitext(base)
        output_file = f"fixed_{name}{ext}"
    with open(output_file, 'w') as f:
        json.dump(data, f, indent=4)
    return output_file

def process_data(input_file, output_file=None, threshold=25, step=20):
    """
    Process a single file by first reformatting it and then fixing timestamps
    """
    # First reformat the data
    reformatted_file = reformat_json(input_file)
    
    # Then fix the timestamps
    if output_file is None:
        base = os.path.basename(input_file)
        name, ext = os.path.splitext(base)
        output_file = f"processed_{name}{ext}"
    
    final_file = fix_timestamps(reformatted_file, output_file, threshold, step)
    
    # Clean up intermediate file
    if os.path.exists(reformatted_file):
        os.remove(reformatted_file)
    
    return final_file

def process_folder(input_folder, output_folder, threshold=25, step=20, label_prefix=""):
    """
    Process all JSON files in a folder
    """
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    
    for filename in os.listdir(input_folder):
        if filename.endswith('.json'):
            input_path = os.path.join(input_folder, filename)
            name, ext = os.path.splitext(filename)
            prefix = f"{label_prefix}_" if label_prefix else ""
            output_path = os.path.join(output_folder, f"{prefix}processed_{name}{ext}")
            process_data(input_path, output_path, threshold, step)
            print(f"Processed {filename}")

if __name__ == "__main__":
    # Example usage
    input_folder = "final_data"
    output_folder = "processed_final_data"
    label = "Resident#3 (Thinner body)"
    process_folder(input_folder, output_folder, label_prefix=label)