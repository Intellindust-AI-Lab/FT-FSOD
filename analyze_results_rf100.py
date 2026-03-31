#!/usr/bin/env python3
import os
import json
import glob
from collections import defaultdict
import sys

# Import category dictionary
Flora_Fauna = [
    "aquarium-combined",
    "bees",
    "deepfruits",
    "exploratorium-daphnia",
    "grapes-5",
    "grass-weeds",
    "gwhd2021",
    "into-the-vale",
    "jellyfish",
    "marine-sharks",
    "orgharvest",
    "peixos-fish",
    "penguin-finder-seg",
    "pig-detection",
    "roboflow-trained-dataset",
    "sea-cucumbers-new-tiles",
    "thermal-cheetah",
    "tomatoes-2",
    "trail-camera",
    "underwater-objects",
    "varroa-mites-detection--test-set",
    "wb-prova",
    "weeds4"
]

Industrial = [
    "-grccs",
    "13-lkc01",
    "2024-frc",
    "aircraft-turnaround-dataset",
    "asphaltdistressdetection",
    "cable-damage",
    "conveyor-t-shirts",
    "dataconvert",
    "deeppcb",
    "defect-detection",
    "fruitjes",
    "infraredimageofpowerequipment",
    "ism-band-packet-detection",
    "l10ul502",
    "needle-base-tip-min-max",
    "recode-waste",
    "screwdetectclassification",
    "smd-components",
    "truck-movement",
    "tube",
    "water-meter",
    "wheel-defect-detection"
]

Document = [
    "activity-diagrams",
    "all-elements",
    "circuit-voltages",
    "invoice-processing",
    "label-printing-defect-version-2",
    "macro-segmentation",
    "paper-parts",
    "signatures",
    "speech-bubbles-detection",
    "wine-labels"
]

Medical = [
    "canalstenosis",
    "crystal-clean-brain-tumors-mri-dataset",
    "dentalai",
    "inbreast",
    "liver-disease",
    "nih-xray",
    "spinefrxnormalvindr",
    "stomata-cells",
    "train",
    "ufba-425",
    "urine-analysis1",
    "x-ray-id",
    "xray"
]

Aerial = [
    "aerial-airport",
    "aerial-cows",
    "aerial-sheep",
    "apoce-aerial-photographs-for-object-detection-of-construction-equipment",
    "electric-pylon-detection-in-rsi",
    "floating-waste",
    "human-detection-in-floods",
    "sssod",
    "uavdet-small",
    "wildfire-smoke",
    "zebrasatasturias"
]

Sports = [
    "actions",
    "aerial-pool",
    "ball",
    "bibdetection",
    "football-player-detection",
    "lacrosse-object-detection"
]

Other = [
    "buoy-onboarding",
    "car-logo-detection",
    "clashroyalechardetector",
    "cod-mw-warzone",
    "countingpills",
    "everdaynew",
    "flir-camera-objects",
    "halo-infinite-angel-videogame",
    "mahjong",
    "new-defects-in-wood",
    "orionproducts",
    "pill",
    "soda-bottles",
    "taco-trash-annotations-in-context",
    "the-dreidel-project"
]

categories = {
    "Flora_Fauna": Flora_Fauna,
    "Industrial": Industrial,
    "Document": Document,
    "Medical": Medical,
    "Aerial": Aerial,
    "Sports": Sports,
    "Other": Other
}



def find_json_files(base_dir):
    """Find all paths containing JSON result files"""
    json_files = []
    
    # Iterate through all dataset folders
    for dataset_dir in os.listdir(base_dir):
        # Match folder prefix, e.g., swinB_all_dataset_10shot
        if not dataset_dir.startswith('swinB_all_'):
            continue
            
        dataset_path = os.path.join(base_dir, dataset_dir)
        if not os.path.isdir(dataset_path):
            continue
            
        # Find timestamp folders containing JSON files in each dataset folder
        for timestamp_dir in os.listdir(dataset_path):
            timestamp_path = os.path.join(dataset_path, timestamp_dir)
            if not os.path.isdir(timestamp_path):
                continue
                
            # Find JSON files (usually result files like bbox_results.json)
            json_pattern = os.path.join(timestamp_path, '*.json')
            json_matches = glob.glob(json_pattern)
            
            if json_matches:
                # Take the first JSON file
                json_file = json_matches[0]
                # Check if last_checkpoint file exists to determine if training is complete
                last_checkpoint_path = os.path.join(dataset_path, 'last_checkpoint')
                has_last_checkpoint = os.path.isfile(last_checkpoint_path)
                json_files.append((dataset_dir, json_file, dataset_path, has_last_checkpoint))
                break  # Break inner loop after finding JSON file
    
    return json_files

def parse_dataset_info(dataset_name):
    """Parse dataset name and shot number from dataset directory name"""
    # Expected format: swinB_all_{dataset}_{shot}shot
    
    # Remove 'shot' suffix
    if not dataset_name.endswith('shot'):
        return None, None
    
    # Remove 'shot' and 'swinB_all_'
    name_without_shot = dataset_name[:-4]
    if not name_without_shot.startswith('swinB_all_'):
         return None, None
    
    name_parts = name_without_shot[len('swinB_all_'):].split('_')
    
    if len(name_parts) >= 2:
        # Last part is the shot number
        shot = name_parts[-1]
        # Remaining parts are the dataset name
        dataset = '_'.join(name_parts[:-1])
        return dataset, shot
        
    return None, None

def main(base_dir):
    # Build dataset to category mapping
    dataset_to_category = {}
    for cat_name, cat_datasets in categories.items():
        for dataset in cat_datasets:
            dataset_to_category[dataset] = cat_name
    
    # Find all JSON files
    json_files = find_json_files(base_dir)
    
    if not json_files:
        print("No JSON result files found")
        return
    
    # Store results
    results = defaultdict(dict)  # {dataset: {shot: mAP}}
    incomplete_training = []  # Store datasets with incomplete training
    
    # Read and parse all JSON files
    for dataset_dir, json_file, dataset_path, has_last_checkpoint in json_files:
        try:
            with open(json_file, 'r') as f:
                data = json.load(f)
            
            # Get mAP value (COCO mAP)
            map_value = data.get('coco/bbox_mAP', None)
            if map_value is None:
                print(f"Warning: 'coco/bbox_mAP' field not found in {json_file}")
                continue
            
            # Parse dataset and shot information
            dataset, shot = parse_dataset_info(dataset_dir)
            if dataset is None or shot is None:
                print(f"Warning: Unable to parse dataset info: {dataset_dir}")
                continue
            
            # Store result (convert to percentage)
            results[dataset][shot] = map_value * 100
            
            # Check if training is complete
            if not has_last_checkpoint:
                incomplete_training.append((dataset, shot, dataset_path))
            
        except Exception as e:
            print(f"Error: Failed to read file {json_file}: {e}")
    
    # Group statistics by shot
    shot_stats = defaultdict(list)  # {shot: [mAP_values]}
    
    for dataset, shots in results.items():
        for shot, map_value in shots.items():
            shot_stats[shot].append(map_value)
    
    # Define shot order for output
    target_shots = ['10']
    
    # Output results grouped by shot
    for shot in target_shots:
        if shot not in shot_stats:
            continue
            
        print("\n" + "=" * 80)
        print(f"{shot}shot Results:")
        print("=" * 80)
        
        # Get all datasets for this shot
        shot_datasets = [(dataset, results[dataset][shot]) 
                        for dataset in results.keys() 
                        if shot in results[dataset]]
        shot_datasets.sort()  # Sort by dataset name
        
        # Print results for each dataset
        for dataset, map_value in shot_datasets:
            # Check if training is complete
            is_complete = not any(d == dataset and s == shot for d, s, _ in incomplete_training)
            status = "✓" if is_complete else "⚠"
            # Get category if available
            category = dataset_to_category.get(dataset, "Unknown")
            print(f"  {status} {dataset:<45} [{category:<12}] mAP = {map_value:>6.3f}")
        
        # Calculate and print average
        values = shot_stats[shot]
        avg_map = sum(values) / len(values)
        print("-" * 80)
        print(f"  {shot}shot Average mAP: {avg_map:>6.3f} (based on {len(values)} datasets)")
    
    # Final summary of average results for each shot
    print("\n" + "=" * 80)
    print("Average Results Summary by Shot:")
    print("=" * 80)
    
    for shot in target_shots:
        if shot in shot_stats:
            values = shot_stats[shot]
            avg_map = sum(values) / len(values)
            print(f"  {shot}shot Average mAP: {avg_map:>6.3f} (based on {len(values)} datasets)")
        else:
            print(f"  {shot}shot: No data")
    
    # Category-wise statistics (if categories are available)
    if categories:
        print("\n" + "=" * 80)
        print("Average Results Summary by Category:")
        print("=" * 80)
        
        # Group statistics by category and shot
        category_stats = defaultdict(lambda: defaultdict(list))  # {category: {shot: [mAP_values]}}
        
        for dataset, shots in results.items():
            category = dataset_to_category.get(dataset, "Unknown")
            for shot, map_value in shots.items():
                category_stats[category][shot].append(map_value)
        
        # Print category statistics
        for category in sorted(category_stats.keys()):
            print(f"\n  Category: {category}")
            for shot in target_shots:
                if shot in category_stats[category]:
                    values = category_stats[category][shot]
                    avg_map = sum(values) / len(values)
                    print(f"    {shot}shot: {avg_map:>6.3f} (based on {len(values)} datasets)")
                else:
                    print(f"    {shot}shot: No data")
    
    print("=" * 80)


if __name__ == '__main__':
    if len(sys.argv) > 1:
        base_dir = sys.argv[1]
        # Convert relative path to absolute path
        if not os.path.isabs(base_dir):
            base_dir = os.path.abspath(base_dir)
    else:
        print("Please provide experiment directory path as command line argument")
        print("Usage: python analyze_results_rf100.py <experiment_directory_path>")
        sys.exit(1)
    
    if not os.path.isdir(base_dir):
        print(f"Error: Directory does not exist: {base_dir}")
        sys.exit(1)
    
    main(base_dir)