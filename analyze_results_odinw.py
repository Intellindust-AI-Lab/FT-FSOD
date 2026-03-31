#!/usr/bin/env python3
import os
import json
import glob
from collections import defaultdict
import sys

def find_json_files(base_dir):
    """Find all paths containing JSON result files"""
    json_files = []
    
    # Iterate through all dataset folders
    for dataset_dir in os.listdir(base_dir):
        if not dataset_dir.startswith('swinB_all_'):
            continue
            
        dataset_path = os.path.join(base_dir, dataset_dir)
        if not os.path.isdir(dataset_path):
            continue
            
        # Find timestamp folders containing JSON files within each dataset folder
        for timestamp_dir in os.listdir(dataset_path):
            timestamp_path = os.path.join(dataset_path, timestamp_dir)
            if not os.path.isdir(timestamp_path):
                continue
                
            # Find JSON files
            json_pattern = os.path.join(timestamp_path, '*.json')
            json_matches = glob.glob(json_pattern)
            
            if json_matches:
                # Take the first JSON file (usually only one)
                json_file = json_matches[0]
                # Check if the last_checkpoint file exists
                last_checkpoint_path = os.path.join(dataset_path, 'last_checkpoint')
                has_last_checkpoint = os.path.isfile(last_checkpoint_path)
                json_files.append((dataset_dir, json_file, dataset_path, has_last_checkpoint))
                break  # Break inner loop after finding a JSON file
    
    return json_files

def parse_dataset_info(dataset_name):
    """Parse dataset name and shot number from dataset directory name"""
    # Format: swinB_all_{dataset}_{shot}shot
    
    # Remove 'shot' suffix
    name_without_shot = dataset_name[:-4]
    parts = name_without_shot.split('_')
    
    if parts[0] == 'swinB':
        # The last part contains the shot number
        shot = parts[-2][:-4]
        # Middle parts form the dataset name
        dataset = '_'.join(parts[2:-2])
        return dataset, shot
    return None, None

def main(base_dir):
    # Expected dataset list
    expected_datasets = {
        'AerialMaritimeDrone',
        'Aquarium',
        'CottontailRabbits',
        'EgoHands',
        'NorthAmericaMushrooms',
        'Packages',
        'PascalVOC',
        'pistols',
        'pothole',
        'Raccoon',
        'ShellfishOpenImages',
        'thermalDogsAndPeople',
        'VehiclesOpenImages'
    }
    
    # Find all JSON files
    json_files = find_json_files(base_dir)
    
    if not json_files:
        print("No JSON result files found")
        return
    
    # Store results
    results = defaultdict(dict)  # {dataset: {shot: mAP}}
    all_results = []  # Store all results for printing
    incomplete_training = []  # Store datasets with incomplete training
    
    # Read and parse all JSON files
    for dataset_dir, json_file, dataset_path, has_last_checkpoint in json_files:
        try:
            with open(json_file, 'r') as f:
                data = json.load(f)
            
            # Get mAP value
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
            all_results.append((dataset, shot, map_value, json_file))
            
            # Check if training is complete
            if not has_last_checkpoint:
                incomplete_training.append((dataset, shot, dataset_path))
            
            # Print individual result
            # print(f"{dataset:15} {shot:>6}shot: mAP = {map_value:.3f} ({os.path.basename(json_file)})")
            
        except Exception as e:
            print(f"Error: Failed to read file {json_file}: {e}")
    
    # print("\n" + "=" * 80)
    # print("Statistics:")
    # print("=" * 80)
    
    # Group statistics by shot
    shot_stats = defaultdict(list)  # {shot: [mAP_values]}
    
    for dataset, shots in results.items():
        for shot, map_value in shots.items():
            shot_stats[shot].append(map_value)
    
    # Compute and print averages
    for shot in sorted(shot_stats.keys(), key=lambda x: int(x)):
        values = shot_stats[shot]
        avg_map = sum(values) / len(values)
        # print(f"{shot} shot Average mAP: {avg_map:.3f} (based on {len(values)} datasets)")
        # print(f"  Datasets: {', '.join([dataset for dataset in results.keys() if shot in results[dataset]])}")
        # print()
    
    # Check for missing datasets
    found_datasets = set(results.keys())
    missing_datasets = expected_datasets - found_datasets
    
    if missing_datasets:
        print("=" * 80)
        print(f"Missing datasets ({len(missing_datasets)}/{len(expected_datasets)}):")
        print("=" * 80)
        for dataset in sorted(missing_datasets):
            print(f"  - {dataset}")
        print()
    else:
        print("=" * 80)
        print("✓ All expected datasets have been found!")
        print("=" * 80)
        print()
    
    # Check for extra datasets (not in the expected list)
    extra_datasets = found_datasets - expected_datasets
    if extra_datasets:
        print("=" * 80)
        print(f"Extra datasets (not in expected list):")
        print("=" * 80)
        for dataset in sorted(extra_datasets):
            print(f"  + {dataset}")
        print()
    
    # Check for incomplete training datasets
    if incomplete_training:
        print("=" * 80)
        print("⚠ Training may be incomplete (missing last_checkpoint file):")
        print("=" * 80)
        for dataset, shot, path in sorted(incomplete_training):
            print(f"  ! {dataset} ({shot}shot)")
            print(f"    Path: {path}")
        print()
    else:
        print("=" * 80)
        print("✓ Training for all datasets is complete!")
        print("=" * 80)
        print()
    
    # Print detailed dataset-shot matrix
    print("=" * 80)
    print("Detailed results matrix:")
    
    datasets = sorted(results.keys())
    shots = sorted(set(shot for shots in results.values() for shot in shots.keys()), key=lambda x: int(x))
    
    # Print header row
    print(f"{'Dataset':<15}", end="")
    for shot in shots:
        print(f"{shot+'shot':>10}", end="")
    print()
    
    print("-" * (15 + 10 * len(shots)))
    
    # Print each dataset's results
    for dataset in datasets:
        print(f"{dataset:<15}", end="")
        for shot in shots:
            if shot in results[dataset]:
                print(f"{results[dataset][shot]:>10.3f}", end="")
            else:
                print(f"{'N/A':>10}", end="")
        print()
    
    # Print average row
    print("-" * (15 + 10 * len(shots)))
    print(f"{'Average':<15}", end="")
    for shot in shots:
        if shot in shot_stats:
            avg = sum(shot_stats[shot]) / len(shot_stats[shot])
            print(f"{avg:>10.3f}", end="")
        else:
            print(f"{'N/A':>10}", end="")
    print()

if __name__ == '__main__':
    if len(sys.argv) > 1:
        base_dir = sys.argv[1]
        # Convert relative path to absolute path
        if not os.path.isabs(base_dir):
            base_dir = os.path.abspath(base_dir)
    else:
        print("Please provide experiment directory path as command line argument")
        print("Usage: python analyze_results_odinw.py <experiment_directory_path>")
        sys.exit(1)

    if not os.path.isdir(base_dir):
        print(f"Error: Directory does not exist: {base_dir}")
        sys.exit(1)

    main(base_dir)


# AerialMaritimeDrone, Aquarium, CottontailRabbits, EgoHands, NorthAmericaMushrooms, Packages, PascalVOC, pistols, pothole, Raccoon, ShellfishOpenImages, thermalDogsAndPeople, VehiclesOpenImages
