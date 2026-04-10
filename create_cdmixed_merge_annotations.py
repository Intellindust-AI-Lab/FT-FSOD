#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import json
import os
from collections import Counter, defaultdict
from typing import Dict, List


DEFAULT_FILE_NAMES = [
    "test_artaxor_new.json",
    "test_clipart1k_new.json",
    "test_dior_new.json",
    "test_neudet_new.json",
    "test_uodd_new.json",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Merge fp annotation JSON files and remap image ids globally."
    )
    parser.add_argument(
        "--base-dir",
        default="./fp_annotations",
        help="Directory that contains input JSON files",
    )
    parser.add_argument(
        "--file-names",
        nargs="+",
        default=DEFAULT_FILE_NAMES,
        help="Input JSON file names to merge",
    )
    parser.add_argument(
        "--output-suffix",
        default="_merged.json",
        help="Output suffix for generated files",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    base_dir = args.base_dir
    file_names = args.file_names
    output_suffix = args.output_suffix

    files = [os.path.join(base_dir, name) for name in file_names]
    for path in files:
        if not os.path.exists(path):
            raise FileNotFoundError(f"File does not exist: {path}")

    # Step 1: Read all COCO files, collect all images, and rebuild global contiguous image ids
    cocos: Dict[str, dict] = {}
    all_images_raw: List[dict] = []
    image_id_map_by_file: Dict[str, Dict[int, int]] = {}

    for fpath in files:
        name = os.path.basename(fpath)
        with open(fpath, "r", encoding="utf-8") as f:
            data = json.load(f)

        cocos[name] = data
        image_id_map_by_file[name] = {}

        for img in data.get("images", []):
            old_id = img["id"]
            new_id = len(all_images_raw) + 1

            new_img = dict(img)
            new_img["id"] = new_id
            all_images_raw.append(new_img)

            if old_id in image_id_map_by_file[name]:
                raise ValueError(f"Duplicate image id found in {name}: {old_id}")
            image_id_map_by_file[name][old_id] = new_id

    # Duplicate file_name statistics
    file_name_values = [img["file_name"] for img in all_images_raw]
    file_name_counts = Counter(file_name_values)
    duplicate_files = {fname: count for fname, count in file_name_counts.items() if count > 1}
    print(f"Total images: {len(all_images_raw)}")
    print(f"Unique file_name count: {len(file_name_counts)}")
    print(f"Duplicate file_name count: {len(duplicate_files)}")
    if duplicate_files:
        print("  Duplicate examples (top 10):")
        for fname, cnt in list(duplicate_files.items())[:10]:
            print(f"    '{fname}' appears {cnt} times")
    else:
        print("  No duplicate file_name")

    # Count duplicate ids after remapping (should be 0)
    id_to_files = defaultdict(list)
    for img in all_images_raw:
        id_to_files[img["id"]].append(img["file_name"])
    duplicate_ids = {id_: files_ for id_, files_ in id_to_files.items() if len(files_) > 1}

    print(f"\nTotal images: {len(all_images_raw)}")
    print(f"Total id count: {len(id_to_files)}")
    print(f"Duplicate id count: {len(duplicate_ids)}")
    if duplicate_ids:
        print("  Duplicate id details (file_names per id):")
        for id_, files_ in list(duplicate_ids.items())[:10]:
            print(f"    id={id_} appears {len(files_)} times")
            for fname in files_:
                print(f"      - {fname}")
            print()
    else:
        print("  No duplicate id")

    print("---------- Statistics completed ----------\n")
    print(f"Collected {len(all_images_raw)} images\n")

    # Step 2: Generate remapped annotations per input file
    for fname, data in cocos.items():
        remapped_annotations = []
        missing_image_id = 0
        image_id_map = image_id_map_by_file[fname]

        for ann in data.get("annotations", []):
            old_image_id = ann.get("image_id")
            if old_image_id not in image_id_map:
                missing_image_id += 1
                continue

            new_ann = dict(ann)
            new_ann["image_id"] = image_id_map[old_image_id]
            remapped_annotations.append(new_ann)

        if missing_image_id > 0:
            print(
                f"[WARN] {fname}: {missing_image_id} annotations have missing image_id and were skipped"
            )

        new_data = {
            "images": all_images_raw,
            "annotations": remapped_annotations,
            "categories": data.get("categories", []),
        }
        for key in ["info", "licenses"]:
            if key in data:
                new_data[key] = data[key]

        out_name = fname.replace(".json", output_suffix)
        out_path = os.path.join(base_dir, out_name)
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(new_data, f, ensure_ascii=False, indent=2)

        print(f"Generated: {out_name}")

    print(f"\nAll files have been generated, output suffix: {output_suffix}")
    print("Original files were not modified")


if __name__ == "__main__":
    main()
