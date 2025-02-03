from multichannel_analyzer import MultichannelAnalyzer
import cv2
import numpy as np
from typing import List, Tuple
import os
import csv
from datetime import datetime

class BatchAnalyzer(MultichannelAnalyzer):
    def __init__(self):
        super().__init__()
        self.results = []

    def process_folder(self, folder_path: str, red_suffix: str = "_red.tif", 
                      cyan_suffix: str = "_cyan.tif", magenta_suffix: str = "_magenta.tif"):
        """Process all image sets in a folder"""
        # Get all unique base filenames (without suffixes)
        files = os.listdir(folder_path)
        base_names = set()
        for file in files:
            if file.endswith(red_suffix):
                base_name = file.replace(red_suffix, "")
                base_names.add(base_name)

        # Process each set of images
        for base_name in base_names:
            red_path = os.path.join(folder_path, base_name + red_suffix)
            cyan_path = os.path.join(folder_path, base_name + cyan_suffix)
            magenta_path = os.path.join(folder_path, base_name + magenta_suffix)

            # Skip if any file is missing
            if not all(os.path.exists(p) for p in [red_path, cyan_path, magenta_path]):
                print(f"Skipping {base_name} - missing one or more files")
                continue

            try:
                result = self.process_image_set(base_name, red_path, cyan_path, magenta_path)
                self.results.append(result)
                print(f"Processed {base_name}")
            except Exception as e:
                print(f"Error processing {base_name}: {e}")

        # Save results
        self.save_results()

    def process_image_set(self, base_name: str, red_path: str, cyan_path: str, magenta_path: str) -> dict:
        """Process a single set of images"""
        # Load images
        red_img, cyan_img, magenta_img = self.load_images(red_path, cyan_path, magenta_path)

        # Auto-detect thresholds
        red_thresh = self.find_optimal_threshold(red_img)
        cyan_thresh = self.find_optimal_threshold(cyan_img)
        magenta_thresh = self.find_optimal_threshold(magenta_img)

        # Update thresholds
        self.update_thresholds(red_threshold=red_thresh,
                             cyan_threshold=cyan_thresh,
                             magenta_threshold=magenta_thresh)

        # Detect structures
        red_contours = self.detect_structures(red_img, self.red_threshold,
                                            self.red_min_area, self.red_max_area)
        cyan_contours = self.detect_structures(cyan_img, self.cyan_threshold)
        magenta_contours = self.detect_structures(magenta_img, self.magenta_threshold)

        # Calculate areas
        areas = self.calculate_areas(red_contours, cyan_contours, magenta_contours, red_img.shape)

        # Compile results
        result = {
            'filename': base_name,
            'red_threshold': red_thresh,
            'cyan_threshold': cyan_thresh,
            'magenta_threshold': magenta_thresh,
            'num_cyan_regions': areas['num_cyan_regions'],
            'num_magenta_regions': areas['num_magenta_regions'],
            'cyan_in_red_area': areas['cyan_in_red'],
            'magenta_in_red_area': areas['magenta_in_red'],
            'cyan_in_red_percentage': areas['cyan_in_red_percentage'],
            'magenta_in_red_percentage': areas['magenta_in_red_percentage'],
            'region_ratio': areas['region_ratio'],
            'area_ratio': areas['area_ratio']
        }

        return result

    def save_results(self):
        """Save results to CSV file"""
        if not self.results:
            print("No results to save")
            return

        # Create filename with timestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"analysis_results_{timestamp}.csv"

        # Write results
        with open(filename, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=self.results[0].keys())
            writer.writeheader()
            writer.writerows(self.results)

        print(f"Results saved to {filename}")

def batch_process(folder_path: str):
    """Process all images in a folder"""
    analyzer = BatchAnalyzer()
    analyzer.process_folder(folder_path)

if __name__ == "__main__":
    # Example usage
    folder_path = input("Enter the path to your images folder: ")
    batch_process(folder_path)