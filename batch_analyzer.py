# batch_analyzer.py

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
        self.folder_path = None

    def process_folder(self, folder_path: str):
        """Process all image sets in a folder"""
        self.folder_path = folder_path
        files = os.listdir(folder_path)
        
        # Define treatments
        treatments = [
            'Antimycin_3',
            'Antimycin_7',
            'UT_3',
            'UT_7'
        ]
        
        # Process each treatment
        for treatment in treatments:
            # Construct exact filenames using f-strings
            base_pattern = f"MAX_WT_mDA_{treatment}"
            cyan_file = f"{base_pattern}_Cyan.tif"
            magenta_file = f"{base_pattern}_Magenta.tif"
            red_file = f"{base_pattern}_Red.tif"
            
            # Check if files exist
            if not all(f in files for f in [cyan_file, magenta_file, red_file]):
                print(f"\nSkipping {treatment} - missing files:")
                if cyan_file not in files:
                    print(f"  Missing {cyan_file}")
                if magenta_file not in files:
                    print(f"  Missing {magenta_file}")
                if red_file not in files:
                    print(f"  Missing {red_file}")
                continue
            
            try:
                # Create full paths
                cyan_path = os.path.join(folder_path, cyan_file)
                magenta_path = os.path.join(folder_path, magenta_file)
                red_path = os.path.join(folder_path, red_file)
                
                print(f"\nProcessing {treatment}:")
                print(f"Files:")
                print(f"  Cyan: {cyan_file}")
                print(f"  Magenta: {magenta_file}")
                print(f"  Red: {red_file}")
                
                result = self.process_image_set(treatment, red_path, cyan_path, magenta_path)
                self.results.append(result)
                print(f"Successfully processed {treatment}")
                
            except Exception as e:
                print(f"Error processing {treatment}: {str(e)}")

        # Save results if any sets were processed
        if self.results:
            self.save_results()
        else:
            print("No results to save - no image sets were successfully processed")

    def process_image_set(self, treatment: str, red_path: str, cyan_path: str, magenta_path: str) -> dict:
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
        
        # Save validation images in the input folder
        output_dir = os.path.join(self.folder_path, "validation_images")
        os.makedirs(output_dir, exist_ok=True)
        
        # Use f-string for output filenames
        red_vis, cyan_vis, magenta_vis, overlap_vis = self.process_channels(red_img, cyan_img, magenta_img)
        cv2.imwrite(os.path.join(output_dir, f"MAX_WT_mDA_{treatment}_red_check.png"), red_vis)
        cv2.imwrite(os.path.join(output_dir, f"MAX_WT_mDA_{treatment}_cyan_check.png"), cyan_vis)
        cv2.imwrite(os.path.join(output_dir, f"MAX_WT_mDA_{treatment}_magenta_check.png"), magenta_vis)
        cv2.imwrite(os.path.join(output_dir, f"MAX_WT_mDA_{treatment}_overlap_check.png"), overlap_vis)

        # Compile results
        result = {
            'treatment': treatment,
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
        """Save results to CSV file in the input folder"""
        # Create filename with timestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = os.path.join(self.folder_path, f"analysis_results_{timestamp}.csv")

        # Write results
        with open(filename, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=self.results[0].keys())
            writer.writeheader()
            writer.writerows(self.results)

        print(f"\nResults saved to: {filename}")

def batch_process(folder_path: str):
    """Process all images in a folder"""
    analyzer = BatchAnalyzer()
    analyzer.process_folder(folder_path)

if __name__ == "__main__":
    folder_path = input("Enter the path to your images folder: ")
    batch_process(folder_path)

