import cv2
import numpy as np
from typing import List, Tuple

class MultichannelAnalyzer:
    def __init__(self):
        """Initialize analyzer with default parameters"""
        self.red_threshold = 200
        self.red_min_area = 50
        self.red_max_area = 100000
        self.cyan_threshold = 200
        self.magenta_threshold = 200

    def load_images(self, red_path: str, cyan_path: str, magenta_path: str) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Load separate channel images and ensure they're grayscale"""
        def load_and_convert(path):
            # Load image with original bit depth
            img = cv2.imread(path, -1)
            if img is None:
                raise ValueError(f"Could not load image at {path}")
            
            # If image is color (3 channels), convert to grayscale
            if len(img.shape) == 3:
                img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            
            # Normalize to 8-bit
            if img.dtype != np.uint8:
                img_min = img.min()
                img_max = img.max()
                if img_max > img_min:
                    img = ((img - img_min) * 255.0 / (img_max - img_min)).astype(np.uint8)
                else:
                    img = np.zeros_like(img, dtype=np.uint8)
            
            return img

        red_img = load_and_convert(red_path)
        cyan_img = load_and_convert(cyan_path)
        magenta_img = load_and_convert(magenta_path)
        
        print(f"Loaded images:")
        print(f"Red channel shape: {red_img.shape}, dtype: {red_img.dtype}")
        print(f"Cyan channel shape: {cyan_img.shape}, dtype: {cyan_img.dtype}")
        print(f"Magenta channel shape: {magenta_img.shape}, dtype: {magenta_img.dtype}")
        
        return red_img, cyan_img, magenta_img

    def update_thresholds(self, red_threshold=None, red_min_area=None, red_max_area=None,
                         cyan_threshold=None, magenta_threshold=None):
        """Update detection thresholds"""
        if red_threshold is not None:
            self.red_threshold = red_threshold
        if red_min_area is not None:
            self.red_min_area = red_min_area
        if red_max_area is not None:
            self.red_max_area = red_max_area
        if cyan_threshold is not None:
            self.cyan_threshold = cyan_threshold
        if magenta_threshold is not None:
            self.magenta_threshold = magenta_threshold

    def detect_structures(self, channel: np.ndarray, threshold: int, 
                         min_area: int = None, max_area: int = None) -> List[np.ndarray]:
        """Detect structures in a channel"""
        # Ensure we're working with 8-bit single channel
        assert channel.dtype == np.uint8
        assert len(channel.shape) == 2
        
        # Create binary mask
        _, binary = cv2.threshold(channel, threshold, 255, cv2.THRESH_BINARY)
        
        # Clean up noise
        kernel = np.ones((3,3), np.uint8)
        binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)
        binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)

        # Find contours
        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # Filter by area if specified
        if min_area is not None and max_area is not None:
            contours = [cnt for cnt in contours 
                       if min_area < cv2.contourArea(cnt) < max_area]

        return contours

    def create_binary_mask(self, contours: List[np.ndarray], shape: Tuple[int, int]) -> np.ndarray:
        """Create a binary mask from contours"""
        mask = np.zeros(shape, dtype=np.uint8)
        cv2.drawContours(mask, contours, -1, 255, -1)  # -1 fills the contours
        return mask

    def calculate_areas(self, red_contours: List[np.ndarray], cyan_contours: List[np.ndarray], 
                       shape: Tuple[int, int]) -> dict:
        """Calculate areas of overlap and difference between red and cyan regions"""
        # Create binary masks
        red_mask = self.create_binary_mask(red_contours, shape)
        cyan_mask = self.create_binary_mask(cyan_contours, shape)

        # Calculate intersections and differences
        cyan_in_red = cv2.bitwise_and(cyan_mask, red_mask)
        cyan_outside_red = cv2.bitwise_and(cyan_mask, cv2.bitwise_not(red_mask))

        # Calculate areas (in pixels)
        total_red_area = np.sum(red_mask > 0)
        total_cyan_area = np.sum(cyan_mask > 0)
        cyan_in_red_area = np.sum(cyan_in_red > 0)
        cyan_outside_red_area = np.sum(cyan_outside_red > 0)

        return {
            'total_red': total_red_area,
            'total_cyan': total_cyan_area,
            'cyan_in_red': cyan_in_red_area,
            'cyan_outside_red': cyan_outside_red_area,
            'cyan_in_red_percentage': (cyan_in_red_area / total_cyan_area * 100) if total_cyan_area > 0 else 0,
            'cyan_outside_red_percentage': (cyan_outside_red_area / total_cyan_area * 100) if total_cyan_area > 0 else 0,
            'overlap_masks': (red_mask, cyan_mask, cyan_in_red, cyan_outside_red)
        }

    def create_visualization(self, image: np.ndarray, contours: List[np.ndarray], 
                           title: str, threshold: int) -> np.ndarray:
        """Create visualization for a channel"""
        # Convert grayscale to BGR for colored visualization
        vis = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
        
        # Draw contours
        cv2.drawContours(vis, contours, -1, (0, 255, 0), 2)
        
        # Add text
        cv2.putText(vis, f'{title} Brightness: {threshold}', (10, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.putText(vis, f'Structures: {len(contours)}', (10, 60), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        if title == 'Red':
            cv2.putText(vis, f'Min Area: {self.red_min_area}', (10, 90), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            cv2.putText(vis, f'Max Area: {self.red_max_area}', (10, 120), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        return vis

    def create_overlap_visualization(self, image: np.ndarray, red_mask: np.ndarray, 
                                   cyan_mask: np.ndarray, cyan_in_red: np.ndarray, 
                                   cyan_outside_red: np.ndarray, areas: dict) -> np.ndarray:
        """Create visualization showing overlapping regions"""
        # Create colored visualization
        vis = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
        
        # Create color overlays
        vis[red_mask > 0] = [0, 0, 200]  # Red regions
        vis[cyan_outside_red > 0] = [200, 200, 0]  # Cyan outside red
        vis[cyan_in_red > 0] = [200, 0, 200]  # Overlap in magenta

        # Add text with measurements
        cv2.putText(vis, f'Total Red Area: {areas["total_red"]}', (10, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.putText(vis, f'Total Cyan Area: {areas["total_cyan"]}', (10, 60), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.putText(vis, f'Cyan in Red: {areas["cyan_in_red"]} ({areas["cyan_in_red_percentage"]:.1f}%)', 
                   (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.putText(vis, f'Cyan outside Red: {areas["cyan_outside_red"]} ({areas["cyan_outside_red_percentage"]:.1f}%)', 
                   (10, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

        return vis

    def process_channels(self, red_img: np.ndarray, cyan_img: np.ndarray, 
                        magenta_img: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Process all channels and create visualizations"""
        # Detect structures
        red_contours = self.detect_structures(red_img, self.red_threshold, 
                                            self.red_min_area, self.red_max_area)
        cyan_contours = self.detect_structures(cyan_img, self.cyan_threshold)
        magenta_contours = self.detect_structures(magenta_img, self.magenta_threshold)

        # Calculate areas and create overlap visualization
        areas = self.calculate_areas(red_contours, cyan_contours, red_img.shape)
        red_mask, cyan_mask, cyan_in_red, cyan_outside_red = areas['overlap_masks']
        overlap_vis = self.create_overlap_visualization(red_img, red_mask, cyan_mask, 
                                                      cyan_in_red, cyan_outside_red, areas)

        # Create standard visualizations
        red_vis = self.create_visualization(red_img, red_contours, 'Red', self.red_threshold)
        cyan_vis = self.create_visualization(cyan_img, cyan_contours, 'Cyan', self.cyan_threshold)
        magenta_vis = self.create_visualization(magenta_img, magenta_contours, 'Magenta', self.magenta_threshold)

        return red_vis, cyan_vis, magenta_vis, overlap_vis

def main():
    """Interactive threshold adjustment"""
    analyzer = MultichannelAnalyzer()
    
    # Load channel images
    red_path = "red_channel.tif"
    cyan_path = "cyan_channel.tif"
    magenta_path = "magenta_channel.tif"
    
    try:
        red_img, cyan_img, magenta_img = analyzer.load_images(red_path, cyan_path, magenta_path)
    except ValueError as e:
        print(f"Error loading images: {e}")
        return

    def update_display():
        red_vis, cyan_vis, magenta_vis, overlap_vis = analyzer.process_channels(red_img, cyan_img, magenta_img)
        cv2.imshow('Red Channel', red_vis)
        cv2.imshow('Cyan Channel', cyan_vis)
        cv2.imshow('Magenta Channel', magenta_vis)
        cv2.imshow('Overlap Analysis', overlap_vis)

    def on_red_threshold(value):
        actual_threshold = 255 - value
        analyzer.update_thresholds(red_threshold=actual_threshold)
        update_display()

    def on_min_area(value):
        analyzer.update_thresholds(red_min_area=value)
        update_display()

    def on_max_area(value):
        analyzer.update_thresholds(red_max_area=value * 1000)
        update_display()

    def on_cyan_threshold(value):
        actual_threshold = 255 - value
        analyzer.update_thresholds(cyan_threshold=actual_threshold)
        update_display()

    def on_magenta_threshold(value):
        actual_threshold = 255 - value
        analyzer.update_thresholds(magenta_threshold=actual_threshold)
        update_display()

    # Create windows with trackbars
    cv2.namedWindow('Red Channel')
    cv2.namedWindow('Cyan Channel')
    cv2.namedWindow('Magenta Channel')
    cv2.namedWindow('Overlap Analysis')
    
    cv2.createTrackbar('Brightness', 'Red Channel', 0, 255, on_red_threshold)
    cv2.createTrackbar('Min Area', 'Red Channel', analyzer.red_min_area, 500, on_min_area)
    cv2.createTrackbar('Max Area', 'Red Channel', analyzer.red_max_area // 1000, 100, on_max_area)
    cv2.createTrackbar('Brightness', 'Cyan Channel', 0, 255, on_cyan_threshold)
    cv2.createTrackbar('Brightness', 'Magenta Channel', 0, 255, on_magenta_threshold)

    # Initial processing
    update_display()

    print("\nControls:")
    print("Adjust the sliders to change detection parameters.")
    print("Move Brightness sliders right for brighter structures.")
    print("The Overlap Analysis window shows:")
    print("  - Red: Red channel regions")
    print("  - Cyan: Cyan regions outside red")
    print("  - Magenta: Cyan regions inside red")
    print("Press 'q' or 'ESC' to quit.")
    
    # Main loop
    while True:
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q') or key == 27:  # q or ESC
            break

    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()