# Fluorescence Image Analysis Tool

This tool analyzes multichannel fluorescence images, specifically designed to work with separate channel TIFF files. It can detect structures in each channel and analyze the overlap between red and cyan channels.

## Installation

1. Make sure you have Python 3.7 or newer installed. You can download it from [python.org](https://python.org)

2. Create and activate a virtual environment (recommended):
   ```bash
   # On Windows
   python -m venv venv
   venv\Scripts\activate

   # On macOS/Linux
   python3 -m venv venv
   source venv/bin/activate
   ```

3. Install the required packages:
   ```bash
   pip install -r requirements.txt
   ```

## Usage

1. Place your TIFF files in the same directory as the script and rename them to:
   - red_channel.tif
   - cyan_channel.tif
   - magenta_channel.tif

   Or modify the file paths in the code to match your file names.

2. Run the script:
   ```bash
   python multichannel-analyzer.py
   ```

3. Use the interface:
   - Adjust brightness sliders to control detection sensitivity
   - For red channel, also adjust Min Area and Max Area to filter structures
   - The Overlap Analysis window shows:
     - Red regions in red
     - Cyan regions outside red in cyan
     - Overlapping regions in magenta
   - Press 'q' or ESC to quit

## Controls

- **Brightness Sliders**: Move right for brighter structures (more selective)
- **Min Area**: Minimum size of structures to detect (red channel only)
- **Max Area**: Maximum size of structures to detect (red channel only)

## Output

The tool provides:
- Visualization of detected structures in each channel
- Overlap analysis showing where cyan structures are inside/outside red regions
- Area measurements:
  - Total red area
  - Total cyan area
  - Area of cyan inside red (with percentage)
  - Area of cyan outside red (with percentage)

## Troubleshooting

1. If you get "No module named 'cv2'" error:
   ```bash
   pip install opencv-python
   ```

2. If you get "No module named 'numpy'" error:
   ```bash
   pip install numpy
   ```

3. For TIFF file loading issues:
   - Make sure your files are single-channel TIFF images
   - The script will automatically convert color TIFFs to grayscale
   - High bit-depth images (12-bit, 16-bit) are automatically normalized to 8-bit
