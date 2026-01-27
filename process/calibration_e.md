
# MASt3R to COLMAP Coordinate Transformation Verification Tool

## Overview

This tool is designed to verify the accuracy of 3D coordinate transformations when converting MASt3R output into COLMAP format.

By generating a synthetic scene with known 3D coordinates and comparing them with the coordinates produced after passing through the conversion pipeline, you can ensure that the transformation logic maintains spatial integrity.

## File Structure

```
calibration_data_generator.py  # Calibration data generator
test_calibration_pipeline.py   # Test pipeline execution script
calibration_usage_example.py   # Usage examples
CALIBRATION_README.md          # This file

```

## Key Features

### 1. CalibrationDataGenerator

Generates synthetic scene data:

* **Known 3D Coordinates**: A cube-shaped point cloud (default: 100 points).
* **Camera Poses**: Multiple viewpoints arranged in a circular configuration (default: 4 views).
* **Ground Truth Storage**: Saves baseline data for verification.

```python
from calibration_data_generator import CalibrationDataGenerator

generator = CalibrationDataGenerator(num_views=4, num_points=100)
scene = generator.create_mock_scene(device='cpu')
generator.save_ground_truth(scene, './calibration_data')

```

### 2. test_calibration_pipeline

Runs a complete end-to-end test pipeline:

```python
from test_calibration_pipeline import test_calibration_pipeline
from your_module import mast3r_to_colmap

# Test your COLMAP conversion function
test_calibration_pipeline(mast3r_to_colmap, output_dir='./test_results')

```

### 3. verify_transformation

Compares Ground Truth with COLMAP output:

```python
from calibration_data_generator import verify_transformation

verify_transformation(
    ground_truth_path='./calibration_data',
    colmap_output_path='./colmap_output'
)

```

---

## Usage Instructions

### Method A: Automated Testing (Recommended)

1. **Import**: Bring in the test pipeline and your existing conversion function.
2. **Execute**: Run the `test_calibration_pipeline`.
3. **Check Results**: Review coordinate statistics, Procrustes alignment disparity, and generated comparison files.

```python
from test_calibration_pipeline import test_calibration_pipeline
from your_module import mast3r_to_colmap 

test_calibration_pipeline(
    colmap_converter_func=mast3r_to_colmap,
    output_dir='./calibration_test'
)

```

### Method B: Manual Testing

1. **Generate Calibration Data**: Create the mock scene and save ground truth.
2. **Process with Existing Pipeline**: Pass the `scene` object through your converter as usual.
3. **Verify**: Run the verification utility.

```python
from calibration_data_generator import CalibrationDataGenerator, verify_transformation

# 1. Setup
generator = CalibrationDataGenerator(num_views=6, num_points=120)
scene = generator.create_mock_scene(device='cpu')
generator.save_ground_truth(scene, './ground_truth')

# 2. Convert
mast3r_to_colmap(scene, images, './colmap_output')

# 3. Verify
verify_transformation('./ground_truth', './colmap_output')

```

---

## Generated Files

### Ground Truth Directory

```
calibration_data/
├── ground_truth_points3d.txt       # 3D Coordinates (X Y Z)
├── ground_truth_poses.json         # Camera Poses (4x4 matrices)
└── ground_truth_intrinsics.json    # Camera Intrinsics

```

### Verification Results

```
test_results/
├── ground_truth/                   # Ground Truth data
├── colmap_output/                  # Post-conversion COLMAP data
│   ├── cameras.txt
│   ├── images.txt
│   ├── points3D.txt
│   └── coordinate_comparison.txt   # Detailed comparison log

```

---

## Verification Metrics

1. **Point Count Consistency**: Compares the number of points in Ground Truth vs. COLMAP output.
2. **Coordinate Statistics**: Compares Mean, Std Dev, Min, and Max values.
3. **Procrustes Alignment**: Measures similarity after rigid transformation (lower is better, 0 is a perfect match).
4. **Coordinate Frame Changes**: Detects shifts in scale, rotation, or translation.

### Interpreting Results

| Result | Meaning |
| --- | --- |
| **✓ Disparity < 0.001** | **Coordinates Retained**: Only rigid transformation (rotation/translation) occurred. Relative positions are perfect. |
| **⚠ Disparity < 0.05** | **Approximate Match**: Minor differences likely due to numerical precision or optimization. Usually acceptable. |
| **❌ Disparity > 0.1** | **Significant Change**: Major differences detected. Check for scaling issues, non-linear transforms, or data loss. |

---

## Troubleshooting

* **"No points found in COLMAP output"**: Check if your conversion script is actually writing to `points3D.txt`.
* **Point counts do not match**: Points might be getting filtered out by thresholding logic in your converter.
* **Large coordinate changes**: Check your coordinate system definitions (Camera vs. World), scale normalization, and the order of matrix multiplications.

## Technical Details

* **World Coordinates**: The origin  is at the center of the synthetic cube.
* **Camera Model**: Pinhole model (Focal length: 500px, Principal Point: 256, 256, Image size: 512x512).
* **Point Cloud**: Includes the 8 vertices of a cube and equidistant points along its 12 edges.

**Would you like me to help you refine the `mast3r_to_colmap` conversion logic based on the results of this tool?**
