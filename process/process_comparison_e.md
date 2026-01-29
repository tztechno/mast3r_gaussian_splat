Based on the source provided, here is the comparison of the primary functions of `process1.py`, `process2.py`, and `process3_04.py`.

---

### Comparison Table: process1, 2, and 3

| Comparison Item | process1.py | process2.py | process3_04.py |
| --- | --- | --- | --- |
| **Intrinsics (Focal Length)** | Uses **scaled MASt3R estimates** | Uses **scaled MASt3R estimates** (supports iso/anisotropic) | **Simplified calculation** based on image size (`max(w, h) * 1.2`) |
| **Intrinsics (Principal Point)** | Uses **scaled MASt3R estimates** | Uses **scaled MASt3R estimates** | Fixed to the **image center** |
| **Camera Pose (Extrinsics)** | **Inverse matrix transform** of MASt3R poses (w2c) | **Inverse matrix transform** of MASt3R poses (w2c) | **Custom pose estimation** based on the median of 3D points |
| **3D Point Extraction & Filtering** | Random sampling (up to 1M pts) and NaN/Inf removal | Filtering based on **confidence threshold** (default 1.5) | **Confidence filtering** and sampling 10k points per image |
| **Color Information** | Colors from resized source images | Colors matching only filtered 3D points | Directly from scene image data |
| **Primary Output Files** | COLMAP sparse reconstruction binaries (cameras, images, points3D) | COLMAP sparse reconstruction binaries (with actual RGB colors) | Sparse binaries + **Depth and Normal maps** |
| **Main Use Case / Features** | Faithful conversion of MASt3R geometry to COLMAP | Emphasis on confidence filtering and accurate color extraction | Designed for integration with **COLMAP Dense Reconstruction** |

---

### Key Differences and Insights

1. **Camera Model Accuracy**
* **process1** and **process2** accurately reflect MASt3R's output by scaling the estimated focal length and principal point to the image dimensions.
* **process3** uses a simplified pinhole model where camera parameters are derived from a fixed formula.


2. **Pose Estimation Approach**
* **process1** and **process2** calculate the inverse of the "camera-to-world" matrix provided by MASt3R to fit the "world-to-camera" format required by COLMAP.
* **process3** employs a unique method of determining translation vectors based on the spatial distribution (median) of the generated 3D point cloud.


3. **Point Cloud Quality and Sampling**
* **process1** is designed to handle a high volume of points (up to 1 million).
* **process2** and **process3** prioritize data quality by using MASt3R's "confidence" scores to filter out unreliable points.


4. **Data Output Depth**
* **process3** provides the most comprehensive output, generating **depth maps and normal maps** in binary format. This makes it specifically optimized for COLMAP’s stereo pipeline, facilitating detailed 3D modeling beyond just sparse reconstruction.



