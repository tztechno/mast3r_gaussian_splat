# mast3r_gaussiansplat

---

### MASt3R-based Gaussian Splatting Pipeline Technical Specifications

#### 1.0 Overview

This document defines the technical specifications for a 3D reconstruction pipeline designed to address two major challenges in modern 3D reconstruction: the complexity of traditional Structure from Motion (SfM) and the performance constraints of general-purpose hardware. The pipeline aims to democratize the process of generating high-fidelity 3D scenes from 2D image collections. At its core is the strategic decision to replace traditional SfM tools with the deep learning model **MASt3R**, significantly streamlining the initial reconstruction process and enabling an end-to-end automated workflow. This specification details the pipeline architecture, technical specifics of each component, and the key technical decisions underpinning its design.

##### 1.1 Purpose and Features

The primary objective of this pipeline is to automate the generation of high-quality 3D scenes from multiple 2D images. It features the following key characteristics:

* **SfM Substitution via MASt3R:** Replaces traditional SfM toolchains (e.g., COLMAP) used for feature matching and camera pose estimation with a single deep learning model, MASt3R. This simplifies the initial 3D reconstruction and increases overall workflow efficiency.
* **DINO-based Intelligent Pair Selection:** Utilizes the **DINOv2** model to extract global features from images and calculate similarity. This intelligently selects image pairs most likely to contribute to the 3D reconstruction, reducing unnecessary computational load.
* **Biplet-Square Normalization:** Employs a unique dual-crop strategy that generates two overlapping square crops based on image orientation (left/right for landscape, top/bottom for portrait). This standardizes image size for downstream processing while maximizing the retention of peripheral information.
* **Rigid Memory Management:** Pursues extreme memory efficiency through image resizing, pair limits, and point cloud downsampling. This ensures stable operation even in environments with limited hardware resources, such as restricted GPU VRAM.
* **End-to-End Automation:** Functions as a comprehensive framework covering every stage from dependency installation and data preprocessing to 3D reconstruction and final Gaussian Splatting model training.

##### 1.2 Architectural Overview

The pipeline adopts a linear data-flow architecture consisting of the following sequential stages:

* **Environment Setup:** Installs and configures system dependencies, MASt3R, and Gaussian Splatting repositories/submodules.
* **Image Preprocessing:** Applies Biplet-Square Normalization to generate standardized square crops from raw 2D images.
* **Image Pair Selection:** Extracts global features using DINOv2 and selects optimal image pairs based on similarity.
* **MASt3R 3D Reconstruction:** The MASt3R model performs inference on selected pairs to output a `scene` object containing globally aligned point clouds () and camera poses ().
* **COLMAP Format Conversion:** Converts MASt3R output into COLMAP binary format. This includes **pose inversion** (to ) and **camera intrinsic scaling**.
* **Gaussian Splatting Training:** Executes the training of the Gaussian Splatting model using the COLMAP data and processed images to produce a renderable 3D model ().

---

#### 2.0 Environment Setup

The setup phase is critical to ensure stable execution. This stage ensures that all dependencies—from basic Python libraries to specialized frameworks like MASt3R and Gaussian Splatting—are accurately configured.

##### 2.1 Base Dependencies

Installed via `setup_base_environment`: `torch`, `torchvision`, `opencv-python`, `transformers` (for DINOv2), `pycolmap`, and `numpy==1.26.4` (fixed version for compatibility).

##### 2.2 MASt3R Setup

The `setup_mast3r` function clones the repository, installs the `dust3r` and `croco` submodules, and downloads the pre-trained weights (`MASt3R_ViTLarge_BaseDecoder_512_catmlpdpt_metric.pth`).

##### 2.3 Gaussian Splatting Setup

The `setup_gaussian_splatting` function clones the repository and builds the performance-critical submodules: `diff-gaussian-rasterization` and `simple-knn`.

##### 2.4 Memory Management Utilities

* **`clear_memory()`:** Releases GPU cache via `torch.cuda.empty_cache()` and forces CPU garbage collection via `gc.collect()` after major steps.
* **`get_memory_info()`:** Logs current GPU/CPU usage to monitor resource consumption and identify bottlenecks.

---

#### 3.0 Pipeline Components and Data Flow

##### 3.1 Step 1: Image Preprocessing (Biplet-Square Normalization)

This stage stabilizes performance for subsequent models by retaining peripheral visual information.

* **Orientation Detection:** Determines if an image is landscape () or portrait ().
* **Dual Cropping:**
* **Landscape:** Two  squares cropped from  and .
* **Portrait:** Two  squares cropped from  and .


* **Resizing:** Uses `LANCZOS` resampling to reach the target resolution (e.g., 1024px).

##### 3.2 Step 2: Image Pair Selection (DINO-based)

1. **Feature Extraction:** DINOv2 extracts a global feature vector summarizing the visual content.
2. **Similarity Calculation:** Computes a similarity matrix ().
3. **Diversity-Aware Selection:** If the number of pairs exceeds `max_pairs`, it prioritizes images with lower appearance counts to prevent visual "hubs" and ensure spatial robustness.

##### 3.3 Step 3: 3D Reconstruction (MASt3R)

1. **Inference:** The `AsymmetricMASt3R` model predicts relative camera poses and depth. To save memory, input is resized to .
2. **Global Alignment:** Uses `GlobalAlignerMode.PointCloudOptimizer` with `niter=150` to integrate all pairs into a consistent 3D scene.

##### 3.4 Step 4: Conversion to COLMAP Format

This stage meticulously bridges MASt3R’s native representation with the strict COLMAP binary format.

* **Point Cloud Downsampling:** Randomly reduces points to `max_points` to prevent memory overflow.
* **Pose Transformation:** Calculates the inverse matrix of MASt3R’s **camera-to-world (c2w)** poses to produce the **world-to-camera (w2c)** format required by COLMAP. Rotation matrices are converted to 4-element quaternions .
* **Intrinsic Scaling:** Scales the focal lengths and principal points from the  inference size to the 1024px processed image dimensions.

##### 3.5 Step 5: Gaussian Splatting Training

* **Performance Tuning:**
* `--resolution 2`: Reduces image resolution by half.
* `--densify_grad_threshold 0.001`: High threshold to limit total Gaussians.
* `--densification_interval 200`: Reduces the frequency of densification to lower computational load.



---

#### 4.0 Configuration Parameters

##### 4.1 Global Settings (Config Class)

| Parameter | Value | Description |
| --- | --- | --- |
| `IMAGE_SIZE` | 1024 | Target size after Biplet-Square Normalization. |
| `GLOBAL_TOPK` | 20 | Max similar pairs per image in DINO selection. |
| `MAST3R_IMAGE_SIZE` | 224 | Inference size for MASt3R (memory saving). |

##### 4.2 Runtime Parameters

| Parameter | Example Default | Description |
| --- | --- | --- |
| `iterations` | 1000 | Training iterations for Gaussian Splatting. |
| `max_pairs` | 1000 | Max image pairs for reconstruction. |
| `max_points` | 1,000,000 | Max 3D points to export to COLMAP. |

---

#### 5.0 Key Technical Decisions

* **MASt3R over COLMAP:** Consolidates a fragile multi-stage process (matching, verification, incremental reconstruction) into a single end-to-end differentiable model, reducing failure points.
* **Biplet-Square Strategy:** Intentionally preserves peripheral data often lost in center crops, providing more context for matching when image overlap is minimal.
* **Resource-Aware Design:** The pipeline is built on a philosophy of hardware survival. By downsampling at the inference level (224px), capping the data structure level (`max_points`), and managing the process level (`clear_memory`), the system remains robust against the constraints of real-world GPU VRAM.

