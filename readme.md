<div align="center">

# Deep Learning Enables Urban Change Profiling Through Alignment of Historical Maps

**Sidi Wu** · **Yizi Chen** · **Maurizio Gribaudi** · **Konrad Schindler** · **Clément Mallet** · **Julien Perret** · **Lorenz Hurni**

</div>

---


## Project Structure and Data Description

This repository contains datasets, processing scripts, models, and outputs for historical map georeferencing, displacement estimation, instance extraction, and change profiling.

Structure of this repository:

```
|
├── data                                              <- Example data used throughout the pipeline
│   ├── maps                                          <- Original historical map sheets
│   ├── gcps                                          <- Ground control points (.json)
│   ├── masks                                         <- District masks (.gpkg)
│   ├── vector_annotations                            <- Manually vectorized shapefiles
│   ├── instance_extraction                           <- Training samples for instance extraction
│   ├── displacement_estimation                       <- Samples for displacement estimation
│   ├── building_blocks                               <- Derived building block examples
│   └── aligned_instances                             <- Instances aligned by displacement fields
├── output                                            <- Created at runtime for pipeline outputs
│   ├── georeferenced                                 <- Georeferenced and rectified map sheets
│   ├── displacement_training                         <- Synthesized training data
│   ├── displacement_inference                        <- Displacement inference results
│   ├── instance_inference                            <- Instance extraction results
│   ├── derived_changes                               <- IoU-based change detection results
│   └── building_blocks_output                        <- Block-level change aggregation
├── code
│   ├── data_processing                               <- Preprocessing, georeferencing, cropping
│   │   ├── georeference_maps.py                      <- Georeference map images with GCPs
│   │   ├── rectify_maps.py                           <- Rectify maps to a common image plane
│   │   ├── crop_foreground_objects.py                <- Crop foreground objects for synthesis
│   │   └── crop_maps_for_training.py                 <- Crop paired map patches for training
│   ├── displacement_estimation                       <- Displacement estimation module
│   │   ├── data                                      <- Data loaders and triplet synthesis
│   │   ├── model                                     <- GluNet / WarpC-GluNet definitions
│   │   ├── train.py                                  <- Training entry point
│   │   ├── test.py                                   <- Testing with pretrained weights
│   │   ├── losses.py                                 <- Loss functions
│   │   ├── log.py                                    <- Logging utilities
│   │   └── plot.py                                   <- Visualization utilities
│   ├── instance_extraction                           <- Multi-temporal instance segmentation
│   │   ├── data                                      <- Data loaders
│   │   ├── utils__                                   <- Map reconstruction from patches
│   │   ├── model.py                                  <- Model with multi-temporal fusion
│   │   ├── train_instance_fuse.py                    <- Training entry point
│   │   ├── test_instance_fuse.py                     <- Testing entry point
│   │   └── infer_instance_fuse.py                    <- Full-dataset inference
│   └── change_profiling                              <- Change derivation and profiling
│       ├── warp_instances.py                         <- Align instances across years
│       ├── derive_changes.py                         <- IoU-based change maps
│       ├── merge_building_blocks.py                  <- Merge instances into blocks
│       ├── coco_pano_ext_demo_new                    <- IoU computation utilities
│       └── building_blocks_iou_and_vectorization.py  <- Block-level IoU and vectorization
├── pretrained_weights                                <- Pretrained weights (download, see 3.5)
├── environment.yml                                   <- Conda environment file
├── requirements.txt                                  <- pip requirements
└── readme.md
```

## Environment Setup

Set up the required environment using either `environment.yml` (Conda) or `requirements.txt` (pip).

**Using Conda:**
```bash
conda env create -f environment.yml
conda activate map_alignment
```

**Using pip:**
```bash
pip install -r requirements.txt
```

## 1. data

This folder contains example data used throughout the pipeline.

| Subfolder | Contents | Format |
|---|---|---|
| `maps` | Samples of original historical map sheets | .jpg |
| `gcps` | Ground control points used for georeferencing | .json |
| `masks` | District masks used to crop the exact spatial extent of interest | .gpkg |
| `vector_annotations` | Maps with manually vectorized annotations; can be visualized in GIS software such as QGIS | .shp |
| `instance_extraction` | Training samples for instance extraction; each sample consists of a map sequence and the ground truth for the target year only | .png |
| `displacement_estimation` | Samples for training and testing displacement estimation models | .png |
| `building_blocks` | Derived building blocks used for spatial aggregation and change analysis | .png |
| `aligned_instances` | Extracted instances aligned using predicted displacement fields; .npy files contain instance IDs, .png visualizations show different instance IDs as different gray values | .npy, .png |

## 2. output

This folder is created at runtime to store outputs generated by the different stages of the pipeline.

| Subfolder | Contents |
|---|---|
| `georeferenced` | Georeferenced and rectified map sheets. Georeferenced maps are organized by year (e.g., 1878), each folder containing georeferenced maps, district masks, and maps cropped by the district masks. Rectified maps are organized by district (e.g., 0006). |
| `displacement_training` | Cropped training pairs (Img_src, Img_tar), foreground objects, and text masks used to synthesize training data for displacement estimation. Each training pair forms a triplet with synthesized object changes and text displacement. |
| `displacement_inference` | Inference results from the pretrained displacement estimation model: **inf_predictions** (original map patches, their overlay, and the overlay of the aligned patches), **inf_displacement** (visualizations of the estimated displacement fields), **inf_img** (original images and images warped using the estimated displacement). |
| `instance_inference` | Inference results for instance extraction using the pretrained model. Instances are stored by ID in .npy format; visualizations are provided in .png format. |
| `derived_changes` | Change detection results derived from IoU scores computed on aligned instances. IoU values are binarized to indicate change vs. no-change. |
| `building_blocks_output` | Building blocks aggregating average IoU values to approximate changes within each block: color-mapped raster outputs (lower IoU = higher change intensity) and vectorized results in GeoJSON format. |

## 3. code

### 3.1. data_processing

Scripts for data preprocessing, georeferencing, rectification, and patch-wise cropping.

| Script | Description | Output location |
|---|---|---|
| `georeference_maps.py` | Converts map images (.jpg) into georeferenced GeoTIFFs (.tiff) using GCPs (.json). District masks (.gpkg) are used to crop maps to specific districts. Outputs are organized by year (e.g., 1878). | `output/georeferenced/` |
| `rectify_maps.py` | Rectifies maps of the same district from different years into a common image plane (same coordinate system and size). The 1878 map is used as the reference. Outputs are organized by district (e.g., 0006). | `output/georeferenced/rectified/` |
| `crop_foreground_objects.py` | Randomly crops foreground objects (up to 5 per sample) along with their masks to simulate object changes for displacement estimation. | `output/displacement_training/Foreground_img`, `output/displacement_training/Foreground_mask` |
| `crop_maps_for_training.py` | Crops pairs of maps from successive editions of the same district at identical spatial locations using district masks. Produces paired map patches with matching indices. | `output/displacement_training/Img_src`, `output/displacement_training/Img_tar` |

### 3.2. displacement_estimation

Code and resources for estimating spatial displacement between maps.

| File / folder | Description |
|---|---|
| `data` | Data loaders and utilities, including synthesis of training triplets using foreground change masks, text masks, and map tiling |
| `model` | Model definitions for displacement estimation; the architecture follows WarpC-GluNet |
| `results` | Created during training to store checkpoints and visualization outputs for training and validation |
| `train.py` | Train the displacement estimation model |
| `test.py` | Run testing using the pretrained model |
| `losses.py` | Loss functions, including the triplet consistency loss |
| `log.py` | Logging utilities |
| `plot.py` | Visualization and plotting utilities |

Train the displacement estimation model:
```bash
python train.py
```

Run testing using the pretrained model:
```bash
python test.py \
  --path ../../output/displacement_inference \
  --model ../../pretrained_weights/displacement_estimation/best_weight.pth \
  --year_sc 1900 \
  --year_tar 1937 \
  --cuda
```

### 3.3. instance_extraction

Code for multi-temporal instance segmentation.

| File / folder | Description |
|---|---|
| `data` | Data loaders for single images and image sequences |
| `utils__` | Utility functions to reconstruct full maps from patch-wise predictions |
| `model.py` | Instance extraction model architecture with multi-temporal feature fusion |
| `train_instance_fuse.py` | Train instance segmentation with multi-temporal feature fusion |
| `test_instance_fuse.py` | Test instance segmentation; produces small testing checkpoint samples |
| `infer_instance_fuse.py` | Run inference on the entire dataset (data/instance_extraction) and stitch patch-level predictions into a complete map |

### 3.4. change_profiling

Code to derive and profile spatial changes based on aligned instances.

| File / folder | Description | Output location |
|---|---|---|
| `warp_instances.py` | Warps extracted instances using estimated displacement fields to align instances across different years | — |
| `derive_changes.py` | Computes Intersection over Union (IoU) between aligned instances from different years to derive change maps | `output/derived_changes/` |
| `merge_building_blocks.py` | Merges extracted building instances into building blocks, divided by roads | — |
| `coco_pano_ext_demo_new` | Utilities for IoU computation | — |
| `building_blocks_iou_and_vectorization.py` | Aggregates IoU values per building block and vectorizes the results | `output/building_blocks_output/` |

### 3.5. pretrained_weights

Pretrained model weights for the different pipeline components.
Download them from [Google Drive](https://drive.google.com/drive/folders/1fJUkb1jj_YmGuyJoEbE6d9A6biy5IKAf?usp=sharing) and place them under `pretrained_weights/`.

## Citation   

If you use this method in your work, please cite our [paper](https://arxiv.org/pdf/2602.02154).

```markdown
@article{wu2026deep,
  title={Deep learning enables urban change profiling through alignment of historical maps},
  author={Wu, Sidi and Chen, Yizi and Gribaudi, Maurizio and Schindler, Konrad and Mallet, Cl{\'e}ment and Perret, Julien and Hurni, Lorenz},
  journal={arXiv preprint arXiv:2602.02154},
  year={2026}
}
```

## Issues and FAQ

Coming soon!