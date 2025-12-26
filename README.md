# Modeling Emergent Vegetation Dynamics: An ARC-NCA Framework for RGB-to-NDVI Estimation from UAV Imagery

## Overview
This repository contains the code and experiments for a Master’s thesis investigating the use of Neural Cellular Automata (ARC-NCA) to reconstruct NDVI-like vegetation indices from UAV RGB imagery. The project explores self-organising, locally interacting models as an interpretable alternative to GAN-based approaches for vegetation modelling in precision agriculture.

## Technical Architecture

The system is organised as a modular pipeline comprising four main components:

1. **Data Preprocessing**: UAV RGB orthomosaics and multispectral data are harmonised through CRS validation, resolution alignment, and window-based sampling. Spectral bands are mapped to reflectance-consistent ranges, and RGB-derived vegetation indices can optionally be computed as auxiliary inputs.

2. **State Representation**: Each image tile is encoded into an initial NCA state tensor consisting of:
    - RGB input channels (and optional RGB-based indices).
    - Auxiliary spatial channels (e.g. normalised coordinates, masks).
    - Hidden state channels used for internal feature propagation.

3. **ARC-NCA Model**: The model applies convolution-based neighbourhood perception and a differentiable update rule to iteratively evolve the state over multiple rollout steps. Residual updates and internal memory enable the emergence of spatial vegetation patterns and NDVI-like representations.

4. **Training and Evaluation**: The automaton is trained on paired RGB–NDVI tiles using rollout-based supervision. Performance is evaluated using standard regression and structural metrics (R², RMSE, MAE, SSIM), and compared against GAN-based and index-based baselines.

This architecture supports scalable experimentation and analysis of both predictive performance and interpretability of learned vegetation dynamics.

**TODO: Add architecture diagram here.**

## Repository Layout
<!-- - `notebooks/nca_model.py`: ARC-NCA model, SSIM-stabilized loss, and `train_nca` loop (channels-last friendly, AMP-ready).
- `notebooks/ARC_NCA_Improved.ipynb`: Main end-to-end notebook (data loading, training, evaluation).
- `notebooks/ARC_NCA-RGB_to_NDVI*.ipynb`: Earlier experiments and variants.
- `notebooks/generate_orthomosaic.ipynb`: Tile predictions back into full-scene NDVI orthomosaics.
- `notebooks/performance.ipynb`: Evaluation and profiling helpers. -->
- `dataset/`: Local UAV scenes (large TIFs; not versioned in detail).
- `output/`: Example predictions / mosaics produced by the notebooks.
- `tests/`: Minimal stability checks for SSIM and the training loop.

## Data
- Place orthomosaic TIFs under `dataset/<scene>/tif/`. Multi-band files are supported; bands `[1,2,3,4,5]` are treated as RGB + NIR + extra.
- Optional shapefiles (e.g., botrytis, trunk masks) can be passed to the dataset for priors; see cells defining `VineyardDataset`.
- Patches are read lazily via rasterio windows; configure `patch_size` and `stride` to balance coverage and throughput.

## Environment Setup
1. Python 3.13+ is recommended. Create an env: `python -m venv .venv && source .venv/bin/activate`.
2. Download & install CUDA toolkit - [docs](https://developer.nvidia.com/cuda-downloads).
3. Install core deps (adjust CUDA wheels as needed) in pyproject.toml

## Quickstart (notebook workflow)
1. Open `notebooks/ARC_NCA-RGB_to_NDVI_v2.ipynb`.
2. Configure data paths in the dataset cell, for example:
   ```python
   dataset = VineyardDataset(
       data_dir="dataset/EscaYard/tif",
       shapefiles={"botrytis": "dataset/EscaYard/botrytis.shp",
                   "trunks": "dataset/EscaYard/trunk_locations/trunks.shp"},
       patch_size=128,
       stride=128,
       bands=[1,2,3,4,5]
   )
   ```
3. Build loaders and train:
   ```python
   from notebooks.nca_model import ARC_NCA, train_nca
   from torch.utils.data import DataLoader
   device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

   train_loader = DataLoader(dataset, batch_size=4, shuffle=True, num_workers=4)
   model = ARC_NCA(input_data_channels=3, hidden_channels=16, out_channels=1, fire_rate=0.5, device=device)

   model, history = train_nca(
       model,
       train_loader,
       train_loader,           # use a held-out loader for real validation
       epochs=10,
       rollout_steps=32,
       device=device,
       save_dir="checkpoints",
       use_amp=True
   )
   ```

## Deliverables
This repository contains all artifacts produced during the development of the thesis “*ARC-NCA for Vegetation Dynamics Reconstruction from RGB UAV Imagery*”.
The deliverables are organized to support reproducibility, experimentation, and future extension.

### Models & Learning Artifacts

| Artifact                  | Description                                                                 | Format                 |
| ------------------------- | --------------------------------------------------------------------------- | ---------------------- |
| ARC-NCA Model             | Adaptive Neural Cellular Automata architecture for RGB → NIR reconstruction | `.ipynb` (Juptyer Notebook)       |
| Trained Model Checkpoints | Saved ARC-NCA weights for different datasets/configurations                 | `.pth` / `.ckpt`        |
| Experiment Memory         | Training logs, hyperparameters, rollout configs                             | `.yaml`, `.json`, logs |

### Data Processing & Ground Truth

| Artifact                  | Description                                                          |
| ------------------------- | -------------------------------------------------------------------- |
| Preprocessing Pipeline    | Georeferencing, resolution matching, normalization, patch extraction |
| Ground-Truth NIR Workflow | Extraction of NIR reflectance from multispectral imagery             |
| NDVI Generation Pipeline  | NDVI computation from reconstructed NIR and RGB Red band             |
| Shapefile Integration     | Support for trunks, Botrytis clusters, and GCPs                      |

### Evaluation & Benchmarking

| Artifact                 | Description                                                 |
| ------------------------ | ----------------------------------------------------------- |
| Evaluation Framework     | Metric computation (RMSE, R², SSIM)                         |
| Baseline Implementations | GAN-based NDVI synthesis and regression-based VI estimation |
| Cross-Dataset Validation | Generalization testing across acquisition conditions        |

## License
Creative Commons Attribution-NonCommercial 4.0 International (CC BY-NC-SA 4.0). See `LICENSE` for details.
