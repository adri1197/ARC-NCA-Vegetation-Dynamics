# Modeling Emergent Vegetation Dynamics: An ARC-NCA Framework for RGB-to-NDVI Estimation from UAV Imagery

## Overview
This project explores auto-regressive convolutional neural cellular automata (ARC-NCA) to estimate NDVI maps directly from RGB UAV orthomosaics. The model rolls out several NCA steps, using fixed 3×3 perception kernels, stochastic firing masks, and a lightweight update network to evolve hidden state, NDVI output, and alive masks. Training and evaluation are organized in a single notebook

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

## License
Creative Commons Attribution-NonCommercial 4.0 International (CC BY-NC-SA 4.0). See `LICENSE` for details.
