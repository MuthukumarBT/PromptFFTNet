# Marine and Coastal Atmospheric Turbulence Removal Using Prompt-Guided Hierarchical Transformers with FFT

## 📘 Description

This project addresses **atmospheric turbulence removal** for **marine and coastal surveillance imagery**. It utilizes a custom-built **Prompt-Guided Hierarchical Transformer network (PromptFFTNet)** integrated with **Fast Fourier Transform (FFT)** for feature-level corrections.

The solution performs tasks such as **atmospheric turbulence removal** through prompt-based adaptation and supports both full-image and tiled inference. Evaluation metrics like **PSNR** and **SSIM** are used to assess output quality.

---

## 📋 Requirements

- Python 3.6.2
- PyTorch
- Lightning (PyTorch Lightning)
- OpenCV
- NumPy
- scikit-image
- tqdm

🛠️ Installation
git clone https://github.com/your-username/your-repo-name.git
cd your-repo-name

🛠️ Installation

python main.py \
  --cuda 0 \
  --mode 2 \
  --test_path /path/to/test/images/ \
  --output_path ./results/ \
  --ckpt_name /path/to/PromptFFTNetModel.ckpt \
  --tile False \
  --tile_size 128 \
  --tile_overlap 32
🔧 Argument Details

| Argument         | Description                                                                 |
| ---------------- | --------------------------------------------------------------------------- |
| `--cuda`         | CUDA device ID to use (default: 0)                                          |
| `--mode`         | Task: `0` = denoise, `1` = derain, `2` = dehaze, `3` = all-in-one (default) |
| `--test_path`    | Path to test images (single image or folder)                                |
| `--output_path`  | Path to save enhanced output                                                |
| `--ckpt_name`    | Checkpoint file path (`.ckpt` from training)                                |
| `--tile`         | Whether to use image tiling (for large images)                              |
| `--tile_size`    | Tile size when tiling is enabled (default: 128)                             |
| `--tile_overlap` | Amount of overlap between tiles (default: 32)                               |
