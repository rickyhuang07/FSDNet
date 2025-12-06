# FSDNet (Frequency-Spectral Deepfake Detection Network)
** FSDNet is a PyTorch-based project for detecting deepfake images and videos using both frequency-domain features and ResNet-50 backbone. The framework supports training and evaluation for binary classification (real vs. fake).

---

## Features

- Train and evaluate multiple models:
  - **FSDNet** (custom model)
  - **ResNet50**
  - **EfficientNet-B0**
  - **MobileNetV2**
  - **HuggingFace pre-trained deepfake classifiers (`hf_df1`, `hf_df2`, `hf_df3`)**
- Support for combined image + video-frame datasets.
- Evaluation reports generation for test datasets.
- Configurable logging and checkpoint loading.

---

## Project Structure
```
FSDNet/
├── main.py                 # Entry point for training/evaluation
├── config/
│   └── config.py           # Configuration for datasets, models, and training
├── data/
│   └── dataset.py          # Dataset loaders and preprocessing
├── models/
│   └── fsdnet.py           # Custom FSDNet model definition
├── training/
│   └── trainer.py          # Training loop and checkpointing
├── evaluation/
│   └── evaluator.py        # Evaluation, threshold tuning, and reports
├── utils/
│   └── device.py           # Device setup and info
├── checkpoints/            # Saved model checkpoints
└── requirements.txt        # Python dependencies
```


## Installation

1. **Clone the repository:**

```bash
git clone https://github.com/powerkracker/FSDNet.git
cd FSDNet
```

2. **Create a virtual environment:**
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. **Install dependencies:**
```bash
pip install -r requirements.txt
```

## Usage

### Training

To train the models:

```bash
python main.py --mode train --model-type fsdnet
```
Options:

--model-type: fsdnet, resnet-50, efficientnet-b0, mobilenet-v2, hf-df1, hf-df2, hf-df3

--log-level: Logging level (DEBUG, INFO, WARNING, ERROR)

--log-file: Optional log file path

### Evaluation

To evaluate a trained model:

```bash
python main.py --mode evaluate --model-type fsdnet --checkpoint checkpoints/best_checkpoint.pth
```
Options:

--checkpoint: Path to model checkpoint for evaluation.

Other options same as training.

### Inference

```bash
python inference_with_evaluator.py \
  --checkpoint checkpoints/best_checkpoint.pth \
  --image-dir ./data/test \
  --device cuda \
  --output-file fsdnet_predictions.csv
```
Folder Structure Expected by the Dataset:

Option 1: Single folder (no labels)
```
dataset/
├── img001.jpg
├── img002.png
├── img003.jpeg
```

Option 2: Labeled (with ground truth)
```
dataset/
├── real/
│   ├── real_001.jpg
│   ├── real_002.png
│   ...
└── fake/
    ├── fake_001.jpg
    ├── fake_002.png
```

### Configuration

The main configuration is in `config.py`. Key parameters include:

- **Data**: Image paths, batch size, augmentation settings
- **Model**: RPSP radii count, ResNet backbone
- **Training**: Learning rate, epochs, optimizer settings
- **Hardware**: Device selection (auto/CUDA/MPS/CPU)


## Data Format

The model expects:
- **Image format**: JPG, PNG, BMP, TIFF
- **Image size**: Will be resized to 224x224 by default
- **Real images & Fake images**: The model expects two types of structure:

A. Explicit split subfolders (preferred)
```
dataset/
├── real/
│   ├── train/
│   │   ├── img001.jpg
│   │   ├── img002.jpg
│   │   └── ...
│   ├── val/
│   │   ├── img101.jpg
│   │   └── ...
│   └── test/
│       ├── img201.jpg
│       └── ...
└── fake/
    ├── train/
    │   ├── fake001.jpg
    │   └── ...
    ├── val/
    │   ├── fake101.jpg
    │   └── ...
    └── test/
        ├── fake201.jpg
        └── ...
```

B. Flat structure (no explicit splits)

If there are no train/val/test folders, the loader will automatically random-split the dataset using ratios in your config (e.g., 0.7/0.15/0.15):
```
dataset/
├── real/
│   ├── img001.jpg
│   ├── img002.jpg
│   └── ...
└── fake/
    ├── fake001.jpg
    ├── fake002.jpg
    └── ...
```

## Hardware Requirements

- **GPU**: CUDA-compatible GPU recommended for training
- **Apple Silicon**: MPS support for Mac users
- **CPU**: Fallback option for inference
- **Memory**: 8GB+ RAM recommended

## Logging

Logs are printed to console by default.

Optional --log-file can be used to save logs.

Adjust log verbosity with --log-level.

## Checkpoints & Evaluation

Model checkpoints are saved in checkpoints/.

Evaluator automatically finds the optimal threshold on validation data using the Youden strategy.

Evaluation reports are saved in outputs/evaluation/.

## Contributing

Contributions are welcome! Please open issues or pull requests for bug fixes, features, or improvements.

## Citation

If you use FSDNet in your research, please cite our project:

```bibtex
@misc{FSDNet2025,
  title={FSDNet: Frequency-Spatial Deepfake Detection Network},
  author={Ricky Huang, Jacob Trentini, Avery chen},
  year={2025},
  howpublished={\url{https://github.com/powerkracker/FSDNet}}
}
```
---

## License

MIT License

---

## Reference
[1] Ricky Huang, Jacob Trentini, Avery Chen, Ziliang Zong, "FSDNet: A Frequency-Spatial Dual Feature Extraction Networkfor Efficient and Generalizable Deepfake Facial Detection", The 41th ACM/SIGAPP Symposium On Applied Computing(<a href='https://www.sigapp.org/sac/sac2026/'>ACM SAC'26</a>), Thessaloniki, Greece
March 23-27, 2026. 



