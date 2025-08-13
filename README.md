# ViT-LCA: Vision Transformer with Local Competition Algorithm

A research implementation of Vision Transformers (ViT) combined with Local Competition Algorithm (LCA) for efficient feature extraction and sparse coding across multiple datasets.

## Overview

This project implements a novel approach combining Vision Transformers with Local Competition Algorithm for feature extraction. The LCA algorithm provides sparse coding capabilities while maintaining the powerful representation learning of Vision Transformers.

## Requirements

- Python 3.7+
- PyTorch 1.12+
- Torchvision
- NumPy
- PIL (Pillow)
- Matplotlib
- Scikit-learn

## Installation

1. Clone the repository:
```bash
git clone https://github.com/Sanaz-Tak/ViT-LCA.git
cd ViT-LCA
```

2. Install dependencies:
```bash
pip install torch torchvision numpy pillow matplotlib scikit-learn
```

## Usage

### Quick Start

Run the interactive multi-dataset script:
```bash
python vit_lca_experiment.py
```

This will guide you through:
- Dataset selection (CIFAR-10, CIFAR-100, ImageNet)
- Model selection (ViT variants)
- Weight variant selection
- Automatic execution with optimal parameters

### Direct Execution

Run the main script directly with custom parameters:
```bash
python vit_lca_main.py --dataset cifar10 --model vit_b_16 --weight_variant imagenet1k
```

### Parameters

- `--dataset`: Dataset name (cifar10, cifar100, imagenet)
- `--model`: Model architecture (vit_b_16, vit_b_32, vit_l_16, vit_l_32, swin_b)
- `--weight_variant`: Pre-trained weights (default, imagenet1k, swag)
- `--data_path`: Path to dataset directory
- `--batch_size`: Batch size for processing (default: 50)
- `--max_samples`: Maximum samples to process (default: 50)
- `--dictionary_num`: Number of dictionary elements (default: 400)
- `--neuron_iter`: Number of neuron iterations (default: 50)
- `--lr_neuron`: Learning rate for neurons (default: 0.01)
- `--landa`: Sparsity parameter (default: 2)

## Dataset Preparation

### CIFAR-10/100
Data will be automatically downloaded to `./data_cifar10/` and `./data_cifar100/`.

### ImageNet
Organize your ImageNet data as follows:
```
data_imagenet/
├── train/
│   ├── class1/
│   ├── class2/
│   └── ...
└── val/
    ├── class1/
    ├── class2/
    └── ...
```

## Project Structure

- **`vit_lca_main.py`**: Core ViT-LCA implementation and main execution pipeline
- **`vit_lca_experiment.py`**: Interactive interface for dataset and model selection
- **`data_*/`**: Dataset directories (automatically created)


## Citation

If you use this code in your research, please cite:

```bibtex
@inproceedings{vitlca2025,
  title={ViT-LCA: A Neuromorphic Approach for Vision Transformers},
  author={Sanaz M. Takaghaj},
  booktitle={Artificial Intelligence Circuits and Systems (AICAS 2025)},
  year={2025}
}
```

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## Issues

If you encounter any issues, please report them on the [GitHub Issues](https://github.com/Sanaz-Tak/ViT-LCA/issues) page.

