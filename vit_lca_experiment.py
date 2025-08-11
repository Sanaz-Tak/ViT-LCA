#!/usr/bin/env python3
"""
ViT-LCA Multi-Dataset Runner

This script provides an interactive interface for running the ViT-LCA project
with different datasets and Vision Transformer model configurations.

Author: Sanaz M. Takaghaj
License: MIT
"""

import subprocess
import sys
import os

def main():
    """
    Main function providing interactive dataset and model selection interface.
    
    Guides users through:
    1. Dataset selection (CIFAR-10, CIFAR-100, ImageNet)
    2. Vision Transformer model selection
    3. Weight variant selection
    4. Execution with optimal parameters
    """
    print("ViT-LCA Multi-Dataset Feature Extraction")
    print("=" * 45)
    print("Available datasets:")
    print("1. CIFAR-10 (10 classes)")
    print("2. CIFAR-100 (100 classes)")
    print("3. ImageNet (1000 classes)")
    print()
    
    while True:
        try:
            choice = input("Select dataset (1-3): ").strip()
            if choice == '1':
                dataset = 'cifar10'
                break
            elif choice == '2':
                dataset = 'cifar100'
                break
            elif choice == '3':
                dataset = 'imagenet'
                break
            else:
                print("Invalid choice. Please enter 1, 2, or 3.")
        except KeyboardInterrupt:
            print("\nExiting...")
            sys.exit(0)
    
    print(f"\nSelected dataset: {dataset}")
    
    # Configure data path and provide dataset-specific instructions
    if dataset == 'imagenet':
        data_path = './data_imagenet'
        print("Note: For ImageNet, ensure your data is organized as:")
        print("  data_imagenet/")
        print("  ├── train/")
        print("  │   ├── class1/")
        print("  │   ├── class2/")
        print("  │   └── ...")
        print("  └── val/")
        print("      ├── class1/")
        print("      ├── class2/")
        print("      └── ...")
    else:
        data_path = f'./data_{dataset}'
    
    print(f"Data will be stored in: {data_path}")
    
    # Present available Vision Transformer model architectures
    print("\nAvailable models:")
    vit_models = ['vit_b_16', 'vit_b_32', 'vit_l_16', 'vit_l_32', 'swin_b']
    for i, model in enumerate(vit_models, 1):
        print(f"{i}. {model}")
    
    while True:
        try:
            choice = input(f"\nSelect model (1-{len(vit_models)}): ").strip()
            choice = int(choice)
            if 1 <= choice <= len(vit_models):
                model = vit_models[choice - 1]
                break
            else:
                print(f"Invalid choice. Please enter a number between 1 and {len(vit_models)}.")
        except (ValueError, KeyboardInterrupt):
            if KeyboardInterrupt:
                print("\nExiting...")
                sys.exit(0)
            print("Invalid input. Please enter a number.")
    
    print(f"Selected model: {model}")
    
    # Present available pre-trained weight variants
    print("\nAvailable weight variants:")
    if model == 'vit_b_16':
        weight_variants = ['default', 'imagenet1k', 'swag']
        print("1. default (latest weights)")
        print("2. imagenet1k (standard ImageNet-1K weights)")
        print("3. swag (SWAG weights for better generalization)")
    else:
        weight_variants = ['default', 'imagenet1k']
        print("1. default (latest weights)")
        print("2. imagenet1k (standard ImageNet-1K weights)")
    
    while True:
        try:
            choice = input(f"\nSelect weight variant (1-{len(weight_variants)}): ").strip()
            choice = int(choice)
            if 1 <= choice <= len(weight_variants):
                weight_variant = weight_variants[choice - 1]
                break
            else:
                print(f"Invalid choice. Please enter a number between 1 and {len(weight_variants)}.")
        except (ValueError, KeyboardInterrupt):
            if KeyboardInterrupt:
                print("\nExiting...")
                sys.exit(0)
            print("Invalid input. Please enter a number.")
    
    print(f"Selected weight variant: {weight_variant}")
    
    # Verify main script availability
    if not os.path.exists('vit_lca_main.py'):
        print("\nError: vit_lca_main.py not found!")
        print("Please ensure vit_lca_main.py is in the current directory.")
        sys.exit(1)
    
    # Construct execution command with optimal parameters
    cmd = [
        'python', 'vit_lca_main.py',
        '--dataset', dataset,
        '--model', model,
        '--weight_variant', weight_variant,
        '--data_path', data_path,
        '--batch_size', '64',
        '--max_samples', '10000',
        '--dictionary_num', '50000',
        '--neuron_iter', '100',
        '--lr_neuron', '0.01',
        '--landa', '2'
    ]
    
    print(f"\nRunning command:")
    print(' '.join(cmd))
    print("\nPress Enter to continue or Ctrl+C to cancel...")
    
    try:
        input()
    except KeyboardInterrupt:
        print("\nCancelled.")
        sys.exit(0)
    
    # Run the command
    try:
        result = subprocess.run(cmd, check=True)
        print("\nExecution completed successfully!")
    except subprocess.CalledProcessError as e:
        print(f"\nExecution failed with error code {e.returncode}")
        sys.exit(1)
    except FileNotFoundError:
        print("\nError: Python executable not found. Make sure Python is installed and in your PATH.")
        sys.exit(1)

if __name__ == '__main__':
    main()
