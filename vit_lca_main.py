#!/usr/bin/env python3
"""
ViT-LCA: Vision Transformer with Local Competition Algorithm

This module implements the core ViT-LCA architecture combining Vision Transformers
with Local Competition Algorithm for sparse feature representation.

Author: Sanaz M. Takaghaj
License: MIT
"""

import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
from torchvision import datasets, transforms
import os
import numpy as np
from PIL import Image
import argparse
import time
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, classification_report
import pickle
import os
import sys
from torchvision.models import vision_transformer, swin_v2_b
from torchvision.models.vision_transformer import ViT_B_16_Weights, ViT_B_32_Weights, ViT_L_16_Weights, ViT_L_32_Weights
from torchvision.models.swin_transformer import Swin_V2_B_Weights

class LCA:
    """
    Local Competition Algorithm (LCA) implementation for sparse coding.
    
    LCA provides sparse feature representation through dictionary learning
    and competitive neuron dynamics.
    """
    def __init__(self, feature_size, dictionary_num, UPDATE_DICT, dictionary_iter, 
                 neuron_iter, lr_dictionary, lr_neuron, landa):
        """
        Initialize LCA parameters.
        
        Args:
            feature_size: Dimension of input features
            dictionary_num: Number of dictionary elements
            UPDATE_DICT: Flag to enable dictionary updates
            dictionary_iter: Number of dictionary update iterations
            neuron_iter: Number of neuron update iterations
            lr_dictionary: Learning rate for dictionary updates
            lr_neuron: Learning rate for neuron updates
            landa: Sparsity parameter (lambda)
        """
        self.feature_size = feature_size
        self.dict_num = dictionary_num
        self.UPDATE_DICT = UPDATE_DICT
        self.dict_iter = dictionary_iter
        self.neuron_iter = neuron_iter
        self.lr_dict = lr_dictionary
        self.lr_neuron = lr_neuron
        self.landa = landa
        self.dictionary = None
        self.data = None
        self.input = None
        self.a = None
        self.u = None

    def lca_update(self, phi, I):
        """
        Perform LCA update step with competitive dynamics.
        
        Args:
            phi: Dictionary matrix
            I: Identity matrix for regularization
        """
        device = self.input.device
        u_list = [torch.zeros([1, self.dict_num]).to(device)]
        a_list = [self.threshold(u_list[0], 'soft', True, self.landa).to(device)]

        batch_size = self.input.shape[0]
        input = self.input.reshape(batch_size, -1)

        S = input.T
        b = torch.matmul(S.T, phi)
        for t in range(self.neuron_iter):
            u = self.neuron_update(u_list[t], a_list[t], b, phi, I)
            u_list.append(u)
            a = self.threshold(u, 'soft', True, self.landa)
            a_list.append(a)

        self.a = a_list[-1]
        self.u = u_list[-1]
        del u_list, a_list, input, S, b, phi, I

    def loss(self):
        """
        Compute LCA loss combining approximation and sparsity terms.
        
        Returns:
            Total loss value
        """
        s = self.input.reshape(-1, 1)
        phi = self.dictionary.reshape(self.dict_num, -1).T
        a = self.a
        residual = s - torch.mm(phi, a.T)
        approximation_loss = .5 * torch.linalg.norm(residual, 'fro')
        sparsity_loss = self.landa * torch.sum(torch.abs(a))
        loss = approximation_loss + sparsity_loss
        print('Loss: {:.2f}'.format(loss.item()), 'approximation loss: {:.2f}'.format(approximation_loss.item()), 'sparsity loss: {:.2f}'.format(sparsity_loss.item()))
        return loss

    def dict_update(self):
        """
        Update dictionary using gradient descent on reconstruction error.
        """
        phi = self.dictionary.reshape(self.dict_num, -1).T
        phi = phi.to(self.input.device)
        S = self.input.reshape(-1, 1)
        d_phi = torch.matmul((S.reshape(-1, 50)-torch.matmul(phi, self.a.T)), self.a)
        d_dict = d_phi.T.reshape([self.dict_num, 1000])
        d_dict = d_dict.cpu()
        self.dictionary = self.dictionary + d_dict * self.lr_dict

    def threshold(self, u, type, rectify, landa):
        """
        Apply thresholding function to neuron activations.
        
        Args:
            u: Input activations
            type: Threshold type ('soft' or 'hard')
            rectify: Whether to apply rectification
            landa: Threshold parameter
            
        Returns:
            Thresholded activations
        """
        u_zeros = torch.zeros_like(u)
        if type == 'soft':
            if rectify:
                a_out = torch.where(torch.greater(u, landa), u - landa, u_zeros)
            else:
                a_out = torch.where(torch.ge(u, landa), u - landa,
                                    torch.where(torch.le(u, - landa), u + landa, u_zeros))
        elif type == 'hard':
            if rectify:
                a_out = torch.where(torch.gt(u, landa), u, u_zeros)
            else:
                a_out = torch.where(torch.ge(u, landa), u,
                                    torch.where(torch.le(u, -landa), u, u_zeros))
        else:
            assert False, (f'Parameter thresh_type must be "soft" or "hard", not {type}')
        return a_out

    def neuron_update(self, u_in, a_in, b, phi, I):
        """
        Update neuron activations using competitive dynamics.
        
        Args:
            u_in: Current neuron activations
            a_in: Current sparse codes
            b: Input projection
            phi: Dictionary matrix
            I: Identity matrix for regularization
            
        Returns:
            Updated neuron activations
        """
        Ga = torch.mm(phi.T, torch.mm(phi, a_in.T)) - torch.mm(I, a_in.T)
        du = b - Ga.T - u_in
        u_out = u_in + self.lr_neuron * du
        return u_out

def normalize(M):
    """
    Normalize matrix M to unit norm.
    
    Args:
        M: Input matrix
        
    Returns:
        Normalized matrix
    """
    sigma = torch.sum(M * M)
    return M / torch.sqrt(sigma)

def get_model_and_transforms(model_name, weight_variant='default'):
    """Get ViT or Swin model and appropriate transforms with weight variant selection."""
    if model_name == 'vit_b_16':
        if weight_variant == 'swag':
            model = vision_transformer.vit_b_16(weights=ViT_B_16_Weights.IMAGENET1K_SWAG_E2E_V1)
            transforms_func = ViT_B_16_Weights.IMAGENET1K_SWAG_E2E_V1.transforms()
        elif weight_variant == 'imagenet1k':
            model = vision_transformer.vit_b_16(weights=ViT_B_16_Weights.IMAGENET1K_V1)
            transforms_func = ViT_B_16_Weights.IMAGENET1K_V1.transforms()
        else:  # default
            model = vision_transformer.vit_b_16(weights=ViT_B_16_Weights.DEFAULT)
            transforms_func = ViT_B_16_Weights.DEFAULT.transforms()
        model.heads = nn.Identity()  # Remove classification head
        
    elif model_name == 'vit_b_32':
        if weight_variant == 'imagenet1k':
            model = vision_transformer.vit_b_32(weights=ViT_B_32_Weights.IMAGENET1K_V1)
            transforms_func = ViT_B_32_Weights.IMAGENET1K_V1.transforms()
        else:  # default
            model = vision_transformer.vit_b_32(weights=ViT_B_32_Weights.DEFAULT)
            transforms_func = ViT_B_32_Weights.DEFAULT.transforms()
        model.heads = nn.Identity()
        
    elif model_name == 'vit_l_16':
        if weight_variant == 'imagenet1k':
            model = vision_transformer.vit_l_16(weights=ViT_L_16_Weights.IMAGENET1K_V1)
            transforms_func = ViT_L_16_Weights.IMAGENET1K_V1.transforms()
        else:  # default
            model = vision_transformer.vit_l_16(weights=ViT_L_16_Weights.DEFAULT)
            transforms_func = ViT_L_16_Weights.DEFAULT.transforms()
        model.heads = nn.Identity()
        
    elif model_name == 'vit_l_32':
        if weight_variant == 'imagenet1k':
            model = vision_transformer.vit_l_32(weights=ViT_L_32_Weights.IMAGENET1K_V1)
            transforms_func = ViT_L_32_Weights.IMAGENET1K_V1.transforms()
        else:  # default
            model = vision_transformer.vit_l_32(weights=ViT_L_32_Weights.DEFAULT)
            transforms_func = ViT_L_32_Weights.DEFAULT.transforms()
        model.heads = nn.Identity()
        
    elif model_name == 'swin_b':
        if weight_variant == 'imagenet1k':
            model = swin_v2_b(weights=Swin_V2_B_Weights.IMAGENET1K_V1)
            transforms_func = Swin_V2_B_Weights.IMAGENET1K_V1.transforms()
        else:  # default
            model = swin_v2_b(weights=Swin_V2_B_Weights.DEFAULT)
            transforms_func = Swin_V2_B_Weights.DEFAULT.transforms()
        model.head = nn.Identity()  # Remove classification head
        
    else:
        raise ValueError(f"Unknown model: {model_name}")
    
    return model, transforms_func

def get_model_feature_dimensions(model_name):
    """Get the feature dimensions for each ViT or Swin model."""
    feature_dims = {
        'vit_b_16': 768,
        'vit_b_32': 768,
        'vit_l_16': 1024,
        'vit_l_32': 1024,
        'swin_b': 1024
    }
    
    if model_name not in feature_dims:
        raise ValueError(f"Unknown model: {model_name}")
    
    return feature_dims[model_name]

def get_dataset_info(dataset_name):
    """Get dataset information including number of classes and dataset class."""
    if dataset_name == 'cifar10':
        return {
            'num_classes': 10,
            'train_class': torchvision.datasets.CIFAR10,
            'test_class': torchvision.datasets.CIFAR10,
            'train_args': {'train': True},
            'test_args': {'train': False}
        }
    elif dataset_name == 'cifar100':
        return {
            'num_classes': 100,
            'train_class': torchvision.datasets.CIFAR100,
            'test_class': torchvision.datasets.CIFAR100,
            'train_args': {'train': True},
            'test_args': {'train': False}
        }
    elif dataset_name == 'imagenet':
        return {
            'num_classes': 1000,
            'train_class': torchvision.datasets.ImageFolder,
            'test_class': torchvision.datasets.ImageFolder,
            'train_args': {'root': './data_imagenet/train'},
            'test_args': {'root': './data_imagenet/val'}
        }
    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")

def main():
    parser = argparse.ArgumentParser(description='Multi-Dataset ViT self-attention Extraction with LCA')
    parser.add_argument('--dataset', type=str, default='cifar10', 
                       choices=['cifar10', 'cifar100', 'imagenet'],
                       help='Dataset to use')
    parser.add_argument('--model', type=str, default='vit_b_16', 
                       choices=['vit_b_16', 'vit_b_32', 'vit_l_16', 'vit_l_32', 'swin_b'],
                       help='ViT or Swin model to use')
    parser.add_argument('--weight_variant', type=str, default='default',
                       choices=['default', 'imagenet1k', 'swag'],
                       help='Weight variant to use (default, imagenet1k, or swag for vit_b_16)')
    parser.add_argument('--data_path', type=str, default='./data_cifar', help='Data path')
    parser.add_argument('--dictionary_num', type=int, default=400, help='Dictionary size')
    parser.add_argument('--neuron_iter', type=int, default=50, help='Neuron iterations')
    parser.add_argument('--lr_neuron', type=float, default=0.01, help='Neuron learning rate')
    parser.add_argument('--landa', type=float, default=2, help='Sparsity coefficient')
    parser.add_argument('--batch_size', type=int, default=50, help='Batch size for data loading and feature extraction')
    parser.add_argument('--max_samples', type=int, default=50, help='Maximum test samples for LCA processing')
    parser.add_argument('--num_workers', type=int, default=0, help='Number of DataLoader workers')
    parser.add_argument('--pin_memory', action='store_true', help='Pin memory for faster GPU transfer')
    
    args = parser.parse_args()
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    print(f"Dataset: {args.dataset}")
    print(f"Model: {args.model}")
    
    torch.manual_seed(1234)
    
    # Get dataset information
    dataset_info = get_dataset_info(args.dataset)
    num_classes = dataset_info['num_classes']
    
    model, transforms_func = get_model_and_transforms(args.model, args.weight_variant)
    model.eval()
    
    # Create data directory if it doesn't exist
    os.makedirs(args.data_path, exist_ok=True)
    
    # Load datasets based on dataset type
    if args.dataset in ['cifar10', 'cifar100']:
        # Load CIFAR datasets
        train_dataset = dataset_info['train_class'](
            root=args.data_path, download=True, transform=None, **dataset_info['train_args'])
        test_dataset = dataset_info['test_class'](
            root=args.data_path, download=True, transform=None, **dataset_info['test_args'])
        
        # Use standard transforms
        train_dataset.transform = transforms_func
        test_dataset.transform = transforms_func
    
    elif args.dataset == 'imagenet':
        # For ImageNet, we need the data to be already organized in folders
        if not os.path.exists('./data_imagenet/train') or not os.path.exists('./data_imagenet/val'):
            print("Error: ImageNet data not found!")
            print("Please organize your ImageNet data as:")
            print("  data_imagenet/")
            print("  ├── train/")
            print("  │   ├── class1/")
            print("  │   ├── class2/")
            print("  │   └── ...")
            print("  └── val/")
            print("      ├── class1/")
            print("      ├── class2/")
            print("      └── ...")
            sys.exit(1)
        
        train_dataset = dataset_info['train_class'](
            **dataset_info['train_args'], transform=transforms_func)
        test_dataset = dataset_info['test_class'](
            **dataset_info['test_args'], transform=transforms_func)
    
    # Create dataloaders
    training_dataloader = torch.utils.data.DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=False, 
        num_workers=args.num_workers, pin_memory=args.pin_memory)
    testing_dataloader = torch.utils.data.DataLoader(
        test_dataset, batch_size=args.batch_size, shuffle=False, 
        num_workers=args.num_workers, pin_memory=args.pin_memory)
    
    # Initialize LCA model
    lca_model = LCA(
        feature_size=32,
        dictionary_num=args.dictionary_num,
        UPDATE_DICT=False,
        dictionary_iter=1,
        neuron_iter=args.neuron_iter,
        lr_dictionary=0.001,
        lr_neuron=args.lr_neuron,
        landa=args.landa
    )
    
    # Note: LCA is a custom class, not nn.Module, so we can't use .to(device)
    # We'll move individual tensors to device as needed
    print(f"LCA model initialized (will use device: {device})")
    
    # Process training data incrementally
    print("Processing training data incrementally...")
    
    all_feature_maps = []
    all_labels = []
    
    sample_count = 0
    for batch_data, batch_labels in training_dataloader:
        print(f"Processing batch {sample_count//args.batch_size + 1}, samples {sample_count}-{sample_count + len(batch_data)}")
        
        with torch.no_grad():
            batch_features = model(batch_data.to(device))
            all_feature_maps.append(batch_features.cpu())
            all_labels.append(batch_labels)
            
        sample_count += len(batch_data)
        
        if sample_count >= args.dictionary_num:
            print(f"Collected {sample_count} samples, stopping data loading")
            break
        
        del batch_data, batch_features
        torch.cuda.empty_cache() if torch.cuda.is_available() else None
    
    # Concatenate all features and labels
    all_feature_maps = torch.cat(all_feature_maps, dim=0)
    all_labels = torch.cat(all_labels, dim=0)
    
    print(f"Feature maps shape: {all_feature_maps.shape}")
    
    # Get correct feature dimensions for the ViT model
    feature_size = get_model_feature_dimensions(args.model)
    print(f"Model {args.model} feature size: {feature_size}")
    
    # Flatten the features if they're not already flattened
    if len(all_feature_maps.shape) > 2:
        all_feature_maps = all_feature_maps.view(all_feature_maps.size(0), -1)
        print(f"Flattened feature maps shape: {all_feature_maps.shape}")
    
    # Create dictionary
    dict_size = min(args.dictionary_num, len(all_feature_maps))
    lca_model.dictionary = torch.zeros(dict_size, feature_size, device=device)
    
    for i in range(dict_size):
        lca_model.dictionary[i] = normalize(all_feature_maps[i].detach()).to(device)
    
    print(f"Dictionary created on device: {lca_model.dictionary.device}")
    torch.cuda.empty_cache() if torch.cuda.is_available() else None
      
    # Process data with LCA
    print("Processing training data with LCA for neural network training...")
    start_time = time.time()
    
    model = model.to(device)
    # Dictionary is already on device, no need to reshape and transfer
    phi = lca_model.dictionary.T  # Already on device
    I = torch.eye(lca_model.dict_num, device=device)  # Create on device
    
    print(f"LCA setup - phi device: {phi.device}, I device: {I.device}")
    
    # Initialize storage for training LCA results
    a_all_train = []
    all_train_labels_for_nn = []
    
    sample_count = 0
    batch_size = min(100, args.batch_size)
    
    for i in range(0, min(dict_size, len(all_labels)), batch_size):
        if sample_count >= dict_size:
            break
            
        end_idx = min(i + batch_size, dict_size, len(all_labels))
        batch_features = all_feature_maps[i:end_idx].to(device)
        batch_labels = all_labels[i:end_idx]
        
        print(f"Processing LCA batch {i//batch_size + 1} (samples {i}-{end_idx})")
        
        try:
            print(f"Running LCA on batch with shape: {batch_features.shape}")
            lca_model.input = batch_features
            lca_model.lca_update(phi, I)
            a = lca_model.a.clone().detach().type(torch.float)
            print(f"LCA completed, output shape: {a.shape}")
            
            a_all_train.append(a.cpu())
            all_train_labels_for_nn.append(batch_labels)
            sample_count += (end_idx - i)
            
        except Exception as e:
            print(f"Error in LCA processing: {e}")
            import traceback
            traceback.print_exc()
            break
        
        del batch_features, a
        torch.cuda.empty_cache() if torch.cuda.is_available() else None
    
    end_time = time.time()
    
    print(f"LCA processing completed in {end_time - start_time:.2f} seconds")
    print(f"Processed {sample_count} training samples for LCA")
    
    del all_feature_maps, all_labels
    torch.cuda.empty_cache() if torch.cuda.is_available() else None
    
    if a_all_train:
        print(f"Training LCA processing completed")
        print(f"Number of LCA batches: {len(a_all_train)}")
        print(f"First batch shape: {a_all_train[0].shape}")
        
        all_train_labels_concatenated = torch.cat(all_train_labels_for_nn, dim=0)
        print(f"Concatenated training labels shape: {all_train_labels_concatenated.shape}")
    else:
        print("No LCA processing completed - dictionary size may be larger than available training samples")
        sys.exit(1)
    
    # Train Neural Network classifier
    print("Training Neural Network classifier on training data...")
    
    nn_model = torch.nn.Sequential(
        torch.nn.Linear(lca_model.dict_num, 1000),
        torch.nn.ReLU(),
        torch.nn.Linear(1000, num_classes),
    ).to(device)
    
    lr = 1e-3
    loss_fn = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(nn_model.parameters(), lr=lr)
    nn_model.zero_grad()
    
    print("Training NN classifier...")
    for epoch in range(1000):
        total_loss = 0
        num_batches = 0
        
        for batch_idx, (batch_a, batch_labels) in enumerate(zip(a_all_train, all_train_labels_for_nn)):
            batch_a_normalized = batch_a.clone()
            for i in range(len(batch_a_normalized)):
                batch_a_normalized[i] = batch_a_normalized[i] / torch.max(batch_a_normalized[i])
            
            y0_batch = batch_labels.clone().detach().type(torch.int64).to(device)
            y_hot_batch = torch.nn.functional.one_hot(y0_batch, num_classes=num_classes).float().to(device)
            
            # Move batch data to device once
            batch_a_device = batch_a_normalized.to(device)
            y_pred = nn_model(batch_a_device)
            loss = loss_fn(y_pred, y_hot_batch)
            
            loss.backward()
            total_loss += loss.item()
            num_batches += 1
        
        optimizer.step()
        optimizer.zero_grad()
        
        if epoch % 100 == 99:
            avg_loss = total_loss / num_batches
            print(f'Epoch {epoch+1}, average classification loss: {avg_loss:.4f}')
    
    # Calculate training accuracy
    print("Calculating training accuracy...")
    correct_predictions = 0
    total_predictions = 0
    
    with torch.no_grad():
        for batch_idx, (batch_a, batch_labels) in enumerate(zip(a_all_train, all_train_labels_for_nn)):
            batch_a_normalized = batch_a.clone()
            for i in range(len(batch_a_normalized)):
                batch_a_normalized[i] = batch_a_normalized[i] / torch.max(batch_a_normalized[i])
            
            y0_batch = batch_labels.clone().detach().type(torch.int64).to(device)
            
            # Move batch data to device once
            batch_a_device = batch_a_normalized.to(device)
            y_pred_batch = nn_model(batch_a_device)
            nn_predictions_batch = torch.argmax(y_pred_batch, dim=1)
            
            correct_predictions += sum(nn_predictions_batch == y0_batch).item()
            total_predictions += len(y0_batch)
    
    accuracy_nn_train = correct_predictions / total_predictions
    print(f'Training accuracy (NN): {accuracy_nn_train:.4f}')
    
    # Process test data
    print("Processing test data and running LCA for testing...")
    start_time = time.time()
    
    a_all_test = []
    all_test_labels = []
    
    sample_count = 0
    for batch_data, batch_labels in testing_dataloader:
        if sample_count >= args.max_samples:
            break
            
        print(f"Processing test batch {sample_count//args.batch_size + 1}, samples {sample_count}-{sample_count + len(batch_data)}")
        
        with torch.no_grad():
            batch_features = model(batch_data.to(device))
            
            if len(batch_features.shape) > 2:
                batch_features = batch_features.view(batch_features.size(0), -1)
            
            lca_model.input = batch_features
            lca_model.lca_update(phi, I)
            a = lca_model.a.clone().detach().type(torch.float)
            
            a_all_test.append(a.cpu())
            all_test_labels.append(batch_labels)
            
        sample_count += len(batch_data)
        
        del batch_data, batch_features, a
        torch.cuda.empty_cache() if torch.cuda.is_available() else None
    
    a_all_test = torch.cat(a_all_test, 0)
    all_test_labels = torch.cat(all_test_labels, dim=0)
    end_time = time.time()
    
    print(f"Test LCA processing completed in {end_time - start_time:.2f} seconds")
    print(f"a_all_test shape: {a_all_test.shape}")
    print(f"Test labels shape: {all_test_labels.shape}")
    
    # Calculate Max-based accuracy
    indices = torch.argmax(a_all_test, dim=1).to('cpu')
    accuracy_max = sum(all_train_labels_concatenated[indices] == all_test_labels) / len(all_test_labels)
    print(f'Test accuracy (max): {accuracy_max:.4f}')
    
    # Calculate Sum-based accuracy
    print("Calculating Sum-based accuracy...")
    
    indices_dict = {}
    for digit in range(num_classes):
        indices = torch.nonzero(all_train_labels_concatenated[0:lca_model.dict_num] == digit).squeeze()
        # Ensure indices is at least 1-dimensional
        if indices.dim() == 0:
            indices = indices.unsqueeze(0)
        indices_dict[digit] = indices
    
    max_indices = []
    for i in range(len(all_test_labels)):
        data = [sum(a_all_test[i, indices_dict[digit]]) for digit in range(num_classes)]
        max_index = max(range(len(data)), key=lambda x: data[x])
        max_indices.append(max_index)
    
    accuracy_sum = sum(torch.tensor(max_indices) == all_test_labels) / len(all_test_labels)
    print(f'Test accuracy (sum): {accuracy_sum:.4f}')
    
    # Test neural network
    print("Testing trained neural network on test data...")
    
    a_all_test_normalized = a_all_test.clone()
    for i in range(len(a_all_test_normalized)):
        a_all_test_normalized[i] = a_all_test_normalized[i] / torch.max(a_all_test_normalized[i])
    
    with torch.no_grad():
        # Move test data to device once
        a_all_test_device = a_all_test_normalized.to(device)
        y_pred_test = nn_model(a_all_test_device)
        nn_predictions_test = torch.argmax(y_pred_test, dim=1)
        accuracy_nn_test = sum(nn_predictions_test == all_test_labels.to(device)) / len(all_test_labels)
        print(f'Test accuracy (NN): {accuracy_nn_test:.4f}')
    
    # Comprehensive Evaluation Metrics
    print("\n" + "="*60)
    print("COMPREHENSIVE EVALUATION METRICS")
    print("="*60)
    
    # Convert predictions to CPU for sklearn compatibility
    nn_predictions_cpu = nn_predictions_test.cpu().numpy()
    test_labels_cpu = all_test_labels.numpy()
    
    # 1. Detailed Classification Report
    print("\n1. DETAILED CLASSIFICATION REPORT:")
    print("-" * 40)
    class_report = classification_report(test_labels_cpu, nn_predictions_cpu, 
                                       target_names=[f'Class_{i}' for i in range(num_classes)],
                                       digits=4)
    print(class_report)
    
    # 2. Per-class Accuracy
    print("\n2. PER-CLASS ACCURACY:")
    print("-" * 40)
    from sklearn.metrics import accuracy_score
    per_class_accuracy = {}
    for class_idx in range(num_classes):
        class_mask = test_labels_cpu == class_idx
        if np.sum(class_mask) > 0:
            class_acc = accuracy_score(test_labels_cpu[class_mask], nn_predictions_cpu[class_mask])
            per_class_accuracy[class_idx] = class_acc
            print(f"Class {class_idx}: {class_acc:.4f} ({np.sum(class_mask)} samples)")
    
    # 3. Confusion Matrix
    print("\n3. CONFUSION MATRIX:")
    print("-" * 40)
    from sklearn.metrics import confusion_matrix
    cm = confusion_matrix(test_labels_cpu, nn_predictions_cpu)
    
    # Print confusion matrix in a readable format
    print("Predicted →")
    print("Actual ↓")
    print("     ", end="")
    for j in range(min(num_classes, 10)):  # Limit display for large number of classes
        print(f"{j:>4}", end="")
    if num_classes > 10:
        print(" ...", end="")
    print()
    
    for i in range(min(num_classes, 10)):
        print(f"{i:>3} ", end="")
        for j in range(min(num_classes, 10)):
            print(f"{cm[i,j]:>4}", end="")
        if num_classes > 10:
            print(" ...", end="")
        print()
    
    if num_classes > 10:
        print("... (showing first 10x10 for readability)")
    
    # 4. Statistical Analysis
    print("\n4. STATISTICAL ANALYSIS:")
    print("-" * 40)
    
    # Calculate confidence intervals using binomial approximation
    from scipy import stats
    
    def confidence_interval(accuracy, n_samples, confidence=0.95):
        """Calculate confidence interval for accuracy using normal approximation."""
        z_score = stats.norm.ppf((1 + confidence) / 2)
        margin_of_error = z_score * np.sqrt(accuracy * (1 - accuracy) / n_samples)
        return accuracy - margin_of_error, accuracy + margin_of_error
    
    # Confidence intervals for different accuracy measures
    n_test = len(test_labels_cpu)
    
    ci_max = confidence_interval(accuracy_max, n_test)
    ci_sum = confidence_interval(accuracy_sum, n_test)
    ci_nn = confidence_interval(accuracy_nn_test, n_test)
    
    print(f"Max-based accuracy: {accuracy_max:.4f} (95% CI: [{ci_max[0]:.4f}, {ci_max[1]:.4f}])")
    print(f"Sum-based accuracy: {accuracy_sum:.4f} (95% CI: [{ci_sum[0]:.4f}, {ci_sum[1]:.4f}])")
    print(f"Neural Network accuracy: {accuracy_nn_test:.4f} (95% CI: [{ci_nn[0]:.4f}, {ci_nn[1]:.4f}])")
    
    # 5. Error Analysis
    print("\n5. ERROR ANALYSIS:")
    print("-" * 40)
    
    # Find misclassified samples
    misclassified_mask = nn_predictions_cpu != test_labels_cpu
    n_misclassified = np.sum(misclassified_mask)
    
    if n_misclassified > 0:
        print(f"Total misclassified samples: {n_misclassified}/{n_test} ({n_misclassified/n_test*100:.2f}%)")
        
        # Most common error patterns
        error_pairs = list(zip(test_labels_cpu[misclassified_mask], nn_predictions_cpu[misclassified_mask]))
        from collections import Counter
        error_counter = Counter(error_pairs)
        
        print("\nTop 5 most common error patterns:")
        print("(True → Predicted): Count")
        for (true, pred), count in error_counter.most_common(5):
            print(f"  {true} → {pred}: {count}")
    else:
        print("Perfect classification! No errors found.")
    
    # 6. Performance Summary
    print("\n6. PERFORMANCE SUMMARY:")
    print("-" * 40)
    print(f"Dataset: {args.dataset.upper()}")
    print(f"Model: {args.model.upper()}")
    print(f"Weight variant: {args.weight_variant}")
    print(f"Number of classes: {num_classes}")
    print(f"Test samples: {n_test}")
    print(f"Best accuracy: {max(accuracy_max, accuracy_sum, accuracy_nn_test):.4f}")
    
    # Determine best method
    accuracies = [('Max-based', accuracy_max), ('Sum-based', accuracy_sum), ('Neural Network', accuracy_nn_test)]
    best_method, best_acc = max(accuracies, key=lambda x: x[1])
    print(f"Best method: {best_method}")
    
    print("\n" + "="*60)
    print("EVALUATION COMPLETE")
    print("="*60)
    
    print(f'\nTotal elapsed time: {end_time - start_time:.2f} seconds')
    print(f"{args.dataset.upper()} + {args.model.upper()} + LCA processing completed successfully!")

if __name__ == '__main__':
    main()
