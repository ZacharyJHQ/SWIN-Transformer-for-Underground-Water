# --------------------------------------------------------
# Swin Transformer
# --------------------------------------------------------

import os
import time
import pickle

import torch
import numpy as np
from torchvision import datasets, transforms

from .samplers import SubsetRandomSampler
from PIL import Image
import io
from random_ground_env_v2 import RandomGroundEnv

def custom_collate_fn(batch):
    """Custom collate function to handle None values in batch"""
    # Filter out None values and items with None components
    filtered_batch = []
    for item in batch:
        if item is not None and len(item) == 3:
            input_tensor, target_tensor, output_path = item
            # Check if tensors are valid
            if input_tensor is not None and target_tensor is not None:
                # Handle None output_path by converting to empty string
                if output_path is None:
                    output_path = ""
                filtered_batch.append((input_tensor, target_tensor, output_path))
    
    if len(filtered_batch) == 0:
        # Return empty batch if all items are None - updated for new data format
        default_input = torch.zeros(1, 6, 224, 224)  # 6 input channels
        default_target = torch.zeros(1, 1, 224, 224)  # 1 output channel
        return default_input, default_target, [""]
    
    # Separate the components
    inputs = [item[0] for item in filtered_batch]
    targets = [item[1] for item in filtered_batch]
    paths = [item[2] for item in filtered_batch]
    
    # Stack tensors
    try:
        input_batch = torch.stack(inputs)
        target_batch = torch.stack(targets)
        return input_batch, target_batch, paths
    except Exception as e:
        print(f"Error in collate function: {e}")
        # Return default batch - updated for new data format
        default_input = torch.zeros(1, 6, 224, 224)  # 6 input channels
        default_target = torch.zeros(1, 1, 224, 224)  # 1 output channel
        return default_input, default_target, [""]

def build_loader(config, dataset_type='train'):
    dataset_train, dataset_val, dataset_test = build_dataset(config)

    if dataset_type == 'train':
        dataset = dataset_train
        # Use RandomSampler for training to shuffle data
        sampler = torch.utils.data.RandomSampler(dataset)
        drop_last = True
    elif dataset_type == 'val':
        dataset = dataset_val
        # Use SequentialSampler for validation to ensure consistent evaluation
        sampler = torch.utils.data.SequentialSampler(dataset)
        drop_last = False
    elif dataset_type == 'test':
        dataset = dataset_test
        # Use SequentialSampler for testing to ensure consistent evaluation
        sampler = torch.utils.data.SequentialSampler(dataset)
        drop_last = False
    else:
        raise ValueError(f"Unknown dataset_type: {dataset_type}")

    data_loader = torch.utils.data.DataLoader(
        dataset, sampler=sampler,
        batch_size=config.DATA.BATCH_SIZE,
        num_workers=config.DATA.NUM_WORKERS,
        pin_memory=config.DATA.PIN_MEMORY,
        drop_last=drop_last,
        collate_fn=custom_collate_fn,
    )

    return dataset, data_loader

def build_dataset(config):
    if config.DATA.DATASET == 'UndergroundWater':
        # Load pickle data
        pickle_path = os.path.join(config.DATA.DATA_PATH, 'mock_data-1754291967.pkl')
        if not os.path.exists(pickle_path):
            raise FileNotFoundError(f"Pickle file not found: {pickle_path}")
        
        with open(pickle_path, 'rb') as fp:
            data_list = pickle.load(fp)
        
        print(f"Loaded {len(data_list)} samples from {pickle_path}")
        
        # Split data into train/val/test (70%/15%/15%)
        total_samples = len(data_list)
        train_size = int(0.7 * total_samples)
        val_size = int(0.15 * total_samples)
        test_size = total_samples - train_size - val_size
        
        train_data = data_list[:train_size]
        val_data = data_list[train_size:train_size + val_size]
        test_data = data_list[train_size + val_size:]
        
        print(f"Data split - Train: {len(train_data)}, Val: {len(val_data)}, Test: {len(test_data)}")
        
        dataset_train = GroundWaterDataset(train_data, config, 'train')
        dataset_val = GroundWaterDataset(val_data, config, 'val')
        dataset_test = GroundWaterDataset(test_data, config, 'test')
    else:
        raise NotImplementedError("We only support UndergroundWater Now.")

    return dataset_train, dataset_val, dataset_test


def export_as_tensor(obj: RandomGroundEnv):
    """Convert RandomGroundEnv object to input and target tensors"""
    well_mesh = np.zeros((obj.num_row, obj.num_col))
    for _w in obj.well_stress_data[0]:
        well_mesh[_w[1], _w[2]] = _w[3]
    
    x = [
        np.ones((obj.num_row, obj.num_col)) * obj.z_top,
        obj.boundary.squeeze(0),
        obj.starting_head.squeeze(0),
        obj.layer_property.squeeze(0),
        well_mesh,
        np.ones((obj.num_row, obj.num_col)) * obj.z_bottom,
    ]
    x = np.stack(x, axis=0)  # Shape: (6, 100, 100)
    x = torch.from_numpy(x).float()
    y = torch.from_numpy(obj.head).float()  # Shape: (1, 100, 100)
    return x, y

class GroundWaterDataset(torch.utils.data.Dataset):
    def __init__(self, data_list, config, dataset_type='train'):
        self.data_list = data_list
        self.config = config
        self.dataset_type = dataset_type
        self.root = f"groundwater_{dataset_type}"
        
    def __len__(self):
        return len(self.data_list)
    
    def __getitem__(self, index):
        try:
            obj = self.data_list[index]
            input_tensor, target_tensor = export_as_tensor(obj)
            
            # Resize to match model input size if needed
            if input_tensor.shape[-1] != self.config.DATA.IMG_SIZE:
                input_tensor = torch.nn.functional.interpolate(
                    input_tensor.unsqueeze(0), 
                    size=(self.config.DATA.IMG_SIZE, self.config.DATA.IMG_SIZE),
                    mode='bilinear', 
                    align_corners=False
                ).squeeze(0)
                
            if target_tensor.shape[-1] != self.config.DATA.IMG_SIZE:
                target_tensor = torch.nn.functional.interpolate(
                    target_tensor.unsqueeze(0), 
                    size=(self.config.DATA.IMG_SIZE, self.config.DATA.IMG_SIZE),
                    mode='bilinear', 
                    align_corners=False
                ).squeeze(0)
            
            return input_tensor, target_tensor, ""
            
        except Exception as e:
            print(f"Error in GroundWaterDataset.__getitem__ at index {index}: {e}")
            import traceback
            traceback.print_exc()
            # Return default tensors
            default_input = torch.zeros(6, self.config.DATA.IMG_SIZE, self.config.DATA.IMG_SIZE)
            default_target = torch.zeros(1, self.config.DATA.IMG_SIZE, self.config.DATA.IMG_SIZE)
            return default_input, default_target, ""