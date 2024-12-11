import tkinter as tk
from tkinter import ttk, filedialog, messagebox, simpledialog
import json
import threading
import logging
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from torch.utils.data import DataLoader
import pandas as pd
import os
from transformers import PreTrainedTokenizerFast, AddedToken
from tokenizers import Tokenizer, models, pre_tokenizers, decoders, trainers, normalizers
from transformers import BertModel
import numpy as np
import psutil
from torch.amp import GradScaler, autocast
import linecache
import torch
from torch.autograd import Function
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')
import seaborn as sns


# Print whether CUDA is available
print(f"CUDA Available: {torch.cuda.is_available()}")

# Global tokenizer variable for multiprocessing
tokenizer = None
scaler = GradScaler()


def log_system_usage():
    cpu_percent = psutil.cpu_percent(interval=1)
    virtual_memory = psutil.virtual_memory()
    ram_used = virtual_memory.used / (1024 ** 3)  # Convert to GB
    ram_total = virtual_memory.total / (1024 ** 3)  # Convert to GB

    logging.info(f"CPU Usage: {cpu_percent}%")
    logging.info(f"RAM Usage: {ram_used:.2f} GB / {ram_total:.2f} GB")

@staticmethod
def convert_model_if_needed(state_dict):
            new_state_dict = {}
            for key, value in state_dict.items():
                # Handle positional encoding -> rotary embedding conversion
                if 'pos_encoder' in key:
                    rotary_key = key.replace('pos_encoder', 'rotary')
                    new_state_dict[rotary_key] = value
                else:
                    new_state_dict[key] = value
            return new_state_dict


def save_checkpoint(model, optimizer, epoch, phase, path):
    checkpoint = {
        'epoch': epoch,
        'phase': phase,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict()
    }
    torch.save(checkpoint, path)

def load_checkpoint(path, model, optimizer=None):
    checkpoint = torch.load(path)
    model.load_state_dict(checkpoint['model_state_dict'])
    if optimizer:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    return checkpoint['epoch'], checkpoint['phase']


def init_tokenizer(tokenizer_path):
    global tokenizer
    tokenizer = PreTrainedTokenizerFast(tokenizer_file=tokenizer_path)
    logging.info(f"Tokenizer pad_token set to: {tokenizer.pad_token}, ID: {tokenizer.pad_token_id}")


@staticmethod
def tokenize_chunk(chunk):
    # Tokenizer is now the global variable initialized in each process
    encoded = tokenizer(chunk, return_attention_mask=False, truncation=True, max_length=1024)
    return encoded['input_ids']

# Collate function
def collate_fn(batch):
    input_ids, attention_masks, labels, seq_lengths = zip(*batch)
    input_ids = torch.stack(input_ids)
    attention_masks = torch.stack(attention_masks)
    labels = torch.stack(labels)
    seq_lengths = torch.tensor(seq_lengths, dtype=torch.long)
    return input_ids, attention_masks, labels, seq_lengths

# Replace CustomTransformerModel with a standard Transformer for testing

class StandardTransformerModel(nn.Module):
    def __init__(self, vocab_size, embed_size, num_heads, num_layers, hidden_size, max_seq_length=1024):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_size)
        self.pos_encoder = PositionalEncoding(embed_size, max_seq_length)
        encoder_layer = nn.TransformerEncoderLayer(d_model=embed_size, nhead=num_heads, dim_feedforward=hidden_size, batch_first=True)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.fc_out = nn.Linear(embed_size, vocab_size)

    def forward(self, src, src_mask=None, src_key_padding_mask=None):
        src = self.embedding(src) * math.sqrt(self.embedding.embedding_dim)
        src = self.pos_encoder(src)
        output = self.transformer_encoder(src, mask=src_mask, src_key_padding_mask=src_key_padding_mask)
        logits = self.fc_out(output)
        return logits


class ChunkedDataset(torch.utils.data.Dataset):
    def __init__(self, tokenized_data_path, tokenizer, max_length=1024):
        self.tokenized_data_path = tokenized_data_path
        self.tokenizer = tokenizer
        self.max_length = max_length

        # Get a list of chunk files
        self.chunk_files = [os.path.join(self.tokenized_data_path, f) 
                            for f in os.listdir(self.tokenized_data_path) 
                            if f.startswith('chunk_') and f.endswith('.jsonl')]
        self.chunk_files.sort()  # Ensure the chunks are in order

        # Build an index mapping from global indices to (chunk_idx, sample_idx)
        self.index_mapping = []
        for chunk_idx, chunk_file in enumerate(self.chunk_files):
            with open(chunk_file, 'r', encoding='utf-8') as f:
                num_lines = sum(1 for _ in f)
            self.index_mapping.extend([(chunk_idx, i) for i in range(num_lines)])

        # Initialize current chunk data
        self.current_chunk_idx = -1  # Indicates no chunk is currently loaded
        self.current_chunk_data = []  # Will hold the data from the current chunk

    def __len__(self):
        return len(self.index_mapping)

    def __getitem__(self, idx):

        if idx < 0 or idx >= len(self.index_mapping):
            raise IndexError(f"Index {idx} out of range for dataset of size {len(self.index_mapping)}")

        chunk_idx, sample_idx = self.index_mapping[idx]

        # Load the appropriate chunk if not already loaded
        if self.current_chunk_idx != chunk_idx:
            self.load_chunk(chunk_idx)

        record = self.current_chunk_data[sample_idx]
        input_ids = record['input_ids']
        labels = record['labels']

        # Calculate original sequence length before padding
        original_seq_length = min(len(input_ids), self.max_length)
        logging.debug(f"original sequence length = {original_seq_length}")
        # Pad sequences to max_length
        input_ids = input_ids[:self.max_length] + [self.tokenizer.pad_token_id] * max(0, self.max_length - len(input_ids))
        labels = labels[:self.max_length] + [self.tokenizer.pad_token_id] * max(0, self.max_length - len(labels))

        assert isinstance(input_ids, list), "input_ids should be a list"
        assert isinstance(labels, list), "labels should be a list"
        assert all(isinstance(id, int) for id in input_ids), "All input_ids should be integers"
        assert all(isinstance(id, int) for id in labels), "All labels should be integers"
        assert len(input_ids) == self.max_length, "input_ids should be padded to max_length"
        assert len(labels) == self.max_length, "labels should be padded to max_length"
        
        input_ids = torch.tensor(input_ids, dtype=torch.long)
        labels = torch.tensor(labels, dtype=torch.long)
        attention_mask = (input_ids != self.tokenizer.pad_token_id).long()
        seq_lengths = torch.tensor(original_seq_length, dtype=torch.long)

        # Check for empty sequences
        if len(input_ids) == 0:
            logging.error(f"Empty input_ids at index {idx}.")
            raise ValueError(f"Empty input_ids at index {idx}.")
        if len(labels) == 0:
            logging.error(f"Empty labels at index {idx}.")
            raise ValueError(f"Empty labels at index {idx}.")
    
        return input_ids, attention_mask, labels, seq_lengths

    def load_chunk(self, idx):
        chunk_file = self.chunk_files[idx]
        with open(chunk_file, 'r', encoding='utf-8') as f:
            self.current_chunk_data = [json.loads(line.strip()) for line in f]
        self.current_chunk_idx = idx


    
# Positional Encoding for the Transformer
class PositionalEncoding(nn.Module):
    def __init__(self, embed_size, max_seq_length=1024):
        super().__init__()
        pe = torch.zeros(max_seq_length, embed_size)
        position = torch.arange(0, max_seq_length, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, embed_size, 2).float() * (-math.log(10000.0) / embed_size))
        pe[:, 0::2] = torch.sin(position * div_term)
        if embed_size % 2 == 0:
            pe[:, 1::2] = torch.cos(position * div_term)
        else:
            pe[:, 1::2] = torch.cos(position * div_term[:-1])
        pe = pe.unsqueeze(0)  # (1, max_seq_length, embed_size)
        self.register_buffer('pe', pe)

    def forward(self, x):
        # x: (batch_size, seq_len, embed_size)
        x = x + self.pe[:, :x.size(1), :]
        return x


# Custom Transformer Model for Language Modeling
class CustomTransformerModel(nn.Module):
    def __init__(self, vocab_size, embed_size, num_heads, num_layers, hidden_size, max_seq_length=1024):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_size)
        self.pos_encoder = PositionalEncoding(embed_size, max_seq_length)
        encoder_layer = nn.TransformerEncoderLayer(d_model=embed_size, nhead=num_heads, dim_feedforward=hidden_size, batch_first=True)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.fc_out = nn.Linear(embed_size, vocab_size)
        self.embed_size = embed_size
        self.copy_linear = nn.Linear(embed_size, embed_size)
        self.initialize_weights()
        self.initialize_copying_weights()

    def initialize_copying_weights(self):
        for name, param in self.named_parameters():
            if "attention" in name:
                param.data.fill_(0.5)  # Bias attention to input
            elif "embedding" in name:
                nn.init.xavier_uniform_(param)

    def initialize_weights(self):
        """Initialize weights with proper scaling for each layer type"""
        for name, param in self.named_parameters():
            if 'W_' in name:
                # Initialize weight matrices using xavier_uniform_
                if param.dim() >= 2:  # Check if parameter has at least 2 dimensions
                    nn.init.xavier_uniform_(param)
                else:
                    # For 1D parameters, use standard normal initialization
                    nn.init.normal_(param, mean=0.0, std=0.01)
            elif 'b_f' in name:  # Forget gate bias
                nn.init.ones_(param)
            elif 'b_' in name:  # Other biases
                nn.init.zeros_(param)
            logging.debug(f"Initialized {name} with shape {param.shape}")


    def forward(self, src, src_mask=None, src_key_padding_mask=None):
        src = self.embedding(src) * math.sqrt(self.embed_size)
        src = self.pos_encoder(src)
        encoder_output = self.transformer_encoder(src, mask=src_mask, src_key_padding_mask=src_key_padding_mask)

        # Gated copying mechanism
        copy_gate = torch.sigmoid(self.copy_linear(encoder_output))
        output = copy_gate * src + (1 - copy_gate) * encoder_output

        logits = self.fc_out(output)
        return logits
    def forward(self, src, src_mask=None, src_key_padding_mask=None):
        src = self.embedding(src)
        
        src = self.pos_encoder(src)
        
        encoder_output = self.transformer_encoder(src, mask=src_mask, src_key_padding_mask=src_key_padding_mask)
        
        # Gated copying mechanism
        copy_gate = torch.sigmoid(self.copy_linear(encoder_output))

        output = copy_gate * src + (1 - copy_gate) * encoder_output

        
        logits = self.fc_out(output)
        return logits

    def resize_token_embeddings(self, new_vocab_size):
        old_vocab_size, embed_size = self.embedding.weight.size()
        if new_vocab_size != old_vocab_size:
            new_embedding = nn.Embedding(new_vocab_size, embed_size)
            new_embedding.weight.data[:old_vocab_size] = self.embedding.weight.data
            self.embedding = new_embedding
            self.fc_out = nn.Linear(embed_size, new_vocab_size)


# MatMul-Free Language Model Components

class TernaryWeightFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, weight, scaling_factor=None):
        if scaling_factor is None:
            # Compute scaling factor based on mean absolute value
            scaling_factor = 1.0 / (weight.abs().mean() + 1e-5)
            # Clamp scaling factor to prevent numerical instability
            scaling_factor = torch.clamp(scaling_factor, 1e-4, 1e4)

        # Scale and ternarize weights
        scaled_weight = weight * scaling_factor
        # Ensure no NaN or Inf values
        if torch.isnan(scaled_weight).any() or torch.isinf(scaled_weight).any():
            logging.error("Quantized weights contain NaN or Inf.")
            raise ValueError("Quantized weights contain NaN or Inf.")
        
        ternary_weight = torch.sign(scaled_weight)
        
        # Save context for backward pass
        ctx.save_for_backward(weight, ternary_weight, scaling_factor.clone().detach().requires_grad_(True))
        return ternary_weight

    @staticmethod
    def backward(ctx, grad_output):
        weight, ternary_weight, scaling_factor = ctx.saved_tensors
        # Straight-through estimator with scaling factor
        grad_input = grad_output * scaling_factor
        return grad_input, None

def ternarize_weight(weight):
    ternary_weight = TernaryWeightFunction.apply(weight)
    if torch.isnan(ternary_weight).any() or torch.isinf(ternary_weight).any():
        logging.error("Ternarized weights contain NaN or Inf.")
        raise ValueError("Ternarized weights contain NaN or Inf.")
    return ternary_weight


# Updated Weight Quantization
class WeightQuantFunction(Function):
    @staticmethod
    def forward(ctx, weight, num_bits=8):
        """Improved weight quantization with better numerical stability"""
        # Calculate scale based on weight statistics
        max_val = weight.abs().max()
        scale = (2 ** (num_bits - 1) - 1) / (max_val + 1e-5)
        
        # Scale and quantize
        weight_scaled = weight * scale
        weight_clipped = torch.clamp(weight_scaled, -2**(num_bits-1), 2**(num_bits-1)-1)
        weight_rounded = torch.round(weight_clipped)
        
        # Rescale back
        weight_quant = weight_rounded / scale
        
        # Save for backward
        ctx.save_for_backward(weight, weight_quant)
        return weight_quant

    @staticmethod
    def backward(ctx, grad_output):
        """Straight-through estimator with gradient clipping"""
        weight, weight_quant = ctx.saved_tensors
        # Clip gradients to stabilize training
        grad_input = torch.clamp(grad_output, -1.0, 1.0)
        return grad_input, None


class MatMulFreeLinearFunction(Function):
    """
    Custom autograd function for a BitNet-style matrix multiplication-free linear layer.
    """
    @staticmethod
    def forward(ctx, input, weight):
        """
        Forward pass using BitNet logic.
        Args:
            input (torch.Tensor): Input tensor of shape (batch_size, input_dim).
            weight (torch.Tensor): Weight tensor of shape (input_dim, output_dim).
        Returns:
            torch.Tensor: Output tensor of shape (batch_size, output_dim).
        """
        logging.debug(f"Input to MatMulFreeLinear: mean={input.mean().item():.4f}, std={input.std().item():.4f}, shape={input.shape}")
        if torch.isnan(input).any() or torch.isinf(input).any():
            logging.error("Input contains NaN or Inf.")
            raise ValueError("Input contains NaN or Inf.")

        logging.debug(f"Sample input for MatMulFreeLinearFunction: {input[:10]}")
        # Convert weights to binary representation (sign-based quantization)
        binary_weight = torch.sign(weight)

        # Compute the linear operation without traditional multiplication
        pos_mask = (binary_weight > 0).float()
        neg_mask = (binary_weight < 0).float()
        logging.debug(f"pos_mask: mean={pos_mask.mean().item():.4f}, std={pos_mask.std().item():.4f}, shape={pos_mask.shape}")
        logging.debug(f"neg_mask: mean={neg_mask.mean().item():.4f}, std={neg_mask.std().item():.4f}, shape={neg_mask.shape}")
        logging.debug(f"Sample pos_mask values: {pos_mask.flatten()[:10]}")
        logging.debug(f"Sample neg_mask values: {neg_mask.flatten()[:10]}")
        pos_contrib = torch.matmul(input, pos_mask)
        neg_contrib = torch.matmul(input, neg_mask)
        # Log a sample of predictions and targets
        logging.debug(f"pos_contrib: mean={pos_contrib.mean().item():.4f}, shape={pos_contrib.shape}")
        logging.debug(f"neg_contrib: mean={neg_contrib.mean().item():.4f}, shape={neg_contrib.shape}")
        logging.debug(f"Sample pos_contrib values: {pos_contrib.flatten()[:10]}")
        logging.debug(f"Sample neg_contrib values: {neg_contrib.flatten()[:10]}")
        output = pos_contrib - neg_contrib
        logging.debug(f"output before skip connection: mean={output.mean().item():.4f}, shape={output.shape}")
        logging.debug(f"Sample output values: {output.flatten()[:10]}")

        # Save tensors for backward pass
        ctx.save_for_backward(input, binary_weight.t())
        logging.debug(f"Saved tensors: input shape={input.shape}, pos_mask shape={pos_mask.shape}, neg_mask shape={neg_mask.shape}")


        # Log details of the input tensor being saved
        logging.debug(f"Input to MatMulFreeLinear saved: mean={input.mean().item():.4f}, std={input.std().item():.4f}")
        logging.debug(f"Sample output values: {output.flatten()[:10]}")
        return output


    @staticmethod
    def backward(ctx, grad_output):
        """
        Backward pass for BitNet-style linear operation.
        Args:
            grad_output (torch.Tensor): Gradient of the loss with respect to the output.
        Returns:
            Tuple[torch.Tensor, torch.Tensor]: Gradients with respect to input and weight.
        """
        input, binary_weight_t = ctx.saved_tensors  # binary_weight was saved transposed
        logging.debug(f"binary_weight_t shape: {binary_weight_t.shape}")
        # Handle 2D and 3D binary_weight cases
        if binary_weight_t.ndim == 2:
            binary_weight = binary_weight_t  # Shape: [embed_size, vocab_size]
        elif binary_weight_t.ndim == 3:
            binary_weight = binary_weight_t.transpose(-2, -1)  # Swap last two dimensions for 3D
        else:
            raise ValueError(f"Unsupported binary_weight_t dimensions: {binary_weight_t.ndim}")

        logging.debug(f"Gradient input (dO) shape: {grad_output.shape}")
        logging.debug(f"Sample gradient input (dO): {grad_output.flatten()[:10]}")
        logging.debug(f"binary_weight mean: {binary_weight.mean().item():.4f}, std: {binary_weight.std().item():.4f}")
        logging.debug(f"Sample binary_weight values: {binary_weight.flatten()[:10]}")
        logging.debug(f"binary_weight shape: {binary_weight.shape}")

        # Compute gradients
        if grad_output.ndim == 2:  # Standard case
            grad_input = grad_output.matmul(binary_weight)  # Shape: [batch_size * seq_len, embed_size]
            logging.debug(f"Grad_input  shape: {grad_input.shape}")

            grad_weight = grad_output.t().matmul(input)  # Shape: [embed_size, vocab_size]
        elif grad_output.ndim == 3:  # Case for batch processing with 3D tensors
            grad_input = grad_output.matmul(binary_weight.transpose(-2, -1))  # Adjust for 3D
            logging.debug(f"Grad_input  shape: {grad_input.shape}")

            grad_weight = grad_output.transpose(-2, -1).matmul(input)  # Adjust for 3D
        else:
            raise ValueError(f"Unsupported grad_output dimensions: {grad_output.ndim}")
        # Log gradients for debugging

        logging.debug(f"grad_input mean: {grad_input.mean().item():.4f}, std: {grad_input.std().item():.4f}")
        logging.debug(f"Gradient weight shape: {grad_weight.shape}")
        logging.debug(f"grad_weight mean: {grad_weight.mean().item():.4f}, std: {grad_weight.std().item():.4f}")
        # Transpose grad_weight back if needed
        if grad_weight.ndim == 2:
            grad_weight = grad_weight.t()  # Ensure it matches the original weight shape
        elif grad_weight.ndim == 3:
            grad_weight = grad_weight.transpose(-2, -1)  # Adjust for 3D if needed

        logging.debug(f"Adjusted grad_weight shape: {grad_weight.shape}")
        return grad_input, grad_weight

class ActivationQuantFunction(Function):
    @staticmethod
    def forward(ctx, X, max_scale=1e3, min_scale=1e-3):
        """
        Forward pass for Activation Quantization using STE.
        """
        # Compute scaling factor s
        max_val = torch.max(torch.abs(X), dim=-1, keepdim=True)[0]
        s = 127.0 / (max_val + 1e-5)  # Prevent division by zero
        
        # Clamp scaling factor to prevent extreme values
        s = torch.clamp(s, min=min_scale, max=max_scale)
        
        # Quantize
        X_scaled = s * X
        X_quant = torch.clamp(torch.round(X_scaled), -128, 127) / s
        
        # Save tensors for backward
        ctx.save_for_backward(X, s)
        ctx.min_scale = min_scale
        ctx.max_scale = max_scale
        
        return X_quant

    @staticmethod
    def backward(ctx, dX_quant):
        """
        Backward pass for Activation Quantization using STE.
        """
        X, s = ctx.saved_tensors
        # Gradient is passed directly through STE
        dX = dX_quant.clone()
        return dX, None, None

class RMSNorm(nn.Module):
    def __init__(self, dim, eps=1e-6):
        super().__init__()
        self.eps = eps
        self.scale = nn.Parameter(torch.ones(dim))
    
    def forward(self, x):
        rms = torch.rsqrt(torch.mean(x * x, dim=-1, keepdim=True) + self.eps)
        return self.scale * (x * rms)

pad_token_id=1

class MLGRULayer(nn.Module):
    """Dirty MLGRUlayer for Matmulfreemodel"""
    def __init__(self, embed_size, hidden_size, eps=1e-5):
        super(MLGRULayer, self).__init__()
        self.cell = MLGRUCell(embed_size, hidden_size, eps)
        self.hidden_size = hidden_size
            
    def forward(self, x, seq_lengths):
        logging.debug(f"Input to MLGRULayer: {x.shape}")
        logging.debug(f"Output of Attention Layer: {x.shape}")

        batch_size, max_seq_len, _ = x.size()
        h_t = torch.zeros(batch_size, self.cell.hidden_size, device=x.device)
        outputs = []

        for t in range(max_seq_len):
            mask = (seq_lengths > t).float().unsqueeze(1)
            x_t = x[:, t, :]
            o_t, h_t = self.cell(x_t, h_t)

            # Add gated copying mechanism
            copy_gate = torch.sigmoid(self.cell.W_g @ x_t.t()).t()
            h_t = copy_gate * x_t + (1 - copy_gate) * h_t

            h_t = h_t * mask + h_t.detach() * (1 - mask)
            outputs.append(o_t.unsqueeze(1) * mask.unsqueeze(2))

        outputs = torch.cat(outputs, dim=1)
        # Log statistics after MLGRULayer
        if torch.isnan(outputs).any() or torch.isinf(outputs).any():
            logging.error("Outputs from MLGRULayer contain NaN or Inf.")
            raise ValueError("Outputs from MLGRULayer contain NaN or Inf.")

        return outputs

class MLGRUCell(nn.Module):
    """MLGRUCell implementation for matmulfreemodel"""
    def __init__(self, input_size, hidden_size, eps=1e-5):
        super(MLGRUCell, self).__init__()
        self.hidden_size = hidden_size
        self.eps = eps
        self.rms_norm = RMSNorm(hidden_size, eps=self.eps)

        # Initialize weights
        self.W_f = nn.Parameter(self.ternary_initialize((input_size, hidden_size)))
        self.W_c = nn.Parameter(self.ternary_initialize((input_size, hidden_size)))
        self.W_g = nn.Parameter(self.ternary_initialize((hidden_size, input_size)))

        self.b_f = nn.Parameter(torch.ones(hidden_size))  # Initialize forget gate bias to 1
        self.b_c = nn.Parameter(torch.zeros(hidden_size))
        self.b_g = nn.Parameter(torch.zeros(hidden_size))

        self.initialize_copying_weights()

    def initialize_copying_weights(self):
        """Initialize copying weights for mirror neuron behavior"""
        for name, param in self.named_parameters():
            if "attention" in name:
                param.data.fill_(0.5)  # Bias attention to input

    def ternary_initialize(self, size):
        """Generate a tensor of ternary weights {-1, 0, +1}."""
        # Randomly assign -1, 0, or +1 with equal probability
        rand_tensor = torch.rand(size)  # Random values in [0, 1)
        ternary_tensor = torch.zeros_like(rand_tensor)
        ternary_tensor[rand_tensor < 1/3] = -1  # ~33% probability for -1
        ternary_tensor[rand_tensor > 2/3] = 1   # ~33% probability for +1
        # Remaining ~33% remain 0
        return ternary_tensor


    def forward(self, x_t, h_t_minus_1):
        # RMS Normalization and Activation Quantization
        x_t = self.rms_norm(x_t)
        x_t = ActivationQuantFunction.apply(x_t)

        # Quantize and ternarize weights
        W_f = WeightQuantFunction.apply(self.W_f)
        W_c = WeightQuantFunction.apply(self.W_c)
        W_g = WeightQuantFunction.apply(self.W_g)
        
        W_f_ternary = ternarize_weight(W_f)
        W_c_ternary = ternarize_weight(W_c)
        W_g_ternary = ternarize_weight(W_g)

        # MatMul-Free Linear Operations
        f_t_linear = MatMulFreeLinearFunction.apply(x_t, W_f_ternary) + self.b_f
        c_t_linear = MatMulFreeLinearFunction.apply(x_t, W_c_ternary) + self.b_c
        g_t_linear = MatMulFreeLinearFunction.apply(x_t, W_g_ternary) + self.b_g

        # Activation Functions
        f_t = torch.sigmoid(f_t_linear)
        if torch.isnan(f_t).any() or torch.isinf(f_t).any():
            logging.error("f_t contains NaN or Inf after sigmoid in MLGRUCell.")
            raise ValueError("f_t contains NaN or Inf after sigmoid in MLGRUCell.")

        c_t = F.silu(c_t_linear)
        g_t = torch.sigmoid(g_t_linear)
        if torch.isnan(g_t).any() or torch.isinf(g_t).any():
            logging.error("g_t contains NaN or Inf after sigmoid in MLGRUCell.")
            raise ValueError("g_t contains NaN or Inf after sigmoid in MLGRUCell.")

        h_t = f_t * h_t_minus_1 + (1 - f_t) * c_t
        o_t = g_t * h_t

        return o_t, h_t

#hadamard product attention
class MatMulFreeAttention(nn.Module):
    """Hadamard product attention for matmulfree transformer"""
    def __init__(self, embed_size, num_heads, seq_length, eps=1e-6):
        super(MatMulFreeAttention, self).__init__()
        self.num_heads = num_heads
        self.head_dim = embed_size // num_heads
        assert (
            embed_size % num_heads == 0
        ), "Embedding size must be divisible by number of heads."

        self.scale = 1 / math.sqrt(self.head_dim)
        self.eps = eps
        self.output_projection = nn.Linear(embed_size, embed_size)

        # Learnable weights for queries, keys, and values
        self.W_q = nn.Parameter(torch.randn(embed_size, embed_size) * 0.01)
        self.W_k = nn.Parameter(torch.randn(embed_size, embed_size) * 0.01)
        self.W_v = nn.Parameter(torch.randn(embed_size, embed_size) * 0.01)

    def forward(self, x, attention_mask=None):
        batch_size, seq_len, embed_size = x.size()

        # Compute queries, keys, values
        q = torch.matmul(x, self.W_q)  # [batch_size, seq_len, embed_size]
        k = torch.matmul(x, self.W_k)
        v = torch.matmul(x, self.W_v)

        # Reshape and transpose for multi-head attention
        q = q.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)  # [batch_size, num_heads, seq_len, head_dim]
        k = k.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)  # [batch_size, num_heads, seq_len, head_dim]
        v = v.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)  # [batch_size, num_heads, seq_len, head_dim]


        # Compute attention scores
        k_t = k.transpose(-2, -1)  # Transpose last two dimensions for matmul
        attn_scores = torch.matmul(q, k_t) * self.scale  # [batch_size, num_heads, seq_len, seq_len]

        # Apply mask if provided
        if attention_mask is not None:
            attn_scores = attn_scores.masked_fill(~attention_mask.unsqueeze(1), float('-inf'))

        # Normalize attention scores
        attn_weights = F.softmax(attn_scores, dim=-1)  # [batch_size, num_heads, seq_len, seq_len]

        # Compute attention outputs
        attn_output = torch.matmul(attn_weights, v)  # [batch_size, num_heads, seq_len, head_dim]

        # Concatenate heads and project back to embed_size
        attn_output = attn_output.transpose(1, 2).contiguous()  # [batch_size, seq_len, num_heads, head_dim]
        attn_output = attn_output.view(batch_size, seq_len, embed_size)  # Combine heads into embed_size

        # Final projection to embed_size
        output = self.output_projection(attn_output)  # [batch_size, seq_len, embed_size]

        return output




class MatMulFreeGLU(nn.Module):
    """MatmulfreeGLU mechanism"""
    def __init__(self, input_size, hidden_size, eps=1e-5):
        super(MatMulFreeGLU, self).__init__()
        self.eps = eps
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.W_g = nn.Parameter(self.ternary_initialize((input_size, hidden_size)))
        self.W_u = nn.Parameter(self.ternary_initialize((input_size, hidden_size)))
        self.W_d = nn.Parameter(self.ternary_initialize((hidden_size, input_size)))

        self.rms_norm = RMSNorm(hidden_size, eps=self.eps)
        self.initialize_copying_weights()

    def initialize_copying_weights(self):
        for name, param in self.named_parameters():
            if "attention" in name:
                param.data.fill_(0.5)  # Bias attention to input

    def ternary_initialize(self, size):
        """Generate a tensor of ternary weights {-1, 0, +1}."""
        # Randomly assign -1, 0, or +1 with equal probability
        rand_tensor = torch.rand(size)  # Random values in [0, 1)
        ternary_tensor = torch.zeros_like(rand_tensor)
        ternary_tensor[rand_tensor < 1/3] = -1  # ~33% probability for -1
        ternary_tensor[rand_tensor > 2/3] = 1   # ~33% probability for +1
        # Remaining ~33% remain 0
        return ternary_tensor
    
    def forward(self, x):
        # Apply RMS normalization using custom Function
        x_norm = self.rms_norm(x)
        if torch.isnan(x_norm).any() or torch.isinf(x_norm).any():
            logging.error("x_norm contains NaN or Inf after rms_norm in MatMulFreeGLU.")
            raise ValueError("x_norm contains NaN or Inf after rms_norm in MatMulFreeGLU.")

        # Activation Quantization using custom Function
        x_quant = ActivationQuantFunction.apply(x_norm)
        if torch.isnan(x_quant).any() or torch.isinf(x_quant).any():
            logging.error("x_quant contains NaN or Inf after activation_quant in MatMulFreeGLU.")
            raise ValueError("x_quant contains NaN or Inf after activation_quant in MatMulFreeGLU.")

        # Weight Quantization
        W_g_bar = WeightQuantFunction.apply(self.W_g)
        W_u_bar = WeightQuantFunction.apply(self.W_u)
        W_d_bar = WeightQuantFunction.apply(self.W_d)

        # MatMul-Free Linear Operations

        g_t = MatMulFreeLinearFunction.apply(x, W_g_bar)
        logging.debug(f"Input to MatMulFreeLinear: mean={g_t.mean().item():.4f}, std={g_t.std().item():.4f}, shape={g_t.shape}")

        u_t = MatMulFreeLinearFunction.apply(x, W_u_bar)
        logging.debug(f"Input to MatMulFreeLinear: mean={u_t.mean().item():.4f}, std={u_t.std().item():.4f}, shape={u_t.shape}")

        p_t = F.silu(g_t) * u_t
        logging.debug(f"Input to MatMulFreeLinear: mean={p_t.mean().item():.4f}, std={p_t.std().item():.4f}, shape={p_t.shape}")

        d_t = MatMulFreeLinearFunction.apply(p_t, W_d_bar)
        logging.debug(f"Input to MatMulFreeLinear: mean={d_t.mean().item():.4f}, std={d_t.std().item():.4f}, shape={d_t.shape}")


        # Check for NaN or Inf in output
        if torch.isnan(d_t).any() or torch.isinf(d_t).any():
            logging.error("Output of MatMulFreeGLU contains NaN or Inf.")
            raise ValueError("Output of MatMulFreeGLU contains NaN or Inf.")


        return d_t


class MatMulFreeLanguageModel(nn.Module):
    """MatmukFreeLangiuagemodel concept with multihead attention"""
    def __init__(self, vocab_size, embed_size, hidden_size, seq_length, num_heads=8, eps=1e-5):
        super(MatMulFreeLanguageModel, self).__init__()
        self.eps=eps
        self.embedding = nn.Embedding(vocab_size, embed_size)
        self.attention = MatMulFreeAttention(embed_size, num_heads, seq_length, eps)  # Add attention
        self.mlgru_layer = MLGRULayer(embed_size, hidden_size, eps)
        self.glu = MatMulFreeGLU(hidden_size, hidden_size, eps)
        self.output_layer = nn.Linear(hidden_size, vocab_size)
        self.rms_norm = RMSNorm(hidden_size, eps=eps)
        self.initialize_ternarized_outputs()
        self.initialize_copying_weights()

    def initialize_copying_weights(self):
        for name, param in self.named_parameters():
            if "attention" in name:
                param.data.fill_(0.5)  # Bias attention to input

    def initialize_ternarized_outputs(self):
        with torch.no_grad():
            for name, param in self.named_parameters():
                if 'W_' in name:
                    # ternary_initialize expects a shape tuple
                    param.data = self.ternary_initialize(param.data.shape)
                elif 'b_f' in name:
                    nn.init.ones_(param)
                elif 'b_' in name:
                    nn.init.zeros_(param)
                logging.debug(f"Initialized {name} with shape {param.shape}")


    def ternary_initialize(self, size):
        """Generate a tensor of ternary weights {-1, 0, +1}."""
        # Randomly assign -1, 0, or +1 with equal probability
        rand_tensor = torch.rand(size)  # Random values in [0, 1)
        ternary_tensor = torch.zeros_like(rand_tensor)
        ternary_tensor[rand_tensor < 1/3] = -1  # ~33% probability for -1
        ternary_tensor[rand_tensor > 2/3] = 1   # ~33% probability for +1
        # Remaining ~33% remain 0
        return ternary_tensor


    def forward(self, input_ids, seq_length):
        logging.debug(f"Input IDs shape: {input_ids.shape}")

        # Embedding Layer
        x = self.embedding(input_ids)
        logging.debug(f"X passed to attention: mean={x.mean().item():.4f}, std={x.std().item():.4f}, shape={x.shape}")

        # Apply MatMul-Free Attention
        x = self.attention(x)
        logging.debug(f"X passed to MLGRU: mean={x.mean().item():.4f}, std={x.std().item():.4f}, shape={x.shape}")

        # MLGRULayer
        x = self.mlgru_layer(x, seq_length)
        logging.debug(f"X passed to GLU: mean={x.mean().item():.4f}, std={x.std().item():.4f}, shape={x.shape}")

        # MatMulFreeGLU

        x = self.glu(x)
        logging.debug(f"X passed to RMSNORM: mean={x.mean().item():.4f}, std={x.std().item():.4f}, shape={x.shape}")

        # Check if x is finite before RMS normalization
        if torch.isnan(x).any() or torch.isinf(x).any():
            logging.error("x contains NaN or Inf before rms_norm.")
            raise ValueError("x contains NaN or Inf before rms_norm.")

        # RMS Normalization using custom autograd Function
        x = self.rms_norm(x)
        logging.debug(f"X passed to Activationquant: mean={x.mean().item():.4f}, std={x.std().item():.4f}, shape={x.shape}")

        # Check for NaN or Inf after RMS normalization
        if torch.isnan(x).any() or torch.isinf(x).any():
            logging.error("x contains NaN or Inf after rms_norm.")
            raise ValueError("x contains NaN or Inf after rms_norm.")

        # Activation Quantization using custom autograd Function
        x = ActivationQuantFunction.apply(x)
        logging.debug(f"X passed to W_bar: mean={x.mean().item():.4f}, std={x.std().item():.4f}, shape={x.shape}")
        batch_size, seq_len, embed_size = x.shape

        # Check for NaN or Inf after activation quantization
        if torch.isnan(x).any() or torch.isinf(x).any():
            logging.error("x contains NaN or Inf after activation_quant.")
            raise ValueError("x contains NaN or Inf after activation_quant.")

        # Weight Quantization using custom autograd Function
        W_bar = WeightQuantFunction.apply(self.output_layer.weight)
        logging.debug(f"W_bar passed to Logits: mean={x.mean().item():.4f}, std={x.std().item():.4f}, shape={x.shape}")


        # MatMul-Free Linear Operation using custom autograd Function

        x = x.view(-1, embed_size)  # [batch_size * seq_len, embed_size]
        logging.debug(f"X passed to logits: mean={x.mean().item():.4f}, std={x.std().item():.4f}, shape={x.shape}")

        logits = MatMulFreeLinearFunction.apply(x, W_bar.t()) + self.output_layer.bias
        logging.debug(f"Logits passed to logits: mean={x.mean().item():.4f}, std={x.std().item():.4f}, shape={x.shape}")

        logits = logits.view(batch_size, seq_len, -1)
        logging.debug(f"2Logits passed to logits: mean={x.mean().item():.4f}, std={x.std().item():.4f}, shape={x.shape}")

        # Check for NaN or Inf in logits
        if torch.isnan(logits).any() or torch.isinf(logits).any():
            logging.error("Logits contain NaN or Inf after matmul_free_linear.")
            raise ValueError("Logits contain NaN or Inf after matmul_free_linear.")

        return logits


class UnifiedTransformerGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("Unified Transformer GUI")

        # Transformer Parameters
        self.num_parameters = tk.IntVar(value=512)
        self.num_heads = tk.IntVar(value=16)
        self.layers = []

        # Model Configuration Variables
        self.model_name = tk.StringVar(value="CustomTransformer")
        self.num_parameters = tk.IntVar(value=1024)

        self.vocab_size = tk.IntVar(value=30000)
        self.hidden_size = tk.IntVar(value=1024)
        self.num_heads = tk.IntVar(value=8)
        self.num_layers = tk.IntVar(value=32)
        self.pad_token_id = 1  # Default value, adjust based on your tokenizer setup

        # Device Selection Variable
        self.device_option = tk.StringVar(value="GPU" if torch.cuda.is_available() else "CPU")

        # Dynamically calculate parameters based on other inputs
        self.vocab_size.trace_add("write", lambda *args: self.update_num_parameters())
        self.hidden_size.trace_add("write", lambda *args: self.update_num_parameters())
        self.num_layers.trace_add("write", lambda *args: self.update_num_parameters())

        # Set initial calculated value
        self.update_num_parameters()

        # Training Parameters
        self.dataset_path = ""
        self.vocab_path = ""
        self.tokenizer_path = ""
        self.batch_size = tk.IntVar(value=12)
        self.learning_rate = tk.DoubleVar(value=0.0002)
        self.epochs = tk.IntVar(value=1)

        # Training Variables
        self.loss_history = []
        self.accuracy_history = []
        self.current_epoch = 0
        self.stop_training = threading.Event()

        # Model and Data Variables
        self.model = None
        self.tokenizer = None
        self.dataset_path = None
        self.vocab_path = None
        self.tokenizer_path = None
        self.model_path = None
        self.train_data = None  # To store the dataset
        self.tokenized_data_path = None  # To store the tokenized data file path
        self.test_bool=False

        # Device (CPU or GPU) - Initially set based on device_option
        self.device = torch.device(self.map_device(self.device_option.get()))

        # Select log file path
        self.select_log_file()

        # Setup logging
        logging.basicConfig(filename=self.log_file_path, level=logging.INFO,
                            format='%(asctime)s - %(levelname)s - %(message)s')

        logging.info(f"Using device: {self.device}")

        self.create_widgets()

    def map_device(self, selected_device):
        device_mapping = {
            "CPU": "cpu",
            "GPU": "cuda"
        }
        return device_mapping.get(selected_device, "cpu")

    def create_widgets(self):
        # Transformer Construction Frame
        transformer_frame = ttk.LabelFrame(self.root, text="Transformer Construction", padding=(10, 10))
        transformer_frame.pack(fill="x", padx=10, pady=5)

        ttk.Label(transformer_frame, text="Number of Parameters:").grid(row=0, column=0, sticky="w")
        ttk.Entry(transformer_frame, textvariable=self.num_parameters, state="readonly").grid(row=0, column=1)

        ttk.Label(transformer_frame, text="Number of Heads:").grid(row=1, column=0, sticky="w")
        ttk.Entry(transformer_frame, textvariable=self.num_heads).grid(row=1, column=1)

        ttk.Label(transformer_frame, text="Vocabulary Size:").grid(row=2, column=0, sticky="w")
        ttk.Entry(transformer_frame, textvariable=self.vocab_size).grid(row=2, column=1)

        ttk.Label(transformer_frame, text="Hidden Size:").grid(row=3, column=0, sticky="w")
        ttk.Entry(transformer_frame, textvariable=self.hidden_size).grid(row=3, column=1)

        ttk.Label(transformer_frame, text="Number of Layers:").grid(row=2, column=4, sticky="w")
        ttk.Entry(transformer_frame, textvariable=self.num_layers).grid(row=2, column=5)

        # Device Selection
        ttk.Label(transformer_frame, text="Select Device:").grid(row=4, column=0, sticky="w", pady=(10, 0))
        device_options = ["CPU"]
        if torch.cuda.is_available():
            device_options.append("GPU")
        device_combo = ttk.Combobox(transformer_frame, textvariable=self.device_option, values=device_options, state="readonly")
        device_combo.grid(row=4, column=1, sticky="w", pady=(10, 0))
        device_combo.bind("<<ComboboxSelected>>", self.on_device_change)

        # Attach parameter calculation to variable updates
        self.vocab_size.trace_add("write", lambda *args: self.update_num_parameters())
        self.hidden_size.trace_add("write", lambda *args: self.update_num_parameters())
        self.num_layers.trace_add("write", lambda *args: self.update_num_parameters())

        # For resuming training
        ttk.Button(transformer_frame, text="Select Model File", command=self.select_model_file).grid(row=3, column=2, pady=5)

        # Architecture selection
        self.architecture = tk.StringVar(value="Transformer")
        ttk.Label(transformer_frame, text="Select Architecture:").grid(row=0, column=2, sticky="w")
        ttk.Combobox(transformer_frame, textvariable=self.architecture, values=["Transformer", "MatMul-Free LM"], state="readonly").grid(row=0, column=3)

        ttk.Button(transformer_frame, text="Add Layer", command=self.add_layer).grid(row=4, column=0, pady=5)
        ttk.Button(transformer_frame, text="Save Transformer and Model", command=self.save_transformer_and_model).grid(row=1, column=3, pady=5)
        ttk.Button(transformer_frame, text="Load Transformer", command=self.load_transformer).grid(row=1, column=2, pady=5)
        ttk.Button(transformer_frame, text="Initialize/Load Model", command=self.load_model).grid(row=2, column=3, pady=5)

        # Data Selection Frame
        data_frame = ttk.LabelFrame(self.root, text="Data Selection", padding=(10, 10))
        data_frame.pack(fill="x", padx=10, pady=5)
        self.use_chunked_dataset = tk.BooleanVar(value=False)
        self.test_bool = tk.BooleanVar(value=False)
        
        ttk.Checkbutton(data_frame, text="Use Chunked Dataset", variable=self.use_chunked_dataset).pack(pady=5)
        ttk.Checkbutton(data_frame, text="Use Std/bert Model", variable=self.test_bool).pack(pady=5)
        ttk.Button(data_frame, text="Select Dataset Directory", command=self.select_dataset).pack(pady=5)
        ttk.Button(data_frame, text="Load Dataset", command=self.load_dataset).pack(pady=5)
        ttk.Button(data_frame, text="Save Dataset as Text File", command=self.save_dataset_as_text).pack(pady=5)
        ttk.Button(data_frame, text="Select Vocabulary File", command=self.select_vocab).pack(pady=5)
        ttk.Button(data_frame, text="Create Tokenizer from Vocab", command=self.create_tokenizer_from_vocab).pack(pady=5)
        ttk.Button(data_frame, text="Load Tokenizer", command=self.load_tokenizer).pack(pady=5)
        ttk.Button(data_frame, text="Test Tokenizer", command=self.test_tokenizer).pack(pady=5)
        ttk.Button(data_frame, text="Test Training", command=self.training_test).pack(pady=5)


        # New buttons for tokenized data
        ttk.Button(data_frame, text="Select/Create Tokenized Data", command=self.select_or_create_tokenized_data).pack(pady=5)
        ttk.Button(data_frame, text="Tokenize Data", command=self.tokenize_data).pack(pady=5)

        # Training Configuration Frame
        train_frame = ttk.LabelFrame(self.root, text="Training Configuration", padding=(10, 10))
        train_frame.pack(fill="x", padx=10, pady=5)

        ttk.Label(train_frame, text="Batch Size:").grid(row=0, column=0, sticky="w")
        ttk.Entry(train_frame, textvariable=self.batch_size).grid(row=0, column=1)

        ttk.Label(train_frame, text="Learning Rate:").grid(row=1, column=0, sticky="w")
        ttk.Entry(train_frame, textvariable=self.learning_rate).grid(row=1, column=1)

        ttk.Label(train_frame, text="Epochs:").grid(row=2, column=0, sticky="w")
        ttk.Entry(train_frame, textvariable=self.epochs).grid(row=2, column=1)

        ttk.Button(train_frame, text="Start Training", command=self.start_training).grid(row=3, column=0, pady=5)
        ttk.Button(train_frame, text="Save Model", command=self.save_model).grid(row=3, column=1, pady=5)
        ttk.Button(train_frame, text="Stop Training", command=self.stop_training_command).grid(row=4, column=0, pady=5)
        self.training_mode = tk.StringVar(value="response")  # Default
        training_modes = ["imitation", "completion", "response"]
        ttk.Combobox(data_frame, textvariable=self.training_mode, values=training_modes, state="readonly").pack(pady=5)
        ttk.Button(train_frame, text="Visualize Gradients", command=self.trigger_gradient_visualization).grid(row=5, column=0, pady=5)
        ttk.Button(train_frame, text="Visualize Outputs", command=self.trigger_output_visualization).grid(row=5, column=1, pady=5)

        # Progress Bar
        self.progress_bar = ttk.Progressbar(self.root, orient='horizontal', length=400, mode='determinate')
        self.progress_bar.pack(pady=10)
        self.status_label = ttk.Label(self.root, text="Status: Ready")
        self.status_label.pack(pady=5)

    def select_log_file(self):
        self.log_file_path = filedialog.asksaveasfilename(
            title="Select Log File Location",
            defaultextension=".log",
            filetypes=[("Log files", "*.log"), ("All files", "*.*")]
        )
        if self.log_file_path:
            print(f"Log file will be saved to: {self.log_file_path}")
        else:
            self.log_file_path = 'training_debug.log'  # Default log file
            print(f"No log file selected. Using default: {self.log_file_path}")

    def calculate_parameters(self, vocab_size, embed_size, num_layers, hidden_size):
        embedding_params = vocab_size * embed_size * 2  # Input and output embeddings
        transformer_params = num_layers * (4 * (hidden_size ** 2) + 2 * embed_size * hidden_size)  # Transformer layers
        total_params = embedding_params + transformer_params
        return total_params

    def update_num_parameters(self):
        vocab_size = self.vocab_size.get()
        embed_size = self.hidden_size.get()
        num_layers = self.num_layers.get()
        hidden_size = self.hidden_size.get()

        total_params = self.calculate_parameters(vocab_size, embed_size, num_layers, hidden_size)
        self.num_parameters.set(total_params)

    def on_device_change(self, event):
        selected_device = self.device_option.get()
        if selected_device == "GPU" and not torch.cuda.is_available():
            messagebox.showerror("Error", "GPU selected but CUDA is not available on this system.")
            self.device_option.set("CPU")
            selected_device = "CPU"
        device_str = self.map_device(selected_device)
        self.device = torch.device(device_str)
        logging.info(f"Device changed to: {self.device}")
        messagebox.showinfo("Device Selection", f"Computation device set to: {selected_device}")

    def resize_checkpoint_weights(self, state_dict, new_vocab_size, embed_size):
        """
        Resize checkpoint weights to match the current model's dimensions.
        """
        # This method may need to be updated depending on the model's state_dict keys
        return state_dict

    def select_model_file(self):
        self.model_path = filedialog.askopenfilename(
            title="Select Model File",
            filetypes=[("Model Files", "*.pth;*.json"), ("All files", "*.*")]
        )
        if self.model_path:
            if self.model_path.endswith('.json'):
                # Load model configuration
                with open(self.model_path, 'r') as f:
                    config = json.load(f)
                # Update GUI parameters
                self.vocab_size.set(config.get("vocab_size", self.vocab_size.get()))
                self.hidden_size.set(config.get("embed_size", self.hidden_size.get()))
                self.num_heads.set(config.get("num_heads", self.num_heads.get()))
                self.num_layers.set(config.get("num_layers", self.num_layers.get()))
                self.architecture.set(config.get("architecture", self.architecture.get()))
                messagebox.showinfo("Success", f"Model configuration loaded from: {self.model_path}")
            elif self.model_path.endswith('.pth'):
                # Load model weights
                config_directory = os.path.dirname(self.model_path)
                config_path = os.path.join(config_directory, 'model_config.json')
                if os.path.exists(config_path):
                    with open(config_path, 'r') as f:
                        config = json.load(f)
                    # Update GUI parameters
                    self.vocab_size.set(config.get("vocab_size", self.vocab_size.get()))
                    self.hidden_size.set(config.get("embed_size", self.hidden_size.get()))
                    self.num_heads.set(config.get("num_heads", self.num_heads.get()))
                    self.num_layers.set(config.get("num_layers", self.num_layers.get()))
                    self.architecture.set(config.get("architecture", self.architecture.get()))
                    # Load the model
                    self.load_model()
                    # Load model state
                    state_dict = torch.load(self.model_path, map_location=self.device)
                    self.model.load_state_dict(state_dict)
                    messagebox.showinfo("Success", f"Model weights and configuration loaded from: {self.model_path}")
                else:
                    messagebox.showwarning("Warning", "Model configuration file not found. Please ensure the configuration is set correctly.")
            else:
                messagebox.showerror("Error", "Unsupported file format selected.")

    def save_transformer_and_model(self):
        if not self.model:
            messagebox.showerror("Error", "Model has not been initialized. Please initialize the model first.")
            return
        if not self.tokenizer:
            messagebox.showerror("Error", "Tokenizer has not been initialized. Please load a tokenizer first.")
            return

        transformer_data = {
            "vocab_size": self.vocab_size.get(),
            "embed_size": self.hidden_size.get(),
            "hidden_size": self.hidden_size.get(),
            "num_heads": self.num_heads.get(),
            "num_layers": self.num_layers.get(),
            "architecture": self.architecture.get(),
            "num_parameters": self.num_parameters.get(),
            "layers": self.layers
        }

        directory = filedialog.askdirectory(title="Select Save Directory")
        if directory:
            # Save configuration
            config_path = os.path.join(directory, "model_config.json")
            with open(config_path, "w") as file:
                json.dump(transformer_data, file, indent=4)

            # Save weights
            if self.architecture.get() == "Transformer":
                model_file_name = 'custom_transformer_model.pth'
            elif self.architecture.get() == "MatMul-Free LM":
                model_file_name = 'matmul_free_lm.pth'
            else:
                messagebox.showerror("Error", f"Unsupported architecture: {self.architecture.get()}")
                return

            model_path = os.path.join(directory, model_file_name)
            torch.save(self.model.state_dict(), model_path)

            # Save tokenizer
            self.tokenizer.save_pretrained(directory)

            messagebox.showinfo("Success", "Model, tokenizer, and configuration saved successfully!")
            logging.info("Model, tokenizer, and configuration saved successfully.")

    def select_dataset(self):
        self.dataset_path = filedialog.askdirectory(title="Select Dataset Directory")
        if self.dataset_path:
            messagebox.showinfo("Success", f"Dataset directory selected: {self.dataset_path}")

    def select_vocab(self):
        self.vocab_path = filedialog.askopenfilename(
            title="Select Vocabulary File",
            filetypes=[("JSON files", "*.json"), ("Text files", "*.txt"), ("All files", "*.*")]
        )
        if self.vocab_path:
            messagebox.showinfo("Success", f"Vocabulary file selected: {self.vocab_path}")

    def select_tokenizer(self):
        self.tokenizer_path = filedialog.askopenfilename(
            title="Select Tokenizer File",
            filetypes=[("JSON files", "*.json"), ("All files", "*.*")]
        )
        if self.tokenizer_path:
            messagebox.showinfo("Success", f"Tokenizer file selected: {self.tokenizer_path}")

    def test_tokenizer(self):
        if not self.tokenizer:
            messagebox.showerror("Error", "Tokenizer not loaded.")
            return
        sample_text = simpledialog.askstring("Test Tokenizer", "Enter a sample text to tokenize:")
        if sample_text:
            tokens = self.tokenizer.tokenize(sample_text)
            token_ids = self.tokenizer.encode(sample_text)
            logging.info(f"Sample Text: {sample_text}")
            logging.info(f"Tokens: {tokens}")
            logging.info(f"Token IDs: {token_ids}")
            messagebox.showinfo("Tokenizer Test", f"Tokens: {tokens}\nToken IDs: {token_ids}")

    def save_dataset_as_text(self):
        if not hasattr(self, 'text_data') or not self.text_data:
            messagebox.showerror("Error", "No dataset loaded or processed to save.")
            return

        save_path = filedialog.asksaveasfilename(
            title="Save Dataset as Text File",
            defaultextension=".txt",
            filetypes=[("Text files", "*.txt"), ("All files", "*.*")]
        )
        if save_path:
            try:
                with open(save_path, 'w', encoding='utf-8') as f:
                    for line in self.text_data:
                        f.write(line + '\n')
                messagebox.showinfo("Success", f"Dataset saved to {save_path}")
                logging.info(f"Dataset saved to {save_path}")
            except Exception as e:
                logging.error(f"Failed to save dataset: {e}")
                messagebox.showerror("Error", f"Failed to save dataset: {e}")



    def create_tokenizer_from_vocab(self):
        try:
            # Ask the user to select the vocabulary file
            vocab_path = filedialog.askopenfilename(
                title="Select Vocabulary File",
                filetypes=[("JSON files", "*.json"), ("All files", "*.*")]
            )
            if not vocab_path:
                messagebox.showerror("Error", "No vocabulary file selected.")
                return

            with open(vocab_path, 'r', encoding='utf-8') as f:
                vocab = json.load(f)

            # Create a word-level tokenizer
            tokenizer = Tokenizer(models.WordLevel(vocab=vocab, unk_token="<UNK>"))
            tokenizer.pre_tokenizer = pre_tokenizers.Whitespace()

            # Wrap with PreTrainedTokenizerFast
            self.tokenizer = PreTrainedTokenizerFast(
                tokenizer_object=tokenizer,
                unk_token='<UNK>',
                pad_token='<PAD>',
                bos_token='<BOS>',
                eos_token='<EOS>',
                model_max_length=1024,
            )

            # Ensure special tokens are added
            self.tokenizer.add_special_tokens({
                'unk_token': '<UNK>',
                'pad_token': '<PAD>',
                'bos_token': '<BOS>',
                'eos_token': '<EOS>'
            })

            # Save the tokenizer
            save_directory = filedialog.askdirectory(title="Select Directory to Save Tokenizer")
            if save_directory:
                os.makedirs(save_directory, exist_ok=True)
                self.tokenizer.save_pretrained(save_directory)
                self.tokenizer_path = os.path.join(save_directory, 'tokenizer.json')
                messagebox.showinfo("Success", f"Tokenizer saved to {self.tokenizer_path}")
                logging.info(f"Tokenizer saved to {self.tokenizer_path}")
            else:
                messagebox.showerror("Error", "No save directory selected for tokenizer.")
                return

            # Test the tokenizer
            test_text = "Hello World!\nThis is a test.\tLet's remove line breaks and tabs."
            tokens = self.tokenizer.tokenize(test_text)
            logging.info(f"Test tokenization of '{test_text}': {tokens}")

            tokenizer_vocab = self.tokenizer.get_vocab()
            sorted_vocab = dict(sorted(tokenizer_vocab.items(), key=lambda item: item[1]))
            logging.info(f"Sorted Tokenizer Vocabulary: {sorted_vocab}")

            logging.info("Tokenizer created and saved successfully")
        except Exception as e:
            logging.error(f"Failed to create tokenizer: {str(e)}")
            messagebox.showerror("Error", f"Failed to create tokenizer: {str(e)}")
            raise

    def load_tokenizer(self):
        try:
            self.tokenizer_path = filedialog.askopenfilename(
                title="Select Tokenizer File",
                filetypes=[("JSON files", "*.json"), ("All files", "*.*")]
            )
            if not self.tokenizer_path or not os.path.exists(self.tokenizer_path):
                raise FileNotFoundError("Tokenizer file not selected or does not exist.")

            self.tokenizer = PreTrainedTokenizerFast(tokenizer_file=self.tokenizer_path)
            logging.info(f"Tokenizer loaded from {self.tokenizer_path}")

            # Load special tokens map
            special_tokens_path = os.path.join(os.path.dirname(self.tokenizer_path), "special_tokens_map.json")
            if os.path.exists(special_tokens_path):
                with open(special_tokens_path, "r") as file:
                    special_tokens = json.load(file)

                for key, value in special_tokens.items():
                    if isinstance(value, dict):
                        special_tokens[key] = AddedToken(value["content"], lstrip=value.get("lstrip", False),
                                                         rstrip=value.get("rstrip", False))
                    elif not isinstance(value, (str, AddedToken)):
                        raise ValueError(f"Invalid token format for key {key}: {value}")

                self.tokenizer.add_special_tokens(special_tokens)
                logging.info(f"Special tokens added: {special_tokens}")

            # Load tokenizer configuration
            tokenizer_config_path = os.path.join(os.path.dirname(self.tokenizer_path), "tokenizer_config.json")
            if os.path.exists(tokenizer_config_path):
                with open(tokenizer_config_path, "r") as file:
                    tokenizer_config = json.load(file)
                    self.tokenizer.init_kwargs.update(tokenizer_config)

                    # Check and set model_max_length
                    if "model_max_length" in tokenizer_config:
                        self.tokenizer.model_max_length = tokenizer_config["model_max_length"]
                    logging.info(f"Tokenizer configuration loaded: {tokenizer_config}")

            # Explicitly set model_max_length if still unset or unreasonable
            if not hasattr(self.tokenizer, "model_max_length") or self.tokenizer.model_max_length > 1024 * 1024:
                self.tokenizer.model_max_length = 1024  # Default to 1024 for character-level tokens

            # Check consistency
            tokenizer_vocab_size = len(self.tokenizer)
            logging.info(f"Loaded tokenizer vocabulary size: {tokenizer_vocab_size}")
            self.vocab_size.set(tokenizer_vocab_size)

            # Ensure special tokens are correctly set
            if not self.tokenizer.pad_token:
                self.tokenizer.pad_token = "<PAD>"
                self.tokenizer.pad_token_id = self.tokenizer.convert_tokens_to_ids("<PAD>")
                logging.warning("Pad token was not set. Defaulting to <PAD>.")
            if not self.tokenizer.unk_token:
                self.tokenizer.unk_token = "<UNK>"
                self.tokenizer.unk_token_id = self.tokenizer.convert_tokens_to_ids("<UNK>")
                logging.warning("UNK token was not set. Defaulting to <UNK>.")
            if not self.tokenizer.bos_token:
                self.tokenizer.bos_token = "<BOS>"
                self.tokenizer.bos_token_id = self.tokenizer.convert_tokens_to_ids("<BOS>")
                logging.warning("BOS token was not set. Defaulting to <BOS>.")
            if not self.tokenizer.eos_token:
                self.tokenizer.eos_token = "<EOS>"
                self.tokenizer.eos_token_id = self.tokenizer.convert_tokens_to_ids("<EOS>")
                logging.warning("EOS token was not set. Defaulting to <EOS>.")
            print("Special tokens map:", self.tokenizer.special_tokens_map)
            print("Pad token ID:", self.tokenizer.pad_token_id)
            print("Model max length:", self.tokenizer.model_max_length)
            

        except Exception as e:
            logging.error(f"Failed to load tokenizer: {str(e)}")
            messagebox.showerror("Error", f"Failed to load tokenizer: {str(e)}")

    def select_or_create_tokenized_data(self):
        use_chunked = self.use_chunked_dataset.get()
        answer = messagebox.askyesno("Select or Create Tokenized Data", "Do you want to use existing tokenized data?")
        
        if answer:
            if use_chunked:
                # User wants to use existing chunked tokenized data, select a directory
                self.tokenized_data_path = filedialog.askdirectory(
                    title="Select Tokenized Data Directory",
                    mustexist=True
                )
                if self.tokenized_data_path:
                    messagebox.showinfo("Success", f"Tokenized data directory selected: {self.tokenized_data_path}")
            else:
                # User wants to use existing single tokenized data file, select a file
                self.tokenized_data_path = filedialog.askopenfilename(
                    title="Select Tokenized Data File",
                    filetypes=[("JSON Lines files", "*.jsonl"), ("All files", "*.*")]
                )
                if self.tokenized_data_path:
                    # Attempt to load the file to validate its content
                    try:
                        with open(self.tokenized_data_path, 'r', encoding='utf-8') as f:
                            self.input_ids, self.labels = [], []
                            for line in f:
                                record = json.loads(line)
                                self.input_ids.append(record['input_ids'])
                                self.labels.append(record['labels'])
                        messagebox.showinfo("Success", f"Tokenized data file loaded: {self.tokenized_data_path}")
                        logging.info(f"Tokenized data file loaded successfully with {len(self.input_ids)} entries.")
                    except Exception as e:
                        messagebox.showerror("Error", f"Failed to load tokenized data file: {str(e)}")
        else:
            if use_chunked:
                # User wants to create new chunked tokenized data, select a directory to save
                self.tokenized_data_path = filedialog.askdirectory(
                    title="Select Directory to Save Tokenized Data"
                )
                if self.tokenized_data_path:
                    os.makedirs(self.tokenized_data_path, exist_ok=True)  # Ensure directory is created
                    messagebox.showinfo("Success", f"Tokenized data will be saved to directory: {self.tokenized_data_path}")
            else:
                # User wants to create new single tokenized data file, select a file path
                self.tokenized_data_path = filedialog.asksaveasfilename(
                    title="Save Tokenized Data As",
                    defaultextension=".jsonl",
                    filetypes=[("JSON Lines files", "*.jsonl"), ("All files", "*.*")]
                )
                if self.tokenized_data_path:
                    messagebox.showinfo("Success", f"Tokenized data will be saved to file: {self.tokenized_data_path}")


            
    def tokenize_data(self):
        if not self.tokenizer:
            messagebox.showerror("Error", "Tokenizer not loaded.")
            return
        if not hasattr(self, 'query_target_pairs') or not self.query_target_pairs:
            messagebox.showerror("Error", "No query-target pairs loaded. Please load the dataset first.")
            return
        if not self.tokenized_data_path:
            messagebox.showerror("Error", "Tokenized data path not set. Please select or create tokenized data.")
            return

        # Select training mode
        training_mode = self.training_mode.get()  # "imitation", "completion", "response"
        self.input_ids = []  # Initialize for unchunked dataset
        self.labels = []  # Initialize for unchunked dataset
        
        try:
            use_chunked = self.use_chunked_dataset.get()
            if use_chunked:
                #create path if none
                os.makedirs(self.tokenized_data_path, exist_ok=True)
                chunk_size = 32
                num_chunks = (len(self.query_target_pairs) + chunk_size - 1) // chunk_size

                for chunk_idx in range(num_chunks):
                    chunk_pairs = self.query_target_pairs[chunk_idx * chunk_size: (chunk_idx + 1) * chunk_size]
                    chunk_file_path = os.path.join(self.tokenized_data_path, f'chunk_{chunk_idx}.jsonl')

                    with open(chunk_file_path, 'w', encoding='utf-8') as f:
                        for query, target in chunk_pairs:
                            input_ids, labels = self._generate_training_pairs(query, target, training_mode)
                            if input_ids and labels:
                                record = {'input_ids': input_ids, 'labels': labels}
                                f.write(json.dumps(record) + '\n')
                logging.info(f"Chunk {chunk_idx} tokenized and saved to {chunk_file_path}")

                messagebox.showinfo("Success", f"Data tokenized into {num_chunks} chunks and saved successfully to {self.tokenized_data_path}.")
                logging.info(f"Data tokenized into {num_chunks} chunks and saved successfully to {self.tokenized_data_path}.")
            else:
                with open(self.tokenized_data_path, 'w', encoding='utf-8') as f:
                    for query, target in self.query_target_pairs:
                        input_ids, labels = self._generate_training_pairs(query, target, training_mode)

                        if input_ids and labels:
                            self.input_ids.append(input_ids)  # Store for training
                            self.labels.append(labels)  # Store for training
                            record = {'input_ids': input_ids, 'labels': labels}


                            f.write(json.dumps(record) + '\n')
                logging.info(f"Input IDs: {len(self.input_ids)} sequences loaded.")
                logging.info(f"Labels: {len(self.labels)} sequences loaded.")
                messagebox.showinfo("Success", f"Data tokenized and saved successfully to {self.tokenized_data_path}.")
                logging.info(f"Data tokenized and saved successfully to {self.tokenized_data_path}.")
        except Exception as e:
            logging.error(f"Tokenization failed: {str(e)}")
            messagebox.showerror("Error", f"Tokenization failed: {str(e)}")

    def _generate_training_pairs(self, query, target, training_mode):
        # Tokenize query and target
        query_ids = self.tokenizer.encode(query, truncation=True, max_length=1024)
        target_ids = self.tokenizer.encode(target, truncation=True, max_length=1024)

        # Convert tokens to integers
        query_ids = [int(token) for token in query_ids]
        target_ids = [int(token) for token in target_ids]

        if training_mode == "imitation":
            input_ids = query_ids + [self.tokenizer.eos_token_id] + target_ids
            labels = query_ids + [self.tokenizer.eos_token_id] + target_ids
        elif training_mode == "completion":
            partial_length = len(query_ids) // 2
            partial_input = query_ids[:partial_length]
            completion = query_ids[partial_length:] + [self.tokenizer.eos_token_id]

            input_ids = partial_input + [self.tokenizer.eos_token_id]
            # For completion, we want labels to represent the completed part only:
            labels = completion  
        else:  # response
            input_ids = query_ids + [self.tokenizer.eos_token_id]
            labels = target_ids + [self.tokenizer.eos_token_id]

        return input_ids, labels


    def add_layer(self):
        layer_type = simpledialog.askstring("Layer Type", "Enter layer type (e.g., attention, feed_forward)")
        if layer_type:
            layer_config = {
                "type": layer_type,
                "parameters": {}  # Placeholder for future parameter configuration
            }
            self.layers.append(layer_config)
            messagebox.showinfo("Layer Added", f"Layer of type '{layer_type}' added.")

    def save_transformer(self):
        transformer_data = {
            "num_parameters": self.num_parameters.get(),
            "num_heads": self.num_heads.get(),
            "layers": self.layers
        }

        file_path = filedialog.asksaveasfilename(defaultextension=".json", filetypes=[("JSON files", "*.json")])
        if file_path:
            with open(file_path, "w") as file:
                json.dump(transformer_data, file, indent=4)
            messagebox.showinfo("Save", "Transformer saved successfully!")
            logging.info(f"Number of layers in the model: {len(self.model.transformer_encoder.layers)}")

    def load_transformer(self):
        file_path = filedialog.askopenfilename(filetypes=[("JSON files", "*.json")])
        if file_path:
            with open(file_path, "r") as file:
                transformer_data = json.load(file)
            self.num_parameters.set(transformer_data["num_parameters"])
            self.num_heads.set(transformer_data["num_heads"])
            self.layers = transformer_data["layers"]
            messagebox.showinfo("Success", "Transformer loaded successfully")

    def load_model(self):
        try:
            if not self.tokenizer:
                vocab_size = self.vocab_size.get()
            else:
                vocab_size = len(self.tokenizer)

            # Log and validate vocab size
            logging.info(f"Tokenizer vocabulary size: {vocab_size}")
            self.vocab_size.set(vocab_size)

            # Initialize the model based on architecture
            if self.architecture.get() == "Transformer":
                self.model = CustomTransformerModel(
                    vocab_size=vocab_size,
                    embed_size=self.hidden_size.get(),
                    num_heads=self.num_heads.get(),
                    num_layers=self.num_layers.get(),
                    hidden_size=self.hidden_size.get(),
                    max_seq_length=1024
                )
            elif self.architecture.get() == "MatMul-Free LM":
                self.model = MatMulFreeLanguageModel(
                    vocab_size=vocab_size,
                    embed_size=self.hidden_size.get(),
                    hidden_size=self.hidden_size.get(),
                    seq_length=1024,  # Add seq_length here
                    eps=1e-8
                )
            else:
                messagebox.showerror("Error", f"Unsupported architecture: {self.architecture.get()}")
                return

            # Move the entire model to the selected device

            self.model.to(self.device)
            logging.info(f"Model moved to device: {self.device}")

            # Resize embeddings to match tokenizer vocabulary size
            if hasattr(self.model, 'resize_token_embeddings'):
                self.model.resize_token_embeddings(vocab_size)
                logging.info("Embeddings resized to match tokenizer vocabulary size.")

            # Load checkpoint if a model file is selected
            if self.model_path and self.model_path.endswith('.pth'):
                state_dict = torch.load(self.model_path, map_location=self.device)
                self.model.load_state_dict(state_dict, strict=True)
                logging.info("Model weights loaded and resized successfully.")

            logging.info(f"Model initialized on device: {self.device}")
            messagebox.showinfo("Success", "Model initialized successfully.")

        except Exception as e:
            logging.error(f"Failed to initialize model: {str(e)}")
            messagebox.showerror("Error", f"Failed to initialize model: {str(e)}")


    def calculate_learning_rate(self, total_params):
        # Calculate learning rate based on total parameters using the derived formula
        # LR = 17.38 * (Model Size)^-0.424
        lr = 17.38 * (total_params ** -0.424)
        return lr

    def start_training(self):
        # Start training in a separate thread to keep the GUI responsive
        self.stop_training.clear()
        training_thread = threading.Thread(target=self.training_loop)
        training_thread.start()

    def update_progress(self, progress_value):
        self.progress_bar['value'] = progress_value

    def update_status(self, message):
        self.status_label.config(text=f"Status: {message}")

    def save_checkpoint(self, model, optimizer, epoch, path):
        if not isinstance(path, (str, os.PathLike)):
            raise TypeError(f"Expected path to be str or os.PathLike, got {type(path).__name__}")
        os.makedirs(os.path.dirname(path), exist_ok=True)
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
        }, path)
        


    def validate_training_parameters(self):
        # Validate batch size
        try:
            batch_size = int(self.batch_size.get())
            if batch_size <= 0:
                raise ValueError
        except (TypeError, ValueError):
            logging.error(f"Invalid batch size: {self.batch_size.get()}")
            messagebox.showerror("Error", "Batch size must be a positive integer.")
            return False

        # Validate epochs
        try:
            epochs = int(self.epochs.get())
            if epochs <= 0:
                raise ValueError
        except (TypeError, ValueError):
            logging.error(f"Invalid epochs value: {self.epochs.get()}")
            messagebox.showerror("Error", "Epochs must be a positive integer.")
            return False

        if not self.tokenized_data_path or not os.path.exists(self.tokenized_data_path):
            logging.error("Tokenized data path is invalid or does not exist.")
            messagebox.showerror("Error", "Tokenized data is not selected or does not exist.")
            return False

        if not hasattr(self.tokenizer, 'pad_token_id') or self.tokenizer.pad_token_id is None:
            logging.error("Tokenizer pad_token_id is not set.")
            messagebox.showerror("Error", "Tokenizer is missing pad_token_id.")
            return False

        return True

    def training_test(self):
        
        if self.test_bool.get()==True:

            # Initialize and test
            self.model = StandardTransformerModel(
                vocab_size=self.vocab_size.get(),
                embed_size=self.hidden_size.get(),
                num_heads=self.num_heads.get(),
                num_layers=self.num_layers.get(),
                hidden_size=self.hidden_size.get(),
                max_seq_length=1024
            ).to(self.device)
        else:
            pass
        try:
            if self.use_chunked_dataset.get():
                    # Initialize the ChunkedDataset
                    dataset = ChunkedDataset(
                        tokenized_data_path=self.tokenized_data_path,
                        tokenizer=self.tokenizer,
                        max_length=1024
                    )
                    dataloader = DataLoader(
                        dataset,
                        batch_size=self.batch_size.get(),
                        shuffle=True,
                        num_workers=0,
                        pin_memory=True,
                        collate_fn=collate_fn
                    )
            else:
                max_length = 1024
                # Convert lists of token IDs to tensors
                input_ids = [
                    torch.tensor(tokens + [self.tokenizer.pad_token_id] * (max_length - len(tokens)), dtype=torch.int64)[:max_length]
                    for tokens in self.input_ids
                ]
                logging.info("input ids torched to tensor")
                input_ids = torch.stack(input_ids)
                logging.info("input ids stacked by torch")
                attention_masks = (input_ids != self.tokenizer.pad_token_id).long()
                logging.info("attention masks set for pad tokens")

                assert isinstance(input_ids, list), "input_ids should be a list"
                assert all(isinstance(id, int) for id in input_ids), "All input_ids should be integers"
                assert len(input_ids) == self.max_length, "input_ids should be padded to max_length"
                dataset = torch.utils.data.TensorDataset(input_ids, attention_masks)
                dataloader = DataLoader(
                        dataset,
                        batch_size=int(self.batch_size.get()),
                        shuffle=True,
                        num_workers=0,  # Set to 0 to prevent multiple workers from loading chunks simultaneously
                        pin_memory=True,
                        collate_fn=collate_fn
                    )
                self.model.train()
                sample_input, sample_attention_mask, sample_labels, sample_seq_length = next(iter(dataloader))
                sample_input = sample_input.to(self.device)
                sample_labels = sample_labels.to(self.device)
                sample_seq_length = sample_seq_length.to(self.device)

                with autocast(device_type="cuda", enabled=torch.cuda.is_available()):
                    if isinstance(self.model, CustomTransformerModel):
                        src_key_padding_mask = (sample_input == self.tokenizer.pad_token_id)
                        outputs = self.model(sample_input, src_key_padding_mask=src_key_padding_mask)
                    elif isinstance(self.model, MatMulFreeLanguageModel):
                        outputs = self.model(sample_input, sample_seq_length)
                    else:
                        raise ValueError("Unsupported model architecture.")

                    logits = outputs.view(-1, outputs.size(-1))
                    targets = sample_labels.view(-1)
                    loss = F.cross_entropy(logits, targets, ignore_index=self.tokenizer.pad_token_id)
                
                loss.backward()
                print("Forward and backward pass successful. Loss:", loss.item())
        except Exception as e:
            print("Error during forward/backward pass:", e)

    def training_loop(self):
        if not self.validate_training_parameters():
            return

        logging.info("All training parameters and data are properly initialized.")
        if not self.model:
            logging.error("Model not initialized before training")
            return

        try:
            if self.use_chunked_dataset.get():
                # Initialize the ChunkedDataset
                dataset = ChunkedDataset(
                    tokenized_data_path=self.tokenized_data_path,
                    tokenizer=self.tokenizer,
                    max_length=1024
                )
                dataloader = DataLoader(
                    dataset,
                    batch_size=self.batch_size.get(),
                    shuffle=True,
                    num_workers=0,
                    pin_memory=True,
                    collate_fn=collate_fn
                )
            else:
                # Initialize the standard dataset and dataloader

                max_length = 1024  # Adjust as needed
                logging.info("max_length set")
                # Convert lists of token IDs to tensors and calculate original sequence lengths
                input_ids, seq_lengths = zip(*[
                    (
                        torch.tensor(tokens + [self.tokenizer.pad_token_id] * (max_length - len(tokens)), dtype=torch.int64)[:max_length],
                        min(len(tokens), max_length)
                    )
                    for tokens in self.input_ids
                ])
                logging.info("input ids torched to tensor")

                labels = [
                    torch.tensor(tokens + [self.tokenizer.pad_token_id] * (max_length - len(tokens)), dtype=torch.int64)[:max_length]
                    for tokens in self.labels
                ]
                logging.info("labels torched to tensor")

                attention_masks = [(ids != self.tokenizer.pad_token_id).long() for ids in input_ids]
                logging.info("attention masks set for pad tokens")

                # Stack tensors
                input_ids = torch.stack(input_ids)
                labels = torch.stack(labels)

                attention_masks = torch.stack(attention_masks)

                seq_lengths = torch.tensor(seq_lengths, dtype=torch.long)
                logging.info("datas stacked and seq lengths torched")

                # Perform assertions to validate tensors
                assert isinstance(input_ids, torch.Tensor), "input_ids should be a tensor"
                assert isinstance(labels, torch.Tensor), "labels should be a tensor"
                assert input_ids.dtype == torch.long, "input_ids should be of type torch.long"
                assert labels.dtype == torch.long, "labels should be of type torch.long"
                assert input_ids.size(1) == max_length, "input_ids should be padded to max_length"
                assert labels.size(1) == max_length, "labels should be padded to max_length"

                dataset = torch.utils.data.TensorDataset(input_ids, attention_masks, labels, seq_lengths)
                logging.info("dataset torched")
                dataloader = DataLoader(
                    dataset,
                    batch_size=int(self.batch_size.get()),
                    shuffle=True,
                    num_workers=0,  # Set to 0 to prevent multiple workers from loading chunks simultaneously
                    pin_memory=True,
                )
                logging.info("dataloader defined")
            ##chunked vs. standard else complete
            # Log dataset samples


            # Adjust learning rate based on architecture
            total_params = self.num_parameters.get()
            lr = self.learning_rate.get()
            logging.info(f"Learning Rate: {lr} for total parameters: {total_params}")

            optimizer = torch.optim.AdamW(self.model.parameters(), lr=lr, weight_decay=1e-5)

            # Learning rate scheduler
            total_steps = self.epochs.get() * len(dataloader)
            logging.info(f"Total training steps: {total_steps}")
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=total_steps, eta_min=lr * 0.1)
            logging.info("Scheduler defined")

            self.model.train()
            logging.info("Model set to training mode")
            progress_step = 0

            for epoch in range(self.epochs.get()):
                if self.stop_training.is_set():
                    logging.info("Training stopped by user.")
                    messagebox.showinfo("Info", "Training stopped by user.")
                    break

                epoch_loss = 0
                logging.info(f"Epoch {epoch+1} started")
                
                save_path = f"visualizations/epochstart_{epoch}"
                os.makedirs(save_path, exist_ok=True)
                self.attention_weights = []  # Clear stored attention at the start of the epoch
                self.register_attention_hooks()  # Ensure hooks are set before any forward pass

                # Save attention heatmaps
                for layer_idx, attention in enumerate(self.attention_weights):
                    self.plot_attention_heatmap(attention, layer=layer_idx, phase="start_epoch", save_path=save_path)

                # Save weight distributions
                for name, param in self.model.named_parameters():
                    if 'weight' in name:
                        self.plot_weight_distribution(param=param, phase="start_epoch", save_path=save_path, name=name)

                # Ensure the tokenizer is loaded and has a valid pad_token_id
                pad_token_id = tokenizer.pad_token_id if tokenizer else 1  # Default to 1 if tokenizer isn't set                
                # Training loop
                for batch_idx, (batch_input_ids, batch_attention_masks, batch_labels, seq_lengths) in enumerate(dataloader):
                    if self.stop_training.is_set():
                        logging.info("Training stopped by user.")
                        messagebox.showinfo("Info", "Training stopped by user.")
                        return
                    
                    optimizer.zero_grad()
                    logging.debug("Optimizer gradients zeroed")

                    # Move batches and targets to the correct device 
                    batch_input_ids = batch_input_ids.to(self.device)
                    batch_labels = batch_labels.to(self.device)
                    batch_attention_masks = batch_attention_masks.to(self.device)

                    seq_lengths = seq_lengths.to(self.device)
                    logging.debug("Batch moved to device")

                    # Logging epoch and batch info
                    logging.debug(f'Epoch: {epoch + 1}, Batch: {batch_idx + 1}')
                    logging.debug(f'Batch input_ids shape: {batch_input_ids.shape}')  # (batch_size, 1024)
                    logging.debug(f'Batch attention_masks shape: {batch_attention_masks.shape}')  # (batch_size, 1024)
                    logging.debug(f'Using device: {self.device}')

                    # Forward pass
                    try:
                        if isinstance(self.model, CustomTransformerModel):
                            # For Transformer model, use src_key_padding_mask
                            src_key_padding_mask = (batch_input_ids == self.tokenizer.pad_token_id)
                            outputs = self.model(batch_input_ids, src_key_padding_mask=src_key_padding_mask)
                        elif isinstance(self.model, MatMulFreeLanguageModel):
                            # For MatMul-Free LM, use seq_lengths
                            outputs = self.model(batch_input_ids, seq_lengths)
                            if torch.isnan(outputs).any() or torch.isinf(outputs).any():
                                logging.error("Model outputs contain NaN or Inf values.")
                                raise ValueError("Model outputs contain NaN or Inf values.")
                        else:
                            raise ValueError("Unsupported model architecture.")
                    except Exception as e:
                        raise ValueError(f"forward pass failed for {str(e)}")

                    logits_holder = []
                    logits_holder = outputs
                    logits = outputs.view(-1, outputs.size(-1))
                    
                    #Prepare targets
                    targets = batch_labels.view(-1)
                    # Ensure targets are in the correct type
                    if targets.dtype != torch.long:
                        targets = targets.long()

                    # Make sure targets are within the valid range
                    vocab_size = self.vocab_size.get()  # Replace with actual vocabulary size

                    targets = torch.clamp(targets, min=0, max=vocab_size - 1)

                    logging.debug(f"batchlabels size: {batch_labels.shape}")
                    logging.debug(f"Target  shape: {targets.shape}")

                    logging.debug(f"Targets after clamping: dtype={targets.dtype}, shape={targets.shape}")
                    logging.debug(f"Clamped targets: mean={targets.float().mean().item():.4f}, "
                                f"std={targets.float().std().item():.4f}, "
                                f"max={targets.max().item()}, min={targets.min().item()}")

                    # Compute loss
                    loss = F.cross_entropy(logits, targets, ignore_index=self.pad_token_id)

                    logging.info(f"Loss computed: {loss.item()}")
                    if batch_idx % 10 == 0:  # Visualize every 10 batches
                        save_path = f"visualizations/epoch_{epoch}/batch_{batch_idx}"
                        os.makedirs(save_path, exist_ok=True)
                        self.visualize_model_output(batch_input_ids, logits_holder, batch_labels, save_path)

                    # Backward pass and optimization
                    scaler.scale(loss).backward()
                    logging.info("Loss backward computed")

                    for param_group in optimizer.param_groups:
                        logging.debug(f"Learning rate: {param_group['lr']}")

                    # Check for NaN or Inf in gradients
                    for name, param in self.model.named_parameters():
                        if param.grad is not None:
                            if torch.isnan(param.grad).any() or torch.isinf(param.grad).any():
                                logging.error(f"Gradient for {name} contains NaN or Inf.")
                                raise ValueError(f"Gradient for {name} contains NaN or Inf.")

                    for name, param in self.model.named_parameters():
                        if param.grad is not None:
                            logging.debug(f"Gradient for {name}: mean={param.grad.mean().item():.4f}, max={param.grad.max().item():.4f}, min={param.grad.min().item():.4f}")
                        else:
                            logging.debug(f"Gradient for {name} is None")

                    # Gradient clipping
                    scaler.unscale_(optimizer)
                    
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=0.1)
                    logging.info("Gradient clipping applied")
                    
                    if batch_idx % 10 == 0:  # Visualize every 10 batches
                        save_path = f"visualizations/epoch_{epoch}/batch_{batch_idx}"
                        os.makedirs(save_path, exist_ok=True)
                        self.visualize_gradients(self.model, save_path, phase=batch_idx)
                                            
                    # Before optimizer step
                    for name, param in self.model.named_parameters():
                        if param.requires_grad:
                            logging.debug(f"Before step - {name}: mean={param.data.mean().item():.4f}, std={param.data.std().item():.4f}")

                    scaler.step(optimizer)
                    

                    # After optimizer step
                    for name, param in self.model.named_parameters():
                        if param.requires_grad:
                            logging.debug(f"After step - {name}: mean={param.data.mean().item():.4f}, std={param.data.std().item():.4f}")

                    scaler.update()

                    logging.info("Optimizer step and scaler update completed")

                    scheduler.step()
                    logging.debug("Scheduler step completed")

                    epoch_loss += loss.item()
                    progress_step += 1
                    progress_value = (progress_step / total_steps) * 100
                    self.root.after(0, self.update_progress, progress_value)

                    # Save checkpoint at specified intervals
                    save_interval = 25  # Save every 25%
                    progress_percentage = (batch_idx + 1) / len(dataloader) * 100
                    if abs(progress_percentage % save_interval) < 1e-6:  # Avoid floating-point issues
                        checkpoint_path = f"checkpoints/epoch_{epoch}_batch_{batch_idx}.pth"
                        self.save_checkpoint(self.model, optimizer, epoch, checkpoint_path)
                        logging.info(f"Checkpoint saved at epoch {epoch}, batch {batch_idx}, progress: {progress_percentage:.2f}%")

                save_path = f"visualizations/epochend_{epoch}"
                os.makedirs(save_path, exist_ok=True)

                # Save attention heatmaps
                for layer_idx, attention in enumerate(self.attention_weights):
                    self.plot_attention_heatmap(attention, layer=layer_idx, phase="end_epoch", save_path=save_path)

                # Save weight distributions
                for name, param in self.model.named_parameters():
                    if 'weight' in name:
                        self.plot_weight_distribution(param=param, phase="end_epoch", save_path=save_path, name=name)


                # Log epoch loss
                average_epoch_loss = epoch_loss / len(dataloader)
                self.loss_history.append(average_epoch_loss)
                logging.info(f"Epoch {epoch + 1}/{self.epochs.get()} completed with average loss: {average_epoch_loss}")
                self.root.after(0, self.update_status, f"Epoch {epoch + 1}/{self.epochs.get()} completed. Current LR = {scheduler.get_last_lr()}")

        except Exception as e:
            logging.error(f"An error occurred during training: {str(e)}")
            messagebox.showerror("Error", f"An error occurred during training: {str(e)}")

    def improved_collate_fn(self, batch):
        input_ids, attention_masks, labels, seq_lengths = zip(*batch)
        
        # Convert sequences to tensors if they aren't already
        input_ids = [x if isinstance(x, torch.Tensor) else torch.tensor(x) for x in input_ids]
        attention_masks = [x if isinstance(x, torch.Tensor) else torch.tensor(x) for x in attention_masks]
        labels = [x if isinstance(x, torch.Tensor) else torch.tensor(x) for x in labels]
        
        # Find max length in batch
        max_len = 1024
        
        # Pad sequences using torch operations
        def pad_sequence(sequences, max_len, pad_value):
            return torch.stack([
                torch.cat([
                    seq,
                    torch.full((max_len - len(seq),), pad_value, dtype=seq.dtype, device=seq.device)
                ]) if len(seq) < max_len else seq[:max_len]
                for seq in sequences
            ])
        
        # Pad all sequences
        padded_input_ids = pad_sequence(input_ids, max_len, self.tokenizer.pad_token_id)
        padded_attention_masks = pad_sequence(attention_masks, max_len, 0)
        padded_labels = pad_sequence(labels, max_len, self.tokenizer.pad_token_id)
        
        # Convert sequence lengths to tensor
        seq_lengths = torch.tensor(seq_lengths, dtype=torch.long)
        
        return padded_input_ids, padded_attention_masks, padded_labels, seq_lengths

    def get_cosine_schedule_with_warmup(self, optimizer, num_warmup_steps, num_training_steps):
        def lr_lambda(current_step):
            if current_step < num_warmup_steps:
                return float(current_step) / float(max(1, num_warmup_steps))
            progress = float(current_step - num_warmup_steps) / float(max(1, num_training_steps - num_warmup_steps))
            return max(0.0, 0.5 * (1.0 + math.cos(math.pi * progress)))
            
        return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    def save_model(self):
        if not self.model:
            messagebox.showerror("Error", "Model has not been initialized. Cannot save.")
            logging.error("Attempted to save model but model was not initialized.")
            return
        if not self.tokenizer:
            messagebox.showerror("Error", "Tokenizer has not been initialized. Cannot save.")
            logging.error("Attempted to save model but tokenizer was not initialized.")
            return

        save_directory = filedialog.askdirectory(title="Select Save Directory")
        if save_directory:
            config = {
                "vocab_size": len(self.tokenizer),
                "embed_size": self.hidden_size.get(),
                "hidden_size": self.hidden_size.get(),
                "num_heads": self.num_heads.get(),
                "num_layers": self.num_layers.get(),
                "architecture": self.architecture.get()
            }
            config_path = os.path.join(save_directory, 'model_config.json')
            with open(config_path, 'w') as f:
                json.dump(config, f)

            # Ensure embeddings match tokenizer
            tokenizer_vocab_size = len(self.tokenizer)
            if hasattr(self.model, 'embedding') and self.model.embedding.num_embeddings != tokenizer_vocab_size:
                self.model.resize_token_embeddings(tokenizer_vocab_size)
                logging.info(f"Resized embeddings to match tokenizer vocabulary size: {tokenizer_vocab_size}")

            # Save the model state dictionary
            if self.architecture.get() == "Transformer":
                model_file_name = 'custom_transformer_model.pth'
            elif self.architecture.get() == "MatMul-Free LM":
                model_file_name = 'matmul_free_lm.pth'
            else:
                messagebox.showerror("Error", f"Unsupported architecture: {self.architecture.get()}")
                return

            model_path = os.path.join(save_directory, model_file_name)
            torch.save(self.model.state_dict(), model_path)

            # Save the tokenizer
            self.tokenizer.save_pretrained(save_directory)

            messagebox.showinfo("Success", "Model, tokenizer, and config saved successfully.")
            logging.info("Model, tokenizer, and config saved successfully.")

    def stop_training_command(self):
        self.stop_training.set()
        messagebox.showinfo("Stop Training", "Training stopped.")
        logging.info("Training stopped by user.")

    def expand_transformer(self):
        # Placeholder method; not used in current implementation
        pass

    
    def load_dataset(self):
        if self.use_chunked_dataset.get():
            # Load data from chunked files
            self.tokenized_data_path = filedialog.askdirectory(
                title="Select Tokenized Data Directory"
            )
            if not self.tokenized_data_path:
                messagebox.showerror("Error", "No tokenized data directory selected.")
                return

            # Check if directory contains chunked data files
            chunk_files = [f for f in os.listdir(self.tokenized_data_path) if f.startswith('chunk_') and f.endswith('.jsonl')]
            if not chunk_files:
                messagebox.showerror("Error", "No chunked data files found in the selected directory.")
                return

            self.chunked_files = [os.path.join(self.tokenized_data_path, f) for f in chunk_files]
            messagebox.showinfo("Success", f"Loaded chunked dataset with {len(self.chunked_files)} files.")
            logging.info(f"Loaded chunked dataset with {len(self.chunked_files)} files.")
        else:
            # Load standard dataset
            if not self.dataset_path:
                messagebox.showerror("Error", "No dataset directory selected.")
                return

            dataset_files = os.listdir(self.dataset_path)
            self.query_target_pairs = []

            for file in dataset_files:
                file_path = os.path.join(self.dataset_path, file)
                if file.endswith('.json') or file.endswith('.jsonl'):
                    try:
                        with open(file_path, 'r', encoding='utf-8') as f:
                            if file.endswith('.jsonl'):
                                for line in f:
                                    conversation = json.loads(line.strip())
                                    self.query_target_pairs.extend(self.extract_query_target_pairs([conversation]))

                                # After loading query_target_pairs
                                for i in range(min(5, len(self.query_target_pairs))):
                                    query, target = self.query_target_pairs[i]


                            else:
                                data = json.load(f)
                                self.query_target_pairs.extend(self.extract_query_target_pairs(data)) 
                                # After loading query_target_pairs
                                for i in range(min(5, len(self.query_target_pairs))):
                                    query, target = self.query_target_pairs[i]
                               
                    except Exception as e:
                        messagebox.showerror("Error", f"Failed to read JSON file '{file}': {str(e)}")
                else:
                    messagebox.showwarning("Warning", f"Unsupported file format: '{file}'")

            if not self.query_target_pairs:
                messagebox.showerror("Error", "No valid query/target pairs found in the dataset.")
                return

            # Store text data for saving as a text file
            self.text_data = []
            for query, target in self.query_target_pairs:
                self.text_data.append(f"User: {query}\nAssistant: {target}")

            messagebox.showinfo("Success", f"Loaded dataset with {len(self.query_target_pairs)} query/target pairs.")
            logging.info(f"Loaded dataset with {len(self.query_target_pairs)} query/target pairs.")

    def extract_query_target_pairs(self, data):
        query_target_pairs = []
        for conversation in data:
            messages = conversation.get("messages", [])
            for i in range(len(messages) - 1):
                if messages[i]["role"] == "user" and messages[i + 1]["role"] == "assistant":
                    query = messages[i]["content"].replace('\n', ' ').strip()
                    target = messages[i + 1]["content"].replace('\n', ' ').strip()
                    query_target_pairs.append((query, target))
        return query_target_pairs

    def register_attention_hooks(self):
        self.attention_weights = []

        def hook_fn(module, input, output):
            if isinstance(module, nn.MultiheadAttention):
                self.attention_weights.append(output[1])  # Assuming the output contains attention weights

        for name, module in self.model.named_modules():
            if isinstance(module, nn.MultiheadAttention):
                module.register_forward_hook(hook_fn)

    def plot_attention_heatmap(self, attention, layer, phase, save_path):
        """
        Visualize attention weights as heatmaps.
        """
        try:
            attention = attention.cpu().detach().numpy()  # Ensure tensor is on the CPU and convert to numpy
            if attention is None or not len(attention):
                logging.warning(f"Attention data missing for layer {layer}. Skipping heatmap.")
                return

            plt.figure(figsize=(12, 6))
            sns.heatmap(attention.cpu().numpy(), cmap="viridis")
            plt.title(f"Attention Heatmap - Layer {layer} - Phase: {phase}")
            plt.xlabel("Heads")
            plt.ylabel("Tokens")
            plt.tight_layout()
            plt.savefig(os.path.join(save_path, f"attention_heatmap_layer_{layer}.png"))
            plt.close()

            logging.info("Attention heatmap saved")

        except Exception as e:
            logging.error(f"Error plotting attention heatmap: {e}")


    def plot_weight_distribution(self, param, phase=None, save_path=None, name=None):
        """
        Visualize the distribution of weights.
        """
        try:
            plt.figure(figsize=(8, 6))
            weights = param.cpu().detach().numpy()
            plt.hist(weights.flatten(), bins=100, alpha=0.75)
            plt.title(f"Weight Distribution: {name}")
            plt.xlabel("Weight Value")
            plt.ylabel("Frequency")
            plt.tight_layout()
            plt.savefig(os.path.join(save_path, f"{name}_weight_distribution.png"))
            plt.close()

            logging.info("Weight distribution plot saved at path")
        except Exception as e:
            logging.error(f"Error plotting weight distribution: {e}")



    def visualize_model_output(self, input_ids, logits, labels, save_path=None):
        """
        Visualize the outpit of model.
        """
        try:
            # Convert logits and labels to CPU
            print("input_ids shape pre-index:", input_ids.shape if hasattr(input_ids, 'shape') else type(input_ids))
            print("logits shape pre-index:", logits.shape if hasattr(logits, 'shape') else type(logits))
            print("labels shape pre-index:", labels.shape if hasattr(labels, 'shape') else type(labels))

            logits = logits[0].cpu().detach().numpy()
            labels = labels[0].cpu().detach().numpy()
            logits=logits.argmax(axis=-1)
            
            # Convert input_ids to NumPy if it's a tensor
            if isinstance(input_ids, torch.Tensor):
                input_ids = input_ids[0].cpu().numpy()

            plt.figure(figsize=(12, 6))
            plt.plot(range(len(input_ids)), input_ids, label="Input IDs", alpha=0.7)
            plt.plot(range(len(logits)), logits, label="Predicted IDs", alpha=0.7)
            plt.plot(range(len(labels)), labels, label="True Labels", alpha=0.7)
            plt.legend()
            plt.xlabel("Sequence Position")
            plt.ylabel("Token ID")
            plt.title("Model Input vs. Prediction vs. True Labels")

            if save_path:
                plt.savefig(f"{save_path}/model_output.png", dpi=300)
                plt.close()
            else:
                plt.close()
        except Exception as e:
            logging.error(f"Error plotting output visualization: {e}")


    def visualize_gradients(self, model, save_path, phase):
        """Visualize gradient flow for model parameters."""
        try:
            plt.figure(figsize=(10, 6))
            layers = []
            max_grads = []
            mean_grads = []
            for name, param in model.named_parameters():
                if param.requires_grad and param.grad is not None:
                    layers.append(name)
                    max_grads.append(param.grad.abs().max().item())
                    mean_grads.append(param.grad.abs().mean().item())
            plt.barh(layers, max_grads, alpha=0.6, color="b", label="max-gradient")
            plt.barh(layers, mean_grads, alpha=0.6, color="r", label="mean-gradient")
            plt.legend()
            plt.title("Gradient flow")
            plt.xlabel("Gradient Magnitude")
            plt.ylabel("Layers")
            plt.tight_layout()
            plt.savefig(os.path.join(save_path, "gradient_flow.png"))
            plt.close()
        except Exception as e:
            logging.error(f"Error visualizing gradients sadface: {e}")
            



    def trigger_gradient_visualization(self):
        save_path = filedialog.askdirectory(title="Select Directory to Save Gradient Visualization")
        self.visualize_gradients(self.model.named_parameters(), save_path)

    def trigger_output_visualization(self):
        save_path = filedialog.askdirectory(title="Select Directory to Save Output Visualization")
        input_ids = ...  # Load sample input IDs
        logits = ...  # Model output logits
        labels = ...  # True labels
        self.visualize_model_output(input_ids, logits, labels, save_path)
        


# Main application entry point
if __name__ == "__main__":
    root = tk.Tk()
    app = UnifiedTransformerGUI(root)
    root.mainloop()
