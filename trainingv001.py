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
import numpy as np
import psutil
from torch.amp import GradScaler, autocast
import linecache
import torch
from torch.autograd import Function

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
from transformers import BertModel

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
        
    def initialize_weights(self):
        for name, param in self.named_parameters():
            if 'weight' in name:
                nn.init.xavier_uniform_(param)
            elif 'bias' in name:
                nn.init.zeros_(param)

    def forward(self, src, src_mask=None, src_key_padding_mask=None):
        # src: (batch_size, seq_len)
        src = self.embedding(src) * math.sqrt(self.embed_size)
        src = self.pos_encoder(src)
        output = self.transformer_encoder(src, mask=src_mask, src_key_padding_mask=src_key_padding_mask)
        logits = self.fc_out(output)  # (batch_size, seq_len, vocab_size)
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
    def forward(_ctx, weight, num_bits=8):
        # Ternarize weights to -1, 0, or +1
        """
        Quantize weights to continuous values within a specified range.
        """
        # Calculate the scale factor based on the mean absolute value
        s = 1.0 / (weight.abs().mean() + 1e-8)
        
        # Scale and quantize the weights
        max_int = 2 ** (num_bits - 1) - 1  # For example, 127 for 8 bits
        weight_scaled = s * weight
        weight_clipped = torch.clamp(weight_scaled, -max_int, max_int)
        weight_quantized = torch.round(weight_clipped)
        
        # De-quantize back to the continuous range
        weight_bar = weight_quantized / s
        
        # Ensure no NaN or Inf values
        if torch.isnan(weight_bar).any() or torch.isinf(weight_bar).any():
            logging.error("Quantized weights contain NaN or Inf.")
            raise ValueError("Quantized weights contain NaN or Inf.")
        
        ternary_weight = torch.sign(weight_bar)
        return ternary_weight

    @staticmethod
    def backward(_ctx, grad_output):
        # Gradient is passed through unchanged
        grad_input = grad_output.clone()
        return grad_input


def ternarize_weight(weight):
    ternary_weight = TernaryWeightFunction.apply(weight)
    if torch.isnan(ternary_weight).any() or torch.isinf(ternary_weight).any():
        logging.error("Ternarized weights contain NaN or Inf.")
        raise ValueError("Ternarized weights contain NaN or Inf.")
    return ternary_weight

class WeightQuantFunction(Function):
    @staticmethod
    def forward(ctx, W, max_scale=1e3, min_scale=1e-3):
        """
        Forward pass for Weight Quantization using STE.
        """
        # Compute scaling factor s
        mean_abs = torch.mean(torch.abs(W))
        s = 1.0 / (mean_abs + 1e-8)  # Prevent division by zero
        
        # Clamp scaling factor
        s = torch.clamp(s, min=min_scale, max=max_scale)
        
        # Quantize
        W_scaled = s * W
        W_quant = torch.clamp(torch.round(W_scaled), -1, 1) / s
        
        # Save tensors for backward
        ctx.save_for_backward(W, s)
        ctx.min_scale = min_scale
        ctx.max_scale = max_scale
        
        return W_quant

    @staticmethod
    def backward(ctx, dW_quant):
        """
        Backward pass for Weight Quantization using STE.
        """
        W, s = ctx.saved_tensors
        # Gradient is passed directly through STE
        dW = dW_quant.clone()
        return dW, None, None


class MatMulFreeLinearFunction(Function):
    @staticmethod
    def forward(ctx, input, weight_quant):
        """
        Forward pass for MatMul-Free Linear operation.
        """
        # Create masks and align dtype with input
        pos_mask = (weight_quant == 1).float().to(input.dtype)  # Convert to input's dtype
        neg_mask = (weight_quant == -1).float().to(input.dtype)  # Convert to input's dtype

        # Transpose to align dimensions for matmul
        pos_mask = pos_mask.t()  # Shape: (512, 30000)
        neg_mask = neg_mask.t()  # Shape: (512, 30000)

        # Perform matrix multiplication
        pos_contrib = torch.matmul(input, pos_mask)  # Shape: (batch_size, 1024, 30000)
        neg_contrib = torch.matmul(input, neg_mask)  # Shape: (batch_size, 1024, 30000)
        output = pos_contrib - neg_contrib         # Shape: (batch_size, 1024, 30000)

        # Save for backward
        ctx.save_for_backward(input, pos_mask, neg_mask)

        return output

    @staticmethod
    def backward(ctx, dO):
        """
        Backward pass for MatMul-Free Linear operation.
        """
        input, pos_mask, neg_mask = ctx.saved_tensors

        # Compute gradients w.r.t. input
        dInput = torch.matmul(dO, pos_mask.t()) - torch.matmul(dO, neg_mask.t())  # Shape: (batch_size, 1024, 512)

        # No gradient w.r.t. weight_quant as it's a quantized constant
        return dInput, None




class ActivationQuantFunction(Function):
    @staticmethod
    def forward(ctx, X, max_scale=1e3, min_scale=1e-3):
        """
        Forward pass for Activation Quantization using STE.
        """
        # Compute scaling factor s
        max_val = torch.max(torch.abs(X), dim=-1, keepdim=True)[0]
        s = 127.0 / (max_val + 1e-8)  # Prevent division by zero
        
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

class RMSNormFunction(Function):
    @staticmethod
    def forward(ctx, X, epsilon=1e-8, max_r=1e3):
        """
        Forward pass for RMS Normalization.
        """
        # Compute mean and variance
        mu = torch.mean(X, dim=-1, keepdim=True)
        sigma2 = torch.var(X, dim=-1, unbiased=False, keepdim=True)

        # Compute scaling factor r
        r = 1.0 / torch.sqrt(sigma2 + epsilon)

        # Clamp r to prevent it from becoming too large
        r = torch.clamp(r, max=max_r)

        # Normalize
        Y = r * (X - mu)

        # Save tensors for backward
        ctx.save_for_backward(Y, r, X - mu)
        ctx.epsilon = epsilon
        ctx.max_r = max_r

        return Y

    @staticmethod
    def backward(ctx, dY):
        """
        Backward pass for RMS Normalization.
        """
        Y, r, X_minus_mu = ctx.saved_tensors
        epsilon = ctx.epsilon
        max_r = ctx.max_r

        # Number of features
        N = Y.size(-1)

        # Compute gradients
        dX = (dY - torch.mean(dY, dim=-1, keepdim=True) - Y * torch.mean(dY * Y, dim=-1, keepdim=True)) * r

        return dX, None, None  # Only gradients w.r.t. X are needed


class MLGRULayer(nn.Module):
    def __init__(self, embed_size, hidden_size, eps=1e-8):
        super(MLGRULayer, self).__init__()
        self.cell = MLGRUCell(embed_size, hidden_size, eps)

    def forward(self, x, seq_lengths):
        batch_size, max_seq_len, _ = x.size()
        h_t = torch.zeros(batch_size, self.cell.hidden_size, device=x.device)
        outputs = []

        for t in range(max_seq_len):
            mask = (seq_lengths > t).float().unsqueeze(1)
            x_t = x[:, t, :]
            o_t, h_t = self.cell(x_t, h_t)
            h_t = h_t * mask + h_t.detach() * (1 - mask)
            outputs.append(o_t.unsqueeze(1) * mask.unsqueeze(2))

        outputs = torch.cat(outputs, dim=1)

        # Log statistics after MLGRULayer
        if torch.isnan(outputs).any() or torch.isinf(outputs).any():
            logging.error("Outputs from MLGRULayer contain NaN or Inf.")
            raise ValueError("Outputs from MLGRULayer contain NaN or Inf.")

        return outputs


class MLGRUCell(nn.Module):
    def __init__(self, input_size, hidden_size, eps=1e-8):
        super(MLGRUCell, self).__init__()
        self.hidden_size = hidden_size
        self.eps = eps

        # Initialize weights
        self.W_f = nn.Parameter(torch.randn(hidden_size, input_size) * 0.01)
        self.W_c = nn.Parameter(torch.randn(hidden_size, input_size) * 0.01)
        self.W_g = nn.Parameter(torch.randn(hidden_size, input_size) * 0.01)
        self.b_f = nn.Parameter(torch.zeros(hidden_size))
        self.b_c = nn.Parameter(torch.zeros(hidden_size))
        self.b_g = nn.Parameter(torch.zeros(hidden_size))

        self.initialize_weights()

    def initialize_weights(self):
        for name, param in self.named_parameters():
            if 'W_' in name:
                nn.init.xavier_uniform_(param)
                logging.debug(f"Initialized {name} with Xavier uniform.")
            elif 'b_' in name:
                nn.init.zeros_(param)
                logging.debug(f"Initialized {name} with zeros.")

    def forward(self, x_t, h_t_minus_1):
        # RMS Normalization and Activation Quantization
        x_t = RMSNormFunction.apply(x_t, self.eps)
        x_t = ActivationQuantFunction.apply(x_t)

        # Ternarize Weights
        W_f_ternary = WeightQuantFunction.apply(self.W_f)
        W_c_ternary = WeightQuantFunction.apply(self.W_c)
        W_g_ternary = WeightQuantFunction.apply(self.W_g)

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





class MatMulFreeGLU(nn.Module):
    def __init__(self, input_size, hidden_size, eps=1e-8):
        super(MatMulFreeGLU, self).__init__()
        self.eps = eps
        self.W_g = nn.Parameter(torch.randn(hidden_size, input_size) * 0.01)
        self.W_u = nn.Parameter(torch.randn(hidden_size, input_size) * 0.01)
        self.W_d = nn.Parameter(torch.randn(hidden_size, hidden_size) * 0.01)

        self.initialize_weights()

    def initialize_weights(self):
        for name, param in self.named_parameters():
            if 'W_' in name:
                nn.init.xavier_uniform_(param)
                logging.debug(f"Initialized {name} with Xavier uniform.")
            elif 'b_' in name:
                nn.init.zeros_(param)
                logging.debug(f"Initialized {name} with zeros.")

    def forward(self, x):
        # Apply RMS normalization using custom Function
        x_norm = RMSNormFunction.apply(x, self.eps)
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
        g_t = MatMulFreeLinearFunction.apply(x_quant, W_g_bar)
        u_t = MatMulFreeLinearFunction.apply(x_quant, W_u_bar)
        d_t = MatMulFreeLinearFunction.apply(g_t, W_d_bar)

        # Activation Function
        p_t = torch.sigmoid(g_t) * u_t

        # Output
        output = torch.sigmoid(d_t) * p_t

        # Check for NaN or Inf in output
        if torch.isnan(output).any() or torch.isinf(output).any():
            logging.error("Output of MatMulFreeGLU contains NaN or Inf.")
            raise ValueError("Output of MatMulFreeGLU contains NaN or Inf.")

        return output



class MatMulFreeLanguageModel(nn.Module):
    def __init__(self, vocab_size, embed_size, hidden_size, eps=1e-8):
        super(MatMulFreeLanguageModel, self).__init__()
        self.eps = eps
        self.embedding = nn.Embedding(vocab_size, embed_size)
        self.mlgru_layer = MLGRULayer(embed_size, hidden_size, eps)
        self.glu = MatMulFreeGLU(hidden_size, hidden_size, eps)
        self.output_layer = nn.Linear(hidden_size, vocab_size)

    def initialize_weights(self):
        for name, param in self.named_parameters():
            if 'weight' in name:
                nn.init.xavier_uniform_(param)
                logging.debug(f"Initialized {name} with Xavier uniform.")
            elif 'bias' in name:
                nn.init.zeros_(param)
                logging.debug(f"Initialized {name} with zeros.")

    def forward(self, input_ids, seq_length):
        # Embedding Layer
        x = self.embedding(input_ids)
        logging.debug(f"Embedding output - mean: {x.mean().item():.4f}, std: {x.std().item():.4f}")

        # MLGRULayer
        x = self.mlgru_layer(x, seq_length)
        logging.debug(f"MLGRULayer output - mean: {x.mean().item():.4f}, std: {x.std().item():.4f}")

        # MatMulFreeGLU
        x = self.glu(x)
        logging.debug(f"GLU output - mean: {x.mean().item():.4f}, std: {x.std().item():.4f}")

        # Check if x is finite before RMS normalization
        if torch.isnan(x).any() or torch.isinf(x).any():
            logging.error("x contains NaN or Inf before rms_norm.")
            raise ValueError("x contains NaN or Inf before rms_norm.")

        # RMS Normalization using custom autograd Function
        x = RMSNormFunction.apply(x, self.eps)
        logging.debug(f"After rms_norm - mean: {x.mean().item():.4f}, std: {x.std().item():.4f}")

        # Check for NaN or Inf after RMS normalization
        if torch.isnan(x).any() or torch.isinf(x).any():
            logging.error("x contains NaN or Inf after rms_norm.")
            raise ValueError("x contains NaN or Inf after rms_norm.")

        # Activation Quantization using custom autograd Function
        x = ActivationQuantFunction.apply(x)
        logging.debug(f"After activation_quant - mean: {x.mean().item():.4f}, std: {x.std().item():.4f}")

        # Check for NaN or Inf after activation quantization
        if torch.isnan(x).any() or torch.isinf(x).any():
            logging.error("x contains NaN or Inf after activation_quant.")
            raise ValueError("x contains NaN or Inf after activation_quant.")

        # Weight Quantization using custom autograd Function
        W_bar = WeightQuantFunction.apply(self.output_layer.weight)
        logging.debug(f"Ternarized output weights - shape: {W_bar.shape}")

        # MatMul-Free Linear Operation using custom autograd Function
        logits = MatMulFreeLinearFunction.apply(x, W_bar) + self.output_layer.bias
        logging.debug(f"Logits - mean: {logits.mean().item():.4f}, std: {logits.std().item():.4f}")

        # Check for NaN or Inf in logits
        if torch.isnan(logits).any() or torch.isinf(logits).any():
            logging.error("Logits contain NaN or Inf after matmul_free_linear.")
            raise ValueError("Logits contain NaN or Inf after matmul_free_linear.")

        return logits






# Custom Dataset Class
class TokenizedDataset(torch.utils.data.Dataset):
    def __init__(self, tokenized_data_path, tokenizer, max_length=1024):
        self.tokenized_data_path = tokenized_data_path
        self.tokenizer = tokenizer
        self.max_length = max_length

        # Get a list of chunk files
        self.chunk_files = [os.path.join(self.tokenized_data_path, f) for f in os.listdir(self.tokenized_data_path) if f.startswith('chunk_') and f.endswith('.jsonl')]
        self.chunk_files.sort()  # Ensure the chunks are in order

        # Initialize counters
        self.current_chunk_idx = 0
        self.current_chunk_data = []
        self.current_chunk_len = 0
        self.total_samples = 0

        # Compute total number of samples
        for chunk_file in self.chunk_files:
            with open(chunk_file, 'r', encoding='utf-8') as f:
                num_lines = sum(1 for _ in f)
                self.total_samples += num_lines

    def __len__(self):
        return self.total_samples

    def load_chunk(self, idx):
        chunk_file = self.chunk_files[idx]
        logging.info(f"Loading chunk file: {chunk_file}")
        try:
            with open(chunk_file, 'r', encoding='utf-8') as f:
                self.current_chunk_data = [json.loads(line.strip()) for line in f]
            self.current_chunk_len = len(self.current_chunk_data)
            self.current_chunk_idx = idx
        except Exception as e:
            logging.error(f"Failed to open chunk file {chunk_file}: {e}")
            raise
        
    def __getitem__(self, idx):

        # Determine which chunk this index falls into
        sample_idx = idx
        chunk_cumulative_sizes = []
        cumulative = 0
        for chunk_file in self.chunk_files:
            with open(chunk_file, 'r', encoding='utf-8') as f:
                num_lines = sum(1 for _ in f)
            cumulative += num_lines
            chunk_cumulative_sizes.append(cumulative)
            if sample_idx < cumulative:
                break

        # Load the appropriate chunk if not already loaded
        chunk_idx = chunk_cumulative_sizes.index(cumulative)
        if self.current_chunk_idx != chunk_idx:
            self.load_chunk(chunk_idx)

        # Adjust the sample index to be within the current chunk
        if chunk_idx > 0:
            sample_idx -= chunk_cumulative_sizes[chunk_idx - 1]

        record = self.current_chunk_data[sample_idx]
        input_ids = record['input_ids']
        labels = record['labels']
        seq_lengths = len(input_ids)

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

        # Check for empty sequences
        if len(input_ids) == 0:
            logging.error(f"Empty input_ids at index {idx}.")
            raise ValueError(f"Empty input_ids at index {idx}.")
        if len(labels) == 0:
            logging.error(f"Empty labels at index {idx}.")
            raise ValueError(f"Empty labels at index {idx}.")
        return input_ids, attention_mask, labels, seq_lengths





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
        self.num_heads = tk.IntVar(value=16)
        self.num_layers = tk.IntVar(value=32)

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
        logging.basicConfig(filename=self.log_file_path, level=logging.DEBUG,
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
                    messagebox.showinfo("Success", f"Tokenized data file selected: {self.tokenized_data_path}")
        else:
            if use_chunked:
                # User wants to create new chunked tokenized data, select a directory to save
                self.tokenized_data_path = filedialog.askdirectory(
                    title="Select Directory to Save Tokenized Data"
                )
                if self.tokenized_data_path:
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
        else:
            # Check if query_target_pairs is loaded
            if not hasattr(self, 'query_target_pairs') or not self.query_target_pairs:
                messagebox.showerror("Error", "No query_target_pairs loaded. Please load the dataset first.")
                return

        if not self.tokenized_data_path:
            messagebox.showerror("Error", "Tokenized data path is not set. Please select or create tokenized data.")
            return

        try:
            use_chunked = self.use_chunked_dataset.get()
            if use_chunked:
                # Ensure the save path is a directory
                if not os.path.isdir(self.tokenized_data_path):
                    os.makedirs(self.tokenized_data_path, exist_ok=True)
                chunk_size = 32  # Adjust based on your memory constraints
                num_chunks = (len(self.query_target_pairs) + chunk_size - 1) // chunk_size

                for chunk_idx in range(num_chunks):
                    chunk_pairs = self.query_target_pairs[chunk_idx * chunk_size : (chunk_idx + 1) * chunk_size]
                    chunk_file_path = os.path.join(self.tokenized_data_path, f'chunk_{chunk_idx}.jsonl')

                    with open(chunk_file_path, 'w', encoding='utf-8') as f:
                        for query, target in chunk_pairs:
                            # Tokenize query and target separately
                            query_ids = self.tokenizer.encode(query, truncation=True, max_length=1024)
                            target_ids = self.tokenizer.encode(target, truncation=True, max_length=1024)

                            if query_ids is None or target_ids is None:
                                logging.error(f"Tokenizer returned None for query or target. Query: {query}, Target: {target}")
                                continue

                            if not isinstance(query_ids, list) or not isinstance(target_ids, list):
                                logging.error(f"Tokenizer returned invalid type. Query IDs: {query_ids}, Target IDs: {target_ids}")
                                continue

                            if self.tokenizer.eos_token_id is None:
                                logging.error("eos_token_id is None.")
                                continue

                            # Combine query and target, adding special tokens if necessary
                            input_ids = query_ids + [self.tokenizer.eos_token_id] + target_ids
                            labels = [self.tokenizer.pad_token_id] * len(query_ids) + [self.tokenizer.eos_token_id] + target_ids

                            # Save as JSON lines
                            record = {
                                'input_ids': input_ids,
                                'labels': labels
                            }
                            f.write(json.dumps(record) + '\n')

                    logging.info(f"Chunk {chunk_idx} tokenized and saved to {chunk_file_path}")

                messagebox.showinfo("Success", f"Data tokenized into {num_chunks} chunks and saved successfully to {self.tokenized_data_path}.")
                logging.info(f"Data tokenized into {num_chunks} chunks and saved successfully to {self.tokenized_data_path}.")

            else:
                # Tokenize and save as a single file
                with open(self.tokenized_data_path, 'w', encoding='utf-8') as f:
                    for query, target in self.query_target_pairs:
                        # Tokenize query and target separately
                        query_ids = self.tokenizer.encode(query, truncation=True, max_length=1024)
                        target_ids = self.tokenizer.encode(target, truncation=True, max_length=1024)

                        if query_ids is None or target_ids is None:
                            logging.error(f"Tokenizer returned None for query or target. Query: {query}, Target: {target}")
                            continue

                        if not isinstance(query_ids, list) or not isinstance(target_ids, list):
                            logging.error(f"Tokenizer returned invalid type. Query IDs: {query_ids}, Target IDs: {target_ids}")
                            continue

                        if self.tokenizer.eos_token_id is None:
                            logging.error("eos_token_id is None.")
                            continue

                        # Combine query and target, adding special tokens if necessary
                        input_ids = query_ids + [self.tokenizer.eos_token_id] + target_ids
                        labels = [self.tokenizer.pad_token_id] * len(query_ids) + [self.tokenizer.eos_token_id] + target_ids

                        # Save as JSON lines
                        record = {
                            'input_ids': input_ids,
                            'labels': labels
                        }
                        f.write(json.dumps(record) + '\n')

                messagebox.showinfo("Success", f"Data tokenized and saved successfully to {self.tokenized_data_path}.")
                logging.info(f"Data tokenized and saved successfully to {self.tokenized_data_path}.")
        except Exception as e:
            logging.error(f"Tokenization failed: {str(e)}")
            messagebox.showerror("Error", f"Tokenization failed: {str(e)}")


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
                    eps=1e-8
                )
            else:
                messagebox.showerror("Error", f"Unsupported architecture: {self.architecture.get()}")
                return

            # Move the entire model to the selected device
            self.model.initialize_weights()

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

                    dataset = TokenizedDataset(
                        tokenized_data_path=self.tokenized_data_path,
                        tokenizer=self.tokenizer,
                        max_length=1024
                    )
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
                # Initialize the standard TokenizedDataset
                dataset = TokenizedDataset(
                    tokenized_data_path=self.tokenized_data_path,
                    tokenizer=self.tokenizer,
                    max_length=1024
                )
                dataloader = DataLoader(
                    dataset,
                    batch_size=int(self.batch_size.get()),
                    shuffle=True,
                    num_workers=0,  # Set to 0 to prevent multiple workers from loading chunks simultaneously
                    pin_memory=True,
                    collate_fn=collate_fn
                )

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

                    logits = outputs.view(-1, outputs.size(-1))
                    targets = batch_labels.view(-1)

                    # Compute loss
                    loss = F.cross_entropy(
                            logits,
                            targets,
                            ignore_index=self.tokenizer.pad_token_id
                        )
                    logging.debug(f"Loss computed: {loss.item()}")

                    # Backward pass and optimization
                    scaler.scale(loss).backward()
                    logging.info("Loss backward computed")

                    # Check for NaN or Inf in gradients
                    for name, param in self.model.named_parameters():
                        if param.grad is not None:
                            if torch.isnan(param.grad).any() or torch.isinf(param.grad).any():
                                logging.error(f"Gradient for {name} contains NaN or Inf.")
                                raise ValueError(f"Gradient for {name} contains NaN or Inf.")

                    # Gradient clipping
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=5.0)
                    logging.info("Gradient clipping applied")

                    scaler.step(optimizer)
                    scaler.update()
                    optimizer.zero_grad()
                    logging.info("Optimizer step and zero_grad completed")

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

                # Log epoch loss
                average_epoch_loss = epoch_loss / len(dataloader)
                self.loss_history.append(average_epoch_loss)
                logging.info(f"Epoch {epoch + 1}/{self.epochs.get()} completed with average loss: {average_epoch_loss}")
                self.root.after(0, self.update_status, f"Epoch {epoch + 1}/{self.epochs.get()} completed. Current LR = {scheduler.get_last_lr()}")

        except Exception as e:
            logging.error(f"An error occurred during training: {str(e)}")
            messagebox.showerror("Error", f"An error occurred during training: {str(e)}")



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
                                    logging.debug(f"Sample {i}: Query Length={len(query)}, Target Length={len(target)}")
                                    logging.debug(f"Sample {i}: Query IDs={self.tokenizer.encode(query, truncation=True, max_length=1024)}")
                                    logging.debug(f"Sample {i}: Target IDs={self.tokenizer.encode(target, truncation=True, max_length=1024)}")

                            else:
                                data = json.load(f)
                                self.query_target_pairs.extend(self.extract_query_target_pairs(data)) 
                                # After loading query_target_pairs
                                for i in range(min(5, len(self.query_target_pairs))):
                                    query, target = self.query_target_pairs[i]
                                    logging.debug(f"Sample {i}: Query Length={len(query)}, Target Length={len(target)}")
                                    logging.debug(f"Sample {i}: Query IDs={self.tokenizer.encode(query, truncation=True, max_length=1024)}")
                                    logging.debug(f"Sample {i}: Target IDs={self.tokenizer.encode(target, truncation=True, max_length=1024)}")

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
                    logging.debug(f"Extracted query-target pair: Query: {query[:50]}... Target: {target[:50]}...")
        return query_target_pairs


# Main application entry point
if __name__ == "__main__":
    root = tk.Tk()
    app = UnifiedTransformerGUI(root)
    root.mainloop()
