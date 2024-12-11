import tkinter as tk
from tkinter import ttk, filedialog, messagebox, simpledialog
import json
import threading
import logging
import torch
import torch.nn as nn
import torch.nn.functional as F
import pandas as pd
from torch.utils.data import DataLoader
import os
from transformers import PreTrainedTokenizerFast, AddedToken
from tokenizers import Tokenizer, models, pre_tokenizers, decoders, trainers, normalizers
import pickle
import numpy as np
import math
import torch.quantization
from torch.quantization import QuantStub, DeQuantStub, fuse_modules
import torch.amp


# Print whether CUDA is available
print(f"CUDA Available: {torch.cuda.is_available()}")



# Global variables
tokenizer = None



# RMS Normalization Function
class RMSNormFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, eps):
        mean = x.mean(dim=-1, keepdim=True)
        variance = x.var(dim=-1, unbiased=False, keepdim=True)
        r = 1 / torch.sqrt(torch.clamp(variance + eps, min=1e-10))  # Prevent division by zero
        y = r * (x - mean)
        ctx.save_for_backward(x, mean, variance, r)
        ctx.eps = eps
        return y


    @staticmethod
    def backward(ctx, grad_output):
        x, mean, variance, r = ctx.saved_tensors
        eps = ctx.eps
        N = x.shape[-1]
        denom = variance + eps
        denom = torch.clamp(denom, min=1e-8)  # Ensure denom is not too small
        grad_input = (1 / N) * r * (
            N * grad_output
            - grad_output.sum(dim=-1, keepdim=True)
            - (x - mean) * ((grad_output * (x - mean)).sum(dim=-1, keepdim=True) / denom)
        )
        return grad_input, None


def rms_norm(x, eps=1e-8):
    return RMSNormFunction.apply(x, eps)

# Activation quantization function
def activation_quant(x, bits=8):
    if torch.isnan(x).any():
        x = torch.nan_to_num(x, nan=0.0)
    qmin = -2 ** (bits - 1)
    qmax = 2 ** (bits - 1) - 1
    x_abs_max = x.abs().max()
    if x_abs_max == 0 or torch.isnan(x_abs_max):
        scale = 1.0  # Avoid division by zero
    else:
        scale = x_abs_max / qmax
    x_quant = torch.clamp((x / scale).round(), qmin, qmax)
    x_dequant = x_quant * scale
    return x_dequant

# Custom Ternary Weight Function
class TernaryWeightFunction(torch.autograd.Function):
    @staticmethod
    def forward(_ctx, weight):
        # Ternarize weights to -1, 0, or +1
        ternary_weight = torch.sign(weight)
        return ternary_weight

    @staticmethod
    def backward(_ctx, grad_output):
        # Gradient is passed through unchanged
        grad_input = grad_output.clone()
        return grad_input

def ternarize_weight(weight):
    return TernaryWeightFunction.apply(weight)

# Matmul-free linear function with quantization
def matmul_free_linear(input, weight):
    # Quantize input and weight
    input_q = activation_quant(input)
    weight_q = ternarize_weight(weight)
    
    # Perform matrix multiplication
    output = input_q.matmul(weight_q.t())
    return output

# MatMul-free Linear Gated Recurrent Unit (MLGRU) Cell with QAT
class MLGRUCell(nn.Module):
    def __init__(self, input_size, hidden_size, eps=1e-8):
        super().__init__()
        self.hidden_size = hidden_size
        self.eps = eps

        # Weights and biases
        self.W_f = nn.Parameter(torch.randn(input_size, hidden_size))
        self.W_c = nn.Parameter(torch.randn(input_size, hidden_size))
        self.W_g = nn.Parameter(torch.randn(input_size, hidden_size))
        self.b_f = nn.Parameter(torch.randn(hidden_size))
        self.b_c = nn.Parameter(torch.randn(hidden_size))
        self.b_g = nn.Parameter(torch.randn(hidden_size))

        # Quantization modules
        self.quant = QuantStub()
        self.dequant = DeQuantStub()

    def forward(self, x_t, h_t_minus_1):
        # Ensure QuantStub and DeQuantStub are on the same device as the input
        self.quant.to(x_t.device)
        self.dequant.to(x_t.device)

        # Quantize input
        x_t = self.quant(x_t)

        # Apply RMS normalization
        x_t = rms_norm(x_t, self.eps)

        # Linear operations
        
        f_t_linear = matmul_free_linear(x_t, self.W_f)+self.b_f
        c_t_linear = matmul_free_linear(x_t, self.W_c)+self.b_c
        g_t_linear = matmul_free_linear(x_t, self.W_g)+self.b_g

        # Activation functions
        sig_f_t = torch.sigmoid(f_t_linear)
        silu_c_t = F.silu(c_t_linear)
        sig_g_t = torch.sigmoid(g_t_linear)

        # Hidden state computations
        h_t = sig_f_t * h_t_minus_1 + (1 - sig_f_t) * silu_c_t
        o_t = h_t * sig_g_t

        # Dequantize output
        o_t = self.dequant(o_t)
        h_t = self.dequant(h_t)

        return o_t, h_t


# MLGRU Layer
class MLGRULayer(nn.Module):
    def __init__(self, input_size, hidden_size, eps=1e-8):
        super().__init__()
        self.cell = MLGRUCell(input_size, hidden_size, eps)
        self.hidden_size = hidden_size

        # Quantization modules
        self.quant = QuantStub()
        self.dequant = DeQuantStub()

    def forward(self, x):
        # Ensure QuantStub and DeQuantStub are on the same device as the input
        self.quant.to(x.device)
        self.dequant.to(x.device)

        # Quantize input
        x = self.quant(x)

        batch_size, seq_len, _ = x.size()
        h_t = torch.zeros(batch_size, self.cell.hidden_size, device=x.device)

        outputs = []
        for t in range(seq_len):
            x_t = x[:, t, :]
            o_t, h_t = self.cell(x_t, h_t)
            outputs.append(o_t.unsqueeze(1))

        output = torch.cat(outputs, dim=1)

        # Dequantize output
        output = self.dequant(output)
        return output


# MatMul-free GLU with QAT
class MatMulFreeGLU(nn.Module):
    def __init__(self, input_size, hidden_size, eps=1e-8):
        super().__init__()
        self.eps = eps

        self.W_g = nn.Parameter(torch.randn(input_size, hidden_size))
        self.W_u = nn.Parameter(torch.randn(input_size, hidden_size))
        self.W_d = nn.Parameter(torch.randn(input_size, hidden_size))
        self.b_g = nn.Parameter(torch.randn(hidden_size))
        self.b_u = nn.Parameter(torch.randn(hidden_size))
        self.b_d = nn.Parameter(torch.randn(hidden_size))
        # Quantization modules
        self.quant = QuantStub()
        self.dequant = DeQuantStub()

    def forward(self, x):
        # Quantize input
        x = self.quant(x)

        # Apply RMS normalization
        x = rms_norm(x, self.eps)
        # Quantize activations
        x = activation_quant(x)

        # Linear operations
        g_t = matmul_free_linear(x, self.W_g) + self.b_g
        u_t = matmul_free_linear(x, self.W_u) + self.b_u

        # Activation functions
        g_t = F.silu(g_t)
        p_t = g_t * u_t  # Assuming linear activation

        # Output layer
        d_t = matmul_free_linear(p_t, self.W_d) + self.b_d

        # Dequantize output
        d_t = self.dequant(d_t)
        return d_t

# MatMul-Free Language Model with QAT
class MatMulFreeLanguageModel(nn.Module):
    def __init__(self, vocab_size, embed_size, hidden_size, eps=1e-8):
        super().__init__()
        self.eps = eps
        self.embedding = nn.Embedding(vocab_size, embed_size)
        self.mlgru_layer = MLGRULayer(embed_size, hidden_size, eps)
        self.glu = MatMulFreeGLU(hidden_size, hidden_size, eps)
        self.output_layer = nn.Linear(hidden_size, vocab_size)

        # Quantization modules
        self.quant = QuantStub()
        self.dequant = DeQuantStub()



    def forward(self, input_ids):

        # Quantize input
        input_ids = self.quant(input_ids.float()).long()

        x = self.embedding(input_ids)
        x = self.mlgru_layer(x)
        x = self.glu(x)

        # Apply RMS normalization and activation quantization before output layer
        x = rms_norm(x, self.eps)
        x = activation_quant(x)

        # Output layer
        logits = self.output_layer(x)

        # Dequantize output
        logits = self.dequant(logits)
        return logits



from torch.quantization import get_default_qconfig, float_qparams_weight_only_qconfig

def prepare_model_for_qat(model, device):
    from torch.quantization import QConfig

    # Define a valid QConfig for Linear layers
    activation_fake_quant = torch.quantization.FakeQuantize.with_args(
        observer=torch.quantization.MovingAverageMinMaxObserver,
        quant_min=0,
        quant_max=255,
        dtype=torch.quint8,
        qscheme=torch.per_tensor_affine,  # Use per-tensor quantization
        reduce_range=False,
    )
    weight_fake_quant = torch.quantization.FakeQuantize.with_args(
        observer=torch.quantization.MovingAverageMinMaxObserver,  # Change from PerChannelMinMaxObserver
        quant_min=-128,
        quant_max=127,
        dtype=torch.qint8,
        qscheme=torch.per_tensor_symmetric,  # Use per-tensor quantization
        reduce_range=False,
    )
    default_qconfig = QConfig(activation=activation_fake_quant, weight=weight_fake_quant)

    # Assign the default_qconfig to the model
    model.qconfig = default_qconfig

    # Exclude nn.Embedding from quantization
    for name, module in model.named_modules():
        if isinstance(module, nn.Embedding):
            module.qconfig = None

    torch.quantization.prepare_qat(model, inplace=True)
    model.to(device)






# Convert model to quantized version
def convert_model_to_quantized(model):
    torch.quantization.convert(model, inplace=True)

# UnifiedTransformerGUI class remains largely the same, with modifications to integrate QAT

class UnifiedTransformerGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("Integrated Transformer GUI")
        
        self.layers = []


        # Model Configuration Variables
        self.model_name = tk.StringVar(value="MatMul-Free LM")
        self.num_parameters = tk.IntVar(value=1024)
        self.vocab_size = tk.IntVar(value=100)
        self.hidden_size = tk.IntVar(value=512)
        self.num_heads = tk.IntVar(value=8)
        self.num_layers = tk.IntVar(value=4)
        
        # Device Selection Variable
        self.device_option = tk.StringVar(value="GPU" if torch.cuda.is_available() else "CPU")
        self.device = torch.device(self.map_device(self.device_option.get()))



        # Dynamically calculate parameters based on other inputs
        self.vocab_size.trace_add("write", lambda *args: self.update_num_parameters())
        self.hidden_size.trace_add("write", lambda *args: self.update_num_parameters())
        self.num_layers.trace_add("write", lambda *args: self.update_num_parameters())

        # Set initial calculated value
        self.update_num_parameters()
        
        # Training Parameters
        self.dataset_path = ""
        self.tokenizer_path = ""
        self.batch_size = tk.IntVar(value=1)
        self.learning_rate = tk.DoubleVar(value=0.001)
        self.epochs = tk.IntVar(value=1)

        # Training Variables
        self.loss_history = []
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
        
        
        # Additional attribute to track if model is prepared for QAT
        self.model_prepared_for_qat = False

        
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
        ttk.Button(transformer_frame, text="Quantize for Inference", command=self.quantize_model_for_inference).grid(row=5, column=0, pady=5)


        # Attach parameter calculation to variable updates
        self.vocab_size.trace_add("write", lambda *args: self.update_num_parameters())
        self.hidden_size.trace_add("write", lambda *args: self.update_num_parameters())
        self.num_layers.trace_add("write", lambda *args: self.update_num_parameters())

        # For resuming training
        ttk.Button(transformer_frame, text="Select Model File", command=self.select_model_file).grid(row=3, column=2, pady=5)

        # Architecture selection
        self.architecture = tk.StringVar(value="MatMul-Free LM")
        ttk.Label(transformer_frame, text="Select Architecture:").grid(row=0, column=2, sticky="w")
        ttk.Combobox(transformer_frame, textvariable=self.architecture, values=["MatMul-Free LM", "MatMul-Free LM"], state="readonly").grid(row=0, column=3)


        ttk.Button(transformer_frame, text="Add Layer", command=self.add_layer).grid(row=4, column=0, pady=5)
        ttk.Button(transformer_frame, text="Save Transformer and Model", command=self.save_transformer_and_model).grid(row=1, column=3, pady=5)
        ttk.Button(transformer_frame, text="Load Transformer", command=self.load_transformer).grid(row=1, column=2, pady=5)
        ttk.Button(transformer_frame, text="Initialize/Load Model", command=self.load_model).grid(row=2, column=3, pady=5)

        # Data Selection Frame
        data_frame = ttk.LabelFrame(self.root, text="Data Selection", padding=(10, 10))
        data_frame.pack(fill="x", padx=10, pady=5)

        ttk.Button(data_frame, text="Select Dataset Directory", command=self.select_dataset).pack(pady=5)
        ttk.Button(data_frame, text="Load Dataset", command=self.load_dataset).pack(pady=5)
        ttk.Button(data_frame, text="Select Vocabulary File", command=self.select_vocab).pack(pady=5)
        ttk.Button(data_frame, text="Create Tokenizer from Vocab", command=self.create_tokenizer_from_vocab).pack(pady=5)
        ttk.Button(data_frame, text="Load Tokenizer", command=self.load_tokenizer).pack(pady=5)
        ttk.Button(data_frame, text="Test Tokenizer", command=self.test_tokenizer).pack(pady=5)

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
            model_file_name = 'matmul_free_lm.pth'
            model_path = os.path.join(directory, model_file_name)
            torch.save(self.model.state_dict(), model_path)

            # Save tokenizer
            self.tokenizer.save_pretrained(directory)

            messagebox.showinfo("Success", "Model, tokenizer, and configuration saved successfully!")
            logging.info("Model, tokenizer, and configuration saved successfully.")


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
            
    def quantize_model_for_inference(self):
        if not self.model:
            messagebox.showerror("Error", "Model not initialized. Cannot quantize for inference.")
            return

        try:
            logging.info("Converting model to quantized version for inference...")
            self.model.eval()
            convert_model_to_quantized(self.model)
            messagebox.showinfo("Success", "Model converted to quantized version for inference.")
            logging.info("Model successfully quantized for inference.")

            # Save the quantized model
            save_directory = filedialog.askdirectory(title="Select Save Directory for Quantized Model")
            if save_directory:
                model_path = os.path.join(save_directory, "quantized_model.pth")
                torch.save(self.model.state_dict(), model_path)
                messagebox.showinfo("Success", f"Quantized model saved to {model_path}.")
                logging.info(f"Quantized model saved to {model_path}.")
        except Exception as e:
            logging.error(f"Failed to quantize model: {str(e)}")
            messagebox.showerror("Error", f"Failed to quantize model: {str(e)}")


    def load_model(self):
        try:
            if not self.tokenizer:
                vocab_size = self.vocab_size.get()
            else:
                vocab_size = len(self.tokenizer)

            self.model = MatMulFreeLanguageModel(
                vocab_size=vocab_size,
                embed_size=self.hidden_size.get(),
                hidden_size=self.hidden_size.get()
            )

            # Move model to the selected device
            self.model.to(self.device)

            # Load model state_dict
            if self.model_path and self.model_path.endswith('.pth'):
                state_dict = torch.load(self.model_path, map_location=self.device)
                
                # Check if the state_dict has QAT-specific keys
                if any("fake_quant" in k or "observer_enabled" in k for k in state_dict.keys()):
                    logging.info("Loaded model is already QAT-prepared.")
                    # Initialize model as QAT-prepared before loading the state_dict
                    prepare_model_for_qat(self.model, self.device)
                    self.model_prepared_for_qat = True
                else:
                    logging.info("Loaded model is not QAT-prepared.")
                    self.model_prepared_for_qat = False

                # Load the state_dict into the model
                self.model.load_state_dict(state_dict)
                logging.info("Model state_dict loaded successfully.")

            # Prepare for QAT if not already prepared
            if not self.model_prepared_for_qat:
                prepare_model_for_qat(self.model, self.device)
                self.model_prepared_for_qat = True
                logging.info("Model prepared for QAT.")

            logging.info("Model loaded and moved to device successfully.")
            messagebox.showinfo("Success", "Model initialized successfully.")

        except Exception as e:
            logging.error(f"Failed to initialize model: {str(e)}")
            messagebox.showerror("Error", f"Failed to initialize model: {str(e)}")



            
    def calculate_learning_rate(self, total_params):
        total_params = max(total_params, 1)  # Prevent division by zero
        lr = 17.38 * (total_params ** -0.424)
        return lr

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

    def load_dataset(self):
                """Load and preprocess dataset"""
                if not self.dataset_path:
                    messagebox.showerror("Error", "No dataset directory selected.")
                    return

                dataset_files = os.listdir(self.dataset_path)
                text_data = []

                for file in dataset_files:
                    file_path = os.path.join(self.dataset_path, file)
                    if file.endswith('.csv'):
                        try:
                            df = pd.read_csv(file_path)
                            if 'text' in df.columns:
                                text_data.extend(df['text'].astype(str).tolist())
                            elif 'instruct' in df.columns and 'output' in df.columns:
                                # Handle 'instruct' and 'output' columns
                                df = df.dropna(subset=['instruct', 'output'])
                                combined_text = (df['instruct'].astype(str) + ' ' + df['output'].astype(str)).tolist()
                                text_data.extend(combined_text)
                            else:
                                messagebox.showerror(
                                    "Error", f"CSV file '{file}' missing 'text' or 'instruct' and 'output' columns."
                                )
                        except Exception as e:
                            messagebox.showerror("Error", f"Failed to read CSV file '{file}': {str(e)}")
                    elif file.endswith('.json'):
                        try:
                            with open(file_path, 'r', encoding='utf-8') as f:
                                data = json.load(f)
                                if isinstance(data, list):
                                    for item in data:
                                        if 'question' in item and 'answer' in item:
                                            text_data.append(f"Question: {item['question']} Answer: {item['answer']}")
                                        elif 'text' in item:
                                            text_data.append(item['text'])
                                        elif 'instruct' in item and 'output' in item:
                                            if item['instruct'] and item['output']:
                                                text_data.append(f"{item['instruct']} {item['output']}")
                                elif isinstance(data, dict):
                                    if 'message_1' in data and 'message_2' in data:
                                        text_data.append(f"Message 1: {data['message_1']} Message 2: {data['message_2']}")
                                    elif 'text' in data:
                                        text_data.append(data['text'])
                                    elif 'instruct' in data and 'output' in data:
                                        if data['instruct'] and data['output']:
                                            text_data.append(f"{data['instruct']} {data['output']}")
                                    else:
                                        messagebox.showerror(
                                            "Error", f"JSON file '{file}' missing 'text' or 'instruct' and 'output' keys."
                                        )
                        except Exception as e:
                            messagebox.showerror("Error", f"Failed to decode JSON file '{file}': {str(e)}")
                    elif file.endswith('.parquet'):
                        try:
                            df = pd.read_parquet(file_path)
                            if 'text' in df.columns:
                                text_data.extend(df['text'].astype(str).tolist())
                            elif 'TEXT' in df.columns:
                                text_data.extend(df['TEXT'].astype(str).tolist())
                            elif 'messages' in df.columns:
                                text_data.extend(df['messages'].astype(str).tolist())
                            elif 'instruct' in df.columns and 'output' in df.columns:
                                # Handle 'instruct' and 'output' columns
                                df = df.dropna(subset=['instruct', 'output'])
                                combined_text = (df['instruct'].astype(str) + ' ' + df['output'].astype(str)).tolist()
                                text_data.extend(combined_text)
                            else:
                                messagebox.showerror(
                                    "Error", f"Parquet file '{file}' missing 'text' or 'instruct' and 'output' columns."
                                )
                        except Exception as e:
                            messagebox.showerror("Error", f"Failed to read Parquet file '{file}': {str(e)}")
        
                    elif file.endswith('.txt'):
                        try:
                            with open(file_path, 'r', encoding='utf-8') as f:
                                text = f.read()
                                text_data.append(text)
                        except Exception as e:
                            messagebox.showerror("Error", f"Failed to read text file '{file}': {str(e)}")
                    else:
                        messagebox.showwarning("Warning", f"Unsupported file format: '{file}'")

                if not text_data:
                    messagebox.showerror("Error", "No valid text data found in the dataset directory.")
                    return

                # Preprocess text_data to remove unwanted whitespaces
                processed_text_data = []
                for text in text_data:
                    text = text.replace('\n', '').replace('\r', '').replace('\t', '')
                    # Replace multiple spaces with a single space
                    text = ' '.join(text.split())
                    processed_text_data.append(text)

                self.text_data = processed_text_data  # Store processed text data
                messagebox.showinfo("Success", f"Loaded dataset with {len(text_data)} texts.")
                logging.info(f"Loaded dataset with {len(text_data)} texts.")
                logging.info(f"Preprocessed text: {text_data[:100]}...")  # Log a preview of the text


    def select_or_create_tokenized_data(self):
        answer = messagebox.askyesno("Select or Create Tokenized Data", "Do you want to use existing tokenized data?")
        if answer:
            # User wants to use existing tokenized data
            self.tokenized_data_path = filedialog.askopenfilename(
                title="Select Tokenized Data File",
                filetypes=[("Pickle files", "*.pkl"), ("All files", "*.*")]
            )
            if self.tokenized_data_path:
                messagebox.showinfo("Success", f"Tokenized data file selected: {self.tokenized_data_path}")
        else:
            # User wants to create new tokenized data
            self.tokenized_data_path = filedialog.asksaveasfilename(
                title="Save Tokenized Data As",
                defaultextension=".pkl",
                filetypes=[("Pickle files", "*.pkl"), ("All files", "*.*")]
            )
            if self.tokenized_data_path:
                messagebox.showinfo("Success", f"Tokenized data will be saved to: {self.tokenized_data_path}")


    def tokenize_data(self):
        if not self.tokenizer:
            messagebox.showerror("Error", "Tokenizer not loaded.")
            return
        if self.tokenized_data_path and os.path.exists(self.tokenized_data_path):
            # Load existing tokenized data
            with open(self.tokenized_data_path, 'rb') as f:
                self.train_data = pickle.load(f)
            messagebox.showinfo("Success", f"Loaded pre-tokenized data from {self.tokenized_data_path}.")
            logging.info(f"Loaded pre-tokenized data from {self.tokenized_data_path}.")
        else:
            # Check if text data is loaded
            if not hasattr(self, 'text_data'):
                messagebox.showerror("Error", "No text data loaded.")
                return

            if not self.tokenized_data_path:
                # Ask for a path to save tokenized data if not already set
                self.tokenized_data_path = filedialog.asksaveasfilename(
                    title="Save Tokenized Data As",
                    defaultextension=".pkl",
                    filetypes=[("Pickle files", "*.pkl"), ("All files", "*.*")]
                )
                if not self.tokenized_data_path:
                    messagebox.showerror("Error", "No tokenized data save path selected.")
                    return

            try:
                tokenized_data = []
                for text in self.text_data:
                    encoded = self.tokenizer.encode(text, truncation=True, max_length=1024)
                    tokenized_data.append(encoded)

                # Save the tokenized data
                with open(self.tokenized_data_path, 'wb') as f:
                    pickle.dump(tokenized_data, f)

                messagebox.showinfo("Success", f"Data tokenized and saved to {self.tokenized_data_path}.")
            except Exception as e:
                logging.error(f"Failed to tokenize data: {str(e)}")
                messagebox.showerror("Error", f"Failed to tokenize data: {str(e)}")


    def create_tokenizer_from_vocab(self):
        try:
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

            # Wrap the tokenizer with PreTrainedTokenizerFast
            self.tokenizer = PreTrainedTokenizerFast(
                tokenizer_object=tokenizer,
                unk_token="<UNK>",
                pad_token="<PAD>",
                bos_token="<BOS>",
                eos_token="<EOS>",
            )

            save_directory = filedialog.askdirectory(title="Select Directory to Save Tokenizer")
            if save_directory:
                os.makedirs(save_directory, exist_ok=True)
                self.tokenizer.save_pretrained(save_directory)
                self.tokenizer_path = os.path.join(save_directory, "tokenizer.json")
                messagebox.showinfo("Success", f"Tokenizer saved to {self.tokenizer_path}")
                logging.info(f"Tokenizer saved to {self.tokenizer_path}")

        except Exception as e:
            logging.error(f"Failed to create tokenizer: {str(e)}")
            messagebox.showerror("Error", f"Failed to create tokenizer: {str(e)}")

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

            if self.tokenizer.pad_token_id is None:
                self.tokenizer.pad_token = "<PAD>"
                self.tokenizer.pad_token_id = self.tokenizer.convert_tokens_to_ids("<PAD>")

            print("Special tokens map:", self.tokenizer.special_tokens_map)
            print("Pad token ID:", self.tokenizer.pad_token_id)
            print("Model max length:", self.tokenizer.model_max_length)

        except Exception as e:
            logging.error(f"Failed to load tokenizer: {str(e)}")
            messagebox.showerror("Error", f"Failed to load tokenizer: {str(e)}")


    def start_training(self):
        # Start the training process in a separate thread
        self.stop_training.clear()
        training_thread = threading.Thread(target=self.training_loop)
        training_thread.start()
        
    def update_progress(self, progress_value):
        self.progress_bar['value'] = progress_value

    def update_status(self, message):
        self.status_label.config(text=f"Status: {message}")

    def training_loop(self):
        if not self.model:
            messagebox.showerror("Error", "Model not initialized.")
            return

        if not self.tokenizer:
            messagebox.showerror("Error", "Tokenizer not loaded.")
            return

        if not self.train_data or len(self.train_data) == 0:
            messagebox.showerror("Error", "Training data is empty or not tokenized.")
            return

        # Prepare model for QAT if not already done
        if not self.model_prepared_for_qat:
            prepare_model_for_qat(self.model, self.device)
            self.model_prepared_for_qat = True

        max_length = 1024  # Adjusted to reduce memory usage
        valid_train_data = [tokens for tokens in self.train_data if len(tokens) <= max_length]

        # Convert lists of token IDs to tensors
        input_ids = [
            torch.tensor(tokens + [self.tokenizer.pad_token_id] * (max_length - len(tokens)), dtype=torch.long)[:max_length]
            for tokens in valid_train_data
        ]
        input_ids = torch.stack(input_ids)
        attention_masks = (input_ids != self.tokenizer.pad_token_id).long()

        dataset = torch.utils.data.TensorDataset(input_ids, attention_masks)
        dataloader = DataLoader(dataset, batch_size=self.batch_size.get(), shuffle=True, num_workers=0, pin_memory=True)

        # Adjust learning rate based on architecture
        # Calculate total parameters
        total_params = self.num_parameters.get()

        # Calculate learning rate dynamically
        lr = self.calculate_learning_rate(total_params)

        logging.info(f"Dynamic Learning Rate calculated: {lr} for total parameters: {total_params}")

        optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)
        logging.info(f"Calculated learning rate: {lr}")

        # Learning rate scheduler
        total_steps = self.epochs.get() * len(dataloader)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=total_steps, eta_min=lr * 0.1)

        self.model.train()
        progress_step = 0

        with torch.autograd.set_detect_anomaly(True):
    # Training code

            for epoch in range(self.epochs.get()):
                if self.stop_training.is_set():
                    logging.info("Training stopped by user.")
                    messagebox.showinfo("Info", "Training stopped by user.")
                    break

                epoch_loss = 0

                for batch_idx, (batch_input_ids, batch_attention_masks) in enumerate(dataloader):
                    if self.stop_training.is_set():
                        logging.info("Training stopped by user.")
                        messagebox.showinfo("Info", "Training stopped by user.")
                        return


                    optimizer.zero_grad()

                    # Move batches and targets to the correct device
                    batch_input_ids = batch_input_ids.to(self.device, dtype=torch.long)
                    batch_attention_masks = batch_attention_masks.to(self.device)

                    # Prepare input and target sequences
                    batch_target_ids = batch_input_ids[:, 1:].contiguous()
                    batch_input_ids = batch_input_ids[:, :-1].contiguous()

                    # Ensure input and target tensors are aligned for batch size
                    if batch_input_ids.size(1) != batch_target_ids.size(1):
                        raise ValueError("Input and target sequence lengths are mismatched.")


                    logits = self.model(batch_input_ids)

                        # Compute loss
                    logits_reshaped = logits[:, :batch_target_ids.size(1), :].reshape(-1, logits.size(-1))
                    targets_reshaped = batch_target_ids.reshape(-1)

                    loss = F.cross_entropy(
                        logits_reshaped,
                        targets_reshaped,
                        ignore_index=self.tokenizer.pad_token_id
                    )
                    logging.info(f"Loss calculated: {loss}")

                    # Backward pass
                    loss.backward()

                    # Gradient clipping
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=0.5)

                    # Optimizer step
                    optimizer.step()

                    # Update scheduler
                    scheduler.step()

                    epoch_loss += loss.item()

                    progress_step += 1
                    progress_value = (progress_step / total_steps) * 100
                    self.root.after(0, self.update_progress, progress_value)


            self.loss_history.append(epoch_loss / len(dataloader))
            logging.info(f"Epoch {epoch + 1}/{self.epochs.get()} completed with average loss: {epoch_loss / len(dataloader)}")
            self.root.after(0, self.update_status, f"Epoch {epoch + 1}/{self.epochs.get()} completed.")

 


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
                "num_layers": self.num_layers.get(),
                "architecture": "MatMul-Free LM"
            }
            config_path = os.path.join(save_directory, 'model_config.json')
            with open(config_path, 'w') as f:
                json.dump(config, f)

            # Save the model state dictionary
            model_path = os.path.join(save_directory, 'matmul_free_lm.pth')
            torch.save(self.model.state_dict(), model_path)

            # Save the tokenizer
            self.tokenizer.save_pretrained(save_directory)

            messagebox.showinfo("Success", "Model, tokenizer, and config saved successfully.")
            logging.info("Model, tokenizer, and config saved successfully.")


    def stop_training_command(self):
        self.stop_training.set()
        messagebox.showinfo("Stop Training", "Training stopped.")
        logging.info("Training stopped by user.")

# Main application entry point
if __name__ == "__main__":
    root = tk.Tk()
    app = UnifiedTransformerGUI(root)
    root.mainloop()
