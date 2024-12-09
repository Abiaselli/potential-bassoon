import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import PreTrainedTokenizerFast
from tkinter import Tk, filedialog, Label, Entry, Button, Text, END, messagebox
import os
import threading
import json
import math

# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

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
    def __init__(self, vocab_size, embed_size, num_heads, num_layers, hidden_size, max_seq_length):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_size)
        self.pos_encoder = PositionalEncoding(embed_size, max_seq_length)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_size,
            nhead=num_heads,
            dim_feedforward=hidden_size,
            batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.fc_out = nn.Linear(embed_size, vocab_size)
        self.embed_size = embed_size

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

# Custom matmul-free linear function
def matmul_free_linear(input, weight):
    # input: (batch_size, seq_len, input_dim)
    # weight: (input_dim, output_dim)

    pos_mask = (weight == 1).float()
    neg_mask = (weight == -1).float()

    # Remove the .t() to correctly align dimensions
    pos_contrib = torch.matmul(input, pos_mask)
    neg_contrib = torch.matmul(input, neg_mask)

    output = pos_contrib - neg_contrib
    return output


# Activation quantization function
def activation_quant(x):
    s = 127 / x.abs().max(dim=-1, keepdim=True)[0]
    x = torch.clamp((s * x).round(), -128, 127) / s
    return x

# RMS normalization function
def rms_norm(x, eps=1e-8):
    mean = x.mean(dim=-1, keepdim=True)
    variance = x.var(dim=-1, unbiased=False, keepdim=True)
    r = 1 / torch.sqrt(variance + eps)
    y = r * (x - mean)
    return y

# MLGRUCell
class MLGRUCell(nn.Module):
    def __init__(self, input_size, hidden_size, eps=1e-8):
        super(MLGRUCell, self).__init__()
        self.hidden_size = hidden_size
        self.eps = eps

        # Initialize weights as full-precision parameters
        self.W_f = nn.Parameter(torch.randn(hidden_size, input_size) * 0.01)
        self.W_c = nn.Parameter(torch.randn(hidden_size, input_size) * 0.01)
        self.W_g = nn.Parameter(torch.randn(hidden_size, input_size) * 0.01)
        self.b_f = nn.Parameter(torch.zeros(hidden_size))
        self.b_c = nn.Parameter(torch.zeros(hidden_size))
        self.b_g = nn.Parameter(torch.zeros(hidden_size))

    def forward(self, x_t, h_t_minus_1):
        # Apply RMS normalization and activation quantization
        x_t = rms_norm(x_t, self.eps)
        x_t = activation_quant(x_t)

        # Ternarize and transpose weights
        W_f_ternary = ternarize_weight(self.W_f).t()
        W_c_ternary = ternarize_weight(self.W_c).t()
        W_g_ternary = ternarize_weight(self.W_g).t()

        # Adjusted matmul_free_linear calls
        f_t_linear = matmul_free_linear(x_t, W_f_ternary) + self.b_f
        c_t_linear = matmul_free_linear(x_t, W_c_ternary) + self.b_c
        g_t_linear = matmul_free_linear(x_t, W_g_ternary) + self.b_g

        # Activation functions
        f_t = torch.sigmoid(f_t_linear)
        c_t = F.silu(c_t_linear)
        g_t = torch.sigmoid(g_t_linear)

        h_t = f_t * h_t_minus_1 + (1 - f_t) * c_t
        o_t = g_t * h_t

        return o_t, h_t


# MLGRULayer
class MLGRULayer(nn.Module):
    def __init__(self, input_size, hidden_size, eps=1e-8):
        super(MLGRULayer, self).__init__()
        self.cell = MLGRUCell(input_size, hidden_size, eps)

    def forward(self, x):
        batch_size, seq_len, _ = x.size()
        h_t = torch.zeros(batch_size, self.cell.hidden_size, device=x.device)
        outputs = []
        for t in range(seq_len):
            x_t = x[:, t, :]
            o_t, h_t = self.cell(x_t, h_t)
            outputs.append(o_t.unsqueeze(1))
        return torch.cat(outputs, dim=1)

# MatMul-free GLU
class MatMulFreeGLU(nn.Module):
    def __init__(self, input_size, hidden_size, eps=1e-8):
        super(MatMulFreeGLU, self).__init__()
        self.eps = eps

        # Initialize weights as full-precision parameters
        self.W_g = nn.Parameter(torch.randn(hidden_size, input_size) * 0.01)
        self.W_u = nn.Parameter(torch.randn(hidden_size, input_size) * 0.01)
        self.W_d = nn.Parameter(torch.randn(input_size, hidden_size) * 0.01)

    def forward(self, x):
        # Apply RMS normalization and activation quantization
        x = rms_norm(x, self.eps)
        x = activation_quant(x)

        # Ternarize and transpose weights
        W_g_ternary = ternarize_weight(self.W_g).t()
        W_u_ternary = ternarize_weight(self.W_u).t()
        W_d_ternary = ternarize_weight(self.W_d).t()

        # Adjusted matmul_free_linear calls
        g_t_linear = matmul_free_linear(x, W_g_ternary)
        u_t_linear = matmul_free_linear(x, W_u_ternary)

        # Activation functions
        g_t = F.silu(g_t_linear)
        u_t = u_t_linear

        p_t = g_t * u_t

        # Output layer
        d_t = matmul_free_linear(p_t, W_d_ternary)

        return d_t


# MatMul-Free Language Model
class MatMulFreeLanguageModel(nn.Module):
    def __init__(self, vocab_size, embed_size, hidden_size, eps=1e-8):
        super(MatMulFreeLanguageModel, self).__init__()
        self.eps = eps
        self.embedding = nn.Embedding(vocab_size, embed_size)
        self.mlgru_layer = MLGRULayer(embed_size, hidden_size, eps)
        self.glu = MatMulFreeGLU(hidden_size, hidden_size, eps)
        self.output_layer = nn.Linear(hidden_size, vocab_size)

    def forward(self, input_ids):
        x = self.embedding(input_ids)
        x = self.mlgru_layer(x)
        x = self.glu(x)

        # Apply RMS normalization and activation quantization before output layer
        x = rms_norm(x, self.eps)
        x = activation_quant(x)

        # Ternarize and transpose output layer weights
        output_weight_ternary = ternarize_weight(self.output_layer.weight).t()

        # Adjusted matmul_free_linear call
        logits = matmul_free_linear(x, output_weight_ternary) + self.output_layer.bias

        return logits

# Top-K and Top-P Filtering
def top_k_top_p_filtering(logits, top_k=0, top_p=0.0, filter_value=-float('Inf')):
    batch_size, vocab_size = logits.size()
    # Apply top-k filtering
    if top_k > 0:
        top_k = min(max(top_k, 1), vocab_size)
        values, _ = torch.topk(logits, top_k, dim=-1)
        min_values = values[:, -1].unsqueeze(-1)
        logits = torch.where(logits < min_values, filter_value, logits)

    # Apply top-p (nucleus) filtering
    if top_p > 0.0:
        sorted_logits, sorted_indices = torch.sort(logits, descending=True, dim=-1)
        cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
        sorted_indices_to_remove = cumulative_probs > top_p

        sorted_indices_to_remove[:, 1:] = sorted_indices_to_remove[:, :-1].clone()
        sorted_indices_to_remove[:, 0] = False

        indices_to_remove = sorted_indices_to_remove.scatter(1, sorted_indices, sorted_indices_to_remove)
        logits = logits.masked_fill(indices_to_remove, filter_value)

    return logits

# Tokenizer Validation and Loading
def validate_tokenizer_folder(tokenizer_path):
    required_files = ["tokenizer.json", "special_tokens_map.json", "tokenizer_config.json"]
    missing_files = [f for f in required_files if not os.path.exists(os.path.join(tokenizer_path, f))]
    if missing_files:
        raise FileNotFoundError(f"Missing files in tokenizer folder: {missing_files}")

def load_tokenizer(tokenizer_path):
    validate_tokenizer_folder(tokenizer_path)
    tokenizer = PreTrainedTokenizerFast.from_pretrained(tokenizer_path)
    print(f"Special tokens loaded: {tokenizer.special_tokens_map}")
    return tokenizer

def ensure_special_tokens(tokenizer):
    special_tokens = {}
    if tokenizer.eos_token is None:
        special_tokens['eos_token'] = '<eos>'
    if tokenizer.pad_token is None:
        special_tokens['pad_token'] = '<pad>'

    if special_tokens:
        tokenizer.add_special_tokens(special_tokens)
        print(f"Added special tokens: {special_tokens}")
    else:
        print("All special tokens are already present.")

    print(f"EOS Token: {tokenizer.eos_token}, ID: {tokenizer.eos_token_id}")
    print(f"PAD Token: {tokenizer.pad_token}, ID: {tokenizer.pad_token_id}")
    return tokenizer

# Model Loading Function
def load_model_parameters(config_path):
    with open(config_path, 'r') as f:
        config = json.load(f)
    return config

def load_model(model_path, device):
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found at {model_path}")

    state_dict = torch.load(model_path, map_location=device)
    return state_dict

# Text Generation Function
def generate_text_gui(model, tokenizer, input_text, max_length=50, temperature=1.0, top_k=0, top_p=0.0, repetition_penalty=1.0):
    model.to(device)
    model.eval()
    input_ids = tokenizer.encode(input_text, return_tensors='pt').to(device)
    generated = input_ids.clone()

    with torch.no_grad():
        for _ in range(max_length):
            if isinstance(model, CustomTransformerModel):
                outputs = model(generated)
            else:
                outputs = model(generated)
            next_token_logits = outputs[:, -1, :]  # Get logits for the last token

            # Apply temperature
            next_token_logits = next_token_logits / temperature

            # Repetition Penalty
            if repetition_penalty != 1.0:
                for token_id in set(generated.view(-1).tolist()):
                    next_token_logits[:, token_id] /= repetition_penalty

            # Filter logits using top-k and/or top-p sampling
            filtered_logits = top_k_top_p_filtering(next_token_logits, top_k=top_k, top_p=top_p)

            # Sample from the filtered distribution
            probabilities = F.softmax(filtered_logits, dim=-1)
            next_token_id = torch.multinomial(probabilities, num_samples=1)

            # Append to generated tokens
            generated = torch.cat((generated, next_token_id), dim=1)

            # Stop if the EOS token is generated
            if next_token_id.item() == tokenizer.eos_token_id:
                break

    # Move generated tokens back to CPU before decoding
    generated = generated.cpu()
    output_text = tokenizer.decode(generated[0], skip_special_tokens=True)
    return output_text

# GUI Implementation
class LanguageModelGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("Language Model Inference")

        # Initialize model and tokenizer as None
        self.model = None
        self.tokenizer = None

        # Define Entry widgets for model and tokenizer paths
        Label(root, text="Model Path:").pack(pady=(10, 0))
        self.model_path_entry = Entry(root, width=60)
        self.model_path_entry.pack(pady=(0, 10))

        Label(root, text="Tokenizer Path:").pack(pady=(0, 0))
        self.tokenizer_path_entry = Entry(root, width=60)
        self.tokenizer_path_entry.pack(pady=(0, 10))

        # Select Folder Button
        self.select_button = Button(root, text="Select Model Folder", command=self.select_folder)
        self.select_button.pack(pady=(0, 10))

        # Input Text
        Label(root, text="Input Text:").pack(pady=(10, 0))
        self.input_box = Text(root, height=5, width=60)
        self.input_box.pack(pady=(0, 10))

        # Generation Parameters
        Label(root, text="Max Length:").pack(pady=(10, 0))
        self.max_length_entry = Entry(root, width=60)
        self.max_length_entry.pack(pady=(0, 10))
        self.max_length_entry.insert(0, "50")

        Label(root, text="Temperature:").pack(pady=(0, 0))
        self.temperature_entry = Entry(root, width=60)
        self.temperature_entry.pack(pady=(0, 10))
        self.temperature_entry.insert(0, "1.0")

        Label(root, text="Top-K:").pack(pady=(0, 0))
        self.top_k_entry = Entry(root, width=60)
        self.top_k_entry.pack(pady=(0, 10))
        self.top_k_entry.insert(0, "0")

        Label(root, text="Top-P:").pack(pady=(0, 0))
        self.top_p_entry = Entry(root, width=60)
        self.top_p_entry.pack(pady=(0, 10))
        self.top_p_entry.insert(0, "0.0")

        Label(root, text="Repetition Penalty:").pack(pady=(0, 0))
        self.repetition_penalty_entry = Entry(root, width=60)
        self.repetition_penalty_entry.pack(pady=(0, 10))
        self.repetition_penalty_entry.insert(0, "1.0")

        # Generate Button
        self.generate_button = Button(root, text="Generate Text", command=self.generate_text_callback)
        self.generate_button.pack(pady=(0, 10))

        # Output Box
        Label(root, text="Generated Output:").pack(pady=(10, 0))
        self.output_box = Text(root, height=10, width=60)
        self.output_box.pack(pady=(0, 10))

    def select_folder(self):
        folder_path = filedialog.askdirectory()
        if folder_path:
            # Set model and tokenizer paths
            model_path = os.path.join(folder_path, "custom_transformer_model.pth")
            tokenizer_path = folder_path  # Assuming tokenizer files are in the same folder

            # Update Entry widgets
            self.model_path_entry.delete(0, END)
            self.model_path_entry.insert(0, model_path)

            self.tokenizer_path_entry.delete(0, END)
            self.tokenizer_path_entry.insert(0, tokenizer_path)

            # Load model and tokenizer
            try:
                self.load_model_and_tokenizer(model_path, tokenizer_path)
                messagebox.showinfo("Success", "Model and Tokenizer loaded successfully.")
            except Exception as e:
                messagebox.showerror("Error", f"Failed to load model/tokenizer:\n{e}")

    def load_model_and_tokenizer(self, model_path, tokenizer_path):
        # Load tokenizer
        tokenizer = load_tokenizer(tokenizer_path)
        tokenizer = ensure_special_tokens(tokenizer)

        # Load model parameters from model_config.json
        config_path = os.path.join(os.path.dirname(model_path), 'model_config.json')
        if not os.path.exists(config_path):
            messagebox.showerror("Error", "model_config.json not found.")
            return

        model_parameters = load_model_parameters(config_path)

        # Create the appropriate model based on the architecture
        architecture = model_parameters.get('architecture', 'CustomTransformerModel')

        if architecture == 'Transformer':
            model = CustomTransformerModel(
                vocab_size=model_parameters['vocab_size'],
                embed_size=model_parameters['embed_size'],
                num_heads=model_parameters['num_heads'],
                num_layers=model_parameters['num_layers'],
                hidden_size=model_parameters['hidden_size'],
                max_seq_length=1024
            )
            # Adjust model path if needed
            model_path = os.path.join(os.path.dirname(model_path), 'custom_transformer_model.pth')
        elif architecture in ['MatMulFreeLanguageModel', 'MatMul-Free LM']:
            model = MatMulFreeLanguageModel(
                vocab_size=model_parameters['vocab_size'],
                embed_size=model_parameters['embed_size'],
                hidden_size=model_parameters['hidden_size']
            )
            # Adjust model path if needed
            model_path = os.path.join(os.path.dirname(model_path), 'matmul_free_lm.pth')
        else:
            messagebox.showerror("Error", f"Unsupported architecture: {architecture}")
            return

        print(f"Model Parameters:")
        print(f"  Vocab Size: {model_parameters['vocab_size']}")
        print(f"  Embed Size: {model_parameters['embed_size']}")
        print(f"  Hidden Size: {model_parameters['hidden_size']}")
        print(f"  Num Heads: {model_parameters['num_heads']}")
        print(f"  Num Layers: {model_parameters['num_layers']}")

        # Load state_dict
        state_dict = load_model(model_path, device)
        model.load_state_dict(state_dict)
        model.to(device)
        model.eval()

        # Update class attributes
        self.tokenizer = tokenizer
        self.model = model

    def generate_text_callback(self):
        if self.model is None or self.tokenizer is None:
            messagebox.showwarning("Warning", "Please load a model and tokenizer first.")
            return

        input_text = self.input_box.get("1.0", END).strip()
        if not input_text:
            messagebox.showwarning("Warning", "Please enter some input text.")
            return

        # Retrieve generation parameters
        try:
            max_length = int(self.max_length_entry.get())
            temperature = float(self.temperature_entry.get())
            top_k = int(self.top_k_entry.get())
            top_p = float(self.top_p_entry.get())
            repetition_penalty = float(self.repetition_penalty_entry.get())
        except ValueError:
            messagebox.showerror("Error", "Invalid generation parameters.")
            return

        # Start generation in a separate thread to keep GUI responsive
        threading.Thread(
            target=self.generate_and_display,
            args=(input_text, max_length, temperature, top_k, top_p, repetition_penalty)
        ).start()

    def generate_and_display(self, input_text, max_length, temperature, top_k, top_p, repetition_penalty):
        try:
            output = generate_text_gui(
                model=self.model,
                tokenizer=self.tokenizer,
                input_text=input_text,
                max_length=max_length,
                temperature=temperature,
                top_k=top_k,
                top_p=top_p,
                repetition_penalty=repetition_penalty
            )
            self.output_box.delete("1.0", END)
            self.output_box.insert(END, output)
        except Exception as e:
            messagebox.showerror("Error", f"Failed to generate text:\n{e}")

def main():
    root = Tk()
    gui = LanguageModelGUI(root)
    root.mainloop()

if __name__ == "__main__":
    main()
