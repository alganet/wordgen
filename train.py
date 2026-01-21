# SPDX-FileCopyrightText: 2026 Alexandre Gomes Gaigalas <alganet@gmail.com>
#
# SPDX-License-Identifier: ISC

import glob
import json
import os
import random
import urllib.request

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm

# Configuration
url = "https://raw.githubusercontent.com/dwyl/english-words/master/words.txt"
data_fname = "train.txt"
data_dir = "data"
out_dir = "out"
max_words = 10000  # 466550
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
generate_len = 16

# Hyperparameters
num_layers = 2
nhead = 4
seq_len = 16
batch_size = 32
d_model = 32
dropout = 0.0
lr = 0.0005
epochs = 2
temperature = 0.7
log_interval = 100

# Data download
os.makedirs(data_dir, exist_ok=True)
os.makedirs(out_dir, exist_ok=True)
file_path = os.path.join(data_dir, data_fname)
if not os.path.exists(file_path):
    print("Downloading data...")
    urllib.request.urlretrieve(url, file_path)

print(f"Device: {device}")

# Load and preprocess text
print("Loading text...")
with open(file_path, "r", encoding="utf-8") as f:
    lines = f.readlines()
words = [line.strip() for line in lines]
words = [w for w in words if w]
random.shuffle(words)  # Shuffle words
words = words[:max_words]

# Build tokens with <sow> as single token
tokens = []
for word in words:
    tokens.append("<sow>")
    tokens.extend(list(word))
    tokens.append(" ")

print("Building/Loading vocab...")
# Load vocab from out/vocab.json if present; otherwise build from data.
vocab_file = os.path.join(out_dir, "vocab.json")
model_checkpoints = glob.glob(f"{out_dir}/checkpoint_epoch_*.pt")
vocab = None

# Try loading separate vocab file
if os.path.exists(vocab_file):
    try:
        with open(vocab_file, "r", encoding="utf-8") as vf:
            vocab = json.load(vf)
            print(f"Loaded vocab from {vocab_file}")
    except Exception as e:
        print(f"Warning: could not load vocab file '{vocab_file}': {e}")

# Build vocab if not loaded
if vocab is None:
    vocab = {"<sow>": 0, "<unk>": 1}
    for c in set(tokens):
        if c not in vocab:
            vocab[c] = len(vocab)

vocab_size = len(vocab)

# Save vocab once if missing
if not os.path.exists(vocab_file):
    try:
        with open(vocab_file, "w", encoding="utf-8") as vf:
            json.dump(vocab, vf, ensure_ascii=False)
            print(f"Saved vocab to {vocab_file}")
    except Exception as e:
        print(f"Warning: failed to write vocab file '{vocab_file}': {e}")

# Map tokens to indices (use <unk>)
data = [vocab.get(c, 1) for c in tokens]

# Create sequences
sequences = []
for i in range(0, len(data) - seq_len):
    sequences.append(data[i : i + seq_len + 1])  # +1 for target

sequences = torch.tensor(sequences, dtype=torch.long)

# Split train/val
print("Splitting data...")
train_size = int(0.9 * len(sequences))
train_data = sequences[:train_size]
val_data = sequences[train_size:]

# DataLoader
print("Creating DataLoaders...")
train_dataset = TensorDataset(train_data)
val_dataset = TensorDataset(val_data)

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)


# Model (depends on vocab_size)
class TransformerLM(nn.Module):
    def __init__(self, vocab_size, d_model, nhead, num_layers, dropout):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.pos_encoder = nn.Embedding(1000, d_model)  # positional embeddings
        encoder_layer = nn.TransformerEncoderLayer(
            d_model, nhead, dropout=dropout, batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer, num_layers
        )
        self.linear = nn.Linear(d_model, vocab_size)

    def forward(self, src):
        seq_len_src = src.size(1)
        pos = torch.arange(0, seq_len_src, device=src.device).unsqueeze(0)
        src = self.embedding(src) + self.pos_encoder(pos)
        mask = (
            torch.triu(torch.ones(seq_len_src, seq_len_src), diagonal=1)
            .bool()
            .to(src.device)
        )
        output = self.transformer_encoder(src, mask=mask)
        output = self.linear(output)
        return output


model = TransformerLM(vocab_size, d_model, nhead, num_layers, dropout).to(
    device
)
print("Initializing model...")
optimizer = optim.Adam(model.parameters(), lr=lr)
criterion = nn.CrossEntropyLoss()

# Check for existing checkpoints and load states if present
if model_checkpoints:
    try:
        # Find the latest checkpoint epoch and path
        epochs_done = [
            int(f.split("_")[-1].split(".")[0]) for f in model_checkpoints
        ]
        ckpt_max_epoch = max(epochs_done)
        ckpt_path = f"{out_dir}/checkpoint_epoch_{ckpt_max_epoch}.pt"

        ckpt = torch.load(ckpt_path, map_location=device)
        model.load_state_dict(ckpt["model_state"])
        if "optimizer_state" in ckpt:
            try:
                optimizer.load_state_dict(ckpt["optimizer_state"])
            except Exception as e:
                print(f"Warning: could not load optimizer state: {e}")
        start_epoch = ckpt_max_epoch + 1
        if start_epoch > epochs:
            print(
                f"Model already trained up to epoch {ckpt_max_epoch}, skipping"
            )
        else:
            print(f"Resuming from epoch {ckpt_max_epoch}")
    except Exception as e:
        cp = ckpt_path if "ckpt_path" in locals() else "unknown"
        print(f"Warning: failed to load checkpoint '{cp}': {e}")
        start_epoch = 1
else:
    start_epoch = 1

# Training loop
for epoch in range(start_epoch, epochs + 1):
    model.train()
    total_loss = 0
    step = 0
    for batch in tqdm(train_loader, desc=f"Epoch {epoch} Train"):
        inputs = batch[0][:, :-1]
        targets = batch[0][:, 1:]
        inputs, targets = inputs.to(device), targets.to(device)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs.reshape(-1, vocab_size), targets.reshape(-1))
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        step += 1
        if step % log_interval == 0:
            avg_loss = total_loss / step
            tqdm.write(f"Epoch {epoch}, Step {step}, Avg Loss: {avg_loss:.4f}")
    print(f"Epoch {epoch}, Train Loss: {total_loss / len(train_loader):.4f}")

    model.eval()
    val_loss = 0
    with torch.no_grad():
        for batch in tqdm(val_loader, desc=f"Epoch {epoch} Val"):
            inputs = batch[0][:, :-1]
            targets = batch[0][:, 1:]
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            loss = criterion(
                outputs.reshape(-1, vocab_size), targets.reshape(-1)
            )
            val_loss += loss.item()
    print(f"Val Loss: {val_loss / len(val_loader):.4f}")

    # Save checkpoint (model + optimizer)
    torch.save(
        {
            "epoch": epoch,
            "model_state": model.state_dict(),
            "optimizer_state": optimizer.state_dict(),
        },
        f"{out_dir}/checkpoint_epoch_{epoch}.pt",
    )
    print(f"Saved checkpoint for epoch {epoch}")


# Generation function
def generate_word(model, vocab, max_len, seq_len, device, temperature):
    model.eval()
    idx_to_char = {idx: char for char, idx in vocab.items()}
    generated = [vocab["<sow>"]]
    for _ in range(max_len):
        input_seq = torch.tensor([generated[-seq_len:]], dtype=torch.long).to(
            device
        )
        with torch.no_grad():
            output = model(input_seq)
        next_token_logits = output[0, -1, :]
        # Apply temperature scaling
        next_token_logits = next_token_logits / temperature
        probs = torch.softmax(next_token_logits, dim=-1)
        next_token = torch.multinomial(probs, 1).item()
        generated.append(next_token)
        if idx_to_char[next_token] == " ":
            break
    # Extract generated word (between <sow> and space)
    start_idx = 1
    end_idx = len(generated) - 1
    generated_word = "".join(
        [idx_to_char[idx] for idx in generated[start_idx:end_idx]]
    )
    return generated_word


# Generate samples
print("Generating samples...")
for i in range(10):
    gen = generate_word(
        model, vocab, generate_len, seq_len, device, temperature
    )
    print(f"Generated word {i+1}: {gen}")
