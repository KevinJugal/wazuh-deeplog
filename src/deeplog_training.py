# train_deeplog.py

import time
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import TensorDataset, DataLoader
import argparse
import os

# Device configuration
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Model definition
class Model(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, num_keys):
        super(Model, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, num_keys)

    def forward(self, x):
        h0 = torch.zeros(self.lstm.num_layers, x.size(0), self.lstm.hidden_size).to(x.device)
        c0 = torch.zeros(self.lstm.num_layers, x.size(0), self.lstm.hidden_size).to(x.device)
        out, _ = self.lstm(x, (h0, c0))
        return self.fc(out[:, -1, :])

# Data loader
def generate(filepath, window_size):
    inputs, outputs = [], []
    with open(filepath, 'r') as f:
        for line in f:
            events = list(map(int, line.strip().split()))
            events = [e - 1 for e in events]  # template ID correction
            if len(events) <= window_size:
                continue
            for i in range(len(events) - window_size):
                inputs.append(events[i:i + window_size])
                outputs.append(events[i + window_size])
    print(f"[+] Loaded {len(inputs)} samples from {filepath}")
    return TensorDataset(
        torch.tensor(inputs, dtype=torch.float32),
        torch.tensor(outputs, dtype=torch.long)
    )

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path', default='data_generator/data/hdfs_train', help='Path to training data')
    parser.add_argument('--num_layers', default=2, type=int)
    parser.add_argument('--hidden_size', default=64, type=int)
    parser.add_argument('--window_size', default=10, type=int)
    parser.add_argument('--batch_size', default=2048, type=int)
    parser.add_argument('--epochs', default=300, type=int)
    parser.add_argument('--model_dir', default='model', help='Directory to save model')
    args = parser.parse_args()

    # Generate data
    dataset = generate(args.data_path, args.window_size)
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, pin_memory=True)

    # Determine number of log keys
    all_labels = [label.item() for _, label in dataset]
    num_classes = max(all_labels) + 1

    # Init model
    model = Model(1, args.hidden_size, args.num_layers, num_classes).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters())
    writer = SummaryWriter(log_dir=f'log/Adam_bs={args.batch_size}_ep={args.epochs}')

    # Training
    print("[*] Starting training...")
    start_time = time.time()
    for epoch in range(args.epochs):
        model.train()
        total_loss = 0
        for seq, label in dataloader:
            seq = seq.view(-1, args.window_size, 1).to(device)
            label = label.to(device)

            optimizer.zero_grad()
            output = model(seq)
            loss = criterion(output, label)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        avg_loss = total_loss / len(dataloader)
        print(f"Epoch [{epoch+1}/{args.epochs}], Loss: {avg_loss:.4f}")
        writer.add_scalar("Loss/train", avg_loss, epoch + 1)

    elapsed = time.time() - start_time
    print(f"[*] Training finished in {elapsed:.2f}s")

    # Save model
    os.makedirs(args.model_dir, exist_ok=True)
    model_path = os.path.join(args.model_dir, f"Adam_bs={args.batch_size}_ep={args.epochs}.pt")
    torch.save(model.state_dict(), model_path)
    print(f"[+] Model saved to {model_path}")
    writer.close()
