import torch
import torch.nn as nn
import time
import argparse

# Device setup
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# --- Hyperparameters ---
input_size = 1  # Each log key is treated as a single feature
model_path = 'model/adeam1.pt'
test_normal_file = 'data_generator/data/hdfs_test_normal'
test_abnormal_file = 'data_generator/data/hdfs_test_abnormal'

# --- Model Definition ---
class Model(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, num_keys):
        super(Model, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, num_keys)

    def forward(self, x):
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(device)
        out, _ = self.lstm(x, (h0, c0))
        return self.fc(out[:, -1, :])

# --- Load Dataset ---
def generate(file_path, window_size):
    sessions = set()
    with open(file_path, 'r') as f:
        for line in f.readlines():
            log_keys = list(map(lambda n: int(n) - 1, line.strip().split()))
            log_keys += [-1] * (window_size + 1 - len(log_keys))  # Padding
            sessions.add(tuple(log_keys))
    print(f"Loaded {len(sessions)} sessions from {file_path}")
    return sessions

# --- Main Evaluation Logic ---
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-num_layers', default=2, type=int)
    parser.add_argument('-hidden_size', default=64, type=int)
    parser.add_argument('-window_size', default=10, type=int)
    parser.add_argument('-num_candidates', default=9, type=int)
    args = parser.parse_args()

    num_layers = args.num_layers
    hidden_size = args.hidden_size
    window_size = args.window_size
    num_candidates = args.num_candidates

    # Set this to the total number of unique log keys used in training
    num_classes = 100  # Replace with the actual number you used

    model = Model(input_size, hidden_size, num_layers, num_classes).to(device)
    model.load_state_dict(torch.load(model_path))
    model.eval()
    print(f"Model loaded from {model_path}")

    test_normal_loader = generate(test_normal_file, window_size)
    test_abnormal_loader = generate(test_abnormal_file, window_size)

    TP = 0
    FP = 0

    print("[*] Running evaluation...")
    start_time = time.time()

    with torch.no_grad():
        for session in test_normal_loader:
            for i in range(len(session) - window_size):
                seq = session[i:i + window_size]
                label = session[i + window_size]
                seq_tensor = torch.tensor(seq, dtype=torch.float).view(-1, window_size, input_size).to(device)
                label_tensor = torch.tensor(label).view(-1).to(device)
                output = model(seq_tensor)
                predicted = torch.argsort(output, 1)[0][-num_candidates:]
                if label not in predicted:
                    FP += 1
                    break

        for session in test_abnormal_loader:
            for i in range(len(session) - window_size):
                seq = session[i:i + window_size]
                label = session[i + window_size]
                seq_tensor = torch.tensor(seq, dtype=torch.float).view(-1, window_size, input_size).to(device)
                label_tensor = torch.tensor(label).view(-1).to(device)
                output = model(seq_tensor)
                predicted = torch.argsort(output, 1)[0][-num_candidates:]
                if label not in predicted:
                    TP += 1
                    break

    elapsed = time.time() - start_time
    FN = len(test_abnormal_loader) - TP
    P = 100 * TP / (TP + FP + 1e-6)
    R = 100 * TP / (TP + FN + 1e-6)
    F1 = 2 * P * R / (P + R + 1e-6)

    print(f"Elapsed Time: {elapsed:.2f}s")
    print(f"True Positives (TP): {TP}")
    print(f"False Positives (FP): {FP}")
    print(f"False Negatives (FN): {FN}")
    print(f"Precision: {P:.3f}%")
    print(f"Recall: {R:.3f}%")
    print(f"F1-score: {F1:.3f}%")
