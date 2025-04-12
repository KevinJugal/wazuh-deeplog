import torch
import torch.nn as nn
import json
import time
from pathlib import Path

# --- Config ---
model_path = '/home/honour/Documents/DeepAI/model/adeam1.pt'
test_file = '/home/honour/Documents/DeepAI/data_generator/data/hdfs_test_abnormal'
template_dict_path = '/home/honour/Documents/DeepAI/data_generator/data/template_id_to_text.json'
output_path = 'anomalies.json'

input_size = 1
hidden_size = 64
num_layers = 2
window_size = 10
num_candidates = 9
num_classes = 100  # Update if different

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# --- Load Template Descriptions ---
with open(template_dict_path, 'r') as f:
    id_to_text = json.load(f)

def get_description(tid):
    return id_to_text.get(str(tid), "[UNKNOWN]")

# --- Model Definition ---
class Model(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, num_keys):
        super(Model, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, num_keys)

    def forward(self, x):
        h0 = torch.zeros(num_layers, x.size(0), hidden_size).to(device)
        c0 = torch.zeros(num_layers, x.size(0), hidden_size).to(device)
        out, _ = self.lstm(x, (h0, c0))
        return self.fc(out[:, -1, :])

# --- Load Test Data ---
def generate(file_path, window_size):
    sessions = set()
    with open(file_path, 'r') as f:
        for line in f:
            events = list(map(lambda n: int(n) - 1, line.strip().split()))
            events += [-1] * (window_size + 1 - len(events))  # pad
            sessions.add(tuple(events))
    return sessions

# --- Main ---
if __name__ == "__main__":
    print("[*] Loading model...")
    model = Model(input_size, hidden_size, num_layers, num_classes).to(device)
    model.load_state_dict(torch.load(model_path))
    model.eval()

    sessions = generate(test_file, window_size)

    anomalies = []
    print("[*] Detecting anomalies...")
    start_time = time.time()

    with torch.no_grad():
        for seq_id, session in enumerate(sessions):
            for i in range(len(session) - window_size):
                seq = session[i:i + window_size]
                label = session[i + window_size]
                seq_tensor = torch.tensor(seq, dtype=torch.float).view(-1, window_size, input_size).to(device)
                output = model(seq_tensor)
                predicted = torch.argsort(output, 1)[0][-num_candidates:]
                if label not in predicted:
                    anomalies.append({
                        "sequence_id": seq_id,
                        "position": i + window_size,
                        "expected": label,
                        "expected_description": get_description(label),
                        "top_k_predicted": predicted.tolist(),
                        "predicted_descriptions": [get_description(tid) for tid in predicted.tolist()]
                    })
                    break  # one anomaly per session is enough

    elapsed = time.time() - start_time
    print(f"[+] Detected {len(anomalies)} anomalies in {elapsed:.2f} seconds.")

    with open(output_path, 'w') as f:
        json.dump(anomalies, f, indent=2)
    print(f"[+] Anomalies saved to {output_path}")
