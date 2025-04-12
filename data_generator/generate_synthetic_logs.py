import random
import os

# Config
NUM_SEQUENCES = 5000
SEQUENCE_LENGTH = 50
NUM_TEMPLATES = 100  # template IDs from 1 to NUM_TEMPLATES
ANOMALY_RATIO = 0.1
WINDOW_SIZE = 10

# Define normal patterns (e.g., login → sudo → update → logout)
normal_patterns = [
    [10, 20, 30, 40, 50],
    [15, 25, 35, 45, 55],
    [12, 22, 32, 42, 52],
    [11, 21, 31, 41, 51],
]

# Generate a single normal sequence with some randomness
def generate_normal_sequence():
    pattern = random.choice(normal_patterns)
    sequence = []
    while len(sequence) < SEQUENCE_LENGTH:
        noisy_pattern = [x if random.random() > 0.1 else random.randint(1, NUM_TEMPLATES) for x in pattern]
        sequence.extend(noisy_pattern)
    return sequence[:SEQUENCE_LENGTH]

# Inject anomaly at a random position by replacing expected log with unexpected one
def inject_anomaly(sequence):
    pos = random.randint(WINDOW_SIZE, len(sequence) - 1)
    original = sequence[pos]
    while True:
        anomaly = random.randint(1, NUM_TEMPLATES)
        if anomaly != original:
            break
    sequence[pos] = anomaly
    return sequence

# Generate datasets
def generate_dataset(filename, include_anomalies=False):
    with open(filename, 'w') as f:
        for _ in range(NUM_SEQUENCES):
            seq = generate_normal_sequence()
            if include_anomalies and random.random() < ANOMALY_RATIO:
                seq = inject_anomaly(seq)
            f.write(" ".join(map(str, seq)) + "\n")
    print(f"[+] Saved {NUM_SEQUENCES} sequences to {filename}")

if __name__ == "__main__":
    os.makedirs("data", exist_ok=True)
    generate_dataset("data/hdfs_train", include_anomalies=False)
    generate_dataset("data/hdfs_test_normal", include_anomalies=False)
    generate_dataset("data/hdfs_test_abnormal", include_anomalies=True)
