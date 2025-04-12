# log_sequence_builder.py

import json
from datetime import datetime
from collections import defaultdict

# Load processed logs from wazuh_log_processor
def load_logs(file_path="clean_logs.json"):
    with open(file_path, 'r') as f:
        return json.load(f)

# Sort logs by timestamp per agent
def build_sequences(logs):
    sequences = defaultdict(list)

    for log in logs:
        agent = log.get("agent", "unknown")
        ts = log.get("timestamp")
        msg = f"{log.get('description', '')} - {log.get('groups', [])}"
        
        if ts:
            try:
                dt = datetime.strptime(ts, "%Y-%m-%dT%H:%M:%S.%fZ")
            except ValueError:
                dt = datetime.strptime(ts, "%Y-%m-%dT%H:%M:%SZ")
            sequences[agent].append((dt, msg))

    # Sort and return only the messages
    sorted_sequences = {
        agent: [msg for dt, msg in sorted(entries)]
        for agent, entries in sequences.items()
    }

    return sorted_sequences

if __name__ == "__main__":
    logs = load_logs()
    sequences = build_sequences(logs)

    # Save for training input
    with open("log_sequences.json", "w") as f:
        json.dump(sequences, f, indent=2)

    print("[✓] Log sequences saved to log_sequences.json")
