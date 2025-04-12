import json

# Load files
with open("anomalies.json") as f:
    anomalies = json.load(f)

with open("template_id_to_text.json") as f:
    id_to_template = json.load(f)

# Map anomalies
for anomaly in anomalies:
    expected = str(anomaly["expected"])
    preds = [str(p) for p in anomaly["top_k_predicted"]]

    print(f"\n[!] Anomaly in Sequence {anomaly['sequence_id']} at Position {anomaly['position']}")
    print(f"    → Expected Template ID {expected}:")
    print(f"      {id_to_template.get(expected, '[UNKNOWN]')}")
    print("    → Top-k Predictions:")
    for pid in preds:
        print(f"      [{pid}] {id_to_template.get(pid, '[UNKNOWN]')}")
