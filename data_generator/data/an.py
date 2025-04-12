import json

# Paths to files
hdfs_train_path = "hdfs_train"
template_dict_path = "/home/honour/Documents/DeepAI/json/template_dict.json"  # Your full template ID to message mapping
output_json_path = "template_id_to_text.json"

# Step 1: Extract unique template IDs used in training
used_ids = set()
with open(hdfs_train_path, 'r') as f:
    for line in f:
        tokens = line.strip().split()
        used_ids.update(tokens)

# Step 2: Load full mapping
with open(template_dict_path, 'r') as f:
    full_mapping = json.load(f)

# Step 3: Filter for used IDs
filtered_mapping = {k: full_mapping[k] for k in used_ids if k in full_mapping}

# Step 4: Save result
with open(output_json_path, 'w') as f:
    json.dump(filtered_mapping, f, indent=2)

print(f"[+] Saved template_id_to_text.json with {len(filtered_mapping)} entries.")
