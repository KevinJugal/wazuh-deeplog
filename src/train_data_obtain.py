import json
from pathlib import Path

# Load the uploaded log_sequences.json
json_path = Path("log_sequences.json")
with json_path.open() as f:
    log_sequences = json.load(f)

# Prepare log template dictionary and sequence mapping
template_to_id = {}
id_to_template = {}
sequences_as_ids = []

template_id = 1
for host, logs in log_sequences.items():
    sequence = []
    for log in logs:
        if log not in template_to_id:
            template_to_id[log] = template_id
            id_to_template[str(template_id)] = log
            template_id += 1
        sequence.append(str(template_to_id[log]))
    sequences_as_ids.append(" ".join(sequence))

# Save the sequences to a .txt file
txt_output_path = "log_sequences.txt"
with open(txt_output_path, "w") as f:
    for seq in sequences_as_ids:
        f.write(seq + "\n")

# Save the template dictionary to a .json file
template_dict_path = "template_dict.json"
with open(template_dict_path, "w") as f:
    json.dump(id_to_template, f, indent=2)

txt_output_path, template_dict_path
