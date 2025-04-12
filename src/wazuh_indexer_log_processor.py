# wazuh_indexer_log_processor.py

import requests
import json
from datetime import datetime, timedelta

# CONFIG
INDEXER_URL = "https://localhost:9200"
USERNAME = "admin"
PASSWORD = "NewPassword123!"  # Replace with working password
VERIFY_SSL = False  # Set to True in production

# Time range (last 24 hours)
end_time = datetime.utcnow()
start_time = end_time - timedelta(hours=24)

def to_indexer_time(dt):
    return dt.strftime('%Y-%m-%dT%H:%M:%S.%fZ')

def fetch_logs():
    url = f"{INDEXER_URL}/wazuh-alerts-*/_search"
    headers = {"Content-Type": "application/json"}
    body = {
        "size": 1000,
        "query": {
            "range": {
                "@timestamp": {
                    "gte": to_indexer_time(start_time),
                    "lte": to_indexer_time(end_time)
                }
            }
        }
    }

    res = requests.post(url, auth=(USERNAME, PASSWORD), headers=headers, json=body, verify=VERIFY_SSL)
    hits = res.json().get('hits', {}).get('hits', [])
    return [hit['_source'] for hit in hits]

def extract_important_fields(logs):
    processed = []
    for log in logs:
        entry = {
            "timestamp": log.get("@timestamp"),
            "agent": log.get("agent", {}).get("name"),
            "rule_id": log.get("rule", {}).get("id"),
            "description": log.get("rule", {}).get("description"),
            "level": log.get("rule", {}).get("level"),
            "groups": log.get("rule", {}).get("groups"),
            "tactic": log.get("rule", {}).get("mitre", {}).get("tactic", [])
        }
        processed.append(entry)
    return processed

if __name__ == "__main__":
    logs = fetch_logs()
    clean_logs = extract_important_fields(logs)
    print(json.dumps(clean_logs, indent=2))

    # Optional: Save to file
    with open("clean_logs.json", "w") as f:
        json.dump(clean_logs, f, indent=2)
