# wazuh_indexer_log_fetcher.py

import requests
import json
from datetime import datetime, timedelta

# CONFIGURATION
INDEXER_URL = "https://localhost:9200/wazuh-alerts-*/_search"
USERNAME = "admin"
PASSWORD = "NewPassword123!"  # Replace with your actual Indexer user password
VERIFY_SSL = False  # Set to True if you’re using trusted certs

# Time range - last 24 hours
end_time = datetime.utcnow()
start_time = end_time - timedelta(hours=24)

# Format time for Elasticsearch query
def to_es_time(dt):
    return dt.strftime('%Y-%m-%dT%H:%M:%SZ')

# Build Elasticsearch query
def build_query():
    return {
        "size": 500,
        "_source": [
            "@timestamp",
            "agent.name",
            "rule.id",
            "rule.description",
            "rule.level",
            "rule.groups",
            "rule.mitre.tactic"
        ],
        "query": {
            "range": {
                "@timestamp": {
                    "gte": to_es_time(start_time),
                    "lte": to_es_time(end_time)
                }
            }
        }
    }

def fetch_logs():
    response = requests.get(
        INDEXER_URL,
        auth=(USERNAME, PASSWORD),
        headers={"Content-Type": "application/json"},
        json=build_query(),
        verify=VERIFY_SSL
    )

    if response.status_code != 200:
        print("[!] Failed to retrieve data")
        print(response.status_code, response.text)
        return []

    hits = response.json()["hits"]["hits"]
    return [hit["_source"] for hit in hits]

if __name__ == "__main__":
    logs = fetch_logs()
    print(f"[+] Retrieved {len(logs)} entries")
    for log in logs:  # Show sample
        print(json.dumps(log, indent=2))
