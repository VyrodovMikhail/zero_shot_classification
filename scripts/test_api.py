"""Test the classification API using samples from the `ai-forever/headline-classification` dataset."""

import argparse

import requests
from datasets import load_dataset


def main() -> None:
    parser = argparse.ArgumentParser(description="Send dataset samples to the classification service")
    parser.add_argument("url", help="Base URL of the service, e.g. http://localhost:8000")
    parser.add_argument("--split", default="test", help="Dataset split to use")
    parser.add_argument("--limit", type=int, default=10, help="Number of samples to send")
    args = parser.parse_args()

    dataset = load_dataset("ai-forever/headline-classification", split=args.split)

    label_feature = dataset.features.get("label")
    candidate_labels = ["экономика", "культура", "спорт", "политика", "наука", "происшествия"]

    texts = dataset["text"][: args.limit]
    payload = {"texts": texts, "labels": candidate_labels, "batch_size": 5}
    resp = requests.post(f"{args.url}/classify", json=payload, timeout=50000)
    for i in range(len(texts)):
        print(texts[i], "->", resp.json()["result"][i])


if __name__ == "__main__":
    main()
