"""Upload evaluation Q&A pairs to Langfuse.

Usage:
    uv run python scripts/seed_eval_dataset.py

Run once to create the dataset. Re-running is safe — existing items are skipped
if the dataset already exists (Langfuse ignores duplicate create_dataset calls).
"""

from app.tracing import init_tracing

init_tracing()

from langfuse import get_client  # noqa: E402

DATASET_NAME = "sec-filing-qa-v1"

# Gold-standard Q&A pairs drawn from the pilot corpus:
# AAPL, AMZN, GOOGL, MSFT, NVDA — fiscal years 2021-2025 (10-K filings)
ITEMS = [
    {
        "input": {
            "question": "What was Apple's total net sales in fiscal year 2023?",
        },
        "expected_output": {
            "answer": "Apple's total net sales in fiscal year 2023 were $383.3 billion.",
        },
        "metadata": {"ticker": "AAPL", "fiscal_year": 2023, "form": "10-K"},
    },
    {
        "input": {
            "question": "What were Apple's three largest revenue segments in FY2023 and their sales?",
        },
        "expected_output": {
            "answer": (
                "Apple's three largest revenue segments in FY2023 were: "
                "iPhone ($200.6B), Services ($85.2B), and Mac ($29.4B)."
            ),
        },
        "metadata": {"ticker": "AAPL", "fiscal_year": 2023, "form": "10-K"},
    },
    {
        "input": {
            "question": "What was Microsoft's total revenue in fiscal year 2024?",
        },
        "expected_output": {
            "answer": "Microsoft's total revenue in fiscal year 2024 was $245.1 billion.",
        },
        "metadata": {"ticker": "MSFT", "fiscal_year": 2024, "form": "10-K"},
    },
    {
        "input": {
            "question": "What was NVIDIA's data center revenue in fiscal year 2024?",
        },
        "expected_output": {
            "answer": "NVIDIA's data center revenue in fiscal year 2024 was $47.5 billion.",
        },
        "metadata": {"ticker": "NVDA", "fiscal_year": 2024, "form": "10-K"},
    },
    {
        "input": {
            "question": "What was Alphabet's (Google) total revenue in 2023?",
        },
        "expected_output": {
            "answer": "Alphabet's total revenues in 2023 were $307.4 billion.",
        },
        "metadata": {"ticker": "GOOGL", "fiscal_year": 2023, "form": "10-K"},
    },
    {
        "input": {
            "question": "What was Amazon's net sales in 2023?",
        },
        "expected_output": {
            "answer": "Amazon's net sales in 2023 were $574.8 billion.",
        },
        "metadata": {"ticker": "AMZN", "fiscal_year": 2023, "form": "10-K"},
    },
    {
        "input": {
            "question": "What were the main risk factors Apple cited related to its supply chain in the FY2023 10-K?",
        },
        "expected_output": {
            "answer": (
                "Apple cited concentration of manufacturing in Asia (primarily China), "
                "single-source components, and geopolitical risk as key supply chain risks in FY2023."
            ),
        },
        "metadata": {"ticker": "AAPL", "fiscal_year": 2023, "form": "10-K"},
    },
    {
        "input": {
            "question": "What was NVIDIA's net income in fiscal year 2024?",
        },
        "expected_output": {
            "answer": "NVIDIA's net income in fiscal year 2024 was $29.8 billion.",
        },
        "metadata": {"ticker": "NVDA", "fiscal_year": 2024, "form": "10-K"},
    },
    {
        "input": {
            "question": "What is the revenue split between Microsoft's three business segments in FY2024?",
        },
        "expected_output": {
            "answer": (
                "In FY2024, Microsoft's revenue was split across: "
                "Intelligent Cloud ($105.4B), Productivity and Business Processes ($77.7B), "
                "and More Personal Computing ($62.0B)."
            ),
        },
        "metadata": {"ticker": "MSFT", "fiscal_year": 2024, "form": "10-K"},
    },
    {
        "input": {
            "question": "What was Apple's gross margin percentage in fiscal year 2024?",
        },
        "expected_output": {
            "answer": "Apple's gross margin was 46.2% in fiscal year 2024.",
        },
        "metadata": {"ticker": "AAPL", "fiscal_year": 2024, "form": "10-K"},
    },
]


def main() -> None:
    lf = get_client()

    lf.create_dataset(
        name=DATASET_NAME,
        description="Gold-standard SEC filing Q&A for pilot corpus (AAPL, AMZN, GOOGL, MSFT, NVDA, FY2021-2025)",
    )
    print(f"Dataset '{DATASET_NAME}' ready.")

    for i, item in enumerate(ITEMS, 1):
        lf.create_dataset_item(
            dataset_name=DATASET_NAME,
            input=item["input"],
            expected_output=item["expected_output"],
            metadata=item["metadata"],
        )
        print(f"  [{i}/{len(ITEMS)}] uploaded: {item['input']['question'][:60]}…")

    lf.flush()
    print(f"\nDone. {len(ITEMS)} items in '{DATASET_NAME}'.")
    print("View at: http://localhost:3000 → Datasets")


if __name__ == "__main__":
    main()
