"""Build the stage-six local BM25 product index."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pyarrow.parquet as pq

from cartwise.retrieval.bm25 import BM25Index
from scripts.paths import PRODUCT_BM25_ARTIFACT_ROOT, PROCESSED_ROOTS


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--scope", choices=PROCESSED_ROOTS, default="dev")
    parser.add_argument("--processed-root", type=Path)
    parser.add_argument("--index-output", type=Path)
    parser.add_argument("--report-output", type=Path)
    return parser.parse_args()


def write_report(path: Path, report: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    partial = path.with_suffix(f"{path.suffix}.part")
    partial.write_text(
        json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    partial.replace(path)


def run_build(*, scope: str, processed_root: Path, index_output: Path) -> dict[str, object]:
    items = pq.read_table(processed_root / "items.parquet").to_pylist()
    index = BM25Index.from_items(items)
    index.save(index_output)
    return {
        "scope": scope,
        "documents": len(index.documents),
        "index": str(index_output),
    }


def main() -> None:
    args = parse_args()
    output_root = PRODUCT_BM25_ARTIFACT_ROOT / args.scope
    index_output = args.index_output or output_root / "bm25.json.gz"
    report_output = args.report_output or output_root / "build_report.json"
    report = run_build(
        scope=args.scope,
        processed_root=args.processed_root or PROCESSED_ROOTS[args.scope],
        index_output=index_output,
    )
    write_report(report_output, report)
    print(f"Indexed {report['documents']:,} products")
    print(f"Wrote index: {index_output}")
    print(f"Wrote report: {report_output}")


if __name__ == "__main__":
    main()
