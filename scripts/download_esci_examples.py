"""Download the official Amazon ESCI query-product examples Parquet file."""

from __future__ import annotations

import argparse
import os
from pathlib import Path
from urllib.request import ProxyHandler, Request, build_opener


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_ROOT = PROJECT_ROOT / "data" / "raw" / "amazon_esci"
BASE_URL = (
    "https://github.com/amazon-science/esci-data/raw/main/"
    "shopping_queries_dataset"
)
RELATIVE_PATH = "shopping_queries_dataset_examples.parquet"
CHUNK_SIZE = 1024 * 1024


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--proxy",
        default=os.getenv("HTTPS_PROXY") or os.getenv("HTTP_PROXY"),
        help="Optional HTTP(S) proxy, for example http://127.0.0.1:9508.",
    )
    parser.add_argument("--timeout", type=float, default=120.0)
    return parser.parse_args()


def create_opener(proxy: str | None):
    proxies = {"http": proxy, "https": proxy} if proxy else {}
    return build_opener(ProxyHandler(proxies))


def download_file(proxy: str | None, timeout: float) -> Path:
    url = f"{BASE_URL}/{RELATIVE_PATH}"
    destination = DATA_ROOT / RELATIVE_PATH
    partial = destination.with_suffix(f"{destination.suffix}.part")
    destination.parent.mkdir(parents=True, exist_ok=True)

    if destination.exists():
        print(f"[skip] {destination}")
        return destination

    downloaded = partial.stat().st_size if partial.exists() else 0
    headers = {"Range": f"bytes={downloaded}-"} if downloaded else {}
    opener = create_opener(proxy)
    with opener.open(Request(url, headers=headers), timeout=timeout) as response:
        append = downloaded > 0 and response.status == 206
        if not append:
            downloaded = 0
        mode = "ab" if append else "wb"
        with partial.open(mode) as output:
            while chunk := response.read(CHUNK_SIZE):
                output.write(chunk)
                downloaded += len(chunk)
                print(f"\r[download] {downloaded / 1024 / 1024:,.1f} MiB", end="")
    print()
    partial.replace(destination)
    print(f"[done] {destination}")
    return destination


def main() -> None:
    args = parse_args()
    download_file(args.proxy, args.timeout)


if __name__ == "__main__":
    main()
