"""Download the Amazon Reviews 2023 Musical_Instruments source files."""

from __future__ import annotations

import argparse
import os
from pathlib import Path
from urllib.request import ProxyHandler, Request, build_opener


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_ROOT = PROJECT_ROOT / "data" / "raw" / "amazon_reviews_2023"
BASE_URL = "https://mcauleylab.ucsd.edu/public_datasets/data/amazon_2023"
FILES = (
    "raw/review_categories/Musical_Instruments.jsonl.gz",
    "raw/meta_categories/meta_Musical_Instruments.jsonl.gz",
    "benchmark/5core/rating_only/Musical_Instruments.csv.gz",
    "benchmark/5core/last_out_w_his/Musical_Instruments.train.csv.gz",
    "benchmark/5core/last_out_w_his/Musical_Instruments.valid.csv.gz",
    "benchmark/5core/last_out_w_his/Musical_Instruments.test.csv.gz",
)
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


def get_remote_size(opener, url: str, timeout: float) -> int:
    request = Request(url, method="HEAD")
    with opener.open(request, timeout=timeout) as response:
        content_length = response.headers.get("Content-Length")
    if content_length is None:
        raise RuntimeError(f"Missing Content-Length header: {url}")
    return int(content_length)


def download_file(opener, relative_path: str, timeout: float) -> None:
    url = f"{BASE_URL}/{relative_path}"
    destination = DATA_ROOT / relative_path
    partial = destination.with_suffix(f"{destination.suffix}.part")
    destination.parent.mkdir(parents=True, exist_ok=True)

    remote_size = get_remote_size(opener, url, timeout)
    if destination.exists():
        if destination.stat().st_size == remote_size:
            print(f"[skip] {relative_path}")
            return
        raise RuntimeError(f"Existing file has an unexpected size: {destination}")

    downloaded = partial.stat().st_size if partial.exists() else 0
    headers = {"Range": f"bytes={downloaded}-"} if downloaded else {}
    request = Request(url, headers=headers)
    with opener.open(request, timeout=timeout) as response:
        append = downloaded > 0 and response.status == 206
        if not append:
            downloaded = 0
        mode = "ab" if append else "wb"
        with partial.open(mode) as output:
            while chunk := response.read(CHUNK_SIZE):
                output.write(chunk)
                downloaded += len(chunk)
                percent = downloaded * 100 / remote_size
                print(f"\r[download] {relative_path}: {percent:6.2f}%", end="")
    print()

    if partial.stat().st_size != remote_size:
        raise RuntimeError(f"Downloaded file has an unexpected size: {partial}")
    partial.replace(destination)


def main() -> None:
    args = parse_args()
    opener = create_opener(args.proxy)
    for relative_path in FILES:
        download_file(opener, relative_path, args.timeout)


if __name__ == "__main__":
    main()
