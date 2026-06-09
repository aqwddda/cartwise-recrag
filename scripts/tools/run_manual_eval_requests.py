from __future__ import annotations

import json
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any

API_BASE_URL = "http://127.0.0.1:8000"
OUTPUT_DIR = Path("artifacts/reports/manual_eval_responses")
TOP_K = 3
TIMEOUT_SECONDS = 180


QUERIES: dict[str, str] = {
    "EN-01": "guitar tuner for beginners",
    "EN-02": "compact clip-on tuner for acoustic guitar and ukulele",
    "EN-03": "chromatic tuner with bright display for live performance",
    "EN-04": "replacement strings for acoustic guitar",
    "EN-05": "light gauge electric guitar strings for easy bending",
    "EN-06": "fender guitar strap for electric guitar",
    "EN-07": "adjustable padded guitar strap for long practice sessions",
    "EN-08": "capo for acoustic guitar with one hand operation",
    "EN-09": "portable microphone stand for home recording",
    "EN-10": "black metal adjustable boom arm microphone stand",
    "EN-11": "desk mount microphone arm for podcast recording",
    "EN-12": "audio technica atr2100 usb microphone",
    "EN-13": "usb condenser microphone for recording vocals on a laptop",
    "EN-14": "xlr cable for condenser microphone",
    "EN-15": "balanced xlr female to male microphone cable 10 feet",
    "EN-16": "sustain pedal compatible with digital piano keyboard",
    "EN-17": "adjustable keyboard stand for stage performance",
    "EN-18": "beginner violin shoulder rest for comfortable practice",
    "EN-19": "violin strings with warm tone for student instrument",
    "EN-20": "drum practice pad for quiet apartment practice",
    "EN-21": "drum sticks for jazz with lightweight wooden feel",
    "EN-22": "alto saxophone reeds strength 2.5 for beginners",
    "EN-23": "trumpet mute for quiet practice at home",
    "EN-24": "gift for someone learning to play guitar",
    "EN-25": "I need something small and reliable to keep my guitar in tune during rehearsals",
    "ZH-01": "适合初学者使用的夹式吉他调音器",
    "ZH-02": "家庭录音用的便携式桌面麦克风支架",
    "ZH-03": "适合电容麦克风的十英尺 XLR 公对母平衡线",
    "ZH-04": "有没有适合公寓安静练习的架子鼓练习垫",
    "ZH-05": "我想给刚开始学吉他的朋友买一个实用的小礼物",
}


def request_json(
    method: str, url: str, payload: dict[str, Any] | None = None
) -> dict[str, Any]:
    body = None
    headers = {"Accept": "application/json"}

    if payload is not None:
        body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        headers["Content-Type"] = "application/json; charset=utf-8"

    request = urllib.request.Request(
        url=url,
        data=body,
        headers=headers,
        method=method,
    )

    with urllib.request.urlopen(request, timeout=TIMEOUT_SECONDS) as response:
        response_body = response.read().decode("utf-8")
        return json.loads(response_body)


def save_json(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(data, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print("Checking backend readiness...")
    try:
        ready = request_json("GET", f"{API_BASE_URL}/health/ready")
        print(f"Backend ready: {ready.get('status')}")
    except Exception as error:
        print("Backend is not ready. Stop.")
        print(error)
        raise SystemExit(1)

    for query_id, query in QUERIES.items():
        print(f"Running {query_id}: {query}")

        payload = {
            "query": query,
            "user_id": None,
            "top_k": TOP_K,
        }

        start = time.perf_counter()

        try:
            response = request_json(
                "POST",
                f"{API_BASE_URL}/api/v1/recommend",
                payload,
            )
            elapsed = round(time.perf_counter() - start, 3)

            output = {
                "query_id": query_id,
                "query": query,
                "top_k": TOP_K,
                "client_elapsed_seconds": elapsed,
                "success": True,
                "response": response,
            }

            output_path = OUTPUT_DIR / f"{query_id}.json"
            save_json(output_path, output)

            api_latency = response.get("latency_ms")
            print(
                f"Saved {output_path} | client={elapsed}s | api_latency_ms={api_latency}"
            )

        except urllib.error.HTTPError as error:
            elapsed = round(time.perf_counter() - start, 3)
            error_body = error.read().decode("utf-8", errors="replace")

            output = {
                "query_id": query_id,
                "query": query,
                "top_k": TOP_K,
                "client_elapsed_seconds": elapsed,
                "success": False,
                "status_code": error.code,
                "error": error.reason,
                "error_body": error_body,
            }

            output_path = OUTPUT_DIR / f"{query_id}.error.json"
            save_json(output_path, output)
            print(f"Failed {query_id}, saved {output_path}")

        except Exception as error:
            elapsed = round(time.perf_counter() - start, 3)

            output = {
                "query_id": query_id,
                "query": query,
                "top_k": TOP_K,
                "client_elapsed_seconds": elapsed,
                "success": False,
                "error": str(error),
            }

            output_path = OUTPUT_DIR / f"{query_id}.error.json"
            save_json(output_path, output)
            print(f"Failed {query_id}, saved {output_path}")


if __name__ == "__main__":
    main()
