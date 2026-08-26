"""Download and normalize Open-Sora-Plan MixKit captions for CacheHead.

The upstream Open-Sora-Plan v1.0 annotation file contains records for several
stock-video sources.  This script downloads the annotation JSON only (not the
27 GB MixKit video archive), selects the MixKit records, and writes the exact
``{"id": ..., "caption": ...}`` JSONL contract expected by
``cache_head_model_training.py``.

Example:
    python download_mixkit_captions.py --output mixkit_captions.jsonl
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import tempfile
import time
from collections.abc import Iterator, Mapping
from pathlib import Path
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen


DEFAULT_SOURCE_URL = (
    "https://huggingface.co/datasets/LanguageBind/Open-Sora-Plan-v1.0.0/resolve/main/"
    "sharegpt4v_path_cap_64x512x512.json?download=true"
)
DEFAULT_EXPECTED_COUNT = 6_484
CHUNK_SIZE = 1024 * 1024
USER_AGENT = "DiffSynth-Studio-MixKit-caption-downloader/1.0"


def iter_json_array(path: Path) -> Iterator[Any]:
    """Stream a top-level JSON array without loading a 474 MB annotation file."""
    decoder = json.JSONDecoder()
    with path.open("r", encoding="utf-8") as handle:
        buffer = ""
        exhausted = False

        def append_chunk() -> None:
            nonlocal buffer, exhausted
            chunk = handle.read(CHUNK_SIZE)
            if chunk:
                buffer += chunk
            else:
                exhausted = True

        append_chunk()
        while not buffer.strip() and not exhausted:
            append_chunk()
        buffer = buffer.lstrip()
        if not buffer.startswith("["):
            raise ValueError(f"{path} is not a top-level JSON array")
        buffer = buffer[1:]

        while True:
            while not buffer.strip() and not exhausted:
                append_chunk()
            buffer = buffer.lstrip()
            if not buffer:
                raise ValueError(f"unexpected end of JSON array in {path}")
            if buffer[0] == "]":
                return

            while True:
                try:
                    record, end = decoder.raw_decode(buffer)
                    break
                except json.JSONDecodeError:
                    if exhausted:
                        raise ValueError(f"invalid or truncated JSON array in {path}") from None
                    append_chunk()
            yield record
            buffer = buffer[end:].lstrip()

            while not buffer and not exhausted:
                append_chunk()
            if not buffer:
                raise ValueError(f"unexpected end of JSON array in {path}")
            if buffer[0] == ",":
                buffer = buffer[1:]
            elif buffer[0] == "]":
                return
            else:
                raise ValueError(f"expected ',' or ']' after an array entry in {path}")


def download_file(url: str, destination: Path, retries: int) -> None:
    """Download ``url`` atomically, with bounded retries and progress output."""
    destination.parent.mkdir(parents=True, exist_ok=True)
    request = Request(url, headers={"User-Agent": USER_AGENT})
    for attempt in range(1, retries + 1):
        temp_name: str | None = None
        try:
            with urlopen(request, timeout=60) as response:
                content_length = response.headers.get("Content-Length")
                total = int(content_length) if content_length and content_length.isdigit() else None
                with tempfile.NamedTemporaryFile(
                    mode="wb", prefix=f".{destination.name}.", suffix=".part",
                    dir=destination.parent, delete=False,
                ) as temporary:
                    temp_name = temporary.name
                    copied = 0
                    next_report = 64 * 1024 * 1024
                    while True:
                        block = response.read(CHUNK_SIZE)
                        if not block:
                            break
                        temporary.write(block)
                        copied += len(block)
                        if copied >= next_report:
                            suffix = f"/{total / 1024 ** 2:.0f} MiB" if total else ""
                            print(f"downloaded {copied / 1024 ** 2:.0f} MiB{suffix}", file=sys.stderr)
                            next_report += 64 * 1024 * 1024
            os.replace(temp_name, destination)
            print(f"downloaded {destination} ({destination.stat().st_size / 1024 ** 2:.1f} MiB)")
            return
        except (HTTPError, URLError, OSError) as error:
            if temp_name is not None:
                Path(temp_name).unlink(missing_ok=True)
            if attempt == retries:
                raise RuntimeError(f"could not download {url}: {error}") from error
            delay = 2 ** (attempt - 1)
            print(f"download attempt {attempt}/{retries} failed ({error}); retrying in {delay}s", file=sys.stderr)
            time.sleep(delay)


def value_as_text(value: Any) -> str | None:
    """Normalize a caption field that may be a string or one-item string list."""
    if isinstance(value, str):
        text = value.strip()
        return text or None
    if isinstance(value, list):
        for item in value:
            text = value_as_text(item)
            if text:
                return text
    return None


def normalize_record(record: Any, source_keyword: str) -> tuple[str, str] | None:
    """Extract ``(id, caption)`` from an Open-Sora-Plan annotation record."""
    if not isinstance(record, Mapping):
        return None

    record_id = value_as_text(record.get("id"))
    source_path = value_as_text(record.get("path")) or value_as_text(record.get("file_name"))
    haystack = " ".join(item for item in (record_id, source_path) if item).lower()
    if source_keyword.lower() not in haystack:
        return None

    caption = None
    for field in ("caption", "cap", "text", "prompt"):
        caption = value_as_text(record.get(field))
        if caption:
            break
    if caption is None:
        return None

    # A clip path is unique across all extracted MixKit segments and remains
    # stable if the source annotation order changes.
    sample_id = record_id or source_path
    if sample_id is None:
        return None
    return sample_id, caption


def write_captions(
    source_path: Path,
    output_path: Path,
    source_keyword: str,
    expected_count: int | None,
    force: bool,
) -> tuple[int, str]:
    """Filter source annotations and atomically write normalized training JSONL."""
    if output_path.exists() and not force:
        raise FileExistsError(f"output already exists: {output_path} (pass --force to replace it)")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    unique_ids: set[str] = set()
    count = 0
    digest = hashlib.sha256()
    with tempfile.NamedTemporaryFile(
        mode="w", encoding="utf-8", prefix=f".{output_path.name}.", suffix=".part",
        dir=output_path.parent, delete=False,
    ) as temporary:
        temp_path = Path(temporary.name)
        try:
            for record in iter_json_array(source_path):
                normalized = normalize_record(record, source_keyword)
                if normalized is None:
                    continue
                sample_id, caption = normalized
                if sample_id in unique_ids:
                    raise ValueError(f"duplicate MixKit id in source annotations: {sample_id!r}")
                unique_ids.add(sample_id)
                line = json.dumps({"id": sample_id, "caption": caption}, ensure_ascii=False, separators=(",", ":"))
                temporary.write(line + "\n")
                digest.update((line + "\n").encode("utf-8"))
                count += 1

            if expected_count is not None and count != expected_count:
                raise ValueError(
                    f"expected {expected_count} {source_keyword} captions, found {count}. "
                    "The upstream annotation schema or source may have changed; inspect the source "
                    "or rerun with --expected-count 0 only after validating it."
                )
            temporary.flush()
            os.fsync(temporary.fileno())
            os.replace(temp_path, output_path)
        except BaseException:
            temp_path.unlink(missing_ok=True)
            raise
    return count, digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Download Open-Sora-Plan MixKit annotations and write CacheHead caption JSONL."
    )
    parser.add_argument("--output", default="mixkit_captions.jsonl", help="destination JSONL path")
    parser.add_argument(
        "--source-url", default=DEFAULT_SOURCE_URL,
        help="upstream annotation JSON URL (used unless --source-path is supplied)",
    )
    parser.add_argument("--source-path", help="use an already downloaded annotation JSON instead")
    parser.add_argument(
        "--cache-dir", default=".cache/mixkit",
        help="where a downloaded annotation JSON is cached (default: %(default)s)",
    )
    parser.add_argument("--source-keyword", default="mixkit", help="case-insensitive source-path filter")
    parser.add_argument(
        "--expected-count", type=int, default=DEFAULT_EXPECTED_COUNT,
        help="fail unless this many captions are written; use 0 to disable the check",
    )
    parser.add_argument("--retries", type=int, default=3, help="network download attempts")
    parser.add_argument("--force", action="store_true", help="replace an existing output JSONL")
    parser.add_argument(
        "--keep-source", action="store_true",
        help="keep a downloaded annotation JSON (it is otherwise deleted after extraction)",
    )
    args = parser.parse_args()

    if args.retries < 1:
        parser.error("--retries must be at least one")
    if args.expected_count < 0:
        parser.error("--expected-count must be non-negative")

    output_path = Path(args.output)
    source_path = Path(args.source_path) if args.source_path else Path(args.cache_dir) / "opensora_mixkit_annotations.json"
    downloaded_here = False
    if not source_path.is_file():
        if args.source_path:
            parser.error(f"--source-path does not exist or is not a file: {source_path}")
        download_file(args.source_url, source_path, args.retries)
        downloaded_here = True

    try:
        expected_count = args.expected_count or None
        count, checksum = write_captions(
            source_path, output_path, args.source_keyword, expected_count, args.force
        )
    finally:
        if downloaded_here and not args.keep_source:
            source_path.unlink(missing_ok=True)
            try:
                source_path.parent.rmdir()
            except OSError:
                pass

    print(f"wrote {count} captions to {output_path} (sha256={checksum})")


if __name__ == "__main__":
    main()
