"""Download and normalize Open-Sora-Plan MixKit captions for CacheHead.

The upstream Open-Sora-Plan v1.0 annotation file contains records for several
stock-video sources. This script fetches the annotation JSON with Hugging
Face's cached downloader (not the 27 GB MixKit video archive), selects the
MixKit records, and writes the
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
import tempfile
from collections.abc import Iterator, Mapping
from pathlib import Path
from typing import Any


DEFAULT_REPO_ID = "LanguageBind/Open-Sora-Plan-v1.0.0"
DEFAULT_FILENAME = "sharegpt4v_path_cap_64x512x512.json"
# The current published annotation revision contains 8,230 MixKit clip-caption
# records.  The CacheHead trainer accepts any non-empty JSONL; the historical
# 6,484 figure described an earlier curated subset, not a parser requirement.
DEFAULT_EXPECTED_COUNT = 8_230
CHUNK_SIZE = 1024 * 1024


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


def download_annotation(
    repo_id: str,
    filename: str,
    revision: str,
    cache_dir: str | None,
    force_download: bool,
) -> Path:
    """Fetch annotations through Hugging Face Hub's resumable local cache."""
    try:
        from huggingface_hub import hf_hub_download
    except ImportError as error:
        raise RuntimeError(
            "huggingface_hub is required. Install this repository's dependencies or run "
            "`pip install huggingface_hub`."
        ) from error

    try:
        path = hf_hub_download(
            repo_id=repo_id,
            repo_type="dataset",
            filename=filename,
            revision=revision,
            cache_dir=cache_dir,
            force_download=force_download,
        )
    except Exception as error:
        raise RuntimeError(
            f"could not download {repo_id}/{filename} from Hugging Face: {error}"
        ) from error
    return Path(path)


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
        "--repo-id", default=DEFAULT_REPO_ID,
        help="Hugging Face dataset repository (used unless --source-path is supplied)",
    )
    parser.add_argument("--filename", default=DEFAULT_FILENAME, help="annotation filename in --repo-id")
    parser.add_argument("--revision", default="main", help="Hugging Face dataset revision")
    parser.add_argument("--source-path", help="use an already downloaded annotation JSON instead")
    parser.add_argument(
        "--cache-dir",
        help="Hugging Face cache directory (default: Hugging Face's standard cache)",
    )
    parser.add_argument("--source-keyword", default="mixkit", help="case-insensitive source-path filter")
    parser.add_argument(
        "--expected-count", type=int, default=DEFAULT_EXPECTED_COUNT,
        help="fail unless this many captions are written; use 0 to disable the check",
    )
    parser.add_argument("--force-download", action="store_true", help="ignore the cached Hub file and download it again")
    parser.add_argument("--force", action="store_true", help="replace an existing output JSONL")
    args = parser.parse_args()

    if args.expected_count < 0:
        parser.error("--expected-count must be non-negative")

    output_path = Path(args.output)
    if args.source_path:
        source_path = Path(args.source_path)
        if not source_path.is_file():
            parser.error(f"--source-path does not exist or is not a file: {source_path}")
    else:
        source_path = download_annotation(
            args.repo_id,
            args.filename,
            args.revision,
            args.cache_dir,
            args.force_download,
        )
        print(f"using Hugging Face cached annotation: {source_path}")

    expected_count = args.expected_count or None
    count, checksum = write_captions(
        source_path, output_path, args.source_keyword, expected_count, args.force
    )
    print(f"wrote {count} captions to {output_path} (sha256={checksum})")


if __name__ == "__main__":
    main()
