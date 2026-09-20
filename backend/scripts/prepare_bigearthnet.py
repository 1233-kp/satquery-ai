"""
backend/scripts/prepare_bigearthnet.py
=======================================
Builds a small, locally-cached image-text VQA dataset for fine-tuning
SmolVLM on remote-sensing imagery, from the official BigEarthNet.txt
annotations paired with a partial sample of the official Sentinel-2 raw
archive.

Sources:
  - Annotations: BIFOLD-BigEarthNetv2-0/BigEarthNet.txt (HF Hub, ~467MB
    parquet, 9.55M QA rows). Downloaded in full via huggingface_hub -- it's
    text-only, no images.
  - Images: BigEarthNet-S2.tar.zst (Zenodo, ~59GB, all 549k patches). We
    stream it over HTTP with on-the-fly zstd decompression and stop early
    once enough patches have been collected, instead of downloading the
    whole archive -- this pulls roughly (patches_needed / 549488) * 59GB.

Since the archive is laid out one Sentinel-2 tile at a time and each tile's
patches share a single official train/val/test split, an early prefix of
the stream is skewed towards whichever tiles happen to come first. This
script keeps reading past a tile once its split's quota is full, so the
output has a real mix of train/validation/test patches -- just drawn from
fewer distinct tiles (countries/seasons) than the full dataset.

Usage:
    python scripts/prepare_bigearthnet.py
    python scripts/prepare_bigearthnet.py --train-quota 900 --val-quota 150 --test-quota 150
"""
from __future__ import annotations

import argparse
import io
import json
import random
import re
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
from PIL import Image

ANNOTATION_REPO = "BIFOLD-BigEarthNetv2-0/BigEarthNet.txt"
ANNOTATION_FILE = "BigEarthNet.txt.parquet"
S2_ARCHIVE_URL = "https://zenodo.org/records/10891137/files/BigEarthNet-S2.tar.zst?download=1"

RGB_BANDS = ("B04", "B03", "B02")  # Red, Green, Blue -- assembled in this order
KEEP_TYPES = {"binary", "mcq", "captioning"}  # excludes "bounding box" (grounding tasks)

# Every captioning answer in BigEarthNet.txt opens with this exact skeleton
# ("This satellite image, captured during the {season} in {country}, showcases
# {descriptor} within the "{climate}" climate zone.") -- 100% of examples, no
# exceptions. Left untouched, a model fine-tuned on these will happily
# memorize the incantation and regurgitate a fabricated country/climate/area
# regardless of what's actually in the image. CAPTION_OPEN_RE lets us detect
# and rewrite that opener so there's no single fixed string to memorize.
CAPTION_OPEN_RE = re.compile(
    r'^This satellite image, captured (?:during|in) the ([\w]+)(?: season)? in ([\w\s]+?), '
    r'showcases (.+?) within the "([^"]+)" climate zone\.\s*',
    re.IGNORECASE | re.DOTALL,
)
ALT_OPENERS = [
    "Looking at this scene, {descriptor}.",
    "{descriptor_cap}.",
    "The imagery reveals {descriptor}.",
    "Here we see {descriptor}.",
]


def diversify_caption(answer: str, rng: random.Random) -> str:
    """Breaks the rigid country/climate-zone opener so it isn't a single
    memorizable incantation, while still teaching the correct fact association
    for a minority of examples (kept verbatim)."""
    m = CAPTION_OPEN_RE.match(answer)
    if not m:
        return answer
    season, country, descriptor, climate = m.groups()
    rest = answer[m.end():].strip()

    roll = rng.random()
    if roll < 0.4:
        result = answer  # keep verbatim for ~40% -- still need some clean signal
    elif roll < 0.75:
        opener = rng.choice(ALT_OPENERS).format(descriptor=descriptor, descriptor_cap=descriptor[0].upper() + descriptor[1:])
        result = f"{opener} {rest}"
    else:
        result = f"{rest} This scene was captured during {season} in {country} (\"{climate}\" climate zone)."

    if rng.random() < 0.5:
        sentences = [s.strip() for s in re.split(r"(?<=[.!?])\s+", result.strip()) if s.strip()]
        if len(sentences) > 3:
            head, *middle, tail = sentences
            rng.shuffle(middle)
            result = " ".join([head] + middle + [tail])
    return result


def parse_args():
    p = argparse.ArgumentParser(description="Prepare a small BigEarthNet.txt image-text subset for VLM fine-tuning.")
    p.add_argument("--output-dir", type=str, default=str(Path(__file__).resolve().parent.parent / "data" / "bigearthnet"))
    p.add_argument("--train-quota", type=int, default=900, help="Stop growing the train split once this many patches are collected.")
    p.add_argument("--val-quota", type=int, default=150)
    p.add_argument("--test-quota", type=int, default=150)
    p.add_argument("--max-patches", type=int, default=1600, help="Hard safety cap on total patches, regardless of quotas.")
    p.add_argument("--max-mb", type=int, default=3000, help="Hard safety cap on MB streamed from the archive.")
    p.add_argument("--max-pairs-per-patch", type=int, default=4, help="Cap QA pairs kept per image patch.")
    p.add_argument("--image-size", type=int, default=448, help="Longest side (px) to resize assembled patches to.")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--skip-download", action="store_true", help="Reuse already-extracted images/manifests if present.")
    p.add_argument("--rebuild-only", action="store_true",
                   help="Skip streaming entirely and rebuild manifests from images already on disk "
                        "(re-downloads only the small annotation parquet, from cache if present).")
    p.add_argument("--captioning-fraction", type=float, default=1.0,
                   help="Fraction of patches allowed to keep a captioning example (0.2 = only 1 in 5 "
                        "patches gets one, so binary/mcq dominate and the model can't shortcut to "
                        "memorizing the one captioning template).")
    return p.parse_args()


def percentile_stretch(ch: np.ndarray, pmin: float = 2.0, pmax: float = 98.0) -> np.ndarray:
    valid = np.isfinite(ch)
    if not valid.any():
        return np.zeros_like(ch, dtype=np.uint8)
    lo, hi = np.percentile(ch[valid], pmin), np.percentile(ch[valid], pmax)
    if hi <= lo:
        hi = lo + 1e-6
    return (np.clip((ch - lo) / (hi - lo), 0.0, 1.0) * 255.0 + 0.5).astype(np.uint8)


def assemble_rgb(band_bytes: Dict[str, bytes], target_size: int) -> Image.Image:
    channels = []
    for b in RGB_BANDS:
        arr = np.array(Image.open(io.BytesIO(band_bytes[b]))).astype(np.float32)
        channels.append(percentile_stretch(arr))
    rgb = np.stack(channels, axis=-1)
    img = Image.fromarray(rgb, mode="RGB")
    img.thumbnail((target_size, target_size), Image.LANCZOS)
    return img


def download_annotations() -> Path:
    from huggingface_hub import hf_hub_download

    print(f"[1/4] Downloading annotations ({ANNOTATION_REPO}/{ANNOTATION_FILE})...", flush=True)
    path = hf_hub_download(repo_id=ANNOTATION_REPO, filename=ANNOTATION_FILE, repo_type="dataset")
    print(f"      -> cached at {path}", flush=True)
    return Path(path)


def build_patch_split_index(parquet_path: Path) -> Dict[str, str]:
    import pyarrow.parquet as pq

    print("      Indexing patch_id -> split (single pass over annotations)...", flush=True)
    index: Dict[str, str] = {}
    pf = pq.ParquetFile(parquet_path)
    for batch in pf.iter_batches(columns=["patch_id", "split"], batch_size=200_000):
        for pid, sp in zip(batch.column("patch_id").to_pylist(), batch.column("split").to_pylist()):
            index[pid] = sp
    print(f"      -> {len(index)} unique patches indexed", flush=True)
    return index


def stream_s2_patches(
    patch_split: Dict[str, str],
    quotas: Dict[str, int],
    max_patches: int,
    max_bytes: int,
    image_size: int,
    images_dir: Path,
) -> Dict[str, str]:
    """Streams the S2 archive, saving RGB JPEGs and stopping once quotas/caps are hit.

    Returns {patch_id: split} for every patch actually saved.
    """
    import tarfile

    import requests
    import zstandard as zstd

    images_dir.mkdir(parents=True, exist_ok=True)
    counts: Dict[str, int] = defaultdict(int)
    accepted: Dict[str, str] = {}
    pending: Dict[str, Dict[str, bytes]] = defaultdict(dict)
    bytes_seen = 0
    t0 = time.perf_counter()

    def quotas_met() -> bool:
        return all(counts[s] >= q for s, q in quotas.items())

    print(f"[2/4] Streaming Sentinel-2 archive from Zenodo (quotas={quotas}, cap={max_patches} patches / {max_bytes // (1024 * 1024)}MB)...", flush=True)
    resp = requests.get(S2_ARCHIVE_URL, stream=True, timeout=60)
    resp.raise_for_status()
    dctx = zstd.ZstdDecompressor()

    try:
        with dctx.stream_reader(resp.raw) as reader:
            tf = tarfile.open(fileobj=reader, mode="r|")
            for member in tf:
                if not member.isfile():
                    continue
                stem = Path(member.name).stem  # "<patch_id>_B04"
                band = next((b for b in RGB_BANDS if stem.endswith("_" + b)), None)
                if band is None:
                    continue
                parts = member.name.split("/")
                patch_id = parts[-2] if len(parts) >= 2 else stem.rsplit("_", 1)[0]

                fobj = tf.extractfile(member)
                if fobj is None:
                    continue
                raw = fobj.read()
                bytes_seen += len(raw)
                pending[patch_id][band] = raw

                if len(pending[patch_id]) < len(RGB_BANDS):
                    continue

                band_bytes = pending.pop(patch_id)
                split = patch_split.get(patch_id)
                if split is None or split == "bench" or counts[split] >= quotas.get(split, 0):
                    continue  # no annotations for this patch, or its split quota is already full

                try:
                    img = assemble_rgb(band_bytes, image_size)
                    img.save(images_dir / f"{patch_id}.jpg", quality=92)
                except Exception as e:
                    print(f"  [warn] failed to assemble {patch_id}: {e}", flush=True)
                    continue

                accepted[patch_id] = split
                counts[split] += 1

                if len(accepted) % 100 == 0:
                    elapsed = time.perf_counter() - t0
                    print(
                        f"  ...{len(accepted)} patches ({dict(counts)}), "
                        f"{bytes_seen / 1e6:.1f} MB streamed, {elapsed:.0f}s elapsed",
                        flush=True,
                    )

                if quotas_met() or len(accepted) >= max_patches or bytes_seen >= max_bytes:
                    break
    finally:
        resp.close()

    print(
        f"      -> collected {len(accepted)} patches {dict(counts)}, "
        f"~{bytes_seen / 1e6:.1f} MB streamed in {time.perf_counter() - t0:.0f}s",
        flush=True,
    )
    return accepted


def build_examples(
    parquet_path: Path,
    accepted: Dict[str, str],
    max_pairs_per_patch: int,
    captioning_fraction: float,
    seed: int,
) -> Dict[str, List[dict]]:
    import pyarrow.compute as pc
    import pyarrow.parquet as pq

    print("[3/4] Joining QA annotations for collected patches...", flush=True)
    table = pq.read_table(parquet_path, columns=["patch_id", "input", "output", "type", "category"])
    mask = pc.is_in(table["patch_id"], value_set=pyarrow_array(list(accepted.keys())))
    mask = pc.and_(mask, pc.is_in(table["type"], value_set=pyarrow_array(list(KEEP_TYPES))))
    table = table.filter(mask)

    rows_by_patch: Dict[str, List[dict]] = defaultdict(list)
    for row in table.to_pylist():
        rows_by_patch[row["patch_id"]].append(row)

    rng = random.Random(seed)
    n_captioning_kept = n_captioning_diversified = 0
    by_split: Dict[str, List[dict]] = defaultdict(list)
    for patch_id, split in accepted.items():
        rows = rows_by_patch.get(patch_id, [])
        if not rows:
            continue
        rng.shuffle(rows)
        captioning = [r for r in rows if r["type"] == "captioning"]
        others = [r for r in rows if r["type"] != "captioning"]
        keep_captioning = captioning[:1] if rng.random() < captioning_fraction else []
        chosen = (keep_captioning + others)[:max_pairs_per_patch]
        for r in chosen:
            answer = r["output"]
            if r["type"] == "captioning":
                n_captioning_kept += 1
                diversified = diversify_caption(answer, rng)
                if diversified != answer:
                    n_captioning_diversified += 1
                answer = diversified
            by_split[split].append(
                {
                    "patch_id": patch_id,
                    "image": f"images/{patch_id}.jpg",
                    "question": r["input"],
                    "answer": answer,
                    "type": r["type"],
                    "category": r["category"],
                }
            )
    if n_captioning_kept:
        print(f"      Captioning examples kept: {n_captioning_kept} "
              f"({n_captioning_diversified} rewritten away from the fixed template, "
              f"{n_captioning_kept - n_captioning_diversified} kept verbatim)", flush=True)
    return by_split


def pyarrow_array(values: List[str]):
    import pyarrow as pa

    return pa.array(values)


def write_manifests(output_dir: Path, by_split: Dict[str, List[dict]]) -> None:
    print("[4/4] Writing manifests...", flush=True)
    name_map = {"train": "train.jsonl", "validation": "val.jsonl", "test": "test.jsonl"}
    summary = {}
    for split, filename in name_map.items():
        examples = by_split.get(split, [])
        with open(output_dir / filename, "w", encoding="utf-8") as f:
            for ex in examples:
                f.write(json.dumps(ex) + "\n")
        summary[split] = len(examples)
        print(f"      {filename}: {len(examples)} pairs", flush=True)

    # A handful of test-split, captioning-first patches for the qualitative before/after demo.
    demo_pool = [ex for ex in by_split.get("test", []) if ex["type"] == "captioning"] or by_split.get("test", [])
    seen_patches = []
    demo = []
    for ex in demo_pool:
        if ex["patch_id"] in seen_patches:
            continue
        seen_patches.append(ex["patch_id"])
        demo.append(ex)
        if len(demo) >= 3:
            break
    with open(output_dir / "demo_samples.jsonl", "w", encoding="utf-8") as f:
        for ex in demo:
            f.write(json.dumps(ex) + "\n")
    print(f"      demo_samples.jsonl: {len(demo)} pairs (for before/after comparison)", flush=True)

    with open(output_dir / "manifest.json", "w", encoding="utf-8") as f:
        json.dump({"counts": summary, "source": ANNOTATION_REPO, "image_archive": S2_ARCHIVE_URL}, f, indent=2)


def already_prepared(output_dir: Path) -> bool:
    return all((output_dir / name).exists() for name in ("train.jsonl", "val.jsonl", "test.jsonl", "manifest.json"))


def main():
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    images_dir = output_dir / "images"

    if args.skip_download and already_prepared(output_dir):
        print(f"Manifests already present in {output_dir}, skipping (--skip-download).", flush=True)
        return

    random.seed(args.seed)
    parquet_path = download_annotations()
    patch_split = build_patch_split_index(parquet_path)

    if args.rebuild_only:
        if not images_dir.exists():
            print(f"ERROR: --rebuild-only given but {images_dir} doesn't exist -- nothing to rebuild from.", file=sys.stderr)
            sys.exit(1)
        accepted = {
            p.stem: patch_split[p.stem]
            for p in images_dir.glob("*.jpg")
            if p.stem in patch_split
        }
        print(f"[2/4] Rebuild-only: reusing {len(accepted)} already-downloaded images (no re-streaming).", flush=True)
    else:
        quotas = {"train": args.train_quota, "validation": args.val_quota, "test": args.test_quota}
        accepted = stream_s2_patches(
            patch_split=patch_split,
            quotas=quotas,
            max_patches=args.max_patches,
            max_bytes=args.max_mb * 1024 * 1024,
            image_size=args.image_size,
            images_dir=images_dir,
        )
    if not accepted:
        print("ERROR: no patches were collected -- check network access to Zenodo.", file=sys.stderr)
        sys.exit(1)

    by_split = build_examples(parquet_path, accepted, args.max_pairs_per_patch, args.captioning_fraction, args.seed)
    write_manifests(output_dir, by_split)
    print(f"\nDone. Dataset ready at {output_dir}", flush=True)


if __name__ == "__main__":
    main()
