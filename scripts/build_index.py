#!/usr/bin/env python
"""Crawl the public BrainGlobe atlas bucket and build a static search index.

The BrainGlobe atlas API (v3) stores every atlas as a set of components on
S3 (https://brainglobe.s3.us-west-2.amazonaws.com/atlas/):

    atlases/<atlas>/<version>/manifest.json   -> points at the components
    templates/<name>/<version>/template.ome.zarr
    annotation-sets/<name>/<version>/annotations_compressed.ome.zarr
    annotation-sets/<name>/<version>/hemispheres.ome.zarr
    annotation-sets/<name>/<version>/annotations.precomputed   (meshes)
    terminologies/<name>/<version>/terminology.csv

This script reads ``last_versions.conf`` and every manifest, inspects the
OME-Zarr metadata of each component, converts each terminology to JSON, and
enriches the atlases with the human-written descriptions from the BrainGlobe
"atlas details" documentation page. The output is written to
``public/data`` and consumed by the static website.

Only the standard library is required. If ``numpy`` and ``zstandard`` are
installed, a display intensity range is also computed for every template
(used to set the neuroglancer contrast).

Usage::

    python scripts/build_index.py [--out public/data] [--docs path/to/atlas-details.md]
"""

from __future__ import annotations

import argparse
import configparser
import csv
import datetime as dt
import io
import json
import re
import sys
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

BUCKET = "https://brainglobe.s3.us-west-2.amazonaws.com"
ROOT = f"{BUCKET}/atlas"
DOCS_URL = (
    "https://raw.githubusercontent.com/brainglobe/brainglobe.github.io/main/"
    "docs/source/documentation/brainglobe-atlasapi/usage/atlas-details.md"
)
DOCS_PAGE = (
    "https://brainglobe.info/documentation/brainglobe-atlasapi/usage/"
    "atlas-details.html"
)
USER_AGENT = "brainglobe-search-index-builder"
VOXEL_BUDGET = 8e6  # max voxels decoded per template for the display range

try:  # optional, for template intensity ranges
    import numpy as np
    import zstandard
except ImportError:  # pragma: no cover
    np = None
    zstandard = None


# --------------------------------------------------------------------------
# HTTP helpers
# --------------------------------------------------------------------------
def fetch(url: str, retries: int = 3) -> bytes | None:
    """GET a URL, returning None on 403/404."""
    for attempt in range(retries):
        try:
            req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
            with urllib.request.urlopen(req, timeout=60) as r:
                return r.read()
        except urllib.error.HTTPError as e:
            if e.code in (403, 404):
                return None
            if attempt == retries - 1:
                raise
        except (urllib.error.URLError, TimeoutError):
            if attempt == retries - 1:
                raise
    return None


def fetch_json(url: str):
    data = fetch(url)
    return None if data is None else json.loads(data)


def loc_url(location: str) -> str:
    """Manifest locations start with '/', relative to the atlas root."""
    return ROOT + location


# --------------------------------------------------------------------------
# OME-Zarr inspection
# --------------------------------------------------------------------------
def inspect_omezarr(url: str) -> dict | None:
    """Return multiscale level info for an OME-Zarr v0.5 (zarr v3) group."""
    group = fetch_json(f"{url}/zarr.json")
    if group is None:
        return None
    attrs = group.get("attributes", {})
    ome = attrs.get("ome", attrs)
    ms = ome["multiscales"][0]
    axes = ms["axes"]
    levels = []
    for ds in ms["datasets"]:
        arr = fetch_json(f"{url}/{ds['path']}/zarr.json")
        if arr is None:
            continue
        scale = next(
            t["scale"] for t in ds["coordinateTransformations"] if t["type"] == "scale"
        )
        trans = next(
            (
                t["translation"]
                for t in ds["coordinateTransformations"]
                if t["type"] == "translation"
            ),
            [0.0] * len(scale),
        )
        codecs = [c["name"] for c in arr.get("codecs", [])]
        cke = arr.get("chunk_key_encoding", {"name": "default"})
        levels.append(
            {
                "path": ds["path"],
                "scale": scale,
                "translation": trans,
                "shape": arr["shape"],
                "chunks": arr["chunk_grid"]["configuration"]["chunk_shape"],
                "dtype": arr["data_type"],
                "codecs": codecs,
                "separator": cke.get("configuration", {}).get(
                    "separator", "/" if cke.get("name") == "default" else "."
                ),
                "fill_value": arr.get("fill_value", 0),
            }
        )
    return {
        "axes": [
            {
                "name": a["name"],
                "unit": a.get("unit"),
                "orientation": (a.get("orientation") or {}).get("value"),
            }
            for a in axes
        ],
        "levels": levels,
    }


def intensity_range(url: str, level: dict) -> list[float] | None:
    """Compute a robust display range from the coarsest pyramid level."""
    if np is None or level["codecs"] not in (["bytes", "zstd"], ["bytes"]):
        return None
    shape, chunks = level["shape"], level["chunks"]
    dtype = np.dtype(
        {
            "uint8": "u1", "uint16": "<u2", "uint32": "<u4", "int8": "i1",
            "int16": "<i2", "int32": "<i4", "float32": "<f4", "float64": "<f8",
        }[level["dtype"]]
    )
    sep = level["separator"]
    n = [-(-s // c) for s, c in zip(shape, chunks)]
    chunk_voxels = int(np.prod(chunks))
    # Sample chunks nearest the volume centre, within a voxel budget, so
    # single-level templates with huge chunks stay cheap.
    centre = [(x - 1) / 2 for x in n]
    coords = sorted(
        ((i, j, k) for i in range(n[0]) for j in range(n[1]) for k in range(n[2])),
        key=lambda c: sum((a - b) ** 2 for a, b in zip(c, centre)),
    )
    budget = max(1, int(VOXEL_BUDGET // chunk_voxels))
    values = []
    dctx = zstandard.ZstdDecompressor()
    for c in coords[:budget]:
        key = sep.join(["c", *map(str, c)])
        raw = fetch(f"{url}/{level['path']}/{key}")
        if raw is None:
            continue
        if "zstd" in level["codecs"]:
            # streaming API: also handles frames without a content size
            raw = dctx.decompressobj().decompress(raw)
        values.append(np.frombuffer(raw, dtype=dtype)[::7])
    if not values:
        return None
    v = np.concatenate(values).astype(np.float64)
    v = v[v > 0]
    if v.size == 0:
        return None
    lo, hi = np.percentile(v, [0.5, 99.8])
    return [float(round(lo, 4)), float(round(hi, 4))]


def match_level(levels: list[dict], resolution_um: list[float]) -> str | None:
    """Pick the pyramid level whose voxel size matches the atlas resolution."""
    if not levels:
        return None
    unit_factor = 1000.0  # OME-Zarr scales are in millimetres
    best, best_err = None, float("inf")
    for lv in levels:
        sc = [s * unit_factor for s in lv["scale"][-3:]]
        err = sum(abs(a - b) / max(b, 1e-9) for a, b in zip(sc, resolution_um))
        if err < best_err:
            best, best_err = lv["path"], err
    return best


# --------------------------------------------------------------------------
# Terminology
# --------------------------------------------------------------------------
def parse_terminology(text: str) -> list[dict]:
    rows = []
    for r in csv.DictReader(io.StringIO(text)):
        def _int(v):
            try:
                return int(float(v))
            except (TypeError, ValueError):
                return None

        path = r.get("root_identifier_path") or ""
        rows.append(
            {
                "id": _int(r.get("identifier")),
                "parent": _int(r.get("parent_identifier")),
                "value": _int(r.get("annotation_value")),
                "name": (r.get("name") or "").strip(),
                "acronym": (r.get("abbreviation") or "").strip(),
                "color": (r.get("color_hex_triplet") or "").strip(),
                "path": [int(x) for x in re.findall(r"-?\d+", path)],
            }
        )
    return rows


def slug(location: str) -> str:
    return location.strip("/").replace("/", "__")


# --------------------------------------------------------------------------
# Documentation enrichment
# --------------------------------------------------------------------------
def md_inline_to_text(s: str) -> str:
    s = re.sub(r"!\[[^\]]*\]\([^)]*\)", "", s)
    s = re.sub(r"\[!\[[^\]]*\]\([^)]*\)\]\([^)]*\)", "", s)
    s = re.sub(r"\[([^\]]+)\]\([^)]*\)", r"\1", s)
    s = s.replace("`", "").replace("**", "")
    return re.sub(r"\s+", " ", s).strip()


def anchor(title: str) -> str:
    t = title.lower()
    t = re.sub(r"[^\w\s-]", "", t)
    return re.sub(r"\s+", "-", t.strip())


def parse_docs(md: str, atlas_keys: set[str]) -> dict[str, dict]:
    """Map atlas key -> {title, group, description, link, docs_url}."""
    out: dict[str, dict] = {}
    group = None
    sections: list[dict] = []
    cur = None
    for line in md.splitlines():
        if line.startswith("## "):
            group = line[3:].strip()
            cur = None
            continue
        if line.startswith("### "):
            head = line[4:].strip()
            m = re.match(r"\[(.+?)\]\((.+?)\)", head)
            title, link = (m.group(1), m.group(2)) if m else (head.rstrip("."), None)
            cur = {"title": title, "link": link, "group": group, "lines": []}
            sections.append(cur)
            continue
        if cur is not None:
            cur["lines"].append(line)
    for sec in sections:
        body = "\n".join(sec["lines"])
        # description = text before "Available versions" / key listing
        desc_part = re.split(
            r"\n\s*(?:Available versions|This atlas is only available|\* `)", body
        )[0]
        paras = [md_inline_to_text(p) for p in re.split(r"\n\s*\n", desc_part)]
        paras = [p for p in paras if p and not p.startswith("|")]
        description = "\n\n".join(paras)
        keys = set(re.findall(r"`([A-Za-z0-9_.\-]+)`", body)) & atlas_keys
        for key in keys:
            out[key] = {
                "title": sec["title"],
                "group": sec["group"],
                "description": description,
                "link": sec["link"],
                "docs_url": f"{DOCS_PAGE}#{anchor(sec['title'])}",
            }
    return out


# --------------------------------------------------------------------------
# Main
# --------------------------------------------------------------------------
def build(out_dir: Path, docs_path: Path | None, workers: int = 16) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    term_dir = out_dir / "terminologies"
    term_dir.mkdir(exist_ok=True)

    conf = configparser.ConfigParser()
    conf.read_string(fetch(f"{ROOT}/atlases/last_versions.conf").decode())
    latest = dict(conf["atlases"])
    print(f"{len(latest)} atlases in last_versions.conf", file=sys.stderr)

    def get_manifest(item):
        key, version = item
        vdir = version.replace(".", "_")
        m = fetch_json(f"{ROOT}/atlases/{key}/{vdir}/manifest.json")
        if m is None:
            print(f"  ! missing manifest for {key} {version}", file=sys.stderr)
        return key, m

    with ThreadPoolExecutor(workers) as ex:
        manifests = {k: m for k, m in ex.map(get_manifest, latest.items()) if m}

    template_locs = {m["template"]["location"] for m in manifests.values()}
    annot_locs = {m["annotation_set"]["location"] for m in manifests.values()}
    term_locs = {m["terminology"]["location"] for m in manifests.values()}
    print(
        f"{len(template_locs)} templates, {len(annot_locs)} annotation sets, "
        f"{len(term_locs)} terminologies",
        file=sys.stderr,
    )

    def get_template(loc):
        url = f"{loc_url(loc)}/template.ome.zarr"
        info = inspect_omezarr(url)
        if info and info["levels"]:
            try:
                info["range"] = intensity_range(url, info["levels"][-1])
            except Exception as e:  # noqa: BLE001
                print(f"  ! range failed for {loc}: {e}", file=sys.stderr)
                info["range"] = None
        print(f"  template {loc}", file=sys.stderr, flush=True)
        return loc, info

    def get_annotation(loc):
        base = loc_url(loc)
        ann = inspect_omezarr(f"{base}/annotations_compressed.ome.zarr")
        hemi = inspect_omezarr(f"{base}/hemispheres.ome.zarr")
        pre = fetch_json(f"{base}/annotations.precomputed/info")
        print(f"  annotation set {loc}", file=sys.stderr, flush=True)
        return loc, {
            "annotation": ann,
            "hemispheres": hemi,
            "precomputed": bool(pre),
            "meshes": bool(pre and pre.get("mesh")),
            "segment_properties": bool(pre and pre.get("segment_properties")),
        }

    def get_terminology(loc):
        data = fetch(f"{loc_url(loc)}/terminology.csv")
        if data is None:
            return loc, None
        rows = parse_terminology(data.decode("utf-8-sig"))
        fname = f"{slug(loc)}.json"
        (term_dir / fname).write_text(json.dumps(rows, separators=(",", ":")))
        return loc, {"file": f"terminologies/{fname}", "count": len(rows), "rows": rows}

    # Submit everything up front so slow templates don't serialise the rest.
    with ThreadPoolExecutor(workers) as ex:
        f_t = [ex.submit(get_template, loc) for loc in template_locs]
        f_a = [ex.submit(get_annotation, loc) for loc in annot_locs]
        f_r = [ex.submit(get_terminology, loc) for loc in term_locs]
        templates = dict(f.result() for f in f_t)
        annotations = dict(f.result() for f in f_a)
        terms_full = dict(f.result() for f in f_r)

    # Region-name search index (deduplicated across terminologies).
    terminologies = {}
    for loc, t in terms_full.items():
        if t is None:
            terminologies[loc] = None
            continue
        terminologies[loc] = {
            "file": t["file"],
            "count": t["count"],
            # compact list used for full-text search: "acronym\tname"
            "regions": [f"{r['acronym']}\t{r['name']}" for r in t["rows"]],
        }

    # Docs enrichment
    md = None
    if docs_path:
        md = docs_path.read_text()
    else:
        raw = fetch(DOCS_URL)
        md = raw.decode() if raw else None
    docs = parse_docs(md, set(manifests)) if md else {}
    print(f"{len(docs)} atlases matched to documentation", file=sys.stderr)

    atlases = []
    for key in sorted(manifests):
        m = manifests[key]
        res = [float(r) for r in m.get("resolution", [])]
        t_loc = m["template"]["location"]
        a_loc = m["annotation_set"]["location"]
        tinfo = templates.get(t_loc) or {}
        ainfo = annotations.get(a_loc) or {}
        entry = {
            "key": key,
            "name": m.get("name"),
            "version": m.get("version"),
            "species": m.get("species"),
            "citation": m.get("citation"),
            "atlas_link": m.get("atlas_link"),
            "symmetric": m.get("symmetric"),
            "resolution": res,
            "orientation": m.get("orientation"),
            "shape": m.get("shape"),
            "additional_references": m.get("additional_references") or [],
            "atlas_packager": m.get("atlas_packager"),
            "coordinate_space": m.get("coordinate_space", {}).get("name"),
            "manifest": f"/atlases/{key}/{m['version'].replace('.', '_')}/manifest.json",
            "template": {
                **m["template"],
                "level": match_level(tinfo.get("levels", []), res),
            },
            "annotation_set": {
                "name": m["annotation_set"]["name"],
                "version": m["annotation_set"]["version"],
                "location": a_loc,
                "level": match_level(
                    ((ainfo.get("annotation") or {}).get("levels") or []), res
                ),
                "hemispheres_level": match_level(
                    ((ainfo.get("hemispheres") or {}).get("levels") or []), res
                ),
            },
            "terminology": m["terminology"],
            **({"docs": docs[key]} if key in docs else {}),
        }
        atlases.append(entry)

    # Atlases missing from the docs page: infer the category from species.
    species_group: dict[str, dict[str, int]] = {}
    for a in atlases:
        g = a.get("docs", {}).get("group")
        if g:
            counts = species_group.setdefault(a["species"], {})
            counts[g] = counts.get(g, 0) + 1
    for a in atlases:
        if "docs" not in a and a["species"] in species_group:
            counts = species_group[a["species"]]
            a["docs"] = {"group": max(counts, key=counts.get)}

    index = {
        "generated": dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"),
        "root": ROOT,
        "atlases": atlases,
        "templates": templates,
        "annotation_sets": annotations,
        "terminologies": terminologies,
    }
    (out_dir / "index.json").write_text(json.dumps(index, separators=(",", ":")))
    size = (out_dir / "index.json").stat().st_size / 1e6
    print(f"wrote {out_dir/'index.json'} ({size:.1f} MB, {len(atlases)} atlases)", file=sys.stderr)


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--out", type=Path, default=Path(__file__).parents[1] / "public" / "data")
    p.add_argument("--docs", type=Path, default=None, help="local atlas-details.md (default: fetch from GitHub)")
    p.add_argument("--workers", type=int, default=16)
    args = p.parse_args(argv)
    build(args.out, args.docs, args.workers)


if __name__ == "__main__":
    main()
