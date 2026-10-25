#!/usr/bin/env python
"""Stage the anonymous P2 dataset release and emit its Croissant 1.0 + RAI metadata.

Builds one self-contained, upload-ready folder per released dataset (each E&D
dataset must be hosted separately with its own Croissant file):

    <out-root>/BdSLW60-SI/   BdSLW60-derived MediaPipe poses, labels, splits
    <out-root>/BdSLW401-SI/  BdSLW401-derived MediaPipe poses, labels, splits

Layout of each folder (this is exactly what gets uploaded):

    README.md                     dataset card (license, terms, citations, RAI)
    croissant.json                Croissant 1.0 + RAI metadata (validated)
    SHA256SUMS                    checksums of every released file
    data/<split>_data.npy         (N, 3, 300, 27, 1) float32 pose tensors
    data/<split>_label.pkl        (clip_ids, labels) tuple -- drop-in for feeders/feeder.py
    metadata/clips.csv            one row per clip (Croissant record set "clips")
    metadata/classes.csv          one row per class (Croissant record set "classes")
    splits/*.json                 canonical / LOSO / signer-mixed clip-ID lists

The .npy files are hard-linked (no extra disk); pass --copy to copy instead.
Every released text file is scanned for absolute paths / usernames before the
Croissant file is written, and the build aborts on a hit.

Usage (stdlib only, any Python >= 3.8):

    # local staging + Croissant with relative contentUrl (validates locally):
    python tools/build_p2_release.py --dataset bdslw60 --out-root /path/to/p2_release
    # final, after the anonymous host exists (absolute URLs => online checker can load records):
    python tools/build_p2_release.py --dataset bdslw60 --out-root ... \
        --base-url https://huggingface.co/datasets/<ANON_ACCOUNT>/BdSLW60-SI

Then validate:  mlcroissant validate --jsonld <out-root>/BdSLW60-SI/croissant.json
"""

import argparse
import ast
import csv
import hashlib
import json
import os
import pickle
import re
import shutil
import sys

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PLACEHOLDER_ACCOUNT = "__ANON_ACCOUNT__"  # must be replaced via --base-url before submission
VERSION = "1.0.0"
DATE_PUBLISHED = "2026-10-22"
LICENSE_URL = "https://creativecommons.org/licenses/by-nc-nd/4.0/"
LICENSE_NAME = "CC BY-NC-ND 4.0 (plus the responsible-use terms in README.md)"

# Third-party (source dataset) citations -- required by the BdSLW401 DUA s.9.
CITE_BDSLW60 = (
    "Rubaiyeat, H. A., Mahmud, H., Habib, A., Hasan, M. K. (2025). BdSLW60: A word-level "
    "Bangla sign language dataset. Multimedia Tools and Applications 84(34):42399-42423. "
    "https://doi.org/10.1007/s11042-025-20792-4"
)
CITE_BDSLW401 = (
    "Transformer-based word level Bangla sign language recognition using relative quantization "
    "encoding. Scientific Reports 16:25259 (2026). https://doi.org/10.1038/s41598-026-53835-3"
)
CITE_PLOS = (
    "Shawon, Hasan, Mahmud. A comparative analysis of video vision transformers on word-level "
    "sign language datasets. PLOS ONE 21(2):e0341909 (2026). "
    "https://doi.org/10.1371/journal.pone.0341909"
)
# Source-dataset Kaggle pages (BdSLW60: the slug the download scripts use).
KAGGLE_BDSLW60 = "https://www.kaggle.com/datasets/hasaniut/bdslw60videoclips"
KAGGLE_BDSLW401 = "https://www.kaggle.com/datasets/hasanssl/bdslw401"

# Absolute home/scratch paths, Windows drive paths, repository URLs and the current user name.
_USER = re.escape(os.environ.get("USER", "") or os.environ.get("USERNAME", "") or "@@no-user@@")
LEAK_PATTERNS = re.compile(
    r"(/(?:users|home|scratch)/|[A-Za-z]:\\|github\.com/|\b" + _USER + r"\b)",
    re.IGNORECASE,
)

# ----------------------------------------------------------------------------------------
# Dataset specifications
# ----------------------------------------------------------------------------------------
BDSLW60_RE = re.compile(r"^U(\d+)W(\d+)([A-Z]*)(?:_(\d+))?_trial_(\d+)_([LR])\.mp4$")
BDSLW401_RE = re.compile(r"^W(\d+)S(\d+)([A-Z])_(\d+)\.mp4$")


def parse_bdslw60(name):
    m = BDSLW60_RE.match(name)
    if not m:
        raise ValueError("unparseable BdSLW60 clip id: %s" % name)
    return {"signer_id": int(m.group(1)), "word_id": int(m.group(2)), "view": m.group(3) or "",
            "trial": int(m.group(5))}


def parse_bdslw401(name):
    m = BDSLW401_RE.match(name)
    if not m:
        raise ValueError("unparseable BdSLW401 clip id: %s" % name)
    return {"signer_id": int(m.group(2)), "word_id": int(m.group(1)), "view": m.group(3),
            "trial": int(m.group(4))}


SPECS = {
    "bdslw60": {
        "name": "BdSLW60-SI",
        "bundle": "data/bdsl_si",
        "splits": ["train", "val", "test", "pretrain"],
        "parse": parse_bdslw60,
        "classes": "bdslw60",
        "split_files": {"signer_independent.json": "splits/signer_independent.json"},
        # extra protocol splits exported as clip-ID lists: out_name -> bundle dir
        "extra_bundles": {
            **{"loso/%s.json" % d: "data/bdsl_si_loso/%s" % d for d in [
                "test_U01_val_U02", "test_U02_val_U01", "test_U04_val_U01", "test_U05_val_U01",
                "test_U06_val_U01", "test_U08_val_U01", "test_U09_val_U01", "test_U11_val_U01",
                "test_U12_val_U01", "test_U13_val_U01", "test_U15_val_U01"]},
            **{"signer_mixed/rand%d.json" % s: "data/bdsl_rand%d" % s for s in range(42, 47)},
        },
        "source_url": KAGGLE_BDSLW60,
        "derived_from": [
            ("BdSLW60 (segmented per-trial video clips)", KAGGLE_BDSLW60, CITE_BDSLW60),
        ],
        "cite": [CITE_BDSLW60, CITE_PLOS],
    },
    "bdslw401": {
        "name": "BdSLW401-SI",
        "bundle": "data/bdslw401_si_sdval",
        "splits": ["train", "val", "test"],
        "parse": parse_bdslw401,
        "classes": "bdslw401",
        "split_files": {"bdslw401_sdval.json": "splits/bdslw401_sdval.json"},
        "extra_bundles": {
            "official_seen_signer_val.json": "data/bdslw401_si",
            "signer_mixed/sd.json": "data/bdslw401_sd",
        },
        "source_url": KAGGLE_BDSLW401,
        "derived_from": [
            ("BdSLW401 (front-view video clips; used under a signed data use agreement)",
             KAGGLE_BDSLW401, CITE_BDSLW401),
        ],
        "cite": [CITE_BDSLW401, CITE_BDSLW60, CITE_PLOS],
    },
}


# ----------------------------------------------------------------------------------------
# Helpers
# ----------------------------------------------------------------------------------------
def sha256(path, buf=1 << 24):
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        while True:
            b = fh.read(buf)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def npy_header(path):
    with open(path, "rb") as fh:
        magic = fh.read(6)
        if magic != b"\x93NUMPY":
            raise ValueError("not an npy file: %s" % path)
        major = fh.read(1)[0]
        fh.read(1)
        hlen = int.from_bytes(fh.read(2 if major == 1 else 4), "little")
        return ast.literal_eval(fh.read(hlen).decode("latin1"))


def load_label_pkl(path):
    with open(path, "rb") as fh:
        obj = pickle.load(fh)
    names, labels = obj
    names = [str(n) for n in names]
    labels = [int(l) for l in labels]
    if len(names) != len(labels):
        raise ValueError("length mismatch in %s" % path)
    return names, labels


def place(src, dst, copy):
    os.makedirs(os.path.dirname(dst), exist_ok=True)
    if os.path.lexists(dst):
        os.remove(dst)
    if copy:
        shutil.copy2(src, dst)
    else:
        try:
            os.link(os.path.realpath(src), dst)
        except OSError:
            shutil.copy2(src, dst)


def load_classes(kind):
    """Return list of (label, word_id, class_name) rows."""
    if kind == "bdslw60":
        cls = json.load(open(os.path.join(REPO, "data/bdsl_si/classes.json")))["classes"]
        c2w = json.load(open(os.path.join(REPO, "splits/word_id_mapping.json")))["class_to_word_id"]
        return [(i, int(c2w[c]), c) for i, c in enumerate(cls)]
    rom = json.load(open(os.path.join(REPO, "data/bdslw401_si_sdval/classes_romanized.json")))
    return [(i, int(w.lstrip("W")), c) for i, (c, w) in
            enumerate(zip(rom["classes"], rom["original_class_ids"]))]


# ----------------------------------------------------------------------------------------
# Build
# ----------------------------------------------------------------------------------------
def build(key, out_root, base_url, copy, skip_data):
    spec = SPECS[key]
    out = os.path.join(out_root, spec["name"])
    os.makedirs(out, exist_ok=True)
    classes = load_classes(spec["classes"])
    n_classes = len(classes)
    label_to_word = {l: w for l, w, _ in classes}

    # ---- pose tensors + label pickles + clips index
    clip_rows, split_counts, universe = [], {}, set()
    npy_meta = {}
    for s in spec["splits"]:
        npy_src = os.path.join(REPO, spec["bundle"], "%s_data.npy" % s)
        pkl_src = os.path.join(REPO, spec["bundle"], "%s_label.pkl" % s)
        hdr = npy_header(npy_src)
        names, labels = load_label_pkl(pkl_src)
        if hdr["shape"][0] != len(names) or tuple(hdr["shape"][1:]) != (3, 300, 27, 1) \
                or hdr["descr"] != "<f4":
            raise ValueError("unexpected npy header %s for %s" % (hdr, npy_src))
        npy_meta[s] = hdr
        for row, (n, l) in enumerate(zip(names, labels)):
            meta = spec["parse"](n)
            if label_to_word[l] != meta["word_id"]:
                raise ValueError("label/word-id mismatch for %s" % n)
            if n in universe:
                raise ValueError("duplicate clip id across splits: %s" % n)
            universe.add(n)
            clip_rows.append({"clip_id": n, "split": s, "signer_id": meta["signer_id"],
                              "word_id": meta["word_id"], "label": l, "trial": meta["trial"],
                              "pose_file": "data/%s_data.npy" % s, "pose_row": row})
        split_counts[s] = len(names)
        if not skip_data:
            place(npy_src, os.path.join(out, "data", "%s_data.npy" % s), copy)
            place(pkl_src, os.path.join(out, "data", "%s_label.pkl" % s), True)

    signers = {s: sorted({r["signer_id"] for r in clip_rows if r["split"] == s}) for s in spec["splits"]}
    for a in spec["splits"]:
        for b in spec["splits"]:
            if a < b and set(signers[a]) & set(signers[b]):
                raise ValueError("signer overlap between %s and %s" % (a, b))

    os.makedirs(os.path.join(out, "metadata"), exist_ok=True)
    fields = ["clip_id", "split", "signer_id", "word_id", "label", "trial", "pose_file", "pose_row"]
    with open(os.path.join(out, "metadata", "clips.csv"), "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=fields)
        w.writeheader()
        w.writerows(clip_rows)
    with open(os.path.join(out, "metadata", "classes.csv"), "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["label", "word_id", "class_name"])
        w.writerows(classes)

    # ---- split definitions
    os.makedirs(os.path.join(out, "splits"), exist_ok=True)
    split_files = []
    for dst, src in spec["split_files"].items():
        place(os.path.join(REPO, src), os.path.join(out, "splits", dst), True)
        split_files.append("splits/" + dst)
    for dst, bundle in spec["extra_bundles"].items():
        bdir = os.path.join(REPO, bundle)
        if not os.path.isdir(bdir):
            print("  [warn] missing bundle %s -- skipped %s" % (bundle, dst))
            continue
        doc = {"protocol": dst[:-5], "derived_from_bundle": os.path.basename(bundle),
               "clip_ids": {}, "counts": {}, "signers": {}}
        for s in ["train", "val", "test"]:
            names, _ = load_label_pkl(os.path.join(bdir, "%s_label.pkl" % s))
            # Clip ids absent from the released pose bundle (e.g. clips excluded from the
            # canonical bundle) are kept but reported, so nothing is silently dropped.
            doc["clip_ids"][s] = names
            doc["counts"][s] = len(names)
            doc["signers"][s] = sorted({spec["parse"](n)["signer_id"] for n in names})
        missing = sum(1 for s in doc["clip_ids"] for n in doc["clip_ids"][s] if n not in universe)
        doc["clips_not_in_released_bundle"] = missing
        p = os.path.join(out, "splits", dst)
        os.makedirs(os.path.dirname(p), exist_ok=True)
        with open(p, "w", encoding="utf-8") as fh:
            json.dump(doc, fh, indent=1)
        split_files.append("splits/" + dst)

    # ---- leak scan of every text/pickle file we ship
    for root, _, files in os.walk(out):
        for f in files:
            p = os.path.join(root, f)
            if f.endswith((".npy",)) or f in ("croissant.json", "SHA256SUMS", "README.md"):
                continue
            with open(p, "rb") as fh:
                txt = fh.read().decode("latin1")
            hit = LEAK_PATTERNS.search(txt)
            if hit:
                raise SystemExit("LEAK: %r found in %s" % (hit.group(0), p))

    # ---- croissant
    def url_of(rel):
        if base_url:
            return "%s/resolve/main/%s" % (base_url.rstrip("/"), rel)
        return rel

    released = (["data/%s_data.npy" % s for s in spec["splits"]]
                + ["data/%s_label.pkl" % s for s in spec["splits"]]
                + ["metadata/clips.csv", "metadata/classes.csv"] + split_files)
    hashes = {}
    for rel in released:
        p = os.path.join(out, rel)
        hashes[rel] = (sha256(p), os.path.getsize(p))
    with open(os.path.join(out, "SHA256SUMS"), "w") as fh:
        for rel in released:
            fh.write("%s  %s\n" % (hashes[rel][0], rel))

    cr = croissant(spec, key, base_url, url_of, hashes, split_counts, signers, n_classes,
                   split_files)
    write_readme(out, spec, key, cr, hashes, split_counts, signers, n_classes)
    with open(os.path.join(out, "croissant.json"), "w", encoding="utf-8") as fh:
        json.dump(cr, fh, indent=2, ensure_ascii=False)
    total = sum(v[1] for v in hashes.values())
    print("%s: %d clips %s, %d classes, %.2f GB -> %s" % (
        spec["name"], len(clip_rows), split_counts, n_classes, total / 1e9, out))
    return out, cr


def write_readme(out, spec, key, cr, hashes, split_counts, signers, n_classes):
    """Dataset card (Hugging Face YAML front matter + human-readable terms)."""
    total = sum(split_counts.values())
    size_cat = "1K<n<10K" if total < 10000 else "10K<n<100K"
    lines = [
        "---",
        "license: cc-by-nc-nd-4.0",
        "pretty_name: %s" % spec["name"],
        "task_categories:",
        "- video-classification",
        "tags:",
        "- sign-language",
        "- bangla-sign-language",
        "- pose",
        "- skeleton",
        "- signer-independent",
        "size_categories:",
        "- %s" % size_cat,
        "configs:",
        "- config_name: clips",
        "  data_files: metadata/clips.csv",
        "  default: true",
        "- config_name: classes",
        "  data_files: metadata/classes.csv",
        "---",
        "",
        "# %s" % spec["name"],
        "",
        cr["description"],
        "",
        "Anonymous release for double-blind review. Machine-readable metadata (Croissant 1.0 + "
        "Responsible-AI fields): `croissant.json`.",
        "",
        "## Files",
        "",
        "| path | size (bytes) | sha256 |",
        "|---|---:|---|",
    ]
    for rel, (h, n) in hashes.items():
        if rel.startswith("splits/loso/") or rel.startswith("splits/signer_mixed/"):
            continue
        lines.append("| `%s` | %d | `%s...` |" % (rel, n, h[:16]))
    lines += [
        "| `splits/loso/*.json`, `splits/signer_mixed/*.json` | see SHA256SUMS | see SHA256SUMS |"
        if key == "bdslw60" else
        "| `splits/signer_mixed/*.json` | see SHA256SUMS | see SHA256SUMS |",
        "",
        "Pose tensor layout: `(N, C=3, T=300, V=27, M=1)` float32; C = (x, y, visibility) in "
        "MediaPipe normalized image coordinates (visibility is 0 for hand joints); frames are "
        "zero-padded after the clip ends and missed detections are zeros. Row *i* of "
        "`data/<split>_data.npy` is the clip with `split=<split>, pose_row=i` in "
        "`metadata/clips.csv`.",
        "",
        "```python",
        "import numpy as np, pandas as pd",
        "clips = pd.read_csv('metadata/clips.csv')",
        "X = np.load('data/test_data.npy', mmap_mode='r')",
        "y = clips[clips.split == 'test'].sort_values('pose_row').label.values",
        "```",
        "",
        "## License and terms of use",
        "",
        "Released under **CC BY-NC-ND 4.0** (non-commercial), matching the source dataset "
        "license" + (" and the data use agreement under which BdSLW401 was obtained, which "
                     "permits non-commercial release of derived pose extractions"
                     if key == "bdslw401" else "") + ". In addition, by using the data you "
        "agree to the source owners' terms:",
        "",
        "- no attempt to re-identify or contact any signer;",
        "- no military, surveillance, biometric-identification, defamatory or otherwise "
        "harmful use;",
        "- no redistribution of the source videos (not included here);",
        "- the source dataset remains the property of its creators; this derived release will "
        "be withdrawn if they revoke permission.",
        "",
        "## Re-identification risk",
        "",
        cr["rai:personalSensitiveInformation"],
        "",
        "## Required citations (source datasets)",
        "",
    ]
    lines += ["- " + c for c in spec["cite"]]
    lines += [
        "",
        "Source dataset page: %s" % spec["source_url"],
        "",
        "## Responsible-AI summary",
        "",
        "- **Limitations:** " + " ".join(cr["rai:dataLimitations"]),
        "- **Biases:** " + cr["rai:dataBiases"],
        "- **Intended uses:** " + " ".join(cr["rai:dataUseCases"]),
        "- **Social impact:** " + cr["rai:dataSocialImpact"],
        "- **Synthetic data:** none.",
        "",
    ]
    with open(os.path.join(out, "README.md"), "w", encoding="utf-8") as fh:
        fh.write("\n".join(lines))


def rai_text(key, split_counts, signers):
    n_sign = len({x for v in signers.values() for x in v})
    if key == "bdslw60":
        leak = ("Signer identity is recoverable from the released poses. Under a recording-grouped "
                "probe on held-out signers, the raw pose features themselves give a "
                "class-conditioned normalized signer-probe score L_norm = 0.52-0.57 and trained "
                "encoders 0.23-0.35; within a class, signer identity is near-perfectly "
                "recoverable from raw poses (L_norm^wc 0.96-0.98) (0 = chance, 1 = perfect; "
                "paper, Sec. 'Signer decodability').")
        bias = ("18 signers only; 11 recorded the full 60-word vocabulary and the other 7 are "
                "routed to an unlabeled pretraining pool (fewer classes), so the labeled splits "
                "rest on 11 people and the canonical test set on 2 of them. Single recording "
                "setup and front view; signer demographics (age, gender, region, hearing status, "
                "handedness) are not documented in the source release and cannot be balanced or "
                "audited. Vocabulary is 60 everyday nouns/kinship/object words, not continuous "
                "signing. MediaPipe detection quality varies with hand pose and occlusion: hand "
                "keypoints are missing (zero) in a substantial fraction of frames, which can bias "
                "models toward body motion.")
    else:
        leak = ("Signer identity is recoverable from the released poses. On BdSLW401, the "
                "class-conditioned signer probe on frozen supervised encoders gives L_norm = "
                "0.63-0.72 and within-class L_norm = 0.99 under the earlier ungrouped protocol, "
                "an upper bound (paper, Sec. 'Signer decodability').")
        bias = ("18 signers; 2 held-out test signers (S04/S08) and 1 validation signer (S15), so "
                "selection rests on one person. Front view only (the lateral view described in "
                "the source paper is not in the public distribution). Several signers have "
                "partial vocabulary coverage. Signer demographics are not documented in the "
                "source release. MediaPipe hand detection is imperfect (zeros for missed hands).")
    return {
        "rai:dataCollection": (
            "No new human-subject data were collected. Source videos were recorded and released "
            "by the original dataset authors (see prov:wasDerivedFrom); we ran MediaPipe Holistic "
            "(model_complexity=1, detection/tracking confidence 0.5) on every clip and kept 27 "
            "keypoints (7 upper-body pose landmarks: nose, shoulders, elbows, wrists; 10 per hand: "
            "wrist, thumb tip, and base+tip of index, middle, ring and little fingers)."),
        "rai:dataCollectionType": ["Secondary Data Analysis", "Software Collection"],
        "rai:dataCollectionRawData": (
            "Raw RGB video is NOT redistributed. It must be obtained from the source dataset "
            "under its own license/terms."),
        "rai:dataCollectionMissingData": (
            "Frames where MediaPipe misses a body part are zero-filled for that part; tensors are "
            "zero-padded to T=300 frames. A clip that fails extraction is logged and excluded, never "
            "dropped silently; no extraction failures are recorded for this release."),
        "rai:dataPreprocessingProtocol": [
            "Per frame: x, y (MediaPipe normalized image coordinates in [0,1]) and channel 2 = "
            "MediaPipe visibility for the 7 body joints (always 0 for hand joints).",
            "Tensor layout (N, C=3, T=300, V=27, M=1), float32, zero-padded after the last frame.",
            "No relabeling: label = source word id mapped to a contiguous 0-based index "
            "(metadata/classes.csv).",
            "Splits are signer-disjoint by construction; see splits/*.json.",
        ],
        "rai:dataAnnotationProtocol": (
            "Labels are inherited from the source dataset (one isolated sign word per clip, from "
            "the filename word id). No new annotation was performed."),
        "rai:dataUseCases": [
            "Intended: signer-independent isolated sign language recognition benchmarking.",
            "Intended: measuring the Signer-Mixing Gap and signer-identity shortcut/leakage "
            "diagnostics; low-resource and cross-lingual pose self-supervised learning research.",
            "Validated only for: isolated word recognition on the provided signer-disjoint "
            "protocols. Not validated for continuous signing, translation, or deployment as an "
            "assistive/communication product.",
            "Out of scope / prohibited: commercial use; re-identification of, or attempts to "
            "contact, the signers; surveillance, military, or biometric identification uses; any "
            "use barred by the source dataset terms.",
            "Discouraged: reporting signer-mixed (random-split) accuracy as a headline number.",
        ],
        "rai:dataLimitations": [
            "Only %d signers; the canonical held-out test set contains %d signer(s), so single-"
            "split accuracies are high-variance (use the LOSO splits and report the "
            "distribution over unseen signers)." % (n_sign, len(signers.get("test", []))),
            "Poses are an automatic MediaPipe estimate, not ground truth; no facial landmarks "
            "(non-manual markers) are included.",
            "Front view, single recording environment; generalization to other cameras, "
            "backgrounds, dialects or signing styles is untested.",
        ],
        "rai:dataBiases": bias,
        "rai:personalSensitiveInformation": (
            "The data are derived from video of identifiable people (sign language users). The "
            "release contains no pixels, faces, names or demographic fields: only 27-keypoint "
            "skeleton trajectories plus an integer signer id taken from the source filenames. "
            "Body proportions and motion style are nevertheless person-specific (biometric-like). "
            "RE-IDENTIFICATION RISK: " + leak + " This enables linking clips to the same "
            "(pseudonymous) signer; linking to a real-world identity would additionally require "
            "reference recordings of that person. Users must not attempt it. Consent and ethics "
            "approval for the original recordings were handled by the source dataset authors "
            "(see their publications)."),
        "rai:dataSocialImpact": (
            "Positive: more trustworthy evaluation of Bangla sign recognition, which matters for "
            "a large, under-resourced Deaf community; exposes inflated random-split claims. "
            "Risks: (i) re-identification of signers from motion (see "
            "rai:personalSensitiveInformation); (ii) over-trust in systems trained on 18 signers "
            "if deployed without community involvement; (iii) misuse for surveillance. "
            "Mitigations: non-commercial license with responsible-use terms prohibiting "
            "re-identification/surveillance, no raw video, documentation of limitations."),
        "rai:hasSyntheticData": False,
        "rai:dataReleaseMaintenancePlan": (
            "Versioned (semantic version + SHA-256 per file). Errata are fixed by a new version; "
            "earlier versions remain identifiable by checksum. Source datasets remain owned by "
            "their authors; derived artifacts will be withdrawn if the source owners revoke "
            "permission."),
    }


def croissant(spec, key, base_url, url_of, hashes, split_counts, signers, n_classes,
              split_files):
    name = spec["name"]
    ds_url = base_url or "https://huggingface.co/datasets/%s/%s" % (PLACEHOLDER_ACCOUNT, name)
    ctx = {
        "@language": "en", "@vocab": "https://schema.org/",
        "citeAs": "cr:citeAs", "column": "cr:column", "conformsTo": "dct:conformsTo",
        "cr": "http://mlcommons.org/croissant/", "rai": "http://mlcommons.org/croissant/RAI/",
        "prov": "http://www.w3.org/ns/prov#",
        "data": {"@id": "cr:data", "@type": "@json"},
        "dataType": {"@id": "cr:dataType", "@type": "@vocab"},
        "dct": "http://purl.org/dc/terms/",
        "equivalentProperty": "cr:equivalentProperty",
        "examples": {"@id": "cr:examples", "@type": "@json"},
        "extract": "cr:extract", "field": "cr:field", "fileProperty": "cr:fileProperty",
        "fileObject": "cr:fileObject", "fileSet": "cr:fileSet", "format": "cr:format",
        "includes": "cr:includes", "isLiveDataset": "cr:isLiveDataset", "jsonPath": "cr:jsonPath",
        "key": "cr:key", "md5": "cr:md5", "parentField": "cr:parentField", "path": "cr:path",
        "recordSet": "cr:recordSet", "references": "cr:references", "regex": "cr:regex",
        "repeated": "cr:repeated", "replace": "cr:replace", "samplingRate": "cr:samplingRate",
        "sc": "https://schema.org/", "separator": "cr:separator", "source": "cr:source",
        "subField": "cr:subField", "transform": "cr:transform",
    }
    total = sum(split_counts.values())
    shape_txt = "(3, 300, 27, 1) float32"
    if key == "bdslw60":
        desc = (
            "Signer-independent (SI) benchmark for word-level Bangla Sign Language recognition "
            "derived from BdSLW60: %d per-trial clips, 60 sign classes, 18 signers, released as "
            "27-keypoint MediaPipe pose sequences (no raw video). Canonical signer-disjoint "
            "split: train signers %s / val %s / test %s (labeled) plus an unlabeled pretraining "
            "pool of signers %s whose clips carry source labels that the protocol does not use. "
            "Also ships 11 leave-one-signer-out folds and 5 matched-size signer-mixed splits for "
            "measuring the Signer-Mixing Gap. Counts: %s."
            % (total, signers["train"], signers["val"], signers["test"], signers["pretrain"],
               split_counts))
        keywords = ["sign language recognition", "signer-independent", "Bangla Sign Language",
                    "BdSLW60", "pose", "skeleton", "MediaPipe", "shortcut learning",
                    "low-resource"]
    else:
        desc = (
            "Signer-independent split of BdSLW401 (401 Bangla sign words, front view) released "
            "as 27-keypoint MediaPipe pose sequences (no raw video): %d clips. Held-out test "
            "signers S04/S08 (the source dataset's own signer-disjoint test), validation on a "
            "held-out training signer S15, train on the remaining 15 signers. Also ships the "
            "source's original seen-signer validation split and a signer-mixed split. "
            "Derived poses are released under the source data use agreement (non-commercial). "
            "Counts: %s." % (total, split_counts))
        keywords = ["sign language recognition", "signer-independent", "Bangla Sign Language",
                    "BdSLW401", "pose", "skeleton", "MediaPipe", "large vocabulary"]

    dist = []
    for s in spec["splits"]:
        rel = "data/%s_data.npy" % s
        dist.append({
            "@type": "cr:FileObject", "@id": "%s_data.npy" % s, "name": "%s_data.npy" % s,
            "description": "Pose tensors of the '%s' split, shape (%d, 3, 300, 27, 1), float32 "
                           "(N, C=[x, y, visibility], T, V=27 joints, M=1 person). Row i is the "
                           "clip in metadata/clips.csv with split=%s and pose_row=i."
                           % (s, split_counts[s], s),
            "contentUrl": url_of(rel), "encodingFormat": "application/x-npy",
            "sha256": hashes[rel][0], "contentSize": "%d B" % hashes[rel][1]})
        rel = "data/%s_label.pkl" % s
        dist.append({
            "@type": "cr:FileObject", "@id": "%s_label.pkl" % s, "name": "%s_label.pkl" % s,
            "description": "Python pickle of a tuple (clip_ids: list[str], labels: list[int]) "
                           "aligned with %s_data.npy; drop-in input for the benchmark harness. "
                           "Same content as metadata/clips.csv." % s,
            "contentUrl": url_of(rel), "encodingFormat": "application/octet-stream",
            "sha256": hashes[rel][0], "contentSize": "%d B" % hashes[rel][1]})
    for rel, fid, d in [("metadata/clips.csv", "clips.csv", "One row per clip: id, split, signer, "
                         "word id, label, trial, and the location of its pose tensor."),
                        ("metadata/classes.csv", "classes.csv", "One row per sign class.")]:
        dist.append({"@type": "cr:FileObject", "@id": fid, "name": fid, "description": d,
                     "contentUrl": url_of(rel), "encodingFormat": "text/csv",
                     "sha256": hashes[rel][0], "contentSize": "%d B" % hashes[rel][1]})
    for rel in split_files:
        fid = rel.replace("splits/", "split:")
        dist.append({"@type": "cr:FileObject", "@id": fid, "name": rel,
                     "description": "Split definition (signer lists and/or clip-ID lists).",
                     "contentUrl": url_of(rel), "encodingFormat": "application/json",
                     "sha256": hashes[rel][0], "contentSize": "%d B" % hashes[rel][1]})

    def fld(rs, col, dtype, desc, fo, ref=None):
        f = {"@type": "cr:Field", "@id": "%s/%s" % (rs, col), "name": col, "description": desc,
             "dataType": dtype, "source": {"fileObject": {"@id": fo}, "extract": {"column": col}}}
        if ref:
            f["references"] = {"field": {"@id": ref}}
        return f

    record_sets = [
        {"@type": "cr:RecordSet", "@id": "classes", "name": "classes",
         "description": "Sign vocabulary (%d classes)." % n_classes, "key": {"@id": "classes/label"},
         "field": [
             fld("classes", "label", "sc:Integer", "Contiguous 0-based class index used as the "
                 "training target.", "classes.csv"),
             fld("classes", "word_id", "sc:Integer", "Word id in the source dataset filenames.",
                 "classes.csv"),
             fld("classes", "class_name", "sc:Text", "Romanized Bangla gloss.", "classes.csv"),
         ]},
        {"@type": "cr:RecordSet", "@id": "clips", "name": "clips",
         "description": "One record per clip. The pose tensor of a clip is row `pose_row` of "
                        "`pose_file` (an .npy FileObject in this distribution), shape %s." % shape_txt,
         "key": {"@id": "clips/clip_id"},
         "field": [
             fld("clips", "clip_id", "sc:Text", "Source clip filename (pseudonymous; encodes "
                 "signer id, word id and trial).", "clips.csv"),
             fld("clips", "split", "sc:Text", "Canonical split: %s." % " | ".join(spec["splits"]),
                 "clips.csv"),
             fld("clips", "signer_id", "sc:Integer", "Pseudonymous integer signer id from the "
                 "source filenames; splits are signer-disjoint.", "clips.csv"),
             fld("clips", "word_id", "sc:Integer", "Source word id.", "clips.csv"),
             fld("clips", "label", "sc:Integer", "Class index in [0, %d]." % (n_classes - 1),
                 "clips.csv", ref="classes/label"),
             fld("clips", "trial", "sc:Integer", "Repetition index within signer and word.",
                 "clips.csv"),
             fld("clips", "pose_file", "sc:Text", "Relative path of the .npy file holding this "
                 "clip's pose tensor.", "clips.csv"),
             fld("clips", "pose_row", "sc:Integer", "Row index into pose_file.", "clips.csv"),
         ]},
    ]

    cr = {
        "@context": ctx,
        "@type": "sc:Dataset",
        "conformsTo": ["http://mlcommons.org/croissant/1.0", "http://mlcommons.org/croissant/RAI/1.0"],
        "name": name,
        "description": desc,
        "license": LICENSE_URL,
        "url": ds_url,
        "version": VERSION,
        "datePublished": DATE_PUBLISHED,
        "creator": {"@type": "sc:Organization", "name": "Anonymous (withheld for double-blind review)"},
        "keywords": keywords,
        "inLanguage": "bn",
        "isAccessibleForFree": True,
        "citeAs": " ".join(spec["cite"]),
        "sdLicense": LICENSE_URL,
        "prov:wasDerivedFrom": [
            {"@id": u, "name": n, "description": "Source dataset. Cite as: " + c} for n, u, c in spec["derived_from"]
        ],
        "prov:wasGeneratedBy": [
            {"@type": "prov:Activity", "name": "Pose extraction",
             "description": "MediaPipe Holistic (legacy mp.solutions.holistic API, mediapipe==0.10.9 "
                            "pinned; model_complexity=1) run on every source clip; 27 "
                            "keypoints kept; per-clip cache. Code: "
                            "preprocessing/preprocess_bdsl.py in the anonymized code repository."},
            {"@type": "prov:Activity", "name": "Signer-disjoint split construction and bundling",
             "description": "Clips assigned to splits by signer id only (see splits/*.json), then "
                            "zero-padded to T=300 and bundled per split. Code: "
                            "preprocessing/generate_signer_split_npy.py, "
                            "preprocessing/bundle_bdslw401_pose_to_npy.py, "
                            "preprocessing/make_w401_sdval_bundle.py, "
                            "preprocessing/generate_loso_split_bundle.py."},
        ],
        "distribution": dist,
        "recordSet": record_sets,
    }
    cr.update(rai_text(key, split_counts, signers))
    return cr


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dataset", choices=sorted(SPECS) + ["all"], default="all")
    ap.add_argument("--out-root", required=True)
    ap.add_argument("--base-url", default=None,
                    help="Final dataset URL on the host (HF: https://huggingface.co/datasets/<acct>/<name>). "
                         "With --dataset all, '{name}' is substituted.")
    ap.add_argument("--copy", action="store_true", help="copy .npy instead of hard-linking")
    ap.add_argument("--metadata-out", default=None,
                    help="also write croissant.json to <dir>/croissant_<name>.json")
    ap.add_argument("--skip-data", action="store_true",
                    help="reuse already-staged data/ files (still re-hashes them)")
    args = ap.parse_args()
    keys = sorted(SPECS) if args.dataset == "all" else [args.dataset]
    for k in keys:
        base = args.base_url.replace("{name}", SPECS[k]["name"]) if args.base_url else None
        out, cr = build(k, args.out_root, base, args.copy, args.skip_data)
        if args.metadata_out:
            os.makedirs(args.metadata_out, exist_ok=True)
            p = os.path.join(args.metadata_out, "croissant_%s.json" % SPECS[k]["name"].lower().replace("-", "_"))
            with open(p, "w", encoding="utf-8") as fh:
                json.dump(cr, fh, indent=2, ensure_ascii=False)
            print("  wrote", p)


if __name__ == "__main__":
    sys.exit(main())
