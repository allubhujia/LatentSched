"""
trace_encoder.py — Converts raw JSON traces into fixed-length tensors for VAE training.

Output: (num_traces, SEQ_LEN, NUM_FEATURES) float32 arrays + padding masks.
"""

import json
import glob
import os
import math
import numpy as np
from typing import Tuple, List, Dict

# ── Constants ─────────────────────────────────────────────────────────────────

SEQ_LEN = 300          # fixed sequence length (pad / truncate every trace to this)
NUM_FEATURES = 5       # features per event step

# Vocabulary: maps event_type strings → integer IDs.
# ID 0 is reserved for PADDING so it is never assigned to a real event type.
EVENT_TYPE_VOCAB: Dict[str, int] = {
    "<PAD>":                0,
    "DEPOSIT_START":        1,
    "DEPOSIT_END":          2,
    "WITHDRAW_START":       3,
    "WITHDRAW_END":         4,
    "LOCK_ACQUIRE_ATTEMPT": 5,
    "LOCK_ACQUIRED":        6,
    "LOCK_RELEASED":        7,
    "READ_BALANCE":         8,
    "WRITE_BALANCE":        9,
    "DEADLOCK_DETECTED":   10,
    "TRANSFER_COMPLETE":   11,
}

NUM_EVENT_TYPES = len(EVENT_TYPE_VOCAB)   # 12 (including <PAD>)


# ── Per‑trace encoding ───────────────────────────────────────────────────────

def _build_thread_index(events: list) -> Dict[str, int]:
    """
    Assign a relative thread index (1, 2, 3 …) based on the order each
    unique thread name first appears in the trace.  ``MainThread`` always
    gets index 0.

    This makes the encoding invariant to run‑specific naming like
    ``Safe-Run01-T1`` vs ``Race-Run42-T3``.
    """
    mapping: Dict[str, int] = {}
    next_idx = 1              # 0 is reserved for MainThread / padding
    for evt in events:
        name = evt["thread_name"]
        if name not in mapping:
            if "main" in name.lower():
                mapping[name] = 0
            else:
                mapping[name] = next_idx
                next_idx += 1
    return mapping


def _safe_log_value(val: float) -> float:
    """
    Log‑scale a balance / amount value.
    Uses log1p(abs(val)) to handle zero gracefully and preserves sign.
    """
    if val is None or val == 0:
        return 0.0
    sign = 1.0 if val > 0 else -1.0
    return sign * math.log1p(abs(val))


def encode_trace(events: list) -> Tuple[np.ndarray, np.ndarray]:
    """
    Encode a single trace's event list into a fixed‑length feature array.

    Parameters
    ----------
    events : list[dict]
        The ``"events"`` list from one trace JSON file.

    Returns
    -------
    features : np.ndarray, shape ``(SEQ_LEN, NUM_FEATURES)``, dtype float32
        Columns: [event_type_id, thread_index, value_norm, has_value, delta_time_norm]
    mask : np.ndarray, shape ``(SEQ_LEN,)``, dtype bool
        True for positions that contain real events, False for padding.
    """
    n_events = len(events)
    actual_len = min(n_events, SEQ_LEN)

    # Pre‑allocate output (zeros = padding by default)
    features = np.zeros((SEQ_LEN, NUM_FEATURES), dtype=np.float32)
    mask = np.zeros(SEQ_LEN, dtype=bool)

    # Build per‑trace thread index
    thread_map = _build_thread_index(events[:actual_len])

    # Compute delta‑times (nanoseconds between consecutive events)
    timestamps = [evt["timestamp_ns"] for evt in events[:actual_len]]
    delta_ns = [0.0]  # first event has zero delta
    for i in range(1, actual_len):
        delta_ns.append(float(timestamps[i] - timestamps[i - 1]))

    # Log‑scale delta times (ns → log1p of ms)
    delta_log = [math.log1p(max(d, 0.0) / 1e6) for d in delta_ns]

    # Normalise delta times to [0, 1] within this trace
    max_delta = max(delta_log) if delta_log else 1.0
    if max_delta == 0:
        max_delta = 1.0

    # Fill features for each real event
    for i in range(actual_len):
        evt = events[i]

        # Feature 0: event type ID
        etype = evt["event_type"]
        features[i, 0] = float(EVENT_TYPE_VOCAB.get(etype, 0))

        # Feature 1: relative thread index
        features[i, 1] = float(thread_map.get(evt["thread_name"], 0))

        # Feature 2: log‑scaled value
        raw_val = evt.get("value")
        features[i, 2] = _safe_log_value(raw_val)

        # Feature 3: has_value binary flag
        features[i, 3] = 0.0 if raw_val is None else 1.0

        # Feature 4: normalised delta time
        features[i, 4] = delta_log[i] / max_delta

        # Mark as real
        mask[i] = True

    return features, mask


# ── Batch loader ──────────────────────────────────────────────────────────────

def load_all_traces(
    traces_base: str,
    variants: Tuple[str, ...] = ("safe", "race_condition", "deadlock"),
) -> Tuple[np.ndarray, np.ndarray, List[str]]:
    """
    Load every trace JSON from the given base directory, encode them, and
    return stacked arrays.

    Parameters
    ----------
    traces_base : str
        Path to the ``trace_harness/traces/`` directory.
    variants : tuple of str
        Sub‑folder names to scan.

    Returns
    -------
    all_features : np.ndarray, shape ``(N, SEQ_LEN, NUM_FEATURES)``
    all_masks    : np.ndarray, shape ``(N, SEQ_LEN)``
    all_labels   : list[str], length ``N`` — the variant name for each trace
                   (kept for later evaluation, NOT used during VAE training).
    """
    features_list: List[np.ndarray] = []
    masks_list: List[np.ndarray] = []
    labels_list: List[str] = []

    for variant in variants:
        folder = os.path.join(traces_base, variant)
        files = sorted(glob.glob(os.path.join(folder, "*.json")))

        if not files:
            print(f"  [WARN] No JSON files found in {folder}")
            continue

        for fpath in files:
            with open(fpath, "r", encoding="utf-8") as fp:
                data = json.load(fp)

            events = data.get("events", [])
            feat, msk = encode_trace(events)
            features_list.append(feat)
            masks_list.append(msk)
            labels_list.append(variant)

    all_features = np.stack(features_list, axis=0)   # (N, SEQ_LEN, NUM_FEATURES)
    all_masks = np.stack(masks_list, axis=0)          # (N, SEQ_LEN)

    print(f"  Loaded {all_features.shape[0]} traces  →  shape {all_features.shape}")
    print(f"  Variants: { {v: labels_list.count(v) for v in variants} }")

    return all_features, all_masks, labels_list
