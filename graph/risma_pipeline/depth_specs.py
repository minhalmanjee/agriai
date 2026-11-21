"""Shared depth label / cm specs for Stage0 multi-depth schemas + loaders."""

from __future__ import annotations

from typing import Any

# (depth_cm for graph geometry, column label without trailing "cm")
DEFAULT_DEPTH_SPECS: tuple[tuple[float, str], ...] = (
    (2.5, "0to5"),
    (5.0, "5"),
    (20.0, "20"),
    (50.0, "50"),
    (100.0, "100"),
    (150.0, "150"),
)

# Config / origins key → column label
DEPTH_KEY_TO_LABEL = {
    "0_to_5": "0to5",
    "5": "5",
    "20": "20",
    "50": "50",
    "100": "100",
    "150": "150",
}

LABEL_TO_DEPTH_KEY = {v: k for k, v in DEPTH_KEY_TO_LABEL.items()}


def schema_depth_specs(schema: dict[str, Any] | None = None) -> list[tuple[float, str]]:
    """Return [(cm, label), ...] from schema depth_specs or depths_cm fallback."""
    if schema is None:
        return list(DEFAULT_DEPTH_SPECS)
    cols = schema.get("columns", {}).get("per_sensor_per_day", {})
    specs = cols.get("depth_specs")
    if specs:
        out: list[tuple[float, str]] = []
        for s in specs:
            out.append((float(s["cm"]), str(s["label"])))
        return out
    depths = cols.get("depths_cm", [5, 20, 50])
    return [(float(d), str(int(d)) if float(d) == int(float(d)) else str(d)) for d in depths]


def depth_labels(schema: dict[str, Any] | None = None) -> list[str]:
    return [lab for _, lab in schema_depth_specs(schema)]


def depth_cms(schema: dict[str, Any] | None = None) -> list[float]:
    return [cm for cm, _ in schema_depth_specs(schema)]
