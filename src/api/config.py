"""Load pinned model ids and prices from configs/api_models.yaml."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG = REPO_ROOT / "configs" / "api_models.yaml"

_REQUIRED = (
    "model_id",
    "model_id_source",
    "model_id_confirmed",
    "input_usd_per_mtok",
    "output_usd_per_mtok",
    "batch_discount",
    "price_source",
    "price_read_date",
    "logprobs",
    "supports_temperature",
    "token_multiplier",
)


@dataclass(frozen=True)
class ModelSpec:
    provider: str
    model_id: str
    model_id_source: str
    model_id_confirmed: bool
    input_usd_per_mtok: float
    output_usd_per_mtok: float
    batch_discount: float
    price_source: str
    price_read_date: str
    logprobs: bool
    supports_temperature: bool
    token_multiplier: float
    chars_per_token: float
    thinking_level: str | None = None

    @property
    def client_kind(self) -> str:
        """Which client class serves this entry (google_25 is still Google)."""
        return "google" if self.provider.startswith("google") else self.provider


def load_config(path: str | Path = DEFAULT_CONFIG) -> dict[str, Any]:
    with open(path, encoding="utf-8") as fh:
        return yaml.safe_load(fh)


def load_specs(path: str | Path = DEFAULT_CONFIG) -> dict[str, ModelSpec]:
    cfg = load_config(path)
    cpt = float(cfg["chars_per_token"])
    specs: dict[str, ModelSpec] = {}
    for name, entry in cfg["providers"].items():
        missing = [k for k in _REQUIRED if k not in entry]
        if missing:
            raise ValueError(f"{path}: provider {name!r} is missing {missing}")
        specs[name] = ModelSpec(
            provider=name,
            model_id=str(entry["model_id"]),
            model_id_source=str(entry["model_id_source"]),
            model_id_confirmed=bool(entry["model_id_confirmed"]),
            input_usd_per_mtok=float(entry["input_usd_per_mtok"]),
            output_usd_per_mtok=float(entry["output_usd_per_mtok"]),
            batch_discount=float(entry["batch_discount"]),
            price_source=str(entry["price_source"]),
            price_read_date=str(entry["price_read_date"]),
            logprobs=bool(entry["logprobs"]),
            supports_temperature=bool(entry["supports_temperature"]),
            token_multiplier=float(entry["token_multiplier"]),
            chars_per_token=cpt,
            thinking_level=entry.get("thinking_level"),
        )
    return specs


def default_grid_providers(path: str | Path = DEFAULT_CONFIG) -> list[str]:
    return list(load_config(path)["default_grid_providers"])
