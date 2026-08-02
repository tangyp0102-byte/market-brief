"""Instrument registry: load and strictly validate config/instruments.yaml.

Validation is deliberately strict. A typo in the registry (wrong change_unit,
missing dollar_direction on an FX pair) produces plausible-looking but wrong
numbers downstream, which is the worst failure mode for this project.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

VALID_ASSET_CLASSES = {"rates", "credit", "equity", "commodity", "fx", "vol"}
VALID_QUOTE_UNITS = {"percent", "index", "usd", "fx_rate"}
VALID_CHANGE_UNITS = {"bp", "pct", "pts"}
VALID_TIERS = {"core", "confirmation"}

# Which exchange calendar governs whether an instrument SHOULD have data on a
# given session. Explicit rather than inferred, for the same reason as
# dollar_direction: guessing produces plausible-looking wrong answers.
VALID_CALENDARS = {"NYSE", "SIFMA_US", "CMEGlobex_EnergyAndMetals", "FX"}

DEFAULT_CALENDAR_BY_ASSET_CLASS = {
    "equity": "NYSE",
    "vol": "NYSE",
    "credit": "NYSE",       # ETFs; the FRED OAS series override this to SIFMA_US
    "rates": "SIFMA_US",
    "commodity": "CMEGlobex_EnergyAndMetals",
    "fx": "FX",             # trades through US holidays
}

VALID_PROVIDERS = {
    "treasury_par_nominal",
    "treasury_par_real",
    "fred",
    "yahoo",
    "stooq",
    "cboe",
}

# bp changes only make sense for values already quoted in percent.
_UNIT_COMPATIBILITY = {
    "bp": {"percent"},
    "pct": {"index", "usd", "fx_rate"},
    "pts": {"index"},
}

DEFAULT_REGISTRY_PATH = Path(__file__).resolve().parent.parent / "config" / "instruments.yaml"


class RegistryError(ValueError):
    """Raised when the instrument registry is malformed."""


@dataclass(frozen=True)
class Source:
    provider: str
    symbol: str


@dataclass(frozen=True)
class Instrument:
    id: str
    name: str
    asset_class: str
    quote_unit: str
    change_unit: str
    sources: tuple[Source, ...]
    bounds_min: float
    bounds_max: float
    max_abs_change: float
    stale_tolerance_days: int
    tier: str = "confirmation"
    calendar: str = "NYSE"
    tradingview: str | None = None
    dollar_direction: int | None = None
    dxy_weight: float | None = None

    @property
    def is_core(self) -> bool:
        return self.tier == "core"

    @property
    def primary(self) -> Source:
        return self.sources[0]

    @property
    def fallbacks(self) -> tuple[Source, ...]:
        return self.sources[1:]

    def tradingview_url(self) -> str | None:
        if not self.tradingview:
            return None
        from urllib.parse import quote

        return f"https://www.tradingview.com/chart/?symbol={quote(self.tradingview, safe='')}"


@dataclass(frozen=True)
class Registry:
    instruments: tuple[Instrument, ...]
    zscore_window: int = 60
    zscore_min_obs: int = 30
    version: int = 1
    _by_id: dict[str, Instrument] = field(default_factory=dict, repr=False, compare=False)

    def __post_init__(self) -> None:
        object.__setattr__(self, "_by_id", {i.id: i for i in self.instruments})

    def __len__(self) -> int:
        return len(self.instruments)

    def __iter__(self):
        return iter(self.instruments)

    def __getitem__(self, instrument_id: str) -> Instrument:
        try:
            return self._by_id[instrument_id]
        except KeyError:
            raise KeyError(f"unknown instrument_id: {instrument_id!r}") from None

    def get(self, instrument_id: str) -> Instrument | None:
        return self._by_id.get(instrument_id)

    @property
    def ids(self) -> tuple[str, ...]:
        return tuple(self._by_id)

    def by_asset_class(self, asset_class: str) -> tuple[Instrument, ...]:
        if asset_class not in VALID_ASSET_CLASSES:
            raise KeyError(f"unknown asset_class: {asset_class!r}")
        return tuple(i for i in self.instruments if i.asset_class == asset_class)

    def by_provider(self, provider: str) -> tuple[Instrument, ...]:
        """Instruments for which `provider` appears in any source slot."""
        return tuple(
            i for i in self.instruments if any(s.provider == provider for s in i.sources)
        )

    def by_tier(self, tier: str) -> tuple[Instrument, ...]:
        if tier not in VALID_TIERS:
            raise KeyError(f"unknown tier: {tier!r}")
        return tuple(i for i in self.instruments if i.tier == tier)

    @property
    def core(self) -> tuple[Instrument, ...]:
        return self.by_tier("core")

    def by_calendar(self, calendar: str) -> tuple[Instrument, ...]:
        return tuple(i for i in self.instruments if i.calendar == calendar)

    def dxy_components(self) -> tuple[Instrument, ...]:
        return tuple(i for i in self.instruments if i.dxy_weight is not None)


def _require(cond: bool, msg: str) -> None:
    if not cond:
        raise RegistryError(msg)


def _parse_instrument(raw: dict[str, Any], index: int) -> Instrument:
    where = raw.get("id", f"<index {index}>")

    for key in ("id", "name", "asset_class", "quote_unit", "change_unit", "sources", "bounds"):
        _require(key in raw, f"{where}: missing required key {key!r}")

    iid = raw["id"]
    _require(
        isinstance(iid, str) and iid == iid.lower() and " " not in iid,
        f"{where}: id must be a lowercase string without spaces",
    )

    asset_class = raw["asset_class"]
    _require(
        asset_class in VALID_ASSET_CLASSES,
        f"{where}: asset_class {asset_class!r} not in {sorted(VALID_ASSET_CLASSES)}",
    )

    quote_unit = raw["quote_unit"]
    _require(
        quote_unit in VALID_QUOTE_UNITS,
        f"{where}: quote_unit {quote_unit!r} not in {sorted(VALID_QUOTE_UNITS)}",
    )

    change_unit = raw["change_unit"]
    _require(
        change_unit in VALID_CHANGE_UNITS,
        f"{where}: change_unit {change_unit!r} not in {sorted(VALID_CHANGE_UNITS)}",
    )
    _require(
        quote_unit in _UNIT_COMPATIBILITY[change_unit],
        f"{where}: change_unit {change_unit!r} is incompatible with "
        f"quote_unit {quote_unit!r} (allowed: {sorted(_UNIT_COMPATIBILITY[change_unit])})",
    )

    raw_sources = raw["sources"]
    _require(
        isinstance(raw_sources, list) and len(raw_sources) >= 1,
        f"{where}: sources must be a non-empty list",
    )
    sources = []
    for s in raw_sources:
        _require(
            isinstance(s, dict) and "provider" in s and "symbol" in s,
            f"{where}: each source needs provider and symbol",
        )
        _require(
            s["provider"] in VALID_PROVIDERS,
            f"{where}: provider {s['provider']!r} not in {sorted(VALID_PROVIDERS)}",
        )
        sources.append(Source(provider=s["provider"], symbol=str(s["symbol"])))

    bounds = raw["bounds"]
    _require(
        isinstance(bounds, dict) and "min" in bounds and "max" in bounds,
        f"{where}: bounds must define min and max",
    )
    bmin, bmax = float(bounds["min"]), float(bounds["max"])
    _require(bmin < bmax, f"{where}: bounds.min must be < bounds.max")

    max_abs_change = float(raw.get("max_abs_change", 0))
    _require(max_abs_change > 0, f"{where}: max_abs_change must be positive")

    stale = int(raw.get("stale_tolerance_days", 1))
    _require(stale >= 1, f"{where}: stale_tolerance_days must be >= 1")

    dollar_direction = raw.get("dollar_direction")
    if asset_class == "fx":
        _require(
            dollar_direction in (1, -1),
            f"{where}: fx instruments must set dollar_direction to +1 or -1. "
            "This is never inferred from the ticker.",
        )
    else:
        _require(
            dollar_direction is None,
            f"{where}: dollar_direction is only valid for asset_class 'fx'",
        )

    tier = raw.get("tier", "confirmation")
    _require(
        tier in VALID_TIERS,
        f"{where}: tier {tier!r} not in {sorted(VALID_TIERS)}",
    )

    calendar = raw.get("calendar") or DEFAULT_CALENDAR_BY_ASSET_CLASS[asset_class]
    _require(
        calendar in VALID_CALENDARS,
        f"{where}: calendar {calendar!r} not in {sorted(VALID_CALENDARS)}",
    )

    dxy_weight = None
    if "dxy_component" in raw:
        _require(
            asset_class == "fx",
            f"{where}: dxy_component is only valid for fx instruments",
        )
        comp = raw["dxy_component"]
        _require(
            isinstance(comp, dict) and "weight" in comp,
            f"{where}: dxy_component must define weight",
        )
        dxy_weight = float(comp["weight"])
        # DXY exponent sign must agree with the quote's dollar direction:
        # a positive exponent means a rise in the quote raises the index.
        _require(
            (dxy_weight > 0) == (dollar_direction == 1),
            f"{where}: dxy_component.weight sign ({dxy_weight}) disagrees with "
            f"dollar_direction ({dollar_direction})",
        )

    return Instrument(
        id=iid,
        name=raw["name"],
        asset_class=asset_class,
        quote_unit=quote_unit,
        change_unit=change_unit,
        sources=tuple(sources),
        bounds_min=bmin,
        bounds_max=bmax,
        max_abs_change=max_abs_change,
        stale_tolerance_days=stale,
        tier=tier,
        calendar=calendar,
        tradingview=raw.get("tradingview"),
        dollar_direction=dollar_direction,
        dxy_weight=dxy_weight,
    )


def load_registry(path: str | Path | None = None) -> Registry:
    """Load and validate the instrument registry."""
    path = Path(path) if path is not None else DEFAULT_REGISTRY_PATH
    if not path.exists():
        raise RegistryError(f"registry file not found: {path}")

    with path.open() as fh:
        doc = yaml.safe_load(fh)

    _require(isinstance(doc, dict), "registry root must be a mapping")
    _require("instruments" in doc, "registry must define 'instruments'")

    raw_instruments = doc["instruments"]
    _require(
        isinstance(raw_instruments, list) and raw_instruments,
        "'instruments' must be a non-empty list",
    )

    instruments = [_parse_instrument(raw, i) for i, raw in enumerate(raw_instruments)]

    seen: set[str] = set()
    for inst in instruments:
        _require(inst.id not in seen, f"duplicate instrument id: {inst.id!r}")
        seen.add(inst.id)

    # Duplicate (provider, symbol) pairs usually mean a copy-paste error.
    primary_keys: dict[tuple[str, str], str] = {}
    for inst in instruments:
        key = (inst.primary.provider, inst.primary.symbol)
        _require(
            key not in primary_keys,
            f"{inst.id!r} shares a primary source {key} with {primary_keys.get(key)!r}",
        )
        primary_keys[key] = inst.id

    meta = doc.get("meta") or {}
    registry = Registry(
        instruments=tuple(instruments),
        zscore_window=int(meta.get("zscore_window", 60)),
        zscore_min_obs=int(meta.get("zscore_min_obs", 30)),
        version=int(meta.get("version", 1)),
    )

    _require(
        bool(registry.core),
        "registry defines no core instruments; the validation gate would have "
        "nothing to require before publishing a brief",
    )

    # DXY weights must sum to +/-1 in absolute terms if any are defined.
    comps = registry.dxy_components()
    if comps:
        total = sum(abs(c.dxy_weight) for c in comps)  # type: ignore[misc]
        _require(
            abs(total - 1.0) < 1e-6,
            f"dxy_component weights must sum to 1.0 in absolute value, got {total:.6f}",
        )

    return registry
