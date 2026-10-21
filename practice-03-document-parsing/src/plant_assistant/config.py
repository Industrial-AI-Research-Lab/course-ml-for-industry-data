"""Model profiles from models.toml plus secrets from .env.

A profile is everything needed to call one model: URL, key, model name, parameters, price.
Switching models means switching the profile name, not editing code.
"""

from __future__ import annotations

import os
import re
import tomllib
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from dotenv import load_dotenv

from . import PROJECT_ROOT

MODELS_FILE = PROJECT_ROOT / "models.toml"

# "${A|B:-default}": first environment variable that is set, otherwise the default.
_ENV_PATTERN = re.compile(r"\$\{([A-Za-z0-9_|]+)(?::-([^}]*))?\}")


def load_env() -> str | None:
    """Load the nearest .env: project folder first, then parent folders. Returns its path."""
    for folder in [PROJECT_ROOT, *PROJECT_ROOT.parents]:
        candidate = folder / ".env"
        if candidate.exists():
            load_dotenv(candidate, override=False)
            return str(candidate)
    return None


def _expand(value: Any) -> Any:
    if isinstance(value, str):

        def repl(match: re.Match) -> str:
            names, default = match.group(1).split("|"), match.group(2)
            for name in names:
                env_value = os.environ.get(name, "").strip().strip('"')
                if env_value:
                    return env_value
            return default if default is not None else ""

        return _ENV_PATTERN.sub(repl, value)
    if isinstance(value, dict):
        return {k: _expand(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_expand(v) for v in value]
    return value


@dataclass
class Profile:
    name: str
    description: str
    base_url: str
    api_key: str
    model: str
    timeout_s: float = 120
    max_tokens: int = 1500
    temperature: float = 0.0
    concurrency: int = 4
    bypass_proxy_cache: bool = True
    price_in_per_m: float = 0.0
    price_out_per_m: float = 0.0
    extra_body: dict[str, Any] = field(default_factory=dict)

    def cost_usd(self, prompt_tokens: int, completion_tokens: int) -> float:
        return (prompt_tokens * self.price_in_per_m + completion_tokens * self.price_out_per_m) / 1e6

    def missing_settings(self) -> list[str]:
        missing = []
        if not self.base_url:
            missing.append("base_url (set COURSE_LLM_BASE_URL in .env)")
        if not self.api_key:
            missing.append("api_key (set the key in .env)")
        return missing

    def public_dict(self) -> dict[str, Any]:
        """Profile without the secret key: safe to print and to log."""
        d = {k: v for k, v in self.__dict__.items() if k != "api_key"}
        d["api_key"] = "<set>" if self.api_key else "<missing>"
        return d


def _read_models_file(path: Path = MODELS_FILE) -> dict[str, Any]:
    with open(path, "rb") as f:
        return tomllib.load(f)


def list_profiles(path: Path = MODELS_FILE) -> dict[str, str]:
    raw = _read_models_file(path)
    return {name: p.get("description", "") for name, p in raw.get("profiles", {}).items()}


def get_profile(name: str, path: Path = MODELS_FILE) -> Profile:
    load_env()
    raw = _read_models_file(path)
    profiles = raw.get("profiles", {})
    if name not in profiles:
        raise KeyError(f"Unknown profile '{name}'. Known profiles: {', '.join(profiles)}")
    merged = {**raw.get("defaults", {}), **profiles[name]}
    merged = _expand(merged)
    return Profile(name=name, **merged)
