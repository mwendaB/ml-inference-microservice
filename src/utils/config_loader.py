import os
import threading
import time
from dataclasses import dataclass
from typing import Any, Callable, Dict, Optional, Type

import yaml
from pydantic import BaseModel, ValidationError

from src.utils.logger import get_logger

logger = get_logger(__name__)


@dataclass
class ConfigHandle:
    data: Dict[str, Any]
    path: str
    mtime: float


def _deep_set(target: Dict[str, Any], keys: list[str], value: Any) -> None:
    current = target
    for key in keys[:-1]:
        if key not in current or not isinstance(current[key], dict):
            current[key] = {}
        current = current[key]
    current[keys[-1]] = value


def _apply_env_overrides(config: Dict[str, Any]) -> Dict[str, Any]:
    for env_key, env_val in os.environ.items():
        if "__" not in env_key:
            continue
        keys = env_key.lower().split("__")
        if not keys:
            continue
        _deep_set(config, keys, _parse_env_value(env_val))
    return config


def _parse_env_value(value: str) -> Any:
    lowered = value.lower()
    if lowered in {"true", "false"}:
        return lowered == "true"
    try:
        if "." in value:
            return float(value)
        return int(value)
    except ValueError:
        return value


def load_yaml(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle) or {}
    if not isinstance(data, dict):
        raise ValueError(f"Config file {path} must contain a mapping")
    return data


def load_config(path: str, schema: Optional[Type[BaseModel]] = None) -> Dict[str, Any]:
    data = load_yaml(path)
    data = _apply_env_overrides(data)
    if schema is not None:
        try:
            schema(**data)
        except ValidationError as exc:
            raise ValueError(f"Invalid config for {path}: {exc}") from exc
    return data


class ConfigLoader:
    def __init__(self, path: str, schema: Optional[Type[BaseModel]] = None) -> None:
        self.path = path
        self.schema = schema
        self._handle = self._load()
        self._stop_event = threading.Event()
        self._thread: Optional[threading.Thread] = None

    def _load(self) -> ConfigHandle:
        data = load_config(self.path, self.schema)
        mtime = os.path.getmtime(self.path)
        return ConfigHandle(data=data, path=self.path, mtime=mtime)

    def get(self) -> Dict[str, Any]:
        return self._handle.data

    def start_hot_reload(self, callback: Optional[Callable[[Dict[str, Any]], None]] = None) -> None:
        if self._thread and self._thread.is_alive():
            return

        def _watch() -> None:
            while not self._stop_event.is_set():
                try:
                    mtime = os.path.getmtime(self.path)
                    if mtime > self._handle.mtime:
                        self._handle = self._load()
                        logger.info("config_reloaded", extra={"extra": {"path": self.path}})
                        if callback:
                            callback(self._handle.data)
                except FileNotFoundError:
                    logger.warning("config_missing", extra={"extra": {"path": self.path}})
                time.sleep(1.0)

        self._thread = threading.Thread(target=_watch, daemon=True)
        self._thread.start()

    def stop_hot_reload(self) -> None:
        self._stop_event.set()
        if self._thread:
            self._thread.join(timeout=2)
