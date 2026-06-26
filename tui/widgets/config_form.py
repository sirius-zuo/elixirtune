from copy import deepcopy
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml
from textual.app import ComposeResult
from textual.message import Message
from textual.widget import Widget
from textual.widgets import Button, Input, Label


@dataclass
class ConfigField:
    label: str
    yaml_file: str   # may contain {domain} placeholder
    key_path: list[str]
    password: bool = False


def _deep_merge(base: dict, override: dict) -> dict:
    out = deepcopy(base)
    for k, v in override.items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = _deep_merge(out[k], v)
        else:
            out[k] = deepcopy(v)
    return out


def _get_nested(data: dict, key_path: list[str]) -> Any:
    for key in key_path:
        if not isinstance(data, dict):
            return ""
        data = data.get(key, "")
    return data if data is not None else ""


def _set_nested(data: dict, key_path: list[str], value: Any) -> dict:
    out = deepcopy(data)
    node = out
    for key in key_path[:-1]:
        node = node.setdefault(key, {})
    node[key_path[-1]] = value
    return out


def _input_id(label: str) -> str:
    return f"cfg-{label.lower().replace(' ', '-')}"


class ConfigForm(Widget):
    class Saved(Message):
        pass

    def __init__(self, fields: list[ConfigField], domain: str = "", **kwargs) -> None:
        super().__init__(**kwargs)
        self._fields = fields
        self._domain = domain
        self._message_handlers: dict = {}

    def on(self, message_class, handler):
        """Register a dynamic message handler."""
        if message_class not in self._message_handlers:
            self._message_handlers[message_class] = []
        self._message_handlers[message_class].append(handler)
        return self

    def _dispatch_message(self, message):
        """Override to call registered handlers."""
        result = super()._dispatch_message(message)
        # Call registered handlers for this message type
        for msg_class, handlers in self._message_handlers.items():
            if isinstance(message, msg_class):
                for handler in handlers:
                    handler(message)
        return result

    def compose(self) -> ComposeResult:
        for f in self._fields:
            yield Label(f.label)
            yaml_file = f.yaml_file.format(domain=self._domain)
            path = Path(yaml_file)
            data = yaml.safe_load(path.read_text()) if path.exists() else {}
            current = str(_get_nested(data, f.key_path))
            yield Input(value=current, password=f.password, id=_input_id(f.label))
        yield Button("Save", id="cfg-save", variant="primary")

    def set_domain(self, domain: str) -> None:
        self._domain = domain
        self.recompose()

    def on_button_pressed(self, event: Button.Pressed) -> None:
        if event.button.id == "cfg-save":
            event.stop()
            self._save()

    def _save(self) -> None:
        for f in self._fields:
            yaml_file = f.yaml_file.format(domain=self._domain)
            path = Path(yaml_file)
            data = yaml.safe_load(path.read_text()) if path.exists() else {}
            value = self.query_one(f"#{_input_id(f.label)}", Input).value
            updated = _set_nested(data, f.key_path, value)
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(yaml.safe_dump(updated))
        self.post_message(self.Saved())
