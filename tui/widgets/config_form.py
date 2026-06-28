import re
from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml
from textual.app import ComposeResult
from textual.containers import Horizontal, Vertical
from textual.message import Message
from textual.widget import Widget
from textual.widgets import Button, Input, Label, Select
from tui.widgets.section_rule import SectionRule


PRESET_DIR = Path("config/presets")


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


def _coerce(value: str) -> Any:
    """Convert a string from an Input widget back to int/float when appropriate."""
    try:
        return int(value)
    except ValueError:
        pass
    try:
        return float(value)
    except ValueError:
        pass
    return value


def _input_id(label: str) -> str:
    return f"cfg-{label.lower().replace(' ', '-')}"


def _safe_filename(name: str) -> str:
    return re.sub(r"[^\w\-]", "_", name).strip("_") or "preset"


def _preset_options() -> list[tuple[str, str]]:
    if not PRESET_DIR.exists():
        return []
    return [(p.stem, p.stem) for p in sorted(PRESET_DIR.glob("*.yaml"))]


class ConfigForm(Widget):
    DEFAULT_CSS = """
ConfigForm { height: auto; }
ConfigForm .preset-section {
    height: auto;
    border: round #0178D4;
    padding: 0 1 1 1;
    margin-bottom: 1;
}
ConfigForm .preset-section-label {
    color: $accent;
    text-style: bold;
}
ConfigForm .preset-select-row {
    height: auto;
    margin-top: 1;
}
ConfigForm .preset-select-row Select { width: 1fr; }
ConfigForm .preset-select-row #cfg-delete-preset { width: auto; }
ConfigForm .preset-save-row { height: auto; margin-top: 1; }
ConfigForm .preset-save-row #preset-name { width: 1fr; }
ConfigForm .preset-save-row #cfg-save-preset { width: auto; }
"""

    class Saved(Message):
        pass

    def __init__(self, fields: list[ConfigField], domain: str = "", **kwargs) -> None:
        super().__init__(**kwargs)
        self._fields = fields
        self._domain = domain

    def compose(self) -> ComposeResult:
        with Vertical(classes="preset-section"):
            yield Label("Choose Config", classes="preset-section-label")
            with Horizontal(classes="preset-select-row"):
                yield Select(
                    _preset_options(),
                    prompt="Load preset…",
                    allow_blank=True,
                    id="preset-select",
                )
                yield Button("Delete", id="cfg-delete-preset", variant="error", disabled=True)
        yield SectionRule("Configuration")
        for f in self._fields:
            yield Label(f.label)
            yaml_file = f.yaml_file.format(domain=self._domain)
            path = Path(yaml_file)
            data = yaml.safe_load(path.read_text()) if path.exists() else {}
            current = str(_get_nested(data, f.key_path))
            yield Input(value=current, password=f.password, id=_input_id(f.label))
        with Horizontal(classes="preset-save-row"):
            yield Input(placeholder="preset name (required)…", id="preset-name")
            yield Button("Save", id="cfg-save-preset", variant="primary")

    def set_domain(self, domain: str) -> None:
        self._domain = domain
        self.call_later(self.recompose)

    def on_select_changed(self, event: Select.Changed) -> None:
        delete_btn = self.query_one("#cfg-delete-preset", Button)
        if event.value is Select.BLANK:
            delete_btn.disabled = True
            return
        delete_btn.disabled = False
        # Populate name field so the user can overwrite or see what's loaded
        self.query_one("#preset-name", Input).value = str(event.value)
        preset_file = PRESET_DIR / f"{event.value}.yaml"
        if not preset_file.exists():
            return
        try:
            data = yaml.safe_load(preset_file.read_text()) or {}
        except yaml.YAMLError:
            return
        for f in self._fields:
            val = data.get(f.label, "")
            if val != "":
                self.query_one(f"#{_input_id(f.label)}", Input).value = str(val)

    def on_button_pressed(self, event: Button.Pressed) -> None:
        if event.button.id == "cfg-save-preset":
            event.stop()
            self._save_preset()
        elif event.button.id == "cfg-delete-preset":
            event.stop()
            self._delete_preset()

    def _save_preset(self) -> None:
        name = self.query_one("#preset-name", Input).value.strip()
        if not name:
            self.app.notify("Enter a preset name first.", severity="warning")
            return
        filename = _safe_filename(name)
        PRESET_DIR.mkdir(parents=True, exist_ok=True)
        # Save preset file
        preset_data = {
            f.label: _coerce(self.query_one(f"#{_input_id(f.label)}", Input).value)
            for f in self._fields
        }
        (PRESET_DIR / f"{filename}.yaml").write_text(yaml.safe_dump(preset_data))
        # Write through to actual config files
        for f in self._fields:
            yaml_file = f.yaml_file.format(domain=self._domain)
            path = Path(yaml_file)
            existing = yaml.safe_load(path.read_text()) if path.exists() else {}
            updated = _set_nested(existing, f.key_path, preset_data[f.label])
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(yaml.safe_dump(updated))
        select = self.query_one("#preset-select", Select)
        select.set_options(_preset_options())
        select.value = filename
        self.app.notify(f"Preset '{filename}' saved.")
        self.post_message(self.Saved())

    def _delete_preset(self) -> None:
        select = self.query_one("#preset-select", Select)
        if select.value is Select.BLANK:
            return
        name = str(select.value)
        preset_file = PRESET_DIR / f"{name}.yaml"
        if preset_file.exists():
            preset_file.unlink()
        select.set_options(_preset_options())
        select.value = Select.BLANK
        self.query_one("#preset-name", Input).value = ""
        self.query_one("#cfg-delete-preset", Button).disabled = True
        self.app.notify(f"Preset '{name}' deleted.")
