from rich.rule import Rule as RichRule
from rich.text import Text
from textual.widget import Widget

_LINE_COLOR = "#0178D4"   # $primary — matches default Textual Rule blue
_TITLE_COLOR = "#ffa62b"  # $accent  — matches "Choose Config" label


class SectionRule(Widget):
    """Horizontal rule with an inline left-aligned title."""

    DEFAULT_CSS = "SectionRule { height: 1; margin: 1 0; }"

    def __init__(self, title: str, **kwargs) -> None:
        super().__init__(**kwargs)
        self._title = title

    def render(self) -> RichRule:
        title = Text(self._title, style=f"bold {_TITLE_COLOR}")
        return RichRule(title, align="left", style=_LINE_COLOR)
