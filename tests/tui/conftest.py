import sys
from pathlib import Path
from unittest.mock import MagicMock

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

if "mlx_lm" not in sys.modules:
    try:
        import mlx_lm  # noqa: F401
    except ImportError:
        sys.modules["mlx_lm"] = MagicMock()
