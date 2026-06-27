import json
from pathlib import Path
from transformers import TrainerCallback


class MetricsWriterCallback(TrainerCallback):
    def __init__(self, path: Path) -> None:
        self._path = Path(path)
        self._data: dict = {"train_loss": [], "val_loss": [], "iterations": []}

    def on_log(self, args, state, control, logs=None, **kwargs) -> None:
        if not logs:
            return
        if "loss" in logs:
            self._data["train_loss"].append(round(float(logs["loss"]), 4))
            self._data["iterations"].append(int(state.global_step))
        if "eval_loss" in logs:
            self._data["val_loss"].append(round(float(logs["eval_loss"]), 4))
        self._path.parent.mkdir(parents=True, exist_ok=True)
        self._path.write_text(json.dumps(self._data))
