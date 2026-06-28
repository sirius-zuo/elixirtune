# tui/panels/deployment.py
import subprocess
from pathlib import Path

from textual.app import ComposeResult
from textual.containers import Horizontal
from textual.widgets import Button, Label
from textual import work

from tui.app import BasePanel
from tui.domain import infer_status, Status, generate_runtime_configs, status_order, resolve_adapters_dir
from tui.runner import RunnerOutput, RunnerDone, stream_subprocess
from tui.widgets.log_view import LogView
from tui.widgets.section_rule import SectionRule


def _dir_size_mb(p: Path) -> str:
    if not p.exists():
        return "—"
    total = sum(f.stat().st_size for f in p.rglob("*") if f.is_file())
    return f"{total / 1_048_576:.1f} MB"


class DeploymentPanel(BasePanel):
    DEFAULT_CSS = "DeploymentPanel { height: 100%; padding: 1; }"

    def compose(self) -> ComposeResult:
        yield SectionRule("Model Status")
        yield Label("Adapters: —", id="adapter-info")
        yield Label("Fused model: —", id="fused-info")
        yield SectionRule("Actions")
        with Horizontal(classes="btn-row"):
            yield Button("▶ Fuse", id="fuse-btn", disabled=True, variant="success")
            yield Button("▶ Export GGUF", id="gguf-btn", disabled=True, variant="success")
            yield Button("Create Ollama Model", id="ollama-btn", disabled=True, variant="success")
            yield Button("Upload to HuggingFace", id="hf-upload-btn", disabled=True, variant="success")
        yield SectionRule("Log")
        yield LogView(id="deploy-log")

    def refresh_content(self) -> None:
        if not self.domain:
            return
        ws = Path("workspaces") / self.domain
        status = infer_status(ws)
        adapter_dir = resolve_adapters_dir(ws)
        fused_dir = ws / "fused"
        self.query_one("#adapter-info", Label).update(
            f"workspaces/{self.domain}/adapters   {_dir_size_mb(adapter_dir)}"
        )
        self.query_one("#fused-info", Label).update(
            f"workspaces/{self.domain}/fused       {_dir_size_mb(fused_dir)}"
        )
        fuse_btn = self.query_one("#fuse-btn", Button)
        fuse_btn.disabled = status_order(status) < status_order(Status.TRAINED)
        if status_order(status) >= status_order(Status.DEPLOYED):
            fuse_btn.label = "Re-Fuse (optional)"
        else:
            fuse_btn.label = "▶ Fuse"
        self.query_one("#ollama-btn", Button).disabled = (
            status_order(status) < status_order(Status.DEPLOYED)
        )
        self.query_one("#hf-upload-btn", Button).disabled = (
            status_order(status) < status_order(Status.DEPLOYED)
        )
        self.query_one("#gguf-btn", Button).disabled = (
            status_order(status) < status_order(Status.DEPLOYED)
        )

    def on_button_pressed(self, event: Button.Pressed) -> None:
        if not self.domain:
            return
        event.stop()
        ws = Path("workspaces") / self.domain
        if event.button.id == "fuse-btn":
            generate_runtime_configs(ws)
            event.button.disabled = True
            self._run_fuse(self.domain)
        elif event.button.id == "ollama-btn":
            generate_runtime_configs(ws)
            event.button.disabled = True
            self._run_ollama(self.domain)
        elif event.button.id == "hf-upload-btn":
            from tui.upload_modal import HFUploadScreen
            captured_domain = self.domain
            def _on_upload_result(result) -> None:
                if result is None:
                    return
                self.query_one("#hf-upload-btn", Button).disabled = True
                self._run_upload(captured_domain, result["repo"], result["private"], result["token"])
            self.app.push_screen(HFUploadScreen(), callback=_on_upload_result)
        elif event.button.id == "gguf-btn":
            event.button.disabled = True
            self._run_export_gguf(self.domain)

    @work(thread=True)
    def _run_fuse(self, domain: str) -> None:
        ws = Path("workspaces") / domain
        cmd = [
            "python3", "cli.py", "fuse", domain,
            "--model-config", str(ws / "runtime_model_config.yaml"),
            "--adapters-path", str(resolve_adapters_dir(ws)),
            "--output-path", str(ws / "fused"),
        ]
        self._stream(cmd)

    @work(thread=True)
    def _run_ollama(self, domain: str) -> None:
        ws = Path("workspaces") / domain
        fused_dir = ws / "fused"

        # Ollama needs a .gguf file, not a directory
        gguf_files = sorted(fused_dir.glob("*.gguf"))
        # Prefer the quantized file (non-f16) if both exist
        quantized = [f for f in gguf_files if "_f16" not in f.name]
        gguf_path = (quantized or gguf_files)[0] if (quantized or gguf_files) else None

        if gguf_path is None:
            self.post_message(RunnerOutput(
                f"No .gguf file found in {fused_dir}. "
                "Export GGUF first using the Export GGUF button."
            ))
            self.post_message(RunnerDone(1))
            return

        # Ollama requires an absolute path in the FROM directive
        modelfile = ws / "Modelfile"
        modelfile.write_text(f"FROM {gguf_path.resolve()}\n")
        cmd = ["ollama", "create", f"{domain}-lora", "-f", str(modelfile)]
        self._stream(cmd)

    @work(thread=True)
    def _run_upload(self, domain: str, repo: str, private: bool, token: str) -> None:
        import huggingface_hub
        ws = Path("workspaces") / domain
        try:
            self.post_message(RunnerOutput(f"Creating repository {repo}…"))
            huggingface_hub.create_repo(
                repo_id=repo, private=private, token=token, exist_ok=True, repo_type="model"
            )
            self.post_message(RunnerOutput(f"Uploading {ws / 'fused'}…"))
            huggingface_hub.upload_folder(
                folder_path=str(ws / "fused"),
                repo_id=repo,
                token=token,
                commit_message=f"Upload fused model for domain: {domain}",
            )
            self.post_message(RunnerOutput(f"Done. https://huggingface.co/{repo}"))
            self.post_message(RunnerDone(0))
        except Exception as e:
            self.post_message(RunnerOutput(f"[red]Upload failed: {e}[/red]"))
            self.post_message(RunnerDone(1))

    @work(thread=True)
    def _run_export_gguf(self, domain: str) -> None:
        cmd = [
            "python3", "cli.py", "export-gguf", domain,
            "--quantization", "Q4_K_M",
        ]
        self._stream(cmd)

    def _stream(self, cmd: list[str]) -> None:
        for line, code in stream_subprocess(cmd):
            if line is not None:
                self.post_message(RunnerOutput(line))
            else:
                self.post_message(RunnerDone(code))

    def on_runner_output(self, event: RunnerOutput) -> None:
        self.query_one(LogView).write_line(event.line)

    def on_runner_done(self, event: RunnerDone) -> None:
        if event.exit_code != 0:
            self.query_one(LogView).write_line(
                f"[red]Failed (exit {event.exit_code})[/red]"
            )
        else:
            self.query_one(LogView).write_line("[green]Done.[/green]")
        self.refresh_content()
        self.call_later(self.app._rescan)
