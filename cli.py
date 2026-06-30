import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

import typer
from commands.init     import app as init_app
from commands.curate   import app as curate_app
from commands.generate import app as generate_app
from commands.prepare  import app as prepare_app
from commands.prepare_dpo import app as prepare_dpo_app
from commands.upload   import app as upload_app
from commands.train    import app as train_app
from commands.evaluate import app as evaluate_app
from commands.fuse     import app as fuse_app
from commands.export_gguf import app as export_gguf_app
from commands.chat     import app as chat_app

app = typer.Typer(pretty_exceptions_enable=False)
app.add_typer(init_app,     name="init")
app.add_typer(curate_app,   name="curate")
app.add_typer(generate_app, name="generate")
app.add_typer(prepare_app,  name="prepare")
app.add_typer(prepare_dpo_app, name="prepare-dpo")
app.add_typer(upload_app,   name="upload")
app.add_typer(train_app,    name="train")
app.add_typer(evaluate_app, name="evaluate")
app.add_typer(fuse_app,     name="fuse")
app.add_typer(export_gguf_app, name="export-gguf")
app.add_typer(chat_app,     name="chat")


@app.command()
def tui(domain: str = typer.Option(None, help="Domain to pre-select on launch")):
    """Launch the ElixirTune TUI."""
    from tui.app import ElixirTuneApp
    ElixirTuneApp(initial_domain=domain).run()


if __name__ == "__main__":
    app()
