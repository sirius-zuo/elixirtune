import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

import typer
from commands.init     import app as init_app
from commands.curate   import app as curate_app
from commands.generate import app as generate_app
from commands.prepare  import app as prepare_app
from commands.upload   import app as upload_app
from commands.train    import app as train_app
from commands.evaluate import app as evaluate_app

app = typer.Typer()
app.add_typer(init_app,     name="init")
app.add_typer(curate_app,   name="curate")
app.add_typer(generate_app, name="generate")
app.add_typer(prepare_app,  name="prepare")
app.add_typer(upload_app,   name="upload")
app.add_typer(train_app,    name="train")
app.add_typer(evaluate_app, name="evaluate")


@app.command()
def tui(domain: str = typer.Option(None, help="Domain to pre-select on launch")):
    """Launch the ElixirLoRA TUI."""
    from tui.app import ElixirLoRAApp
    ElixirLoRAApp(initial_domain=domain).run()


if __name__ == "__main__":
    app()
