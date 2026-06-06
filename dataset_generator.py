import typer
from sonicmorph.pipeline import Pipeline
from sonicmorph.database import init_db
from sonicmorph.logging_setup import setup_logging

app = typer.Typer()


@app.command()
def status():
    """Initialize DB and show basic status."""
    setup_logging()
    db = init_db()
    typer.echo(f"Initialized DB at: {db}")


@app.command()
def run(stage: str = typer.Option(None, help="Stage to run (default: all)")):
    p = Pipeline()
    if stage:
        ok = p.run_stage(stage)
        typer.echo(f"Stage {stage}: {'ok' if ok else 'failed'}")
    else:
        res = p.run_all()
        typer.echo(str(res))


if __name__ == "__main__":
    app()
