import typer
from dataset_generation.pipeline import Pipeline
from dataset_generation.database import init_db
from dataset_generation.logging_setup import setup_logging

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
