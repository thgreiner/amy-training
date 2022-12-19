import click

from amy.pgn.split import split_pgn_by_date
from amy.pickle.converter import convert_pgn_to_pickle
from amy.quantization.quantize import quantize_model
from amy.training.train_pkl import train_from_pkl


@click.group()
def cli():
    pass


@click.command()
@click.argument("file_name")
def pgn_split(file_name: str):
    """Split a PGN file into separate files by their date tag."""
    split_pgn_by_date(file_name)


@click.command()
@click.argument(
    "model-name",
    default=None,
)
@click.option("--batch-size", help="Batch size", type=int, default=256)
def train(model_name: str, batch_size: int):
    """Train a model."""
    train_from_pkl(model_name, batch_size, False)


@click.command()
@click.argument(
    "model-name",
    default=None,
)
@click.option("--batch-size", help="Batch size", type=int, default=256)
def validate(model_name: str, batch_size: int):
    """Validate a model."""
    train_from_pkl(model_name, batch_size, True)


@click.command()
@click.option("--file-name", type=str, help="PGN input file", required=True)
@click.option("--output-dir", type=str, help="Output directory", required=True)
@click.option("--nfiles", type=int, help="Files to split into", default=25)
@click.option("--split", type=int, help="Training/validation split", default=10)
def pgn_to_pickle(file_name: str, output_dir: str, nfiles: int, split: int):
    """Convert a PGN file to a pickle training file."""
    convert_pgn_to_pickle(file_name, output_dir, nfiles, split)


@click.command()
@click.argument("file_name")
def quantize(file_name: str) -> None:
    """Quantize a Keras model."""
    quantize_model(file_name)


cli.add_command(pgn_split)
cli.add_command(pgn_to_pickle)
cli.add_command(quantize)
cli.add_command(train)
cli.add_command(validate)

if __name__ == "__main__":
    cli()
