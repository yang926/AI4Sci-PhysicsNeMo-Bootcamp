"""Publish a run only after its result files have been generated successfully."""
from contextlib import contextmanager
from pathlib import Path
from tempfile import TemporaryDirectory


@contextmanager
def staged_output(destination):
    """Yield a private staging directory, then publish without replacing a run.

    Normal write failures remove only this call's temporary artifacts. Final
    directory creation is exclusive; a concurrent successful writer wins.
    Publishing several files is not an atomic transaction or crash recovery.
    """
    # Do not resolve the final component: even a dangling symlink is occupied.
    destination = Path(destination).absolute()
    if destination.exists() or destination.is_symlink():
        raise FileExistsError(f"Output directory already exists: {destination}. Choose a new output directory.")
    destination.parent.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(prefix=f".{destination.name}-", dir=destination.parent) as temporary:
        staging = Path(temporary)
        yield staging
        destination.mkdir()  # Refuses any file, directory, or symlink created meanwhile.
        published = []
        try:
            for source in staging.iterdir():
                target = destination / source.name
                source.rename(target)
                published.append(target)
        except BaseException:
            # Move back only artifacts this call published, never foreign files.
            for target in reversed(published):
                target.rename(staging / target.name)
            destination.rmdir()
            raise
