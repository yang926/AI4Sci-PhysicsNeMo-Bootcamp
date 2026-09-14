"""Level 1: learn the periodic reaction-diffusion solution operator with FNO.

Edit the two FIXME functions, save this file, then run it from the notebook.
Use --reference for the instructor implementations and a runnable baseline.
"""
from torch.utils.data import TensorDataset
from physicsnemo.models.fno import FNO

from operator_training import UnfinishedExerciseError, run


def build_datasets(train_pairs, val_pairs, test_pairs):
    """Return three TensorDataset objects; pairs contain normalized (f, u)."""
    # FIXME: keep the train / validation / test splits separate.
    # Hint: TensorDataset(*pairs) converts one (f_tensor, u_tensor) pair.
    raise UnfinishedExerciseError(
        "Level 1: implement build_datasets in fno_physicsnemo_l1.py and save the file. "
        "The instructor can run with --reference."
    )


def build_model(model_config, grid_size):
    """Return FNO mapping [batch, 1, n, n] forcing to a solution of the same shape."""
    # FIXME: instantiate FNO with in_channels=1, out_channels=1, dimension=2
    # and the keyword settings in model_config. This is a torch.nn.Module.
    raise UnfinishedExerciseError(
        "Level 1: implement build_model in fno_physicsnemo_l1.py and save the file. "
        "The instructor can run with --reference."
    )


if __name__ == "__main__":
    try:
        run(1, build_model, dataset_builder=build_datasets)
    except UnfinishedExerciseError as error:
        raise SystemExit(str(error)) from None
