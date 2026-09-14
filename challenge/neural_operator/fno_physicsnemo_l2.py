"""Level 2: replace the FNO backbone with PhysicsNeMo AFNO.

Edit the FIXME functions, or use --reference to run the instructor baseline.
Grid dimensions must be divisible by the patch dimensions: do not crop the
periodic domain, because that would change the PDE boundary conditions.
"""
from torch.utils.data import TensorDataset
from physicsnemo.models.afno import AFNO

from operator_training import UnfinishedExerciseError, run


def build_datasets(train_pairs, val_pairs, test_pairs):
    # FIXME: create the three TensorDataset objects as in Level 1.
    raise UnfinishedExerciseError(
        "Level 2: implement build_datasets in fno_physicsnemo_l2.py and save. "
        "Use --reference for the instructor baseline."
    )


def build_model(model_config, grid_size):
    # FIXME: check that grid_size is divisible by each patch_size value, then
    # instantiate AFNO(inp_shape=[grid_size, grid_size], in_channels=1,
    #                  out_channels=1, **model_config).
    raise UnfinishedExerciseError(
        "Level 2: implement build_model in fno_physicsnemo_l2.py and save. "
        "Use --reference for the instructor baseline."
    )


if __name__ == "__main__":
    try:
        run(2, build_model, dataset_builder=build_datasets)
    except UnfinishedExerciseError as error:
        raise SystemExit(str(error)) from None
