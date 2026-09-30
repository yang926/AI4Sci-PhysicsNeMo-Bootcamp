"""Level 2: replace the FNO backbone with PhysicsNeMo AFNO.

Edit the FIXME functions and save before running.
Grid dimensions must be divisible by the patch dimensions: do not crop the
periodic domain, because that would change the PDE boundary conditions.
"""
from torch.utils.data import TensorDataset
from physicsnemo.models.afno import AFNO

from operator_training import UnfinishedExerciseError, run


############################################################################
# EDIT HERE 1/2: build_datasets (dataset factories)
# Keep the function signature. Replace the raise ... placeholder below
# with your implementation; do not leave it after your return/assignment.
############################################################################
def build_datasets(train_pairs, val_pairs, test_pairs):
    # FIXME: create the three TensorDataset objects as in Level 1.
    raise UnfinishedExerciseError(
        "Level 2: implement build_datasets in fno_physicsnemo_l2.py and save."
    )
# END EDIT HERE 1/2: build_datasets
############################################################################


############################################################################
# EDIT HERE 2/2: build_model (model factory)
# Keep the function signature. Replace the raise ... placeholder below
# with your implementation; do not leave it after your return/assignment.
############################################################################
def build_model(model_config, grid_size):
    # FIXME: check that grid_size is divisible by each patch_size value, then
    # instantiate AFNO for the square grid with one input/output channel.
    # Pass every model_config setting through to the model.
    raise UnfinishedExerciseError(
        "Level 2: implement build_model in fno_physicsnemo_l2.py and save."
    )
# END EDIT HERE 2/2: build_model
############################################################################


if __name__ == "__main__":
    try:
        run(2, build_model, dataset_builder=build_datasets)
    except UnfinishedExerciseError as error:
        raise SystemExit(str(error)) from None
