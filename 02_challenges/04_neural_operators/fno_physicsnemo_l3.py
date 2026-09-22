"""Level 3: FNO plus PhysicsInformer spectral residuals implements PINO.

Keep u - Laplacian(u) = f on [0,1)^2, with periodic boundary conditions.
Edit the FIXME functions and save, or run --reference for the instructor path.
The training loop differentiates both data and physics losses through FNO.
"""
from sympy import Function, Symbol
from torch.utils.data import TensorDataset
from physicsnemo.models.fno import FNO
from physicsnemo.sym.eq.pde import PDE
from physicsnemo.sym.eq.phy_informer import PhysicsInformer

from operator_training import BatchedPhysicsInformer, UnfinishedExerciseError, run


class ReactionDiffusionPDE(PDE):
    def __init__(self):
        self.dim = 2
        x, y = Symbol("x"), Symbol("y")
        u, f = Function("u")(x, y), Function("f")(x, y)
        # FIXME: define self.equations = {"reaction_diffusion": <residual>}
        # using u, f, u.diff(x, 2) and u.diff(y, 2). The residual is zero
        # when the prediction satisfies u - Delta(u) = f.
        raise UnfinishedExerciseError(
            "Level 3: implement ReactionDiffusionPDE.equations in fno_physicsnemo_l3.py and save. "
            "Use --reference for the instructor baseline."
        )


def build_datasets(train_pairs, val_pairs, test_pairs):
    # FIXME: return the three separate TensorDataset objects as in Level 1.
    raise UnfinishedExerciseError(
        "Level 3: implement build_datasets in fno_physicsnemo_l3.py and save, or use --reference."
    )


def build_model(model_config, grid_size):
    # FIXME: create the same FNO backbone as Level 1, with **model_config.
    raise UnfinishedExerciseError(
        "Level 3: implement build_model in fno_physicsnemo_l3.py and save, or use --reference."
    )


def build_physics(device):
    return BatchedPhysicsInformer(PhysicsInformer(
        required_outputs=["reaction_diffusion"], equations=ReactionDiffusionPDE(),
        grad_method="spectral", bounds=[1.0, 1.0], device=str(device)))


if __name__ == "__main__":
    try:
        run(3, build_model, dataset_builder=build_datasets, physics_builder=build_physics)
    except UnfinishedExerciseError as error:
        raise SystemExit(str(error)) from None
