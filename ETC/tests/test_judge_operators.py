"""Operator source extraction uses intentionally incorrect transport fixtures.

Authoritative model/equation answer tests belong in the private judge checkout.
"""
import ast

import pytest

from ETC.judge.operators import FactoryInterpreter, source_nodes
from ETC.judge.expressions import SubmissionError
from ETC.runtime.submission import collect_submission


def dummy_operator_source(level):
    """Non-answer source: valid signatures, deliberately wrong return values."""
    source = """def build_datasets(train_pairs, val_pairs, test_pairs):
    return ()

def build_model(model_config, grid_size):
    return 0
"""
    if level == 3:
        source += """
class ReactionDiffusionPDE(PDE):
    def __init__(self):
        self.dim = 2
        self.equations = {}
"""
    return source


@pytest.mark.parametrize("level", (1, 2, 3))
def test_operator_source_extraction_keeps_only_contract_nodes(level):
    nodes = source_nodes(dummy_operator_source(level) + "\nprivate_value = 42\n", level)
    assert [node.name for node in nodes] == [
        "build_datasets", "build_model", *(["ReactionDiffusionPDE"] if level == 3 else [])]
    assert all(isinstance(node, (ast.FunctionDef, ast.ClassDef)) for node in nodes)


@pytest.mark.parametrize("level", (1, 2, 3))
def test_export_filters_nonexercise_code_without_claiming_correctness(tmp_path, level):
    filename = f"fno_physicsnemo_l{level}.py"
    (tmp_path / filename).write_text(dummy_operator_source(level) + '\nPRIVATE_VALUE="not exported"\n')
    payload = collect_submission("4", tmp_path, levels=(level,))
    assert set(payload["sources"]) == {filename}
    assert "PRIVATE_VALUE" not in payload["sources"][filename]


@pytest.mark.parametrize("expression", [
    '__import__("os").system("id")', 'open("/etc/passwd").read()', '(0).__class__',
])
def test_factory_interpreter_rejects_arbitrary_python(expression):
    with pytest.raises(SubmissionError):
        FactoryInterpreter({}).value(ast.parse(expression, mode="eval").body)
