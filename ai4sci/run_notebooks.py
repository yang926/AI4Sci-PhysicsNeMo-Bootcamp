"""Execute authentic course notebooks in the instructor's reference mode.

No cells are replaced or exercise functions filled in. Environment variables
are documented parameters in the lesson notebooks. Saves executed copies and
HTML outside the source lessons. Use a fresh output directory for every run.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
import time
import traceback

import nbformat
from nbclient import NotebookClient
from nbconvert import HTMLExporter

ROOT = Path(__file__).resolve().parents[1]
MINIMUM_IMAGES = {'lab1': 3, 'lab2': 1, 'lab3': 2, 'lab4': 1,
                  'wave': 3, 'fluid': 3, 'climate': 2, 'operators': 3, 'wave_reference': 1}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--steps', type=int, default=2)
    parser.add_argument('--device', choices=['cpu', 'cuda'], default='cpu')
    parser.add_argument('--case', action='append')
    args = parser.parse_args()
    if args.steps < 1:
        parser.error('--steps must be positive')
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=False)
    manifest = json.loads((ROOT / 'ai4sci/course_manifest.json').read_text())
    cases = [{'id': 'start', 'notebook': 'Start_Here.ipynb'},
             {'id': 'preflight', 'notebook': 'ai4sci/00_environment_check.ipynb'}, *manifest['course'],
             {'id': 'wave_reference', 'notebook': 'ai4sci/01_Wave_PINN.ipynb'}]
    if args.case:
        unknown = set(args.case) - {c['id'] for c in cases}
        if unknown:
            parser.error(f'Unknown cases: {unknown}')
        cases = [c for c in cases if c['id'] in args.case]
    environment = dict(os.environ, AI4SCI_DEVICE=args.device, AI4SCI_STEPS=str(args.steps),
                       AI4SCI_REFERENCE='1', MPLBACKEND='Agg',
                       PYTHONUNBUFFERED='1', OMP_NUM_THREADS='2', MKL_NUM_THREADS='2')
    environment['PATH'] = str(Path(sys.executable).parent) + os.pathsep + environment.get('PATH', '')
    environment['IPYTHONDIR'] = str(output / 'ipython')
    environment['JUPYTER_RUNTIME_DIR'] = str(output / 'jupyter-runtime')
    report = {'scope': 'Reference-mode notebook execution and plotting; short run, not convergence',
              'device_requested': args.device, 'steps': args.steps, 'cases': [], 'passed': False}
    for case in cases:
        path = ROOT / case['notebook']
        nb = nbformat.read(path, as_version=4)
        target = output / case['id']
        target.mkdir()
        env = dict(environment, AI4SCI_OUTPUT_DIR=str(target / 'training'),
                   AI4SCI_DATA_DIR=str(target / 'data'))
        started = time.perf_counter()
        record = {'id': case['id'], 'source': case['notebook'],
                  'sha256': hashlib.sha256(path.read_bytes()).hexdigest(), 'passed': False}
        try:
            client = NotebookClient(nb, timeout=600, kernel_name='python3',
                                    resources={'metadata': {'path': str(path.parent)}},
                                    allow_errors=False)
            client.execute(env=env)
            if not all(c.execution_count is not None for c in nb.cells if c.cell_type == 'code'):
                raise AssertionError('Not all code cells executed')
            record['embedded_images'] = sum(
                any(mime in item.get('data', {}) for mime in ('image/png', 'image/svg+xml', 'image/jpeg'))
                for cell in nb.cells if cell.cell_type == 'code' for item in cell.get('outputs', []))
            minimum = MINIMUM_IMAGES.get(case['id'], 0)
            if record['embedded_images'] < minimum:
                raise AssertionError(f"Expected at least {minimum} embedded plots, got {record['embedded_images']}")
            record['passed'] = True
        except Exception:
            record['error'] = traceback.format_exc()
            (target / 'error.txt').write_text(record['error'])
        record['seconds'] = time.perf_counter() - started
        nbformat.write(nb, target / 'executed.ipynb')
        html, _ = HTMLExporter().from_notebook_node(nb)
        (target / 'executed.html').write_text(html)
        report['cases'].append(record)
        (output / 'report.json').write_text(json.dumps(report, indent=2) + '\n')
        print(f"{case['id']}: {'PASS' if record['passed'] else 'FAIL'} ({record['seconds']:.1f}s)", flush=True)
    report['passed'] = all(c['passed'] for c in report['cases'])
    (output / 'report.json').write_text(json.dumps(report, indent=2) + '\n')
    return 0 if report['passed'] else 1


if __name__ == '__main__':
    raise SystemExit(main())
