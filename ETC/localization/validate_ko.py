"""Check the Korean edition against the frozen English execution baseline."""
import hashlib
import json
from pathlib import Path
import re
import xml.etree.ElementTree as ET

ROOT = Path(__file__).resolve().parents[2]


def digest(value):
    return hashlib.sha256(value).hexdigest()


def git_blob_digest(value):
    return hashlib.sha1(b'blob ' + str(len(value)).encode() + b'\0' + value).hexdigest()


def code_digest(notebook):
    cells = [cell for cell in notebook['cells'] if cell['cell_type'] == 'code']
    return digest(json.dumps(cells, sort_keys=True, ensure_ascii=False).encode())


def main():
    baseline = json.loads((ROOT / 'ETC/localization/english-baseline.json').read_text())
    errors = []
    for relative, expected in baseline['execution_files'].items():
        path = ROOT / relative
        if not path.is_file() or git_blob_digest(path.read_bytes()) != expected:
            errors.append('Execution/data file changed: ' + relative)
    for relative, expected in baseline['notebooks'].items():
        notebook = json.loads((ROOT / relative).read_text())
        if code_digest(notebook) != expected:
            errors.append('Executable notebook cells changed: ' + relative)
        markdown = '\n'.join(''.join(c['source']) for c in notebook['cells'] if c['cell_type'] == 'markdown')
        if not re.search(r'[가-힣]', markdown):
            errors.append('Missing Korean explanation: ' + relative)
    for path in sorted((ROOT / 'ETC/assets/teaching').glob('*.svg')):
        svg = ET.parse(path).getroot()
        for element in svg.iter():
            if element.tag.rsplit('}', 1)[-1] in {'script', 'foreignObject'}:
                errors.append('Unsafe SVG element: ' + path.name)
            if any(key.lower().startswith('on') for key in element.attrib):
                errors.append('Unsafe SVG event handler: ' + path.name)
        if not re.search(r'[가-힣]', ''.join(svg.itertext())):
            errors.append('Missing Korean diagram labels: ' + path.name)
    for relative in ('ETC/launchable/setup.sh', 'ETC/launchable/update.sh'):
        script = (ROOT / relative).read_text()
        if '/ko/ETC/launchable/bootstrap.py' not in script or '--ref ko' not in script:
            errors.append('Updater does not stay on Korean branch: ' + relative)
    print(json.dumps({'source_commit': baseline['source_commit'],
                      'execution_files': len(baseline['execution_files']),
                      'notebooks': len(baseline['notebooks']),
                      'errors': errors, 'status': 'fail' if errors else 'pass'}, indent=2))
    raise SystemExit(bool(errors))


if __name__ == '__main__':
    main()
