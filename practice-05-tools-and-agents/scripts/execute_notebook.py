"""Execute the complete notebook with a key-free, network-blocked kernel."""
import argparse
import json
import os
import tempfile
import sys
from pathlib import Path

import nbformat
from nbclient import NotebookClient

ROOT = Path(__file__).resolve().parents[1]
GUARD = '''import os
import socket
for name in list(os.environ):
    if any(word in name for word in ("API_KEY", "TOKEN", "BASE_URL", "COURSE_ENV_FILE", "LITE_LLM")):
        os.environ.pop(name, None)
os.environ["PRACTICE_MODE"] = "replay"
os.environ["LANGCHAIN_TRACING_V2"] = "false"
os.environ["LANGSMITH_TRACING"] = "false"
def deny(*args, **kwargs):
    raise RuntimeError("Network is blocked for the replay notebook check")
socket.socket.connect = deny
socket.create_connection = deny
'''


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=ROOT / 'runs/practice_05_executed.ipynb')
    args = parser.parse_args()
    os.environ.setdefault('IPYTHONDIR', tempfile.mkdtemp(prefix='practice05-runner-'))
    notebook = nbformat.read(ROOT / 'notebooks/practice_05_tools_and_agents.ipynb', as_version=4)
    notebook.cells.insert(0, nbformat.v4.new_code_cell(GUARD))
    env = dict(os.environ, PRACTICE_MODE='replay', IPYTHONDIR=tempfile.mkdtemp(prefix='practice05-ipython-'))
    client = NotebookClient(notebook, timeout=120, kernel_name='python3',
                            resources={'metadata': {'path': str(ROOT)}})
    client.create_kernel_manager()
    client.km.kernel_spec.argv = [sys.executable, '-m', 'ipykernel_launcher', '-f', '{connection_file}']
    client.execute(env=env)
    notebook.cells.pop(0)
    errors = [o for c in notebook.cells for o in c.get('outputs', []) if o.output_type == 'error']
    assert not errors
    path = args.output
    path.parent.mkdir(parents=True, exist_ok=True)
    def ascii_display(value):
        if isinstance(value, dict):
            return {k: ascii_display(v) for k, v in value.items()}
        if isinstance(value, list):
            return [ascii_display(v) for v in value]
        if isinstance(value, str):
            return value.replace(chr(0x200b), '').replace(chr(0x2026), '...')
        return value
    path.write_text(json.dumps(ascii_display(dict(notebook)), indent=1, ensure_ascii=True)+'\n')
    summary = dict(mode='replay', network='blocked in kernel', credentials='removed',
                   code_cells=sum(c.cell_type == 'code' for c in notebook.cells), errors=len(errors))
    print(summary)


if __name__ == '__main__':
    main()
