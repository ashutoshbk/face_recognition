"""Portable local paths and loopback ports. No credentials belong in this file."""
import os
from pathlib import Path

ROOT=Path(__file__).resolve().parent
HOME=Path(os.environ.get('CAMEX_LAB_HOME',ROOT/'.runtime')).expanduser().resolve()
DATA=Path(os.environ.get('LOCAL_LAB_DATA',HOME/'data')).expanduser().resolve()
MODELS=Path(os.environ.get('CAMEX_MODELS_DIR',HOME/'models')).expanduser().resolve()
DATASET=Path(os.environ.get('CAMEX_DATASET_DIR',HOME/'datasets/DigiFace1M')).expanduser().resolve()
WEB_PORT=int(os.environ.get('CAMEX_PORT','9160'))
GALLERY_PORT=int(os.environ.get('CAMEX_GALLERY_PORT','9161'))
QDRANT_PORT=int(os.environ.get('CAMEX_QDRANT_PORT','16333'))
QDRANT_GRPC_PORT=int(os.environ.get('CAMEX_QDRANT_GRPC_PORT','16334'))
GALLERY_RUN=os.environ.get('CAMEX_GALLERY_RUN','all_identities')
if not GALLERY_RUN.replace('_','').isalnum():raise ValueError('Invalid gallery run name')
GALLERY_ROOT=Path(os.environ.get('CAMEX_GALLERY_ROOT',DATA/'digiface_research')).expanduser().resolve()
RESULT=GALLERY_ROOT/GALLERY_RUN/'result.json'
GALLERY_DB=RESULT.parent/'embeddings.sqlite3'
WEB_URL=f'http://127.0.0.1:{WEB_PORT}'
GALLERY_URL=f'http://127.0.0.1:{GALLERY_PORT}'


def process_owned(pid,needle):
    import subprocess
    result=subprocess.run(['ps','-p',str(pid),'-o','command='],capture_output=True,text=True)
    if needle not in result.stdout:return False
    if Path('/proc').exists():
        try:return Path(f'/proc/{pid}/cwd').resolve()==ROOT
        except OSError:return False
    import shutil
    executable=shutil.which('lsof')
    if not executable:return False
    result=subprocess.run([executable,'-a','-p',str(pid),'-d','cwd','-Fn'],capture_output=True,text=True)
    return f'n{ROOT}' in result.stdout.splitlines()
