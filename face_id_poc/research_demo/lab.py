"""Portable operator entry point; research data and weights remain external."""
import argparse
import hashlib
import json
import platform
import shutil
import subprocess
import sys
import tarfile
import zipfile
from pathlib import Path
from settings import ROOT,HOME,DATA,DATASET,MODELS,RESULT

QDRANT={
 ('Darwin','arm64'):('aarch64-apple-darwin','d24dfd25f4584684b337dd30ee016472cd25f53fc2501f1cd058ab3d7ee0b019'),
 ('Darwin','x86_64'):('x86_64-apple-darwin','e095b682ad7460baff43827eaa13647b5ec97c48edcb676646ae2f4108395044'),
 ('Linux','x86_64'):('x86_64-unknown-linux-musl','50b253243309ed0ae50a19f678f0d0adcc4f0b567b3bd2c5c0c7e02fa8404bb6'),
 ('Linux','aarch64'):('aarch64-unknown-linux-musl','6970b93b56fa1203f0cea47d2f330fdb654988fd56617aa77e8478cffd3c0ccb')}

def digest(path):
    with Path(path).open('rb') as stream:return hashlib.file_digest(stream,'sha256').hexdigest()

def download(url,target,expected=None):
    target=Path(target);target.parent.mkdir(parents=True,exist_ok=True)
    if target.exists() and (expected is None or digest(target)==expected):return
    partial=target.with_suffix(target.suffix+'.part')
    subprocess.run(['curl','--fail','--location','--retry','3','--continue-at','-','--output',str(partial),url],check=True)
    if expected and digest(partial)!=expected:raise ValueError(f'Checksum mismatch: {target.name}; partial retained')
    partial.replace(target)

def assets(args):
    if not args.accept_research_terms:raise ValueError('Review THIRD_PARTY_NOTICES.md, then supply --accept-research-terms')
    from vision.networks import MODEL_HASHES
    MODELS.mkdir(parents=True,exist_ok=True)
    if args.model_bundle:
        bundle=Path(args.model_bundle).expanduser()
        if args.model_bundle.startswith('https://'):
            bundle=HOME/'downloads/model_bundle.zip';download(args.model_bundle,bundle)
        if bundle.is_dir():
            for name,sha in MODEL_HASHES.items():
                source=bundle/name
                if digest(source)!=sha:raise ValueError(f'Model checksum mismatch: {name}')
                if source.resolve()!=(MODELS/name).resolve():shutil.copy2(source,MODELS/name)
        else:
            with zipfile.ZipFile(bundle) as archive:
                for name,sha in MODEL_HASHES.items():
                    matches=[x for x in archive.infolist() if Path(x.filename).name==name]
                    if len(matches)!=1:raise ValueError(f'Expected one {name} in model bundle')
                    temporary=MODELS/(name+'.part')
                    with archive.open(matches[0]) as source,temporary.open('wb') as target:shutil.copyfileobj(source,target)
                    if digest(temporary)!=sha:raise ValueError(f'Model checksum mismatch: {name}')
                    temporary.replace(MODELS/name)
    for name,sha in MODEL_HASHES.items():
        if not (MODELS/name).exists() or digest(MODELS/name)!=sha:raise ValueError(f'Missing model: {name}; supply the separate model bundle')
    key=(platform.system(),platform.machine())
    if key not in QDRANT:raise ValueError(f'Unsupported platform {key}')
    target,sha=QDRANT[key];package=HOME/'downloads/qdrant.tar.gz'
    download(f'https://github.com/qdrant/qdrant/releases/download/v1.19.2/qdrant-{target}.tar.gz',package,sha)
    binary=HOME/'vendor/qdrant';binary.parent.mkdir(parents=True,exist_ok=True)
    with tarfile.open(package) as archive:
        member=next(x for x in archive.getmembers() if x.isfile() and Path(x.name).name=='qdrant')
        with archive.extractfile(member) as source,binary.open('wb') as output:shutil.copyfileobj(source,output)
    binary.chmod(0o755)
    download('https://raw.githubusercontent.com/qdrant/qdrant/v1.19.2/LICENSE',HOME/'vendor/QDRANT_LICENSE.txt')
    print('Model checksums verified; native Qdrant installed. No GPU runtime required.')

def dataset(args):
    if not args.accept_research_terms:raise ValueError('Review the DigiFace R-UDA, then supply --accept-research-terms for non-commercial research')
    DATASET.mkdir(parents=True,exist_ok=True)
    manifest=json.loads((ROOT/'assets/digiface_manifest.json').read_text())
    if args.parts:
        manifest['archives']=[entry for i,entry in enumerate(manifest['archives'],1) if i in args.parts]
        manifest['total_png_count']=sum(e['png_count'] for e in manifest['archives'])
        manifest['total_identity_folders_across_archives']=sum(e['identity_folder_count'] for e in manifest['archives'])
        manifest['total_download_bytes']=sum(e['bytes'] for e in manifest['archives'])
        manifest['selection']='subset; download all eight parts for the full benchmark'
    download('https://raw.githubusercontent.com/microsoft/DigiFace1M/main/README.md',DATASET/'README.official.md')
    download('https://raw.githubusercontent.com/microsoft/DigiFace1M/main/LICENSE',DATASET/'LICENSE.official.txt')
    for entry in manifest['archives']:
        destination=DATASET/entry['file']
        if not destination.exists() and shutil.disk_usage(DATASET).free<entry['bytes']+2*1024**3:raise ValueError('Insufficient free space')
        print(f"Downloading / verifying {entry['file']}",flush=True)
        download(entry['url'],destination,entry['sha256'])
        if destination.stat().st_size!=entry['bytes']:raise ValueError('Archive size mismatch')
        if args.check_crc:
            with zipfile.ZipFile(destination) as archive:
                if archive.testzip():raise ValueError(f"CRC failure: {entry['file']}")
    manifest['status']='complete'
    (DATASET/'download_manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    print(f"{len(manifest['archives'])} verified archives in {DATASET}; not extracted.")

def run(command):subprocess.run([sys.executable,*command],cwd=ROOT,check=True)

def main():
    parser=argparse.ArgumentParser(description=__doc__);sub=parser.add_subparsers(dest='command',required=True)
    a=sub.add_parser('assets');a.add_argument('--model-bundle');a.add_argument('--accept-research-terms',action='store_true')
    d=sub.add_parser('dataset');d.add_argument('--accept-research-terms',action='store_true');d.add_argument('--check-crc',action='store_true')
    d.add_argument('--parts',nargs='+',type=int,choices=range(1,9))
    for name in ['doctor','start','stop','status','build','pause','fixtures','test']:sub.add_parser(name)
    args=parser.parse_args()
    if args.command=='assets':assets(args)
    elif args.command=='dataset':dataset(args)
    elif args.command=='doctor':
        from vision.networks import MODEL_HASHES
        print(json.dumps({'python':sys.version.split()[0],'platform':platform.platform(),'home':str(HOME),
            'models':{n:(MODELS/n).is_file() and digest(MODELS/n)==h for n,h in MODEL_HASHES.items()},
            'qdrant_binary':(HOME/'vendor/qdrant').is_file(),'dataset_manifest':(DATASET/'download_manifest.json').is_file(),
            'gallery_result':RESULT.is_file()},indent=2))
    elif args.command in ('start','stop','status'):run(['manage.py',args.command])
    elif args.command=='build':run(['manage.py','start-db']);run(['-m','synthetic_bench.pipeline']);run(['manage.py','start'])
    elif args.command=='pause':run(['-m','synthetic_bench.control','pause'])
    elif args.command=='fixtures':run(['qa_fixtures.py']);run(['qa_sync_fixture.py'])
    elif args.command=='test':run(['-m','pytest','-q'])

if __name__=='__main__':
    try:main()
    except (ValueError,OSError,subprocess.CalledProcessError) as error:raise SystemExit(str(error))
