#!/usr/bin/env python3
"""Install only hash-verified model files; requires no host ML packages."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import stat
import tempfile
import urllib.request
import zipfile

MANIFEST = Path(__file__).resolve().parents[1] / 'models_manifest.json'


def sha256(path):
    value = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1024*1024), b''): value.update(block)
    return value.hexdigest()


def copy_member(source, name, destination):
    source = Path(source).expanduser().resolve(strict=True)
    if source.is_dir():
        candidates = [source/name, source/'models'/name]
        found = [p for p in candidates if p.is_file() and not p.is_symlink()]
        if len(found) != 1: raise ValueError('Expected one model file: '+name)
        shutil.copyfile(found[0], destination)
        return
    if not zipfile.is_zipfile(source):
        if source.name != name: raise ValueError('Not the expected model file: '+name)
        shutil.copyfile(source,destination)
        return
    with zipfile.ZipFile(source) as archive:
        matches = [i for i in archive.infolist() if Path(i.filename).name == name and not i.is_dir()]
        if len(matches) != 1: raise ValueError('Expected exactly one ZIP member: '+name)
        item = matches[0]
        if item.file_size > 250_000_000 or stat.S_ISLNK(item.external_attr >> 16):
            raise ValueError('Invalid model member: '+name)
        # Copy an allowlisted basename, never extract archive-controlled paths.
        with archive.open(item) as inp, destination.open('wb') as out:
            shutil.copyfileobj(inp,out,1024*1024)


def install(bundle, output, specs, extra_source=None, extra_spec=None):
    output = Path(output).expanduser().absolute()
    if output.is_symlink(): raise ValueError('Model output must not be a symbolic link')
    wanted = dict(specs)
    if extra_spec: wanted[extra_spec['file']] = extra_spec['sha256']
    if output.exists():
        if output.is_dir() and all((output/n).is_file() and sha256(output/n)==h for n,h in wanted.items()):
            return {'output':str(output),'files':sorted(wanted),'reused':True}
        raise ValueError('Output exists with missing/different files. Choose a NEW output directory; nothing overwritten.')
    output.parent.mkdir(parents=True,exist_ok=True)
    with tempfile.TemporaryDirectory(prefix='.office-models-',dir=output.parent) as temp:
        stage=Path(temp)/'models'; stage.mkdir(mode=0o700)
        for name,expected in wanted.items():
            source=extra_source if extra_spec and name==extra_spec['file'] else bundle
            copy_member(source,name,stage/name)
            if sha256(stage/name)!=expected: raise ValueError('Model checksum mismatch: '+name)
            (stage/name).chmod(0o600)
        (stage/'RESEARCH_USE.txt').write_text('Pretrained model and custom checkpoint rights require separate clearance.\n'
            'This installer does not grant commercial or paid-client evaluation rights.\n'
            'See office_demo/README.md and research_demo/THIRD_PARTY_NOTICES.md.\n')
        stage.rename(output)
    return {'output':str(output),'files':sorted(wanted),'reused':False}


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--bundle',type=Path,required=True,help='Companion ZIP or its extracted models directory')
    parser.add_argument('--output',type=Path,required=True,help='New private models directory outside the checkout')
    group=parser.add_mutually_exclusive_group()
    group.add_argument('--scrfd10g',type=Path,help='Existing det_10g.onnx or official buffalo_l.zip')
    group.add_argument('--download-scrfd10g',action='store_true',help='Download the pinned official InsightFace archive')
    parser.add_argument('--acknowledge-model-restrictions',action='store_true')
    args=parser.parse_args()
    if not args.acknowledge_model_restrictions:
        parser.error('Read model-use notices first, then explicitly --acknowledge-model-restrictions')
    os.umask(0o077)
    manifest=json.loads(MANIFEST.read_text())
    with tempfile.TemporaryDirectory(prefix='office-model-download-') as temp:
        extra=args.scrfd10g
        if args.download_scrfd10g:
            extra=Path(temp)/'buffalo_l.zip'
            request=urllib.request.Request(manifest['scrfd_10g']['archive_url'],headers={'User-Agent':'CAMEX-model-setup/1.0'})
            with urllib.request.urlopen(request,timeout=90) as response, extra.open('wb') as out:
                total=0
                while block:=response.read(1024*1024):
                    total+=len(block)
                    if total>350_000_000: raise ValueError('Model archive exceeds size limit')
                    out.write(block)
            if sha256(extra)!=manifest['scrfd_10g']['archive_sha256']:
                raise ValueError('Official archive checksum mismatch; nothing installed')
        result=install(args.bundle,args.output,manifest['models'],extra,
                       manifest['scrfd_10g'] if extra else None)
    print(json.dumps(result,indent=2))


if __name__=='__main__': main()
