import argparse
import json
import subprocess
import sys
import urllib.request
from settings import ROOT,GALLERY_URL

def request(path,method='GET'):
    with urllib.request.urlopen(urllib.request.Request(GALLERY_URL+path,method=method),timeout=10) as response:return json.load(response)

def main():
    parser=argparse.ArgumentParser();parser.add_argument('action',choices=['start','resume','pause','status']);a=parser.parse_args()
    if a.action in ('start','resume'):subprocess.run([sys.executable,str(ROOT/'manage.py'),'start'],cwd=ROOT,check=True)
    if a.action=='resume':print(json.dumps(request('/api/build/start','POST'),indent=2))
    elif a.action=='pause':print(json.dumps(request('/api/build/pause','POST'),indent=2))
    else:print(json.dumps(request('/api/status'),indent=2))

if __name__=='__main__':main()
