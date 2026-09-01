#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

forbidden='\.(mp4|avi|mov|mkv|pt|pth|onnx|engine|jsonl)$'
if git ls-files | grep -E "$forbidden"; then
  echo "forbidden media, model, or event file is tracked" >&2
  exit 1
fi

while IFS= read -r file; do
  size="$(wc -c < "$file")"
  if (( size > 10000000 )); then
    echo "tracked file exceeds 10 MB: $file ($size bytes)" >&2
    exit 1
  fi
done < <(git ls-files)

if git grep -I -E '(gho_|ghp_|github_pat_|BEGIN (RSA|OPENSSH|EC) PRIVATE KEY|password[[:space:]]*[:=])' -- ':!scripts/audit_release.sh'; then
  echo "possible credential found in tracked files" >&2
  exit 1
fi

if git grep -I -E '(/Users/|/Volumes/|/home/[[:alnum:]_.-]+/|192\.168\.)' -- ':!scripts/audit_release.sh'; then
  echo "local machine path or private network address found in tracked files" >&2
  exit 1
fi

echo "release audit passed"
