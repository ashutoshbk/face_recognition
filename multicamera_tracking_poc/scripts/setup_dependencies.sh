#!/usr/bin/env bash
set -euo pipefail

ROOT="${CAMEX_MCT_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
DESTINATION="${CAMEX_MCT_THIRD_PARTY:-$ROOT/third_party}"
LOCK_FILE="$ROOT/third_party.lock"

mkdir -p "$DESTINATION"

while read -r name url commit; do
  [[ -z "$name" ]] && continue
  target="$DESTINATION/$name"
  if [[ -d "$target/.git" ]]; then
    current="$(git -C "$target" rev-parse HEAD)"
    if [[ "$current" != "$commit" ]]; then
      echo "$name is at $current; expected $commit" >&2
      exit 1
    fi
    echo "reusing $name at $commit"
    continue
  fi
  if [[ -e "$target" ]]; then
    echo "refusing to overwrite existing non-Git path: $target" >&2
    exit 1
  fi
  git clone --filter=blob:none --no-checkout "$url" "$target"
  git -C "$target" fetch --depth 1 origin "$commit"
  git -C "$target" checkout --detach "$commit"
done < "$LOCK_FILE"

bot_sort_patch="$ROOT/patches/bot-sort-external-reid.patch"
bot_sort_checkout="$DESTINATION/BoT-SORT"
if git -C "$bot_sort_checkout" apply --check "$bot_sort_patch" 2>/dev/null; then
  git -C "$bot_sort_checkout" apply "$bot_sort_patch"
  echo "applied BoT-SORT external-ReID compatibility patch"
elif git -C "$bot_sort_checkout" apply --reverse --check "$bot_sort_patch" 2>/dev/null; then
  echo "reusing patched BoT-SORT checkout"
else
  echo "BoT-SORT compatibility patch does not apply cleanly" >&2
  exit 1
fi
