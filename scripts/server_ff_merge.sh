#!/bin/bash
# Fast-forward the server worktree to FETCH_HEAD, clearing the one blocker that has
# bitten this workflow five times: files the incoming commit adds as tracked still
# exist untracked on the server (generated there, or written by a teammate who does
# not commit). git refuses to overwrite them, and `tail -1` of the merge output hides
# the error behind the "Updating ..." line.
#
# For every path the incoming commit touches that is currently untracked, back it up
# to pb_recovery/untracked_backup/ and remove it, then merge. Afterwards each backup
# is diffed against the merged file so any local content that would have been lost is
# named rather than silently discarded.
set -u
cd /home/user/pb_sensor
BK=/home/user/pb_recovery/untracked_backup
mkdir -p "$BK"

echo "=== incoming paths that are untracked here ==="
conflicts=()
while IFS= read -r p; do
  [ -z "$p" ] && continue
  if [ -e "$p" ] && ! git ls-files --error-unmatch "$p" >/dev/null 2>&1; then
    conflicts+=("$p")
  fi
done < <(git diff --name-only HEAD FETCH_HEAD)

if [ ${#conflicts[@]} -gt 0 ]; then
  for p in "${conflicts[@]}"; do
    dest="$BK/$(echo "$p" | tr '/' '_').pre-merge"
    cp -p "$p" "$dest"
    rm -f "$p"
    echo "  moved $p -> $dest"
  done
else
  echo "  none"
fi

echo "=== merge ==="
git merge --ff-only FETCH_HEAD
echo "=== head ==="
git log --oneline -1

echo "=== did any backed-up file differ from the merged version? ==="
for p in "${conflicts[@]:-}"; do
  [ -z "$p" ] && continue
  dest="$BK/$(echo "$p" | tr '/' '_').pre-merge"
  if [ -e "$p" ] && ! diff -q "$dest" "$p" >/dev/null 2>&1; then
    echo "  DIFFERS (local content not in the commit): $p"
  fi
done
echo "  (nothing listed above = every backup matched the merged file)"
