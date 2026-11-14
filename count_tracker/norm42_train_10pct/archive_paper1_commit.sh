#!/usr/bin/env bash
# Create an immutable source archive of the Paper1 commit used by the baseline.
# The destination must not already exist.

set -euo pipefail

if [ "$#" -ne 2 ]; then
  echo "usage: $0 SOURCE_REPOSITORY DESTINATION" >&2
  exit 2
fi

source_repository=$(realpath "$1")
destination=$2
commit=90bd576c619416dbd7889eb1900ec80ac687ab92

if [ -e "$destination" ]; then
  echo "refusing to replace $destination" >&2
  exit 1
fi

mkdir -p "$destination"
git --git-dir="$source_repository/.git" archive "$commit" \
  | tar -xf - -C "$destination"
printf '%s\n' "$commit" > "$destination/.source_commit"

echo "Archived Paper1 commit $commit -> $destination"
