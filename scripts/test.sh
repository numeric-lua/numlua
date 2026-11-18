#!/usr/bin/env bash
#
# Run the dependency-free numlua test suite.
#
# The default layout expects luatorch either beside this checkout or two levels
# above it (the old staging layout).  Linked worktrees are resolved through the
# main checkout, whose sibling luatorch is still found.  Override with
# LUATORCH_ROOT.
#
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
luatorch_root="${LUATORCH_ROOT:-}"

if [[ -n "$luatorch_root" ]]; then
  if [[ ! -f "$luatorch_root/lua/torch/init.lua" ]]; then
    echo "numlua: LUATORCH_ROOT='$luatorch_root' does not contain lua/torch/init.lua" >&2
    exit 1
  fi
  luatorch_root="$(cd "$luatorch_root" && pwd)"
else
  candidates=(
    "$repo_root/../luatorch"
    "$repo_root/../.."
    "$repo_root/../../luatorch"
  )

  git_common_dir="$(git -C "$repo_root" rev-parse --git-common-dir 2>/dev/null || true)"
  if [[ -n "$git_common_dir" ]]; then
    if [[ "$git_common_dir" != /* ]]; then
      git_common_dir="$repo_root/$git_common_dir"
    fi
    if [[ -d "$git_common_dir" ]]; then
      main_checkout="$(cd "$git_common_dir/.." && pwd)"
      candidates+=("$main_checkout/../luatorch")
    fi
  fi

  for candidate in "${candidates[@]}"; do
    if [[ -f "$candidate/lua/torch/init.lua" ]]; then
      luatorch_root="$(cd "$candidate" && pwd)"
      break
    fi
  done
fi

if [[ -z "$luatorch_root" ]]; then
  echo "numlua: cannot find luatorch. Set LUATORCH_ROOT to its checkout." >&2
  exit 1
fi

mkdir -p "$repo_root/tests/.tmp"
cd "$repo_root"

export LUA_PATH="$luatorch_root/lua/?.lua;$luatorch_root/lua/?/init.lua;$repo_root/?.lua;$repo_root/?/init.lua;;"
exec "${LUAJIT:-luajit}" tests/run.lua "$@"
