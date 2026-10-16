#!/bin/bash
# After a push: the remote branch must equal the local working tree. Fails when anything is modified, untracked or missing, or when
# origin/<branch> is not at HEAD. Files that stay local on purpose (listed in .gitignore) are printed at the end.
set -e
cd "$(dirname "$0")/.."
BRANCH=${1:-main}
git fetch --quiet origin
if [ -n "$(git status --porcelain)" ]; then
  echo "working tree differs from HEAD:"; git status --short; exit 1
fi
if [ "$(git rev-parse HEAD)" != "$(git rev-parse "origin/$BRANCH")" ]; then
  echo "origin/$BRANCH is not at HEAD"; exit 1
fi
echo "origin/$BRANCH == HEAD == working tree"
echo "local only (ignored):"; git status --short --ignored | grep '^!!' | sed 's/^!! /  /'
