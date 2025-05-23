#!/usr/bin/env zsh

BPREVEAL_PATH=..

cd $BPREVEAL_PATH

if git status | grep -q "nothing to commit"
then
    echo "Git is clean. Performing uv build."
    uv build
else
    echo "Git has unstaged changes."
fi
