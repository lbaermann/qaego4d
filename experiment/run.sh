#!/bin/bash
set -e

# Run this within the project main directory (where the directories experiment, model, ... reside)

if [[ -z $TRANSFORMERS_CACHE ]]; then
  echo "TRANSFORMERS_CACHE env var has to be set!"
  exit 1
fi
if [[ ! -d $TRANSFORMERS_CACHE ]]; then
  echo "TRANSFORMERS_CACHE directory should exist (to avoid unintended downloads). Current: '$TRANSFORMERS_CACHE'"
  exit 1
fi

python run.py "${@:1}"
