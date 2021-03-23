#!/bin/bash

DIR_CFG="$(cd "$(dirname "${BASH_SCRIPT[0]}")" && pwd)"
HARD_STORAGE_CFG=${DIR_CFG}/hard_storage.json
QUANT_HOME=$(cd $(dirname ${DIR_CFG}) && pwd)

# data folder
PYTHON_READ_HARD_STORAGE_DATA="import sys; import json; fp = open('${HARD_STORAGE_CFG}', 'r'); d = json.load(fp); fp.close(); print(d['data'])"
HARD_HOME_DATA=$(python -c "${PYTHON_READ_HARD_STORAGE_DATA}")/$(basename ${QUANT_HOME})
mkdir -p ${HARD_HOME_DATA}/problems

# logs folder
PYTHON_READ_HARD_STORAGE_LOGS="import sys; import json; fp = open('${HARD_STORAGE_CFG}', 'r'); d = json.load(fp); fp.close(); print(d['logs'])"
HARD_HOME_LOGS=$(python -c "${PYTHON_READ_HARD_STORAGE_LOGS}")/$(basename ${QUANT_HOME})
mkdir -p ${HARD_HOME_LOGS}/problems
