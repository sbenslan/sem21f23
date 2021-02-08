#!/bin/bash

DIR_CFG=$(cd $(dirname ${BASH_SOURCE[0]}) && pwd)
HARD_STORAGE_CFG=${DIR_CFG}/hard_storage.json


# data folder

PYTHON_READ_HARD_STORAGE_DATA="import sys; import json; fp = open('${HARD_STORAGE_CFG}', 'r'); d = json.load(fp); fp.close(); print(d['data'])"
HARD_QUANTLAB_HOME_DATA=$(python -c "${PYTHON_READ_HARD_STORAGE_DATA}")/QuantLab
mkdir -p ${HARD_QUANTLAB_HOME_DATA}


# logs folder
PYTHON_READ_HARD_STORAGE_LOGS="import sys; import json; fp = open('${HARD_STORAGE_CFG}', 'r'); d = json.load(fp); fp.close(); print(d['logs'])"
HARD_QUANTLAB_HOME_LOGS=$(python -c "${PYTHON_READ_HARD_STORAGE_LOGS}")/QuantLab
mkdir -p ${HARD_QUANTLAB_HOME_LOGS}
