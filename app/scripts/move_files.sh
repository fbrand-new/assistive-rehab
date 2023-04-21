#!/bin/bash

HOME_DIR="/home/r1-user/experimentData_27_4_23"
ETAPAS_DIR="/home/r1-user/etapas-results/"
ETAPAS_MOTION_DIR="${ETAPAS_DIR}/motion-analysis"
ETAPAS_EVENT_DIR="${ETAPAS_DIR}/events"

SUBJECT=$1
TRIAL=$2

TRIAL_DIR="${HOME_DIR}/subj_${SUBJECT}/tr_${TRIAL}"
mkdir -p ${TRIAL_DIR}

LAST_MOTION_FILE=$(ls -t ${ETAPAS_MOTION_DIR} | head -n 1)
LAST_EVENT_FILE=$(ls -t ${ETAPAS_EVENT_DIR} | head -n 1)

cp ${ETAPAS_MOTION_DIR}/${LAST_MOTION_FILE} ${TRIAL_DIR}/data.mat
cp ${ETAPAS_EVENT_DIR}/${LAST_EVENT_FILE} ${TRIAL_DIR}

#TODO: complete with the dump files