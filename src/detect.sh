#!/bin/bash
#-----------------------------------------------------------------------------------------------------------------------------------
# The scope of this script is to
# * create a conda environment on the local SSD from a predeined env file
# * copy tar compressed files from a specific folder to the local SSD and extract them
# * run a predefined python script
# * write various information to a log file
#
# NOTE:
#      * This script is not a simple copy and paste scripts! If you have no idea how it works ask the author of the script before
#        you execute it.
#      * This script will - as it is - work only on the STYX GPU Cluster at FRaunhofer ITWM
#
# author: Dr. Dominik Straßel, 05/2021
# copyright: GPLv3
#-----------------------------------------------------------------------------------------------------------------------------------


# define parameters ----------------------------------------------------------------------------------------------------------------

# define data directory
DATA_ROOT_PATH="/home/DATA/ImageNet_raw_tar"

DATA_PATH="/home/SSD/data"

# define conda env
ENV_FILES_ROOT="/home/ENV-FILES"
ENV_NAME="cuda--11-1-1--pytorch--1-8-1"


# define python command including script and all parameters to execute

###################### cif10 
## cif10 Logistic Regression
PYTHON_COMMAND="python -u detect_adversarials_imagenet32.py -clf LR --net cif10 --attack pgd --detector LayerPFS"
# PYTHON_COMMAND="python -u detect_adversarials_imagenet32.py --net cif10 --detector LayerPFS --attack df"
# PYTHON_COMMAND="python -u detect_adversarials_imagenet32.py --net cif10 --detector LayerPFS --attack cw"
# PYTHON_COMMAND="python -u detect_adversarials_imagenet32.py --net cif10 --detector LayerPFS --attack std"

## cif10 Random Forest


###################### cif100
## cif100 Logistic Regression


## cif100 Random Forest



###################### imagenet32
## imagenet32 Logistic Regression


## imagenet32 Random Forest


# NOTE: the "-u" flag is essential as we want the output of the python script directly and not the moment the buffer it full
#-----------------------------------------------------------------------------------------------------------------------------------

echo "${PYTHON_COMMAND}"

# define function die that is called if a command fails ----------------------------------------------------------------------------
function die () {
  echo "ERROR: ${1}"
  exit 200
}
#-----------------------------------------------------------------------------------------------------------------------------------


# define function to print time and date -------------------------------------------------------------------------------------------
function currenttime () {
  date +"[%F %T.%3N]"
}
#-----------------------------------------------------------------------------------------------------------------------------------


# define logfile and log function --------------------------------------------------------------------------------------------------
ROOT_DIR="$(pwd)"

[[ ! -d logs ]] && mkdir "logs"

LOG_FILE_TIME="$(date +"[%F-%T]")"
LOG_FILE="${ROOT_DIR}/logs/$(hostname -s)--${LOG_FILE_TIME}.log"
[[ -f "${LOG_FILE}" ]] && mv "${LOG_FILE}" "${LOG_FILE}.bak"
#-----------------------------------------------------------------------------------------------------------------------------------


# define log function --------------------------------------------------------------------------------------------------------------
function log () {
  echo "$(currenttime): ${1}" >> "${LOG_FILE}"
  echo "$(currenttime): ${1}"
}
#-----------------------------------------------------------------------------------------------------------------------------------


# define function to format time in seconds to readable output ---------------------------------------------------------------------
format_seconds() {
  local SEC TOT RET

  SEC="${1}"

  RET="$((SEC%60))s"
  TOT=$((SEC%60))

  if [[ "${SEC}" -gt "${TOT}" ]]; then
    RET="$((SEC%3600/60))m:${RET}"
    (( TOT+=$((SEC%3600)) ))
  fi

  if [[ "${SEC}" -gt "${TOT}" ]]; then
    RET="$((SEC%86400/3600))h:${RET}"
    (( TOT+=$((SEC%86400)) ))
  fi

  if [[ "${SEC}" -gt "${TOT}" ]]; then
    RET="$((SEC/86400))d:${RET}"
  fi

  echo "${RET}"
}
#-----------------------------------------------------------------------------------------------------------------------------------


# load conda init file -------------------------------------------------------------------------------------------------------------
# CONDA_INIT_FILE="/opt/anaconda3/etc/profile.d/conda.sh"
# if [[ -f "${CONDA_INIT_FILE}" ]];then
#   source "${CONDA_INIT_FILE}"
# else
#   die "cannot find conda init file '${CONDA_INIT_FILE}'"
# fi
#-----------------------------------------------------------------------------------------------------------------------------------


# preform checks -------------------------------------------------------------------------------------------------------------------
# data directories
# [[ ! -d "${DATA_ROOT_PATH}" ]] && die "${DATA_ROOT_PATH} does not exist"


# conda envs
# [[ ! -d "${ENV_FILES_ROOT}" ]] && die "${ENV_FILES_ROOT} does not exist"
# [[ ! -f "${ENV_FILES_ROOT}/${ENV_NAME}.yml" ]] && die "${ENV_FILES_ROOT}/${ENV_NAME}.yml does not exist"


# check if folder to local SSD exists
# [[ ! -d "/home/SSD" ]] && die "/home/SSD does not exist"
#-----------------------------------------------------------------------------------------------------------------------------------


# start experiment -----------------------------------------------------------------------------------------------------------------
log "start experiment on $(hostname -s)"
#-----------------------------------------------------------------------------------------------------------------------------------


# check if env exists --------------------------------------------------------------------------------------------------------------
# ENV_JOB_DIR="/home/SSD/conda"
# ENV="${ENV_JOB_DIR}/${ENV_NAME}"

# if [[ ! -d "${ENV}" ]];then
#   log "create conda env '${ENV}' from ${ENV_FILES_ROOT}/${ENV_NAME}.yml"
#   START="${SECONDS}"
#   conda env create --prefix "${ENV}" -f "${ENV_FILES_ROOT}/${ENV_NAME}.yml"
#   END="${SECONDS}"
#   ELAPSED_TIME=$((END-START))
#   log "creating conda env '${ENV}' took $(format_seconds ${ELAPSED_TIME})"
# else
#   log "use existing conda env '${ENV}'"
# fi
#-----------------------------------------------------------------------------------------------------------------------------------


# print conda pacakges to log file -------------------------------------------------------------------------------------------------
# log "activate conda env '${ENV}'"
# echo "$(currenttime): activate conda env '${ENV}'"
# conda activate ${ENV}


# echo "$(currenttime): list all packages in '${ENV}'" >> "${LOG_FILE}"
# conda list >> "${LOG_FILE}"
#-----------------------------------------------------------------------------------------------------------------------------------


# export env to yml file -----------------------------------------------------------------------------------------------------------
# YML_FILE="$(hostname -s)--${LOG_FILE_TIME}.yml"
# [[ -f "${YML_FILE}" ]] && mv "${YML_FILE}" "${YML_FILE}.bak"

# log "export conda env ${ENV} to ${YML_FILE}"

# [[ ! -d "job-envs" ]] && mkdir "job-envs"

# conda env export > "${ROOT_DIR}/job-envs/${YML_FILE}"
#-----------------------------------------------------------------------------------------------------------------------------------


# copy and extract data to SSD (if needed) -----------------------------------------------------------------------------------------
# if [[ ! -d "${DATA_PATH}" ]];then
#   log "create data folder ${DATA_PATH}"
#   mkdir -p "${DATA_PATH}" || die "cannot create '${DATA_PATH}'"
#   cd "${DATA_PATH}" || die "canno-workerst open '${DATA_PATH}'"


#   log "copy tar files from ${DATA_ROOT_PATH}"
#   START="${SECONDS}"
#   cp -v ${DATA_ROOT_PATH}/* . || die "cannot copy files from '${DATA_ROOT_PATH}'"
#   END="${SECONDS}"
#   ELAPSED_TIME=$((END-START))
#   log "coping took $(format_seconds ${ELAPSED_TIME})"

#   mapfile -t TAR_FILES < <(find . -maxdepth 1 -name "*.tar" | cut -c3-)

#   log "extract tar files"
#   START="${SECONDS}"
#   for TAR_FILE in "${TAR_FILES[@]}";do
#     tar -xf "${TAR_FILE}"
#   done
#   END="${SECONDS}"
#   ELAPSED_TIME=$((END-START))
#   log "extracting tar files took $(format_seconds ${ELAPSED_TIME})"


#   log "remove tar files"
#   START="${SECONDS}"
#   for TAR_FILE in "${TAR_FILES[@]}";do
#     rm "${TAR_FILE:?}"
#   done
#   END="${SECONDS}"
#   ELAPSED_TIME=$((END-START))
#   log "removing tar files took $(format_seconds ${ELAPSED_TIME})"


#   cd "${ROOT_DIR}" || die "cannot open '${ROOT_DIR}'"
# fi
#-----------------------------------------------------------------------------------------------------------------------------------


# run simulation -------------------------------------------------------------------------------------------------------------------
log "start simulation"
START="${SECONDS}"

${PYTHON_COMMAND} >> "${LOG_FILE}" 2>&1
EXIT_STATE="$?"

END="${SECONDS}"
ELAPSED_TIME=$((END-START))

log "simulation run for $(format_seconds ${ELAPSED_TIME})"

log "finished simulation with exit code '${EXIT_STATE}'"

# echo -e "$(currenttime): $(hostname -s): ${PYTHON_COMMAND} \t| env: ${ENV_NAME} \t| runtime: ${ELAPSED_TIME} \t| exitcode ${EXIT_STATE}" >> "${ROOT_DIR}/logs/experiment-overview.out"
echo -e "$(currenttime): $(hostname -s): ${PYTHON_COMMAND} \t| runtime: ${ELAPSED_TIME} \t| exitcode ${EXIT_STATE}" >> "${ROOT_DIR}/logs/experiment-overview.out"

#-----------------------------------------------------------------------------------------------------------------------------------


#-----------------------------------------------------------------------------------------------------------------------------------
log "finished experiment"

exit 0
#-----------------------------------------------------------------------------------------------------------------------------------