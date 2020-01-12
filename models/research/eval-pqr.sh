CURRENT_DIR=$(pwd)
WORK_DIR="${CURRENT_DIR}/deeplab"
DATASET_DIR="datasets"

PQR_FOLDER="PQR"
EXP_FOLDER="exp/train_on_trainval_set"
INIT_FOLDER="${WORK_DIR}/${DATASET_DIR}/${PQR_FOLDER}/${EXP_FOLDER}/init_models"
TRAIN_LOGDIR="${WORK_DIR}/${DATASET_DIR}/${PQR_FOLDER}/${EXP_FOLDER}/train"
EVAL_LOGDIR="${WORK_DIR}/${DATASET_DIR}/${PQR_FOLDER}/${EXP_FOLDER}/eval"
DATASET="${WORK_DIR}/${DATASET_DIR}/${PQR_FOLDER}/tfrecord"

python3 "${WORK_DIR}"/eval.py \
--logtostderr \
--eval_split="val" \
--model_variant="xception_65" \
--atrous_rates=6 \
--atrous_rates=12 \
--atrous_rates=18 \
--output_stride=16 \
--decoder_output_stride=4 \
--eval_crop_size=1000,667 \
--checkpoint_dir="${TRAIN_LOGDIR}" \
--eval_logdir="${EVAL_LOGDIR}" \
--dataset_dir="${DATASET}" \
--max_number_of_evaluations=1