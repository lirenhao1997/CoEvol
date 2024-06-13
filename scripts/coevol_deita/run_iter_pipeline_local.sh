WORK_DIR=$(dirname $(dirname $(dirname $(readlink -f $0))))
EDIT_DIR="${WORK_DIR}"/edit
cd "${EDIT_DIR}"

# Run coevol for multi-turn deita-6k
JOB_NAME="multi-deita6k_evol-mixtral"
python main.py \
    --task_name edit \
    --root_path ../data/ \
    --save_folder_name ${JOB_NAME} \
    --dataset_name deita/multi-deita6k.json \
    --dataset_format sharegpt \
    --conv_wind_size 3 \
    --max_optimize_len 8192 \
    --use_local_model \
    --model_name mixtral \
    --proxy_api_url http://0.0.0.0:8000/v1 \
    --num_workers 100 \
    --max_evol_iter 3 \
    --max_tokens 1000 \
    --completion_number 1 \
    --temperature 0.0 \
    --top_p 1.0 \
    --edit_mode 4 \
    "$@"

# Run coevol for single-turn deita-6k
JOB_NAME="single-deita6k_evol-mixtral"
python main.py \
    --task_name edit \
    --root_path ../data/ \
    --save_folder_name ${JOB_NAME} \
    --dataset_name deita/single-deita6k.json \
    --dataset_format sharegpt \
    --max_optimize_turn 1 \
    --use_local_model \
    --model_name mixtral \
    --proxy_api_url http://0.0.0.0:8000/v1 \
    --num_workers 100 \
    --max_evol_iter 3 \
    --max_tokens 1000 \
    --completion_number 1 \
    --temperature 0.0 \
    --top_p 1.0 \
    --edit_mode 4 \
    "$@"