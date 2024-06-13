WORK_DIR=$(dirname $(dirname $(dirname $(readlink -f $0))))
EDIT_DIR="${WORK_DIR}"/edit
cd "${EDIT_DIR}"

# Run coevol for multi-turn deita-6k
JOB_NAME="multi-deita6k_evol-chatgpt"
python main.py \
    --task_name edit \
    --root_path ../data/ \
    --save_folder_name ${JOB_NAME} \
    --dataset_name deita/multi-deita6k.json \
    --dataset_format sharegpt \
    --conv_wind_size 3 \
    --max_optimize_len 8192 \
    --model_name gpt-3.5-turbo-1106 \
    --proxy_api_url https://api.d1chun.com/v1 \
    --api_keys ./api_keys.json \
    --num_workers 50 \
    --max_evol_iter 3 \
    --max_tokens 1000 \
    --completion_number 1 \
    --temperature 0.0 \
    --top_p 1.0 \
    --edit_mode 4 \
    "$@"

# Run coevol for single-turn deita-6k
JOB_NAME="single-deita6k_evol-chatgpt"
python main.py \
    --task_name edit \
    --root_path ../data/ \
    --save_folder_name ${JOB_NAME} \
    --dataset_name deita/single-deita6k.json \
    --dataset_format sharegpt \
    --conv_wind_size 3 \
    --max_optimize_len 8192 \
    --model_name gpt-3.5-turbo-1106 \
    --proxy_api_url https://api.d1chun.com/v1 \
    --api_keys ./api_keys.json \
    --num_workers 50 \
    --max_evol_iter 3 \
    --max_tokens 1000 \
    --completion_number 1 \
    --temperature 0.0 \
    --top_p 1.0 \
    --edit_mode 4 \
    "$@"