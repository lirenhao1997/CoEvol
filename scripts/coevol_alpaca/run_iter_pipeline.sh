WORK_DIR=$(dirname $(dirname $(dirname $(readlink -f $0))))
EDIT_DIR="${WORK_DIR}"/edit
cd "${EDIT_DIR}"

JOB_NAME="alpaca_random9k_evol-chatgpt"
python main.py \
    --task_name edit \
    --root_path ../data/ \
    --save_folder_name ${JOB_NAME} \
    --dataset_name alpaca/random_9k.json \
    --dataset_format alpaca \
    --max_optimize_turn 1 \
    --model_name gpt-3.5-turbo-1106 \
    --api_keys ./api_keys.json \
    --num_workers 10 \
    --max_evol_iter 3 \
    --max_tokens 1000 \
    --completion_number 1 \
    --temperature 0.0 \
    --top_p 1.0 \
    --edit_mode 4 \
    "$@"