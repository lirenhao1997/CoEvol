from tqdm import tqdm
import jsonlines
import tiktoken
import json
import os


def num_tokens_from_string(string: str=None, disallowed_special=False):
    """Returns the number of tokens in a text string."""
    encoding = tiktoken.encoding_for_model("gpt-3.5-turbo")
    if disallowed_special:
        num_tokens = len(encoding.encode(string, disallowed_special=()))
    else:
        num_tokens = len(encoding.encode(string))
    return num_tokens


def json_dump(path, res):
    with open(path, 'w') as file:
        json.dump(res, file)


def json_load(path):
    with open(path, 'r') as file:
        tmp = json.load(file)
    return tmp


def jsonl_load(f):
    data = []
    with jsonlines.open(f, 'r') as reader:
        for item in reader:
            data.append(item)
    return data


def jsonl_dump(f, data):
    with jsonlines.open(f, "w") as f:
        f.write_all(data)


def gen_multi_conv_train_dataset(res_name: str = None, output_format="json"):
    res_path = os.path.join("./res", res_name)
    sft_base="../data/sft"
    if not os.path.exists(sft_base):
        os.mkdir(sft_base)
    sft_path_json = os.path.join(sft_base, res_name)+".json"
    sft_path_jsonl = os.path.join(sft_base, res_name)+".jsonl"
    sft_list = []
    total_count = 0
    error_count=0
    total_evol_conv_round = 0
    sum_ori_resp_len=0
    sum_evol_resp_len=0
    evol_round_dist={
        0:0,
        1:0,
        2:0,
        3:0
    }

    for f_name in tqdm(os.listdir(res_path)):
        res = json_load(os.path.join(res_path, f_name))
        total_count += 1
        if "edit_error" in res:
            error_count+=1
            res["evol_conversations"] = res["conversations"]
            res["total_evol_conv_round"] = 0
        
        total_evol_conv_round += res["total_evol_conv_round"]

        for indx in range(res["total_evol_conv_round"]):
            sum_ori_resp_len += num_tokens_from_string(res["conversations"][indx*2+1]["value"], disallowed_special=True)
            sum_evol_resp_len += num_tokens_from_string(res["evol_conversations"][indx*2+1]["value"], disallowed_special=True)
            evol_round_dist[res["optimization_steps"][indx]["evol_round"]] += 1

        sft_list.append(
            {
                "id": res["id"],
                "conversations": res["evol_conversations"],
                "source": res["source"],
            }
        )
    sft_list = sorted(sft_list, key=lambda x:x["id"])

    avg_ori_resp_len = sum_ori_resp_len / total_evol_conv_round
    avg_evol_resp_len = sum_evol_resp_len / total_evol_conv_round
    print(f"Total number of samples: {total_count}")
    print(f"Total number of error samples (keeped): {error_count}")
    print(f"Total number of evolved turns: {total_evol_conv_round} (Avg.: {total_evol_conv_round/total_count:.2f})")
    print(f"Average number of tokens in origin responses: {avg_ori_resp_len:.2f}")
    print(f"Average number of tokens in evolved responses: {avg_evol_resp_len:.2f}")
    print(f"Distribution of evol rounds for each sample: {evol_round_dist}")

    if output_format=="json":
        json_dump(sft_path_json, sft_list)
    elif output_format=="jsonl":
        jsonl_dump(sft_path_jsonl, sft_list)
    else:
        raise NotImplementedError


def gen_single_conv_train_dataset(res_name: str = None, output_format="json", data_format="alpaca"):
    res_path = os.path.join("./res", res_name)
    sft_base="../data/sft"
    if not os.path.exists(sft_base):
        os.mkdir(sft_base)
    sft_path_json = os.path.join(sft_base, res_name)+".json"
    sft_path_jsonl = os.path.join(sft_base, res_name)+".jsonl"
    sft_list = []
    total_count = 0
    error_count=0
    sum_ori_resp_len=0
    sum_evol_resp_len=0
    evol_round_dist={
        0:0,
        1:0,
        2:0,
        3:0
    }

    for f_name in tqdm(os.listdir(res_path)):
        res = json_load(os.path.join(res_path, f_name))
        total_count += 1
        if "edit_error" in res:
            error_count += 1
            if data_format == "alpaca":
                res["evol_output"] = res["output"]
                res["evol_round"] = 0
            elif data_format == "sharegpt":
                res["evol_output"] = res["conversations"][1]["value"]
                res["evol_round"] = 0
        
        if data_format == "alpaca":
            sum_ori_resp_len += num_tokens_from_string(res["output"], disallowed_special=True)
            sum_evol_resp_len += num_tokens_from_string(res["evol_output"], disallowed_special=True)
            evol_round_dist[res["evol_round"]] += 1
            sft_list.append(
                {
                    "id": res["id"],
                    "instruction": res["instruction"],
                    "input": res["input"],
                    "output": res["evol_output"],
                }
            )
        elif data_format == "sharegpt":
            sum_ori_resp_len += num_tokens_from_string(res["conversations"][1]["value"], disallowed_special=True)
            sum_evol_resp_len += num_tokens_from_string(res["evol_output"], disallowed_special=True)
            evol_round_dist[res["evol_round"]] += 1
            evol_conversations = [
                res["conversations"][0],
                {
                    "from": "gpt",
                    "value": res["evol_output"]
                }
            ]
            sft_list.append(
                {
                    "id": res["id"],
                    "conversations": evol_conversations,
                    "source": res["source"],
                }
            )
        else:
            raise NotImplementedError

    sft_list = sorted(sft_list, key=lambda x:x["id"])

    avg_ori_resp_len = sum_ori_resp_len / total_count
    avg_evol_resp_len = sum_evol_resp_len / total_count
    print(f"Total number of samples: {total_count}")
    print(f"Total number of error samples (keeped): {error_count}")
    print(f"Average number of tokens in origin responses: {avg_ori_resp_len:.2f}")
    print(f"Average number of tokens in evolved responses: {avg_evol_resp_len:.2f}")
    print(f"Distribution of evol rounds for each sample: {evol_round_dist}")

    if output_format=="json":
        json_dump(sft_path_json, sft_list)
    elif output_format=="jsonl":
        jsonl_dump(sft_path_jsonl, sft_list)
    else:
        raise NotImplementedError


def gen_ablation_train_dataset(res_name: str=None):
    res_path = os.path.join("./res", res_name)
    sft_base="../data/sft"
    if not os.path.exists(sft_base):
        os.mkdir(sft_base)
    sft_path_mode0 = os.path.join(sft_base, res_name.replace("ablation-chatgpt", "edit"))+".json"
    sft_path_mode1 = os.path.join(sft_base, res_name.replace("ablation-chatgpt", "advise-edit_wo-resp"))+".json"
    sft_path_mode2 = os.path.join(sft_base, res_name.replace("ablation-chatgpt", "advise-edit"))+".json"
    sft_path_mode3 = os.path.join(sft_base, res_name.replace("ablation-chatgpt", "debate-advise-edit"))+".json"
    sft_list = {
        "0":[],
        "1":[],
        "2":[],
        "3":[]
    }
    sum_ori_resp_len=0
    sum_evol_resp_len=[0,0,0,0]
    total_count = 0
    error_count=0


    for f_name in tqdm(os.listdir(res_path)):
        res = json_load(os.path.join(res_path, f_name))
        total_count += 1
        sum_ori_resp_len += num_tokens_from_string(res["output"], disallowed_special=True)
        for m in ["0","1","2","3"]:
            tmp_res = res[f"mode_{m}"]
            if not ("evol_output" in tmp_res and tmp_res["evol_output"]):
                error_count += 1
                tmp_res["evol_output"] = res["output"]
            
            sum_evol_resp_len[int(m)] += num_tokens_from_string(tmp_res["evol_output"], disallowed_special=True)
            sft_list[m].append(
                {
                    "id": res["id"],
                    "instruction": res["instruction"],
                    "input": res["input"],
                    "output": tmp_res["evol_output"],
                }
            )
    for m in ["0","1","2","3"]:
        sft_list[m] = sorted(sft_list[m], key=lambda x:x["id"])

    avg_ori_resp_len = sum_ori_resp_len / total_count
    avg_evol_resp_len = [round(l/total_count, 2) for l in sum_evol_resp_len]
    print(f"Total number of samples: {total_count}")
    print(f"Total number of error samples (keeped): {error_count}")
    print(f"Average number of tokens in origin responses: {avg_ori_resp_len:.2f}")
    print(f"Average number of tokens in evolved responses: {avg_evol_resp_len}")

    json_dump(sft_path_mode0, sft_list["0"])
    json_dump(sft_path_mode1, sft_list["1"])
    json_dump(sft_path_mode2, sft_list["2"])
    json_dump(sft_path_mode3, sft_list["3"])


if __name__ == "__main__":
    gen_single_conv_train_dataset(
        res_name="single-deita6k_evol-mixtral",
        data_format="sharegpt"
    )
    gen_single_conv_train_dataset(
        res_name="multi-deita6k_evol-mixtral",
        data_format="sharegpt"
    )