import tiktoken
import json
import time
import os


def num_tokens_from_string(string: str=None) -> int:
    """Returns the number of tokens in a text string."""
    encoding = tiktoken.encoding_for_model("gpt-3.5-turbo")
    num_tokens = len(encoding.encode(string))
    return num_tokens


def check_conv_max_len(conv: list=None, max_len:str=None) -> bool:
    conv=[item["value"] for item in conv]
    cur_len = num_tokens_from_string((" ").join(conv))
    # print(f"current length: {cur_len}")
    return cur_len >= max_len


def single_sample2query(sample: dict=None, sample_format: str=None) -> dict:
    if sample_format == "alpaca":
        query = {
            "id": sample["id"],
            "instruction": sample["instruction"],
            "input": sample["input"],
            "output": sample["output"],
        }
    elif sample_format == "sharegpt":
        query = {
            "id": sample["id"],
            "instruction": sample["conversations"][0]["value"],
            "input": "",
            "output": sample["conversations"][1]["value"],
        }

    return query


def multi_sample2query(sample: dict=None, cur_turn: str=None, conv_wind_size: str=2) -> dict:
    from fastchat.conversation import get_conv_template
    conv = get_conv_template("vicuna_v1.1")
    indx_start = max(0, (cur_turn-conv_wind_size)*2)
    indx_end = (cur_turn-1)*2
    for i in range(indx_start, indx_end):
        content = sample["conversations"][i]["value"]
        if i % 2 == 0:
            conv.append_message(conv.roles[0], content)
        else:
            conv.append_message(conv.roles[1], content)
    conv.append_message(conv.roles[0], sample["conversations"][indx_end]["value"])
    conv.append_message(conv.roles[1], None)

    instruction = conv.get_prompt()
    output = sample["conversations"][indx_end+1]["value"]
    query = {
        "id": sample["id"],
        "instruction": instruction,
        "input": "",
        "output": output,
    }
    return query


def format_mistral_prompt(mem_hist: list=None) -> list:
    if mem_hist[0]["role"] == "system":
        mem_hist[1]["content"] = mem_hist[0]["content"] + "\n\n" + mem_hist[1]["content"]
        return mem_hist[1:]
    return mem_hist


def get_sample_prompt(ori_sample: dict[str, str] = None) -> tuple[str, bool]:
    have_input = ori_sample['input'] if (
        ori_sample['input'] != "" or ori_sample['input'] != "<no input>") else None
    ful_tmpl = "### Instruction:\n{instruction}\n\n### Input:\n{input}\n\n### Response:\n{output}" if have_input else \
        "### Instruction:\n{instruction}\n\n### Response:\n{output}"
    req_tmpl = "### Instruction:\n{instruction}\n\n### Input:\n{input}\n\n" if have_input else \
        "### Instruction:\n{instruction}\n\n"
    sample_ful = ful_tmpl.format_map(ori_sample)
    sample_req = req_tmpl.format_map(ori_sample)
    return sample_ful, sample_req, have_input


def get_role_prompt(agent_role: str = None) -> str:
    pos_persona = "You are an optimistic person who embodies a mindset that looks for the best in every situation, maintains a positive attitude, and embraces challenges as opportunities for growth and success. "
    crt_persona = "You are a critical person who tends to view things through critical thinking and provide feedback for improvement or identify areas of concern. "
    adv_persona = "You are an experienced advisor who possesses a high level of expertise in summarizing and giving advice. "
    edt_persona = "You are a professional editor who possesses a high level of expertise in refining and improving writing content. "
    jdg_persona = "You are a helpful and precise assistant for checking the quality of the response. "

    if agent_role == "positive":
        role_prompt = pos_persona
    elif agent_role == "critical":
        role_prompt = crt_persona
    elif agent_role == "advisor":
        role_prompt = adv_persona
    elif agent_role == "editor":
        role_prompt = edt_persona
    elif agent_role == "judge":
        role_prompt = jdg_persona
    else:
        raise NotImplementedError

    return role_prompt


def get_task_prompt(agent_role: str = None, ctx_info: dict = None, edt_mode: str = "0", judge_mode: str= "compare", reverse_jdg: bool = False) -> str:
    if "positive" in agent_role:
        dbt_phase = agent_role.split("_")[1]
        if dbt_phase == "pred":
            tmpl = "{sample}\n\nIn your opinion, the above response accurately answers the instruction and the input. Please state reasons why the response is accurate if it is used for supervised fine-tuning."
        elif dbt_phase == "free":
            tmpl = "### Review from others:\n{crt_pred}\n\nAbove is another review from others, please evaluate the plausibility of each point according to the given instruction and input."
    elif "critical" in agent_role:
        dbt_phase = agent_role.split("_")[1]
        if dbt_phase == "pred":
            tmpl = "{sample}\n\nIn your opinion, the above response does not accurately answer the instruction and the input. Please offer suggestions on how to improve the response if it is used for supervised fine-tuning."
        elif dbt_phase == "free":
            tmpl = "### Review from others:\n{pos_pred}\n\nAbove is another review from others, please evaluate the plausibility of each point according to the given instruction and input."
    elif agent_role == "advisor":
        if edt_mode == "1":
            desc_tmpl = "Below is an instruction that describes a task, paired with an input that provides further context.\n\n{sample_request} " if ctx_info["have_input"] else \
                "Below is an instruction that describes a task.\n\n{sample_request} "
            task_tmpl = "Propose no more than 3 writing suggestions for others to better complete the request. Directly output these suggestions in separate lines without any foreword or explanation."
            tmpl = desc_tmpl+task_tmpl
        if edt_mode == "2":
            desc_tmpl = "Below is an instruction that describes a task, paired with an input that provides further context.\n\n{sample} " if ctx_info["have_input"] else \
                "Below is an instruction that describes a task.\n\n{sample} "
            task_tmpl = "Propose no more than 3 writing suggestions for improving the given response. Directly output these suggestions in separate lines without any foreword or explanation."
            tmpl = desc_tmpl+task_tmpl
        elif edt_mode in ["3","4"]:
            desc_tmpl = "Below is an instruction that describes a task, paired with an input that provides further context.\n\n{sample} " if ctx_info["have_input"] else \
                "Below is an instruction that describes a task.\n\n{sample} "
            dbt_hist_tmpl = "The following is a discussion about the given request and response by two reviewers.\n\n### Reviewer 1:\n{pos_pred}\n\n### Reviewer 2:\n{crt_pred}\n\n### Reviewer 1:\n{pos_free}\n\n### Reviewer 2:\n{crt_free}\n\n"
            task_tmpl = "Extract and summarize credible ideas from the above dialogue and rewrite them into no more than 3 writing suggestions for improving the given response. Directly output these suggestions in separate lines without any foreword or explanation."
            tmpl = desc_tmpl+dbt_hist_tmpl+task_tmpl
    elif agent_role == "editor":
        desc_tmpl = "Below is an instruction that describes a task, paired with an input that provides further context. " if ctx_info["have_input"] else \
            "Below is an instruction that describes a task. "
        if edt_mode == "0":
            tmpl = desc_tmpl + \
                "Write a response that appropriately completes the request.\n\n{sample_request}### Response:\n"
        elif edt_mode == "1":
            tmpl = "### Writing Suggestions:\n{adv_sugg}\n\n" + \
                desc_tmpl + \
                "\nReferring to the above writing suggestions (MUST ignore suggestions beyond your capabilities), modify the previous response and make sure that it appropriately completes the request.\n\n{sample_request}### Response:\n"
        elif edt_mode in ["2", "3", "4"]:
            tmpl = "### Writing Suggestions:\n{adv_sugg}\n\n" + \
                "### Previous Response:\n{pre_resp}\n\n" + \
                desc_tmpl + \
                "\nReferring to the above writing suggestions (MUST ignore suggestions beyond your capabilities), modify the previous response and make sure that it appropriately completes the request.\n\n{sample_request}### Response:\n"
        else:
            raise NotImplementedError
    elif agent_role == "judge":
        desc_tmpl = "Below is an instruction that describes a task, paired with an input that provides further context.\n\n{sample_request}" if ctx_info["have_input"] else \
            "Below is an instruction that describes a task.\n\n{sample_request}"
        if reverse_jdg:
            resp_tmpl = "[The Start of Assistant 1's Response]\n{new_resp}\n\n[The End of Assistant 1's Response]\n\n[The Start of Assistant 2's Response]\n{pre_resp}\n\n[The End of Assistant 2's Response]\n\n[System]\n"
        else:
            resp_tmpl = "[The Start of Assistant 1's Response]\n{pre_resp}\n\n[The End of Assistant 1's Response]\n\n[The Start of Assistant 2's Response]\n{new_resp}\n\n[The End of Assistant 2's Response]\n\n[System]\n"
        if judge_mode=="compare":
            task_tmpl = "We would like to request your comparison of the performance of two AI assistants in response to the user request displayed above.\nPlease compare the helpfulness, relevance, accuracy, and level of detail of their responses.\nPlease first output a single line containing a name indicating whose response is better, <assistant 1> or <assistant 2> or <equal>. In the subsequent line, please provide a comprehensive explanation of your comparison, avoiding any potential bias and ensuring that the order in which the responses were presented does not affect your judgment\n\n"
        elif judge_mode=="score":
            task_tmpl = "We would like to request your feedback on the performance of two AI assistants in response to the user request displayed above.\nPlease rate the helpfulness, relevance, accuracy, and level of detail of their responses. Each assistant receives an overall score on a scale of 1 to 10, where a higher score indicates better overall performance.\nPlease first output a single line containing only two values indicating the scores for Assistant 1 and 2, respectively. The two scores are separated by a space. In the subsequent line, please provide a comprehensive explanation of your evaluation, avoiding any potential bias and ensuring that the order in which the responses were presented does not affect your judgment\n\n"
        tmpl = desc_tmpl + resp_tmpl + task_tmpl
    else:
        raise NotImplementedError
    prompt = tmpl.format_map(ctx_info)
    return prompt


def get_mem_path(args) -> str:
    save_folder_name = args.save_folder_name
    rq = time.strftime('%Y%m%d%H%M', time.localtime(time.time()))
    log_date = rq[:8]
    if save_folder_name:
        log_path = os.getcwd() + '/mems/{}/'.format(save_folder_name)
    else:
        log_path = os.getcwd() + '/mems/{}/'.format(log_date)
    if not os.path.exists(log_path):
        os.makedirs(log_path)
    mem_pref = log_path + rq + '_'

    return mem_pref


def get_res_path(args) -> str:
    save_folder_name = args.save_folder_name
    rq = time.strftime('%Y%m%d%H%M', time.localtime(time.time()))
    log_date = rq[:8]
    if save_folder_name:
        log_path = os.getcwd() + '/res/{}/'.format(save_folder_name)
    else:
        log_path = os.getcwd() + '/res/{}/'.format(log_date)
    if not os.path.exists(log_path):
        os.makedirs(log_path)
    res_pref = log_path + rq + '_'

    return res_pref


def get_error_path(args) -> str:
    save_folder_name = args.save_folder_name
    rq = time.strftime('%Y%m%d%H%M', time.localtime(time.time()))
    log_date = rq[:8]
    if save_folder_name:
        log_path = os.getcwd() + '/error_case/{}/'.format(save_folder_name)
    else:
        log_path = os.getcwd() + '/error_case/{}/'.format(log_date)
    if not os.path.exists(log_path):
        os.makedirs(log_path)
    error_pref = log_path + rq + '_'

    return error_pref


def save_args(args, save_log_path) -> dict:
    args_dict = args.__dict__
    save_args_path = save_log_path.replace('.log', '.json')
    with open(save_args_path, 'w') as file:
        json.dump(args_dict, file)
    return args


def parse_jdg(resp: str = None, judge_mode:str=None) -> dict:
    error = None
    if judge_mode == "compare":
        try:
            resp = resp.lstrip("Output:").lstrip("output:").strip()
            option = resp.split("\n")[0].lower()
            review = ("\n").join(resp.split("\n")[1:]).strip()
            if "assistant 1" in option:
                jdg_sp = [1, 0]
            elif "assistant 2" in option:
                jdg_sp = [0, 1]
            elif "equal" in option:
                jdg_sp = [1, 1]
            else:
                jdg_sp = [-1, -1]
                review = None
                error = "wrong option in judgement"
        except Exception as e:
            jdg_sp = [-1, -1]
            review = None
            error = str(e)
    elif judge_mode == "score":
        try:
            resp = resp.lstrip("Output:").lstrip("output:").strip()
            score_pair = resp.split("\n")[0]
            review = ("\n").join(resp.split("\n")[1:]).strip()
            score_pair = score_pair.replace(",", " ").strip()
            sp = score_pair.split(" ")
            assert len(sp) == 2, "wrong number of scores"
            jdg_sp = [float(sp[0]), float(sp[1])]
        except Exception as e:
            jdg_sp = [-1, -1]
            review = None
            error = str(e)

    res = {
        "jdg_score_pair": jdg_sp,
        "jdg_review": review,
        "jdg_parse_error": error
    }

    return res


def merge_jdg_res(res1: dict, res2: dict) -> dict:
    sp1 = res1["jdg_score_pair"]
    sp2 = res2["jdg_score_pair"]

    if (-1 in sp1) and (-1 in sp2):
        sp = [-1, -1]
    elif (-1 in sp1) or (-1 in sp2):
        sp = sp1 if (-1 not in sp1) else list(reversed(sp2))
    else:
        sp = [(sp1[0]+sp2[1])/2, (sp1[1]+sp2[0])/2]
    return sp
