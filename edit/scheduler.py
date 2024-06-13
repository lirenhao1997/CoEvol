from tqdm.asyncio import tqdm_asyncio
import concurrent.futures
import functools
import asyncio
import time
import json
import copy
import os

from models import GPTModel, ProxyGPTModel, ERNIEModel, GLMModel, LocalModel
from dataloader import SFTDataLoader
from agents import LLMAgent
from utils import check_conv_max_len
from utils import single_sample2query, multi_sample2query
from utils import get_sample_prompt, get_task_prompt
from utils import parse_jdg, merge_jdg_res
from mylogging import my_log


class ConCurAgentScheduler():
    def __init__(self, args):
        self.data = SFTDataLoader(args)
        self.data_format=args.dataset_format
        self.task_name = args.task_name
        self.save_log =args.save_log
        self.save_mem =args.save_mem
        if isinstance(args.api_keys, list):
            self.api_key = args.api_keys[args.api_key_indx]
        else:
            self.api_key = args.api_keys

        self.mem_pref = args.mem_path
        self.res_pref = args.res_path
        self.error_pref = args.error_path
        self.num_workers = args.num_workers

        self.agent_wind_size = args.agent_wind_size
        self.max_agent_len = args.max_agent_len
        self.agent_names = args.agent_names
        self.edit_mode = args.edit_mode
        self.judge_mode = args.judge_mode
        self.max_evol_iter = args.max_evol_iter

        self.conv_wind_size = args.conv_wind_size
        self.max_optimize_turn = args.max_optimize_turn
        self.max_optimize_len = args.max_optimize_len
        self.__init_model(args)
        self.__init_excutor()

    def __init_model(self, args):
        if args.use_local_model:
            self.model = LocalModel(args)
        elif "gpt" in args.model_name:
            if args.proxy_api_url:
                self.model = ProxyGPTModel(args)
            else:
                self.model = GPTModel(args)
        elif "ernie" in args.model_name:
            self.model = ERNIEModel(args)
        elif "glm" in args.model_name:
            self.model = GLMModel(args)
        else:
            raise NotImplementedError

    def __init_excutor(self):
        self.excutor = concurrent.futures.ThreadPoolExecutor(
            max_workers=self.num_workers)

    def __return_agents(self, key):
        pos_agent = LLMAgent(
            self.model,
            agent_role='positive',
            agent_names=self.agent_names,
            api_key=key,
            agent_wind_size=self.agent_wind_size,
            max_agent_len=self.max_agent_len,
        )
        crt_agent = LLMAgent(
            self.model,
            agent_role='critical',
            agent_names=self.agent_names,
            api_key=key,
            agent_wind_size=self.agent_wind_size,
            max_agent_len=self.max_agent_len,
        )
        adv_agent = LLMAgent(
            self.model,
            agent_role='advisor',
            agent_names=self.agent_names,
            api_key=key,
            agent_wind_size=self.agent_wind_size,
            max_agent_len=self.max_agent_len,
        )
        edt_agent = LLMAgent(
            self.model,
            agent_role='editor',
            agent_names=self.agent_names,
            api_key=key,
            agent_wind_size=self.agent_wind_size,
            max_agent_len=self.max_agent_len,
        )
        jdg_agent = LLMAgent(
            self.model,
            agent_role='judge',
            agent_names=self.agent_names,
            api_key=key,
            agent_wind_size=self.agent_wind_size,
            max_agent_len=self.max_agent_len,
        )

        return pos_agent, crt_agent, adv_agent, edt_agent, jdg_agent

    def main_run(self):
        start_t = time.time()
        loop = asyncio.get_event_loop()
        loop.run_until_complete(self.asyn_run())
        end_t = time.time()
        print(f"Total time: {end_t-start_t}")

    async def asyn_run(self):
        data_iter = iter(self.data)
        loop = asyncio.get_event_loop()
        if self.task_name == "edit":
            agent_func = self.__return_agents
            run_func = self.run_edit_proc
        else:
            raise NotImplementedError

        tasks = [loop.run_in_executor(self.excutor, functools.partial(
            run_func, agents=agent_func(self.api_key), sample=sample)) for _, sample in data_iter]

        my_log.info(f"Total number of tasks: {len(tasks)}")
        await tqdm_asyncio.gather(*tasks)
        my_log.info("Task Completed!")

    def run_edit_proc(self, agents, sample):
        pos_agent, crt_agent, adv_agent, edt_agent, jdg_agent = agents
        if self.data_format == "alpaca":
            num_turn = 1
        elif self.data_format == "sharegpt":
            assert len(sample["conversations"]) % 2 == 0, \
                "The number of turns should be a multiple of 2"
            num_turn = int(len(sample["conversations"])/2)
        else:
            raise NotImplementedError

        if self.data_format=="sharegpt" and num_turn >= 2:
            opt_steps=[]
            updated_sample = copy.deepcopy(sample)
            error_flag = False
            cur_turn = 1
            
            while True:
                # print(f"optimizing turn {cur_turn}...")
                try:
                    query = multi_sample2query(
                        sample=updated_sample,
                        cur_turn=cur_turn,
                        conv_wind_size=self.conv_wind_size
                    )
                    if "4" in self.edit_mode:
                        edit_res = self.run_iter_pipeline(agents, query)
                    else:
                        edit_res = self.run_sep_pipeline(agents, query)
                    opt_steps.append(edit_res)
                except Exception as e:
                    edit_res={
                        "evol_output": None,
                        "evol_round": 0
                    }
                    opt_steps.append(edit_res)
                    total_edit_res={
                        "optimization_steps": opt_steps,
                        "edit_error": str(e)
                    }
                    error_flag = True
                    break
                else:
                    # update optimized history
                    update_indx = (cur_turn - 1) * 2 + 1
                    # "None" for evol_output:
                    # error occured in judge process -> keep the last evol_output or origin response
                    if edit_res["evol_output"]:
                        updated_sample["conversations"][update_indx]["value"] = edit_res["evol_output"]
                    cur_turn += 1
                    for agent in agents:
                        agent.clear_mem()
                    # whether to stop optimization
                    if cur_turn > num_turn:
                        break
                    if self.max_optimize_turn:
                        if cur_turn > self.max_optimize_turn:
                            break
                        else:
                            continue
                    else:
                        # adaptive optimization length
                        if check_conv_max_len(
                            updated_sample["conversations"][:update_indx+1], 
                            self.max_optimize_len
                            ):
                            break
                        else:
                            continue
            if not error_flag:
                total_edit_res={
                    "optimization_steps": opt_steps,
                    "evol_conversations": updated_sample["conversations"]
                    }
            self.save_res(total_edit_res, sample)
        else:
            try:
                query = single_sample2query(
                    sample=sample,
                    sample_format=self.data_format
                )
                if "4" in self.edit_mode:
                    edit_res = self.run_iter_pipeline(agents, query)
                else:
                    edit_res = self.run_sep_pipeline(agents, query)
            except Exception as e:
                edit_res={
                    "edit_error": str(e)
                }
            finally:
                self.save_res(edit_res, sample)
        
        if not self.save_mem:
            sample_id = sample["id"]
            session_name = self.mem_pref + f'{sample_id}_hist-session.json'
            mem_path = os.getcwd() + '/mems/'
            tmp_file = os.path.join(mem_path, session_name)
            os.remove(tmp_file)
        
        del pos_agent
        del crt_agent
        del adv_agent
        del edt_agent
        del jdg_agent

    def run_sep_pipeline(self, agents, cur_query) -> dict:
        # Edit Mode
        # 0 - editor
        # 1 - advisor (visible: instruction) + editor 
        # 2 - advisor (visible: instruction & response) + editor 
        # 3 - MAD + advisor (visible: instruction & response) + editor 
        pos_agent, crt_agent, adv_agent, edt_agent, _ = agents
        sample_id = cur_query["id"]
        session_name = self.mem_pref + f'{sample_id}_hist-session.json'
        sample_ful, sample_req, have_input = get_sample_prompt(cur_query)
        ctx_info = {
            "sample": sample_ful,
            "sample_request": sample_req,
            "have_input": have_input,
            "pre_resp": cur_query["output"]
        }
        edit_res = {}

        if "0" in self.edit_mode:
            edt_task_prompt = get_task_prompt(
                agent_role="editor",
                ctx_info=ctx_info,
                edt_mode="0"
            )
            edt_agent.update_mem(
                new_mem=edt_task_prompt,
                role="user"
            )
            edt_resp = edt_agent.act(session_name).strip()
            my_log.info(edt_resp)
            edit_res.update(
                {
                    "mode_0": {
                        "evol_output": edt_resp,
                    }
                }
            )

            edt_agent.clear_mem()

        if "1" in self.edit_mode:
            # speak of advisor
            adv_task_prompt = get_task_prompt(
                agent_role="advisor",
                ctx_info=ctx_info,
                edt_mode="1"
            )
            adv_agent.update_mem(
                new_mem=adv_task_prompt,
                role="user"
            )
            adv_resp = adv_agent.act(session_name).strip()
            my_log.info(adv_resp)
            ctx_info["adv_sugg"] = adv_resp

            # speak of editor
            edt_task_prompt = get_task_prompt(
                agent_role="editor",
                ctx_info=ctx_info,
                edt_mode="1"
            )
            edt_agent.update_mem(
                new_mem=edt_task_prompt,
                role="user"
            )
            edt_resp = edt_agent.act(session_name).strip()
            my_log.info(edt_resp)
            edit_res.update(
                {
                    "mode_1": {
                        "evol_output": edt_resp,
                        "suggestions": adv_resp
                    }
                }
            )

            adv_agent.clear_mem()
            edt_agent.clear_mem()

        if "2" in self.edit_mode:
            # speak of advisor
            adv_task_prompt = get_task_prompt(
                agent_role="advisor",
                ctx_info=ctx_info,
                edt_mode="2"
            )
            adv_agent.update_mem(
                new_mem=adv_task_prompt,
                role="user"
            )
            adv_resp = adv_agent.act(session_name).strip()
            my_log.info(adv_resp)
            ctx_info["adv_sugg"] = adv_resp

            # speak of editor
            edt_task_prompt = get_task_prompt(
                agent_role="editor",
                ctx_info=ctx_info,
                edt_mode="2"
            )
            edt_agent.update_mem(
                new_mem=edt_task_prompt,
                role="user"
            )
            edt_resp = edt_agent.act(session_name).strip()
            my_log.info(edt_resp)
            edit_res.update(
                {
                    "mode_2": {
                        "evol_output": edt_resp,
                        "suggestions": adv_resp
                    }
                }
            )

            adv_agent.clear_mem()
            edt_agent.clear_mem()

        if "3" in self.edit_mode:
            # Debate Phase-Round 1: Predetermined Position Debate
            # speak of positive debater
            pos_task_prompt = get_task_prompt(
                agent_role="positive_pred",
                ctx_info=ctx_info
            )
            pos_agent.update_mem(
                new_mem=pos_task_prompt,
                role="user"
            )
            pos_resp = pos_agent.act(session_name).strip()
            pos_agent.update_mem(
                new_mem=pos_resp,
                role="assistant"
            )
            my_log.info(pos_resp)

            # speak of critical debater
            crt_task_prompt = get_task_prompt(
                agent_role="critical_pred",
                ctx_info=ctx_info
            )
            crt_agent.update_mem(
                new_mem=crt_task_prompt,
                role="user"
            )
            crt_resp = crt_agent.act(session_name).strip()
            crt_agent.update_mem(
                new_mem=crt_resp,
                role="assistant"
            )
            my_log.info(crt_resp)

            ctx_info["pos_pred"] = pos_resp
            ctx_info["crt_pred"] = crt_resp

            # Debate Phase-Round 2: Free Debate
            # speak of positive debater
            pos_task_prompt = get_task_prompt(
                agent_role="positive_free",
                ctx_info=ctx_info
            )
            pos_agent.update_mem(
                new_mem=pos_task_prompt,
                role="user"
            )
            pos_resp = pos_agent.act(session_name).strip()
            my_log.info(pos_resp)
            pos_agent.update_mem(
                new_mem=pos_resp,
                role="assistant"
            )

            # speak of critical debater
            crt_task_prompt = get_task_prompt(
                agent_role="critical_free",
                ctx_info=ctx_info
            )
            crt_agent.update_mem(
                new_mem=crt_task_prompt,
                role="user"
            )
            crt_resp = crt_agent.act(session_name).strip()
            my_log.info(crt_resp)
            crt_agent.update_mem(
                new_mem=crt_resp,
                role="assistant"
            )

            ctx_info["pos_free"] = pos_resp
            ctx_info["crt_free"] = crt_resp

            # Edit Phase
            # speak of advisor
            adv_task_prompt = get_task_prompt(
                agent_role="advisor",
                ctx_info=ctx_info,
                edt_mode="3"
            )
            adv_agent.update_mem(
                new_mem=adv_task_prompt,
                role="user"
            )
            adv_resp = adv_agent.act(session_name).strip()
            my_log.info(adv_resp)
            ctx_info["adv_sugg"] = adv_resp

            # speak of editor
            edt_task_prompt = get_task_prompt(
                agent_role="editor",
                ctx_info=ctx_info,
                edt_mode="3"
            )
            edt_agent.update_mem(
                new_mem=edt_task_prompt,
                role="user"
            )
            edt_resp = edt_agent.act(session_name).strip()
            my_log.info(edt_resp)
            edit_res.update(
                {
                    "mode_3": {
                        "evol_output": edt_resp,
                        "suggestions": adv_resp
                    }
                }
            )

        return edit_res

    def run_iter_pipeline(self, agents, cur_query) -> dict:
        # Edit Mode
        # 4 - iterative: MAD + advisor + editor + judge
        pos_agent, crt_agent, adv_agent, edt_agent, jdg_agent = agents
        sample_id = cur_query["id"]
        session_name = self.mem_pref + f'{sample_id}_hist-session.json'
        sample_ful, sample_req, have_input = get_sample_prompt(cur_query)
        ctx_info = {
            "sample": sample_ful,
            "sample_request": sample_req,
            "have_input": have_input,
            "pre_resp": cur_query["output"]
        }
        edit_res = {
            "evol_output": "",
            "evol_round": 0,
        }
        max_iter = self.max_evol_iter

        for cur_round in range(max_iter):
            # speak of positive debater
            pos_task_prompt = get_task_prompt(
                agent_role="positive_pred",
                ctx_info=ctx_info
            )
            pos_agent.update_mem(
                new_mem=pos_task_prompt,
                role="user"
            )
            pos_resp = pos_agent.act(session_name).strip()
            pos_agent.update_mem(
                new_mem=pos_resp,
                role="assistant"
            )
            my_log.info(pos_resp)

            # speak of critical debater
            crt_task_prompt = get_task_prompt(
                agent_role="critical_pred",
                ctx_info=ctx_info
            )
            crt_agent.update_mem(
                new_mem=crt_task_prompt,
                role="user"
            )
            crt_resp = crt_agent.act(session_name).strip()
            crt_agent.update_mem(
                new_mem=crt_resp,
                role="assistant"
            )
            my_log.info(crt_resp)

            ctx_info["pos_pred"] = pos_resp
            ctx_info["crt_pred"] = crt_resp

            # Debate Phase-Round 2: Free Debate
            # speak of positive debater
            pos_task_prompt = get_task_prompt(
                agent_role="positive_free",
                ctx_info=ctx_info
            )
            pos_agent.update_mem(
                new_mem=pos_task_prompt,
                role="user"
            )
            pos_resp = pos_agent.act(session_name).strip()
            my_log.info(pos_resp)
            pos_agent.update_mem(
                new_mem=pos_resp,
                role="assistant"
            )

            # speak of critical debater
            crt_task_prompt = get_task_prompt(
                agent_role="critical_free",
                ctx_info=ctx_info
            )
            crt_agent.update_mem(
                new_mem=crt_task_prompt,
                role="user"
            )
            crt_resp = crt_agent.act(session_name).strip()
            my_log.info(crt_resp)
            crt_agent.update_mem(
                new_mem=crt_resp,
                role="assistant"
            )

            ctx_info["pos_free"] = pos_resp
            ctx_info["crt_free"] = crt_resp

            # Edit Phase
            # speak of advisor
            adv_task_prompt = get_task_prompt(
                agent_role="advisor",
                ctx_info=ctx_info,
                edt_mode="4"
            )
            adv_agent.update_mem(
                new_mem=adv_task_prompt,
                role="user"
            )
            adv_resp = adv_agent.act(session_name).strip()
            my_log.info(adv_resp)
            ctx_info["adv_sugg"] = adv_resp

            # speak of editor
            edt_task_prompt = get_task_prompt(
                agent_role="editor",
                ctx_info=ctx_info,
                edt_mode="4"
            )
            edt_agent.update_mem(
                new_mem=edt_task_prompt,
                role="user"
            )
            edt_resp = edt_agent.act(session_name).strip()
            my_log.info(edt_resp)

            ctx_info["new_resp"] = edt_resp

            # speak of judge
            jdg_task_prompt = get_task_prompt(
                agent_role="judge",
                ctx_info=ctx_info,
            )
            jdg_agent.update_mem(
                new_mem=jdg_task_prompt,
                role="user"
            )
            jdg_resp1 = jdg_agent.act(session_name).strip()
            my_log.info(jdg_resp1)
            jdg_agent.clear_mem()

            jdg_task_prompt = get_task_prompt(
                agent_role="judge",
                ctx_info=ctx_info,
                reverse_jdg=True
            )
            jdg_agent.update_mem(
                new_mem=jdg_task_prompt,
                role="user"
            )
            jdg_resp2 = jdg_agent.act(session_name).strip()
            my_log.info(jdg_resp2)

            # whether to end the iteration
            jdg_res1 = parse_jdg(jdg_resp1, self.judge_mode)
            jdg_res2 = parse_jdg(jdg_resp2, self.judge_mode)
            jdg_sp = merge_jdg_res(jdg_res1, jdg_res2)
            edit_res.update(
                {   
                    f"round_{cur_round}": {
                        "output": edt_resp,
                        "suggestions": adv_resp,
                        "judge": jdg_sp
                    }
                }
            )
            if -1 not in jdg_sp:
                if jdg_sp[1] <= jdg_sp[0]:
                    edit_res["evol_output"] = ctx_info["pre_resp"]
                    edit_res["evol_round"] = cur_round
                    return edit_res
                else:
                    edit_res["evol_output"] = edt_resp
                    edit_res["evol_round"] = cur_round + 1
            else:
                edit_res["evol_output"] = ctx_info["pre_resp"]
                edit_res["evol_round"] = cur_round
                edit_res["evol_error"] = "<JudgeError>"
                return edit_res

            next_iter_query = {
                "instruction": cur_query["instruction"],
                "input": cur_query["input"],
                "output": edt_resp
            }
            sample_ful, sample_req, have_input = get_sample_prompt(
                next_iter_query)
            ctx_info = {
                "sample": sample_ful,
                "sample_request": sample_req,
                "have_input": have_input,
                "pre_resp": edt_resp
            }

            pos_agent.clear_mem()
            crt_agent.clear_mem()
            adv_agent.clear_mem()
            edt_agent.clear_mem()
            jdg_agent.clear_mem()

        return edit_res

    def save_res(self, edit_res, ori_sample):
        sample_id = ori_sample["id"]
        res_dict = copy.deepcopy(ori_sample)
        res_dict.update(edit_res)

        # append memory history to result
        mem_path = os.getcwd() + '/mems/'
        session_name = self.mem_pref + f'{sample_id}_hist-session.json'
        with open(os.path.join(mem_path, session_name), 'r') as f:
            hist_mem = json.load(f)
        res_dict["memory_history"] = hist_mem

        res_name = self.res_pref + str(sample_id) + '.json'
        with open(res_name, 'w') as file:
            json.dump(res_dict, file)
        my_log.info(f'Save edit result to {res_name}')
    