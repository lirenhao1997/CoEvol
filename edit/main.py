from scheduler import ConCurAgentScheduler
from utils import get_mem_path, get_res_path, get_error_path, save_args
from mylogging import my_log, save_log_path
import argparse
import json


def parse_args():
    parser = argparse.ArgumentParser()

    # Arguments for Logger
    parser.add_argument("--save_log", action='store_true',
                        help="Whether to save log file (default path: ../logs/).")
    parser.add_argument("--save_mem", action='store_true',
                        help="Whether to save agent memory file (default path: ../mems/).")
    parser.add_argument("--save_folder_name", type=str, default="None",
                        help="If not set, current date will be used for saved files' folder name.")

    # Arguments for Data Loding
    parser.add_argument("--task_name", type=str, default="edit",
                        help="The specific name of excuting task.")
    parser.add_argument("--root_path", type=str, default='../data/',
                        help="The root path of loading data.")
    parser.add_argument("--dataset_name", type=str, default="alpaca.json",
                        help="The name of dataset of loading data.")
    parser.add_argument("--dataset_format", type=str, default="alpaca",
                        choices=["alpaca", "sharegpt"],
                        help="The name of dataset of loading data.")
    parser.add_argument("--start_indx", type=int, default=0,
                        help="The start indx of loading data.")
    parser.add_argument("--end_indx", type=int, default=None,
                        help="The end indx of loading data.")

    # Arguments for Model Setting
    parser.add_argument("--model_name", type=str, default="gpt-3.5-turbo",
                        choices=["gpt-3.5-turbo", "gpt-3.5-turbo-0301", "gpt-3.5-turbo-0613", "gpt-3.5-turbo-1106",
                                 "gpt-4", "gpt-4-0613", "gpt-4-1106-preview",
                                 "ernie-bot-4", "ernie-bot-turbo",
                                 "chatglm_turbo",
                                 "mixtral"],
                        help="ID of the model to use")
    parser.add_argument("--use_local_model", action='store_true',
                        help="If set, the local model will be used and api_key will be set to 'None'.")
    parser.add_argument("--proxy_api_url", type=str, default=None,
                        help="Proxy api url for requesting gpt model.")
    parser.add_argument("--api_keys", type=str, default=None,
                        help="Path of a txt file which contains api keys to communicate with gpt models.")
    parser.add_argument("--api_key_indx", type=int, default=0,
                        help="The indx of api key used for request this time within the file.")
    parser.add_argument("--api_base", type=str, default=None,
                        help="If deployed locally, the IP address of the chat model api is required.")
    parser.add_argument("--max_tokens", type=int, default=1000,
                        help="The maximum number of tokens to generate in the chat completion.")
    parser.add_argument("--completion_number", type=int, default=1,
                        help="How many chat completion choices to generate for each input message.")
    parser.add_argument("--temperature", type=float, default=1,
                        help="What sampling temperature to use, between 0 and 2.\
                        Higher values like 0.8 will make the output more random, while lower values like 0.2 will make it more focused and deterministic.")
    parser.add_argument("--top_p", type=float, default=1,
                        help="An alternative to sampling with temperature, called nucleus sampling, where the model considers the results of the tokens with top_p probability mass.\
                        So 0.1 means only the tokens comprising the top 10% probability mass are considered.")

    # Arguments for Agent Scheduler
    parser.add_argument("--num_workers", type=int, default=10,
                        help="Number of multi-threads used for edit.")
    parser.add_argument("--edit_mode", nargs='+', type=str,
                        default=["0", "3"],
                        help="List of mode to get output for instruction data samples.")
    parser.add_argument("--judge_mode", type=str, default="compare",
                        choices=["compare", "score"],
                        help="Way of judge to express their judgement.")
    parser.add_argument("--max_evol_iter", type=int, default=3,
                        help="Max number of iterations to evol for the framework.")

    # Arguments for Agents
    parser.add_argument("--agent_names", nargs='+', type=str,
                        default=["<Anna>", "<Bruno>",
                                 "<Charles>", "<David>", "<Emma>"],
                        help="Names of agents.")
    parser.add_argument("--agent_wind_size", type=int, default=0,
                        help="Size of window to control agents' visible memory for each action. 0 means all memory are visible.")
    parser.add_argument("--max_agent_len", type=int, default=16000,
                        help="The maximum number of input tokens for each agent (set to prevent exceeding the maximum input length limit of LLMs).")
    
    # Arguments for Multi-turn Conversations
    parser.add_argument("--conv_wind_size", type=int, default=2,
                        help="Size of window to control visible history for multi-turn dialogue data.")
    parser.add_argument("--max_optimize_turn", type=int, default=None,
                        help="Max number of rounds to optimize in multi-turn dialogue data.")
    parser.add_argument("--max_optimize_len", type=int, default=2048,
                        help="Max number of tokens to optimize in multi-turn dialogue data.")
    return parser.parse_args()


def prepare_for_loading(args):
    args.agent_names = {
        "positive": args.agent_names[0],
        "critical": args.agent_names[1],
        "advisor": args.agent_names[2],
        "editor": args.agent_names[3],
        "judge": args.agent_names[4],
    }
    # convert reverse indx of loading data
    if args.end_indx is not None:
        assert args.start_indx < args.end_indx, "The start_indx should be smaller than end_indx for data loading!"
    args.mem_path = get_mem_path(args)
    args.res_path = get_res_path(args)
    args.error_path = get_error_path(args)
    args_dict = save_args(args, save_log_path)
    my_log.info(args_dict)

    load_api_keys(args)
    my_log.info("Ready to start agent scheduler!")


def load_api_keys(args):
    if args.use_local_model:
        args.api_keys = "<LOCAL_API_KEY>"
    else:
        assert args.api_keys, "Keys should be set for proprietary model!"
        with open(args.api_keys, 'r') as file:
            api_dict = json.load(file)
 
        if args.proxy_api_url:
            args.api_keys = api_dict["proxy"]
        else:
            if "gpt" in args.model_name:
                args.api_keys = api_dict["openai"]
            elif "ernie" in args.model_name:
                args.api_keys = api_dict["ernie"]
            elif "glm" in args.model_name:
                args.api_keys = api_dict["glm"]
            else:
                raise NotImplementedError


def main(args):
    prepare_for_loading(args)
    sched = ConCurAgentScheduler(args)
    sched.main_run()


if __name__ == "__main__":
    args = parse_args()
    main(args)
