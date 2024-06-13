from utils import get_role_prompt, format_mistral_prompt
from copy import deepcopy
import json
import os


class LLMAgent():
    def __init__(self,
                 model,
                 agent_role: str = None,
                 agent_names: dict = None,
                 api_key: str = None,
                 agent_wind_size: int = None,
                 max_agent_len: int = None,
                 ) -> None:
        self.model = model
        self.api_key = api_key
        self.memory = []
        self.agent_role = agent_role
        self.all_agent_names = agent_names
        self.agent_name = agent_names[agent_role]
        self.agent_wind_size = agent_wind_size
        self.max_agent_len = max_agent_len
        self.__init_agent_role()

    def __init_agent_role(self) -> None:
        role_prompt = get_role_prompt(self.agent_role)
        self.update_mem(new_mem=role_prompt, role='system')
        self.role_prompt = role_prompt

    def __save_session(self, mem_session, session_name: str = None) -> None:
        mem_path = os.getcwd() + '/mems/'
        if not os.path.exists(mem_path):
            os.makedirs(mem_path)
        tmp_file = os.path.join(mem_path, session_name)
        if not os.path.isfile(tmp_file):
            tmp = []
        else:
            with open(tmp_file, 'r') as f:
                tmp = json.load(f)

        cur_session = {
            "agent_role": self.agent_role,
            "agent_name": self.agent_name,
            "mem_session": mem_session
        }
        tmp.append(cur_session)

        with open(tmp_file, 'w') as f:
            json.dump(tmp, f)

    def __check_mem(self) -> str:
        if "glm" in self.model.model_name:
            self.memory = self.model.check_hist(self.memory)
        elif "ernie" in self.model.model_name:
            self.memory, _ = self.model.check_hist(self.memory)

    def act(self, session_name: str = None) -> str:
        self.__check_mem()
        if (not self.agent_wind_size) or (self.agent_wind_size == 0):
            vis_mem = deepcopy(self.memory)
        else:
            vis_mem = []
            tmp_count = 0
            for mem in self.memory[::-1]:
                if mem["role"] == "system":
                    vis_mem.append(mem)
                else:
                    if tmp_count < self.agent_wind_size:
                        vis_mem.append(mem)
                        tmp_count += 1
                    continue
            vis_mem = list(reversed(vis_mem))
        
        if "ernie" in self.model.model_name:
            resp = self.model.query(
                prompt=vis_mem,
                api_key=self.api_key,
                system=self.role_prompt
            )
        elif any(key in self.model.model_name for key in ["mixtral", "mistral"]):
            # mistral/mixtral does not accept "system" prompt
            vis_mem=format_mistral_prompt(vis_mem)
            resp = self.model.query(
                prompt=vis_mem,
                api_key=self.api_key,
            )
        else:
            resp = self.model.query(
                prompt=vis_mem,
                api_key=self.api_key
            )
        
        # trace every action
        mem_session = vis_mem
        resp_dict = {
            "role": "assistant",
            "content": resp,
        }
        mem_session.append(resp_dict)
        self.__save_session(mem_session, session_name)
        assert resp != "__error__", "An error occurred during model generation."

        return resp

    def update_mem(self, new_mem: str = None, role: str = None, name: str = None) -> None:
        self.memory.append({
            "role": role,
            "content": f"{name}:\n" + new_mem if name is not None else new_mem
        }
        )

    def clear_mem(self, clear_sys: bool = False) -> None:
        if clear_sys:
            self.memory = []
        else:
            new_mem = []
            for mem in self.memory:
                if mem["role"] == "system":
                    new_mem.append(mem)
            self.memory = new_mem

    def remind_agent_role(self) -> None:
        self.update_mem(new_mem=self.role_prompt, role='system')
