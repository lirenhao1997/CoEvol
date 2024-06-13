from mylogging import my_log
import requests
import zhipuai
from openai import OpenAI
import openai
import time
import json


class GPTModel():
    def __init__(self, args) -> None:
        self.model_name = args.model_name
        self.max_tokens = args.max_tokens
        self.n = args.completion_number
        self.temp = args.temperature
        self.top_p = args.top_p

    def query(self, prompt: list[dict[str, str]] = None, api_key: str = None) -> str:
        error_cnt = 1
        response = '__error__'
        cnt = 0
        while error_cnt == 1 and cnt < 3:
            try:
                completion = openai.ChatCompletion.create(
                    api_key=api_key,
                    model=self.model_name,
                    max_tokens=self.max_tokens,
                    temperature=self.temp,
                    top_p=self.top_p,
                    n=self.n,
                    messages=prompt
                )
                error_cnt = 0
            except Exception as e:
                my_log.error(f'Error: {e}')
                time.sleep(3)
                cnt += 1
        if cnt == 3:
            time.sleep(3)
            response = '__error__'
        else:
            response = completion.choices[0].message['content']
        time.sleep(3)

        return response


class ProxyGPTModel():
    def __init__(self, args) -> None:
        self.proxy_api_url = args.proxy_api_url
        self.model_name = args.model_name
        self.use_local_model = args.use_local_model
        self.max_tokens = args.max_tokens
        self.n = args.completion_number
        self.temp = args.temperature
        self.top_p = args.top_p

    def query(self, prompt: list[dict[str, str]] = None, api_key: str = None) -> str:
        response_content=self.__query_chat_completion(prompt, api_key)

        return response_content

    def __query_requests(self, prompt: list[dict[str, str]] = None, api_key: str = None)->str:
        headers = {
            'Content-Type': 'application/json',
            'Authorization': f'Bearer {api_key}',
        }
        data = {
            'model': self.model_name,
            'max_tokens': self.max_tokens,
            'temperature': self.temp,
            'top_p': self.top_p,
            'n': self.n,
            'messages': prompt
        }

        has_error = True
        retries = 0
        response_content = "__error__"

        while has_error and retries < 3:
            try:
                response = requests.post(
                    self.proxy_api_url, headers=headers, json=data)
                response.raise_for_status()
                response_content = response.json().get("choices", [{}])[0].get(
                    "message", {}).get("content", "__error__")
                has_error = False
            except Exception as e:
                my_log.error(f"Error: {e}")
                time.sleep(3)
                retries += 1
        return response_content
    
    def __query_chat_completion(self, prompt: list[dict[str, str]] = None, api_key: str = None)->str:
        client = OpenAI(
            api_key=api_key,
            base_url=self.proxy_api_url,
        )
        error_cnt = 1
        response_content = "__error__"
        cnt = 0
        while error_cnt == 1 and cnt < 3:
            try:
                completion = client.chat.completions.create(
                    model=self.model_name,
                    max_tokens=self.max_tokens,
                    temperature=self.temp,
                    top_p=self.top_p,
                    n=self.n,
                    messages=prompt
                )
                error_cnt = 0
            except Exception as e:
                my_log.error(f"Error: {e}")
                time.sleep(3)
                cnt += 1
        if cnt == 3:
            time.sleep(3)
            response_content = "__error__"
        else:
            response_content = completion.choices[0].message.content
        return response_content


class ERNIEModel():
    def __init__(self, args) -> None:
        self.model_name = args.model_name
        self.temp = args.temperature
        self.top_p = args.top_p
        self.post_addr = self.__get_post_addr()

    def __get_access_token(self, api_key: dict = None):
        url = "https://aip.baidubce.com/oauth/2.0/token?grant_type=client_credentials&client_id={client_id}&client_secret={client_secret}".format_map(
            api_key)
        payload = json.dumps("")
        headers = {
            'Content-Type': 'application/json',
            'Accept': 'application/json'
        }
        response = requests.request("POST", url, headers=headers, data=payload)
        return response.json().get("access_token")

    def __get_post_addr(self):
        if self.model_name == "ernie-bot-4":
            post_addr = "https://aip.baidubce.com/rpc/2.0/ai_custom/v1/wenxinworkshop/chat/completions_pro"
        elif self.model_name == "ernie-bot-turbo":
            post_addr = "https://aip.baidubce.com/rpc/2.0/ai_custom/v1/wenxinworkshop/chat/eb-instant"
        else:
            raise NotImplementedError
        return post_addr

    def check_hist(self, prompt: list = None) -> tuple[list, str]:
        # "system" field in ernie model is split with messages
        sys_content = prompt[0]["content"] if prompt[0]["role"] == "system" else None
        new_prompt = prompt[1:] if sys_content else prompt
        return new_prompt, sys_content

    def query(self, prompt: list = None, api_key: dict = None, system: str = None) -> str:
        access_token = self.__get_access_token(api_key)
        url = self.post_addr + f"?access_token={access_token}"
        headers = {
            'Content-Type': 'application/json'
        }
        data = {
            "temperature": self.temp,
            "top_p": self.top_p,
            "system": system if system is not None else "",
            "messages": prompt,
        }
        payload = json.dumps(data)
        has_error = True
        sleep_time = 3
        retries = 0
        response_content = '__error__'

        while has_error and retries < 3:
            try:
                response = requests.request(
                    "POST", url, headers=headers, data=payload)
                response.raise_for_status()
                response_content = json.loads(
                    response.text).get("result", "__error__")
                has_error = False
            except Exception as e:
                my_log.error(f'Error: {e}')
                time.sleep(sleep_time)
                retries += 1

        time.sleep(sleep_time)
        return response_content


class GLMModel():
    def __init__(self, args) -> None:
        self.model_name = args.model_name
        self.temp = args.temperature
        self.top_p = args.top_p

    def query(self, prompt: list[dict[str, str]] = None, api_key: str = None) -> str:
        zhipuai.api_key = api_key
        response = '__error__'
        error_cnt = 1
        cnt = 0
        sleep_time = 3

        while error_cnt == 1 and cnt < 3:
            try:
                completion = zhipuai.model_api.invoke(
                    model=self.model_name,
                    prompt=prompt,
                    temperature=self.temp,
                    top_p=self.top_p
                )
                error_code = completion["code"]
                error_msg = completion["msg"]
                assert error_code == 200, error_msg
                error_cnt = 0
            except Exception as e:
                my_log.error(f'Error: {e}')
                time.sleep(sleep_time)
                cnt += 1

        if cnt == 3:
            response = '__error__'
        else:
            response = completion["data"]["choices"][0]["content"]
            try:
                response = json.loads(response)
            except Exception as e:
                time.sleep(sleep_time)
                return response

        time.sleep(sleep_time)
        return response

    def check_hist(self, hist: list = None) -> list:
        # there is no "system" role in glm model
        # convert and combine system-prompt with the first user-prompt
        new_hist = []
        is_first = True
        sys_p = None
        for indx, p in enumerate(hist):
            if p["role"] == "system":
                assert indx == 0, "system prompt should be the first in prompt history."
                sys_p = p["content"]
                continue
            if p["role"] == "user" and is_first:
                first_usr_p = p["content"]
                new_user_p = sys_p+"\n\n"+first_usr_p if sys_p else first_usr_p
                new_hist.append(
                    {
                        "role": "user",
                        "content": new_user_p
                    }
                )
                is_first = False
            else:
                new_hist.append(p)

        return new_hist


class LocalModel():
    def __init__(self, args) -> None:
        self.proxy_api_url = args.proxy_api_url
        self.model_name = args.model_name
        self.use_local_model = args.use_local_model
        self.max_tokens = args.max_tokens
        self.n = args.completion_number
        self.temp = args.temperature
        self.top_p = args.top_p

    def query(self, prompt: list[dict[str, str]] = None, api_key: str = None) -> str:
        response_content=self.__query_chat_completion(prompt)
        return response_content
    
    def __query_chat_completion(self, prompt: list[dict[str, str]] = None, api_key: str = None)->str:
        openai_api_key = "EMPTY"
        openai_api_base = self.proxy_api_url

        client = OpenAI(
            api_key=openai_api_key,
            base_url=openai_api_base,
        )
        error_cnt = 1
        response_content = "__error__"
        cnt = 0
        while error_cnt == 1 and cnt < 3:
            try:
                completion = client.chat.completions.create(
                    model=self.model_name,
                    max_tokens=self.max_tokens,
                    temperature=self.temp,
                    top_p=self.top_p,
                    n=self.n,
                    messages=prompt
                )
                error_cnt = 0
            except Exception as e:
                my_log.error(f"Error: {e}")
                time.sleep(3)
                cnt += 1
        if cnt == 3:
            time.sleep(3)
            response_content = "__error__"
        else:
            response_content = completion.choices[0].message.content
        return response_content
