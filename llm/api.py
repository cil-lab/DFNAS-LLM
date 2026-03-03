import time
import requests
import json
import torch
import numpy as np
from icecream import ic
import asyncio
import aiohttp
from typing import List

from yflog import Logger





class API(object):
    def __init__(self, dataset='cifar10',multi=False,pre_epoch=10,model='v3',params=None,prompt_path='llm/predict.md') -> None:
        
        # 目前只接受v3 和 r1 的模型
        self.dataset = dataset
        self.model_name=model
        self.multi=True
        self.pre_epoch=pre_epoch
        self.max_retry=5
        self.prompt_path=prompt_path
        self.load_model_config()
        
        self.headers = {
            "Content-Type": "application/json",
            "Authorization": "Bearer {}".format(self.token)
        }
        
        self.params={}
        if params is not None:
            self.params.update(params)
        
        
        # load model config
        
        
        # load the prompt
        
        with open(self.prompt_path,'r') as f:
            self.prompt=f.read()
            
        self.init_logger()
 
    def init_logger(self):
        import logging
        import os
        from datetime import datetime

        log_dir = './llm/log'
        if not os.path.exists(log_dir):
            os.makedirs(log_dir)
        
        # Log file name
        log_file = os.path.join(log_dir, f'api_{datetime.now().strftime("%Y%m%d")}.log')
        
        self.logger = logging.getLogger('API_Logger')
        self.logger.setLevel(logging.INFO)
        
        # Clear existing handlers
        if self.logger.hasHandlers():
            self.logger.handlers.clear()
            
        file_handler = logging.FileHandler(log_file)
        formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
        file_handler.setFormatter(formatter)
        
        self.logger.addHandler(file_handler)
        # Prevent propagation to root logger to avoid console output
        self.logger.propagate = False

    def construct_prompt(self,num,pre_epoch):
        prompt=self.prompt.format(num=num,pre_epoch=pre_epoch)
        return prompt
    

    def post(self, system_prompt,user_prompt) -> float:
        # model='claude-instant-1.2'
        # msg='什么是人工智能'
        # print(ques)
        data = {
            "model": self.model,
            "messages": [{"role": "system", "content": system_prompt},{"role": "user", "content": user_prompt}],
            # "stream":False,
            "temperature":0.0,
            'stream': False,
        }
        self.logger.info(data)
        if self.params is not None:
            data.update(self.params)

        r = requests.post(self.url, headers=self.headers,
                          data=json.dumps(data))
        # print(r.content)
        # print(r.json()['choices'][0]['message']['content'])
        print(r.json())
        if self.multi:
            return self.parse_rank(r.json())
        else:
            return self.parse(r.json())
            
    def parse(self, content: dict) -> float:
        out = content['choices'][0]['message']['content']
        st = out.find('[')
        ed = out.find(']')
        return float(out[st+1:ed])
    
    def parse_rank(self,content:dict)->list:
        out = content['choices'][0]['message']['content']
        try:
            ans=json.loads(out)
        except:
            st = out.find('[')
            ed = out.find(']')
            ans=json.loads(out[st+1:ed])

        return ans
    
    
    def _call_multi(self,train_acc:list,valid_acc:list):
        # format x

        num=len(train_acc)
        # text_x=''
        # for item in x:
        #     text_x+=str(item.tolist())
        #     text_x+='\n'
            
        system_prompt=self.construct_prompt(num,self.pre_epoch)
        
        # construct user prompt
        input_dict={}
        for idx in range(len(train_acc)):
            train_acc_item = train_acc[idx]
            valid_acc_item = valid_acc[idx]
            input_dict[idx+1] = {
                'train_acc':train_acc_item,
                'valid_acc':valid_acc_item
            }
            
        input_json=json.dumps(input_dict)
        user_prompt='# input data:\n\n'+input_json
        
        ans=self.pred_multi(system_prompt,user_prompt)

        return ans

    
    def pred_multi(self, system_prompt,user_prompt):
        
        flag = False
        for i in range(self.max_retry):
            try:
                out = self.post(system_prompt,user_prompt)
                flag = True
            except Exception as e:
                time.sleep(0.5)
                print("Retry")
                print(e)

            if flag:
                break

        return out
    
    def __call__(self, train_acc:list,valid_acc:list) -> np.ndarray:
        if self.multi:
            return self._call_multi(train_acc,valid_acc)

            
        
    def load_model_config(self):
        import os
        if self.model_name not in ['v3','r1','gpt-4o','gemini-3-flash-preview','gemini-3-pro-preview','gpt-5']:
            raise ValueError('Model not supported')
        
        # 优先从环境变量读取，如果没有则使用占位符
        self.token = os.getenv("LLM_API_TOKEN", "YOUR_API_TOKEN")
        self.url = os.getenv("LLM_API_URL", "https://api3.wlai.vip/v1/chat/completions/")

        if self.model_name=='v3':
            self.model = os.getenv("V3_MODEL_ENDPOINT", "YOUR_V3_ENDPOINT")
            self.token = os.getenv("V3_API_TOKEN", self.token)
            self.url = os.getenv("V3_API_URL", "https://ark.cn-beijing.volces.com/api/v3/chat/completions/")
        elif self.model_name=='r1':
            self.model = os.getenv("R1_MODEL_ENDPOINT", "YOUR_R1_ENDPOINT")
            self.token = os.getenv("R1_API_TOKEN", self.token)
            self.url = os.getenv("R1_API_URL", "https://ark.cn-beijing.volces.com/api/v3/chat/completions/")
        else:
            self.model = self.model_name
            
            


