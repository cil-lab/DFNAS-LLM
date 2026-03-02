import os
import numpy as np
from llm.api import API
from icecream import ic
from yflog import Logger
import json
import time


class Evaluator(object):
    def __init__(self,f, max_eval=500,benchmark='cifar10',proxy=False, pre_epoch=10, stop_epoch=50,reeval_num=1, log_file=None):
        # params
        self.max_eval=max_eval
        self.num_eval=0
        self.f=f
        self.dim=self.f.dim
        self.train=True
        self.cur_x=None
        self.cur_y=None
        self.best_x=None
        self.best_y=None
        self.proxy=proxy
        self.proxy_ratio=0.2
        self.benchmark=benchmark
        self.pre_epoch = pre_epoch
        self.full_epoch =stop_epoch
        self.storage = []
        self.reeval_num=reeval_num
        self.full_recorder=[]
        self.log_file = log_file

        if self.log_file:
            with open(self.log_file, 'w') as f:
                f.write(f"Start Experiment: {time.ctime()}\n")
        
        # ...


        self.traj=[]
        self.traj_iter=[]

        # init the netfunction
        if self.proxy:
            self.init_proxy()

    def init_proxy(self):
        # init llm proxy
        self.api=API(dataset=self.benchmark,multi=True,pre_epoch=self.pre_epoch)


    def _proxy_call(self,x):
        n=x.shape[0]
        # first layer
        # ic(x)
        # ic(n)
        tmp=self.f(x,epoch=self.pre_epoch)
        pre_train_acc = [item[0][:self.pre_epoch] for item in tmp]
        pre_valid_acc=[item[1][:self.pre_epoch] for item in tmp]
        # print(pre_train)
        rank=self.api(pre_train_acc,pre_valid_acc)
        
        idx=np.argsort(np.array(rank))
        top_idxs=idx[:self.reeval_num]

        re_eval_x=x[top_idxs]
        # ic(re_eval_x)

        best_y_tmp=self.f(re_eval_x,epoch=self.full_epoch)
        self.full_recorder.extend(best_y_tmp)
        self.log.info(best_y_tmp)
        

        best_y=np.array([np.max(item[1]) for item in best_y_tmp])
        
        best_idx=np.argmax(best_y)
        best_y=best_y[best_idx]


  
        return best_y, best_idx
        

    def __call__(self,x:np.ndarray,return_idx=False, full=False):
        # judge the shape
        if len(x.shape)==1:
            n=1
            x=x.reshape((1,-1))
            ans = self.f(x,epoch=self.full_epoch)
            acc = np.max(ans[1])
            
            if self.cur_y is None or self.cur_y < acc:
                self.cur_y=acc.item()
                self.cur_x= x[0] 
                if self.log_file:
                    with open(self.log_file, 'a') as f:
                        f.write(f"Eval: {self.num_eval}, Acc: {self.cur_y}, Code: {self.cur_x.tolist()}\n")

            self.traj.append(self.cur_y)



        else:
            n=x.shape[0]
            if self.proxy==True and n > 2 and not full:
                acc, best_idx =self._proxy_call(x)
                best_acc=acc.item()
            else:
                ans=self.f(x,epoch=self.full_epoch)
                self.full_recorder.extend(ans)
                acc=np.array([np.max(item[1]) for item in ans])
                best_idx=np.argmax(acc)
                best_acc=acc[best_idx]
                
            # max_acc=acc[best_idx]
            if self.cur_y is None or self.cur_y < best_acc:
                self.cur_x=x[best_idx]
                self.cur_y=best_acc
                if self.log_file:
                    with open(self.log_file, 'a') as f:
                        f.write(f"Eval: {self.num_eval}, Acc: {self.cur_y}, Code: {self.cur_x.tolist()}\n")

            # save to storage
            tmp={'x':x[best_idx].tolist(),'acc':best_acc}
            self.storage.append(tmp)

            self.traj.extend([self.cur_y]*n)
            
        self.num_eval+=n
        self.traj_iter.append(self.cur_y)
        self.log.info(self.cur_y)
        self.log.info(self.cur_x)
        print(f"[{time.strftime('%H:%M:%S')}] Progress: {self.num_eval:3d}/{self.max_eval} ({(self.num_eval/self.max_eval)*100:5.1f}%) | Best Acc: {self.cur_y:.2f}")


        # return train final acc
        if return_idx:
            return best_acc, best_idx
        else:
            if full:
                return acc
            else:
                return best_acc

        
    def terminate(self):
        if self.num_eval>= self.max_eval:
            self.best_x=self.cur_x
            self.best_y=self.cur_y
            try:
                with open('./output_data/full_recorder.json','w') as f:
                    json.dump(self.full_recorder,f)
            except Exception as e:
                print(e)
                
            return True
        else:
            return False
        









# if __name__=="__main__":
#     x=np.random.randint(0,2,(3,22))
#     function=F_cifar10()
#     e=Evaluator(function)
#     ans=e(x)
#     print(ans)
    
