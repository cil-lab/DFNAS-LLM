import numpy as np


class Archive(object):
    def __init__(self,dim):
        self.cap=1000
        self.dim=dim

        # storage
        self.data=set()
        self.tmp=set()
        
        
    def check(self,x:np.ndarray):
        n,dim=x.shape
        idx=[]
        for i in range(n):
            s=self.encode(x[i])
            if s in self.data:
                idx.append(i)
            else:
                self.save(s)
        return idx
    

    def check_mut(self,x:np.ndarray):
        n,dim=x.shape
        idx=[]
        for i in range(n):
            s=self.encode(x[i])
            if s in self.data:
                return True
            else:
                self.save(s)
        return False
    
    
    

    
    def update(self):
        self.data.update(self.tmp)
        self.tmp=set()

        


    def save(self,s:str):
        self.tmp.add(s)

        
    def encode(self,x:np.ndarray):
        # to string:
        a_list=x.tolist()
        a_string=''.join(map(str,a_list))
        return a_string
    


    


    
        

        
        

    