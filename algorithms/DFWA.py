# use new mutation operator
# Use discrete encoder

import os
import time
import numpy as np
from .archive import Archive
from icecream import ic
# from operators import operator as opt
# from operators import remap
# from population.bin import BaseBinFirework
# from operators.plot import Plot
from yflog import Logger


EPS = 1e-8


class DFWA(object):

    def __init__(self):
        # Definition of all parameters and states
        # each object has a choose probability
        # use improvement information to modify prob
        # params
        self.fw_size = None
        self.sp_size = None
        self.init_amp = None

        self.wm_ratio = None

        # states
        self.pop = None
        self.fit = None
        self.amps = None
        self.nspk = None
        self.prob1 = None
        self.prob2 = None
        # num of objects
        self.s=None

        # problem related params
        self.dim = None
        self.lb = None
        self.ub = None
        self.max = False
        self.iter=0
        self.logger=Logger(filename='./yflog/fwa.log')



        # load default params
        self.set_params(self.default_params())

    def default_params(self, benchmark=None):
        params = {}
        params['fw_size'] = 2
        params['sp_size'] = 16
        params['wm_ratio'] = 0.10
        params['init_ratio'] = 0.5

        return params

    def set_params(self, params):
        for param in params:
            setattr(self, param, params[param])

    def optimize(self, e):

        self.init(e)
        # init recorder
        self.amp_recorder=[]



        while not e.terminate():

            # explode
            spk_pops= self.explode()

            # mutate
            mut_pops= self.mutate(spk_pops)

            # merge and select
            n_pop, n_fit = self.select(spk_pops, mut_pops, e)


            # restrict min and max amp
            for idx in range(self.fw_size):
                if n_fit[idx] > self.fit[idx]:
                    # improvement
                    self.amps[idx] *= 1.2
                    self.amps[idx] = np.minimum(self.amps[idx], self.init_amp)
                    # decrease prob to half or other ratio
                    ind = n_pop[idx ]-self.pop[idx]
                    ind1=ind>0 # 0->1
                    ind2=ind<0 # 1->0

                    self.prob1[idx, ind1] = self.prob1[idx, ind1]*1
                    self.prob1[idx, ind2] = self.prob1[idx, ind2]*1

                    self.prob2[idx, ind1] = self.prob2[idx, ind1]*1
                    self.prob2[idx, ind2] = self.prob2[idx, ind2]*1



                    self.prob1[idx] = (
                        self.amps[idx]/np.sum(self.prob1[idx]))*self.prob1[idx]
                    self.prob2[idx] = (
                        self.amps[idx]/np.sum(self.prob2[idx]))*self.prob2[idx]

                else:
                    self.amps[idx] *=0.7
                    self.amps[idx] = np.maximum(self.amps[idx], self.min_bit)
                    # # normalize prob
                    self.prob1[idx] = (
                        self.amps[idx]/np.sum(self.prob1[idx]))*self.prob1[idx]
                    self.prob2[idx] = (
                        self.amps[idx]/np.sum(self.prob2[idx]))*self.prob2[idx]
                    
                # append amp

            
            self.amp_recorder.append(self.amps.tolist())
            self.logger.info('Iter: {}, amp: {}'.format(self.iter, self.amps))

            self.pop = n_pop
            self.fit = n_fit
            self.iter+=1

        # save_path_amp='/home/liuyifan/code/knapsack/data/amp/{}'.format(self.__class__.__name__)
        
        # if not os.path.exists(save_path_amp):
        #     os.mkdir(save_path_amp)
        # name_amp='amp-{}.npy'.format(e.bench_id)
        # np.save(os.path.join(save_path_amp,name_amp),np.array(record_amp))
        # # add plot function
        # if e.traj_mod:
        #     p.plot_traj_save(e)

        return e.best_y, e.best_x

    def init(self, e):

        # record problem related params
        # self.fun_id = e.bench_id
        self.dim = e.dim
        self.lb = 0
        self.ub = 1
        self.init_amp = self.dim/2.0
        # self.max = e.max
        # remap function from evaluator
        self.min_bit = 1.0
        # ic(self.dim)

        # init states
        self.pop = np.random.randint(
            self.lb, self.ub+1, [self.fw_size, self.dim])
        self.fit = e(self.pop,full=True)
        self.amps = np.array([self.dim*self.init_ratio] * self.fw_size)
        self.nspk = np.array([int(self.sp_size / self.fw_size)-2]*self.fw_size)
        # ic(self.nspk)
        tmp = (self.amps/self.dim).reshape(-1, 1)
        self.prob1 = np.tile(tmp, (1, self.dim))
        self.prob2 = np.tile(tmp, (1, self.dim))

        # self.prob=np.one/np.tile(self.amps,(self.fw_size,1))
        # init random seed
        self.seed = int(os.getpid()*time.time()) % 233280
        np.random.seed(self.seed)

        # init the archive:

        self.arcive=Archive(self.dim)
        

    def explode(self):
        spk_pops = []
        for idx in range(self.fw_size):
            spk_pop = self.box_explode_d_prob2(
                self.pop[idx, :], self.prob1[idx], self.prob2[idx], self.nspk[idx])
            ck=self.arcive.check(spk_pop)
            # check the archive and regen if idv is in the archive
            it=0
            while ck:
                num_regen=len(ck)
                regen_pop= self.box_explode_d_prob2(
                self.pop[idx, :], self.prob1[idx], self.prob2[idx], num_regen)
                spk_pop[ck]=regen_pop
                ck=self.arcive.check(spk_pop)
                it+=1
                if it > 100:
                    # print archive and regen
                    print(self.arcive.data)
                    print(spk_pop)
                    print(ck)
                    raise 'Error: regen too many times'
            self.arcive.update()

            # spk_fit,spk_real = e(spk_pop)
            spk_pops.append(spk_pop)
            # spk_fits.append(spk_fit)
        return spk_pops
    
    def box_explode_d_prob2(self,idv, prob1 ,prob2, num_spk):
        # prob1 => increase
        # prob2 => decrease
        dim = idv.shape[-1]
        # ratio = amp/dim
        bias1 = np.random.rand(num_spk, dim) < prob1
        spks = np.tile(idv, (num_spk, 1))
        spks = spks+bias1
        bias2 = np.random.rand(num_spk, dim) < prob2
        spks = spks-bias2
        # remap accroding s

        spks=np.clip(spks,0,1)

        return spks
    


    def mutate(self, spk_pops):
        mut_pops = []
        mut_fits = []
        for idx in range(self.fw_size):
            mut_pop = self.crossover(
                spk_pops[idx])

            ck=self.arcive.check_mut(mut_pop)
            # check the archive and regen if idv is in the archive
            it=0
            while ck:
                regen_pop= self.crossover(
                spk_pops[idx])
                ck=self.arcive.check_mut(regen_pop)
                if it > 100:
                    # print archive and regen
                    print(self.arcive.data)
                    raise 'Error: regen too many times'
            self.arcive.update()


            
            mut_pops.append(mut_pop)
            # mut_fits.append(mut_fit)
        return mut_pops
    
    def crossover(self, spk_pop):
        # random choose two spk and perform crossover
        # sort 
        n,d=spk_pop.shape
        # idx=np.argsort(spk_fits)
        # rank=np.argsort(idx)
        # weight=(np.arange(n)+1)**2
        # prob=weight/np.sum(weight)
        choose_two=np.random.choice(n,2)
        # choose on bit
        bit=np.random.randint(1,n-1,1)[0]
        new_1=spk_pop[choose_two[0]].copy()
        new_2=spk_pop[choose_two[1]].copy()
        tmp=new_1[:bit].copy()
        new_1[:bit]=new_2[:bit]
        new_2[:bit]=tmp
        return np.vstack([new_1,new_2])
        
        
        
    


    def select(self, spk_pops,mut_pops, e):
        n_pop = np.empty_like(self.pop)
        n_fit = np.empty_like(self.fit)
        # ic(self.pop)
        # ic(self.fit)
        # ic(n_fit)


        
        max_flag = self.max

        for idx in range(self.fw_size):
            tot_pop = np.vstack(
                [self.pop[idx, :], spk_pops[idx], mut_pops[idx]])
            tot_fit,tot_idx=e(tot_pop,return_idx=True)
            # ic(tot_fit)
            # ic(tot_idx)
            n_pop[idx]=tot_pop[tot_idx]
            n_fit[idx]=tot_fit

        return n_pop, n_fit

    def elite_select(self, pop, fit, max_flag, topk=1):
        if max_flag:
            if topk == 1:
                max_idx = np.argmax(fit)
                return pop[max_idx, :], fit[max_idx]
            else:
                sort_idx = np.argsort(fit)
                top_idx = sort_idx[-topk:]
                return pop[top_idx, :], fit[top_idx]
        else:
            if topk == 1:
                min_idx = np.argmin(fit)
                return pop[min_idx, :], fit[min_idx]
            else:
                sort_idx = np.argsort(fit)
                top_idx = sort_idx[:topk]
                return pop[top_idx, :], fit[top_idx]

    # NOTE: use clip instead of remap
    # def remap(self, samples):
    #     return e.remap(samples)
