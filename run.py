import os
import argparse
from algorithms.DFWA import DFWA
from objective import Evaluator
from net.func import F_cifar10, F_cifar100
from net.func54 import F_cifar10 as F_cifar10_54
import time 
import json



def parsing():
    # experiment setting for benchmark testing
    parser = argparse.ArgumentParser(description="Run FWA for NAS")


    # agortihm params
    parser.add_argument("--alg", "-a", help="Algorithm name")

    # testing params
    parser.add_argument("--name", "-n", default="", help="Name of experiment")

    parser.add_argument("--traj", "-t", default=0, type=int, choices=[0, 1], help="Choose whether to record the trajactory, 1 for true and 0 for false")

    # parser.add_argument("--proxy", "-p", default=0, type=int, choices=[0, 1], help="Choose whether to use proxy model, 1 for true and 0 for false.")


    parser.add_argument(
        "--rep", "-r", default=1, type=int, help="Repetition of each problem"
    )
    parser.add_argument(
        '--gpu',
        help='Using gpu device, "0" for example is the device id.'
    )
    parser.add_argument(
        '--ndev',
        default=1,
        help='The number of algorithm runs on a gpu, needs to be specified using gpu first',
        type=int
    )
    parser.add_argument(
        '--epoch',
        default=10,
        help='The number of early stop epoch',
        type=int
    )

    parser.add_argument(
        '--max_eval',
        default=200,
        help='The number of evaluation',
        type=int
    )
    
    parser.add_argument(
        '--stop_epoch',
        default=100,
        help='The number of evaluation',
        type=int
    )

    parser.add_argument(
        '--pre_epoch',
        default=10,
        help='The number of evaluation',
        type=int
    )
    parser.add_argument(
        '--reeval_num',
        default=1,
        help='The number of reeval',
        type=int
    )
    

    parser.add_argument(
        '--bench',
        help='Benchmark for nas search, option: cifar10 or cifar100',
        type=str
    )
    parser.add_argument(
        '--proxy',
        help='Whether to use a proxy.',
        action='store_true',
        default=False
    )
    parser.add_argument(
        '--dim',
        help='The dimension of the network',
        type=int,
        default=72,
        choices=[54, 72],
    )


    return parser.parse_args()


def main(args):
    start_time=time.time()

    run_name=args.name
    device='cuda:{}'.format(args.gpu)

    algorithm=args.alg

    benchmark=args.bench
    stop_epoch= args.stop_epoch
    pre_epoch= args.pre_epoch

    max_eval=args.max_eval
    
    
    alg=DFWA()
    params=alg.default_params()
    alg.set_params(params)

    # save result
    save_dir='./logs/{}'.format(algorithm)
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
        print(save_dir)
    
    log_file = os.path.join(save_dir, 'traj_{}_{}.txt'.format(benchmark, run_name))

    # load function
    if benchmark=='cifar10':
        if args.dim==54:
            print('cifar-10-54')
            f=F_cifar10_54(epoch=stop_epoch,device=device)
        else:
            f=F_cifar10(epoch=stop_epoch,device=device)
    else:
        print('cifar-100')
        f=F_cifar100(epoch=stop_epoch,device=device)
    # print(args.proxy)
    e=Evaluator(f=f,max_eval=max_eval,proxy=args.proxy,stop_epoch=args.stop_epoch,pre_epoch=pre_epoch,reeval_num=args.reeval_num, log_file=log_file)
    final_acc, final_net_x = alg.optimize(e)
    # final_acc, final_net_x = 1,2
    end_time=time.time()
    time_used=end_time-start_time

    # print(final_acc,final_net_x)

    
    save_name='{}-{}.json'.format(benchmark,run_name)
    traj=e.traj
    print(f"Time used: {time_used:.2f}s")
    # print(alg.arcive.data)
    storage=e.storage
    
    save_data={'run_name':run_name,'benchmark':benchmark,'algorithm':algorithm,'traj':traj,'acc':final_acc,'net':final_net_x.tolist(),'time':time_used,'storage':storage}

    json_name=os.path.join(save_dir,save_name)
    with open(json_name,'w') as f:
        json.dump(save_data,f)



if __name__=="__main__":
    args=parsing()
    main(args)



