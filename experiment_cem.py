#from scipy import special
#from loss.grad_2p import Grad_2p
import numpy as np
import mxnet as mx
import os
from loss.cifer_resnet  import Grad_2p
#from loss.grad_2p  import Laplace as Grad_2p

def l1_norm(x):
    return np.linalg.norm(x.flatten(),ord=1)

def trace_norm(x):
    s=np.linalg.svd(x,full_matrices=False,compute_uv=False)
    return np.linalg.norm(s.flatten(),ord=1)
    
def main(args):
    num_samples=args['num_samples']
    num_iter=args['num_iterations']
    spectral=args['spectral']
    device=args['device']
    l1=args['l1']
    l2=args['l2']
    mode=args['mode']
    data=args['data']
    kappa=args['kappa']
    delta=args['smooth']
    n=args['batch']
    seed=args['seed']
    result_path=args['path']
    root=args['imagenet']
    zo=False
    if spectral:
        print("use spectral algorithm")
        import solvers.spectral.ao_grad as ao_grad
        import solvers.spectral.ao_ftrl as ao_ftrl
        import solvers.spectral.ao_exp_grad as ao_exp_grad
        import solvers.spectral.ao_exp_ftrl as ao_exp_ftrl
        get_l1=trace_norm
    else:
        print("use R^d algorithm")
        import solvers.ao_grad as ao_grad
        import solvers.ao_ftrl as ao_ftrl
        import solvers.ao_exp_grad as ao_exp_grad
        import solvers.ao_exp_ftrl as ao_exp_ftrl
        import solvers.fista as fista
        get_l1=l1_norm
    if device <0:
        ctx=mx.cpu()
    else:
        ctx=mx.gpu(device)
    
    if data=="CIFER":
        import loss.cifer_resnet as model
        max_index=10000
    else:
        import loss.imagenet_resnet as model
        max_index=50000
    if delta>0.0:
        zo=True
    if not os.path.exists(result_path):
        os.makedirs(result_path)
    for alg in [fista]:
    #for alg in [ao_exp_grad,ao_exp_ftrl,ao_grad,ao_ftrl]:
        file_path=data+'_'+mode+'_'+alg.__name__+'_'+str(zo)
        file_path=os.path.join(result_path,file_path)
        if not os.path.exists(file_path):
            os.makedirs(file_path)
        print("start algorithm ",alg.__name__)
        train_loss=[]
        attack_loss=[]
        average_train_loss=np.zeros(num_iter+1)
        attack_array=np.zeros(num_iter+1)
        for class_idx in range(0,10):
            train_loss_array=np.zeros(num_iter+1)
            attack_array=np.zeros(num_iter+1)
            print("class ",class_idx)
            np.random.seed(seed+class_idx)   
            shuffled_indices=np.random.permutation(max_index)
            current_index=0
            for sample_index in range(0,num_samples):
                print("sample",sample_index)
                cem=None
                found=False
                while not found and current_index<max_index:            
                    if data=="CIFER":
                        cem=model.CEM(index=shuffled_indices[current_index],ctx=ctx,kappa=kappa)
                    else:
                        cem=model.CEM(index=shuffled_indices[current_index],ctx=ctx,kappa=kappa,folder_path=root)
                    current_index+=1
                    if data=="CIFER" and (cem.label== class_idx and cem.correct):
                        found=True
                        break
                    elif data=="IMAGENET" and cem.correct:                        
                        found=True
                        break
                if not found:
                    print("not enough sample of class: ", class_idx)
                    return
                if mode=="PP":
                    func=model.PP_Loss(cem)
                    func_p=model.PP_Grad(cem)
                    init=cem.pp_init
                    upper=cem.pp_upper
                    lower=cem.pp_lower
                else:
                    func=model.PN_Loss(cem)
                    func_p=model.PN_Grad(cem)
                    init=cem.pn_init
                    upper=cem.pn_upper
                    lower=cem.pn_lower
                if zo:
                    func_p=Grad_2p(n=n,func=func,delta=delta)
                regl1=get_l1(init)
                regl2=np.linalg.norm(init.flatten(),ord=2)**2
                attack=func(init)
                print('iteration: ',0,' attack: ', attack,' loss: ',1.0,' l1: ',regl1,' l2: ',regl2 )
                init_loss=attack+l1*regl1+0.5*l2*regl2+kappa
                train_loss.append(1.0)
                attack_loss.append(attack)
                def callback(res):
                    regl1=get_l1(res.x)
                    regl2=np.linalg.norm(res.x.flatten(),ord=2)**2
                    cur_loss=res.func+l1*regl1+0.5*l2*regl2+kappa
                    if res.nit%(num_iter//10)==0:
                        print('iteration: ',res.nit,' attack: ', res.func,' loss: ',cur_loss/init_loss,' l1: ',regl1,' l2: ',regl2 )
                    train_loss.append(cur_loss/init_loss)
                    attack_loss.append(res.func)
                alg.fmin(func=func, func_p=func_p, x0=init, lower=lower, upper=upper,l1=l1,l2=l2,maxfev=num_iter,callback=callback,epoch_size=1)
                train_loss_array+=((1.0/num_samples)*np.array(train_loss))
                attack_array+=((1.0/num_samples)*np.array(attack_loss))
                loss_file = os.path.join(file_path,'class_'+str(class_idx)+str(sample_index)+'_loss.csv')
                attack_file = os.path.join(file_path,'class_'+str(class_idx)+str(sample_index)+'_attack.csv')
                np.savetxt(loss_file, np.array(train_loss), delimiter=",")
                np.savetxt(attack_file, np.array(attack_loss), delimiter=",")
                train_loss=[]
                attack_loss=[]
        average_train_loss+=(train_loss_array/10.0)
        loss_file = os.path.join(file_path,'average_loss.csv')
        np.savetxt(loss_file, average_train_loss, delimiter=",")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("-n", "--num_samples", type=int, default=20)
    parser.add_argument("-t", "--num_iterations", type=int, default=1000)
    parser.add_argument("-s", "--spectral", type=bool, default=False)
    parser.add_argument("-d", "--device", type=int, default=-1)
    parser.add_argument("--l1", type=float, default=0.5)
    parser.add_argument("--l2", type=float, default=0.5)
    parser.add_argument("--mode", choices=["PN", "PP"], default="PN")
    parser.add_argument("--data", choices=["IMAGENET", "CIFER"], default="CIFER")
    parser.add_argument("--kappa", type=float, default=10)
    parser.add_argument("--smooth", type=float, default=0.0)
    parser.add_argument("-b","--batch", type=int, default=10)
    parser.add_argument("--seed", type=int, default=48)
    parser.add_argument("--path", type=str, default='experiment_results')
    parser.add_argument("--imagenet", type=str, default='./datasets/imagenet')
    args = vars(parser.parse_args())
    main(args)