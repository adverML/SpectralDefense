
import torch
import numpy as np



def plot_model(args, model):
    
    characteristics  = np.asarray([])
    characteristics_adv  = np.asarray([])
    
    with torch.no_grad():
        print("")
        
    return characteristics, characteristics_adv

# with torch.no_grad():
#     for i, m in enumerate(filter(lambda m: type(m) == torch.nn.relu and m.kernel_size == (3, 3), model.modules())):
        
#         import pdb; pdb.set_trace()

# with torch.no_grad():
#     for i, m in enumerate(filter(lambda m: type(m) == torch.nn.Conv2d and m.kernel_size == (3, 3), model.modules())):
#         weight = m.weight.detach().cpu().numpy().copy()
#         shape = weight.shape
#         w = ( weight.reshape(-1, 9) )
#         t = np.abs(w).max() / 100    
#         cache = np.ones_like(w)
#         cache[np.abs(w) < t] = 0
#         dead_mask = (cache.sum(axis=1) == 0)
#         # with np.printoptions(edgeitems=1000):
#         dead_filter = np.where(dead_mask == True)[0]
#         # print("sparsity: ", dead_filter)
#         # dead_filter_reshaped = dead_filter.flatten().reshape(shape)
        
#         if len(dead_filter) > 0:
#             w[dead_filter, :] = 0
            # import pdb; pdb.set_trace()
        # m.weight = weight
    # usv = np.linalg.svd(w - w.mean(axis=0), full_matrices=False, compute_uv=True)
    # u = usv[0]
    # s = usv[1]
    # v = s**2 / (w.shape[0]-1)
    # e = scipy.stats.entropy(v, base=10)
    # def H_T(n):
    #     L, x0, k, b = (1.2618047,2.30436435,0.88767525,-0.31050834)  # min distr.
    #     return L / (1 + np.exp(-k * (np.log2(n) - x0))) + b
    
    # dead = (e >= H_T(w.shape[0])) | (e < 0.5)
    # print("dead_mask: ", )
    # print("i: ", i, ",dead: ", dead, ", e: ", e, ", H_T: ", H_T(w.shape[0]))

