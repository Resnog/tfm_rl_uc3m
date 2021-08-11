import numpy as np 

main_path = "/home/greatceph/myRepos/tfm_rl_uc3m/"

a = np.arange(10)
print(a)
save_path = main_path + "experiments/results/"

np.save(save_path+"a.npy",a)

b = np.load(save_path+"a.npy")

print(b)