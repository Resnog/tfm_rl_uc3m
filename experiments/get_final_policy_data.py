import numpy as np
import matplotlib.pyplot as plt
import os

from numpy.core.fromnumeric import shape
from numpy.lib.npyio import load

def get_policy_graph(ql_values):

    best_action = np.argmax(ql_values,axis=0)
    worst_action = np.argmin(ql_values,axis=0)

    policy = np.zeros(ql_values.shape)

    for a in range(ql_values.shape[0]):
        for s in range(ql_values.shape[1]):
            if a == best_action[s]:
                policy[a,s] = 1
            elif a == worst_action[s]:
                policy[a,s] = 0
            else:
                policy[a,s] = 0.5

    return policy

def save_policy_image(ql_func, policy, data_path, par_type):
    
    fig, (ax0,ax1) = plt.subplots(2,1)

    ax0.plot(ql_func,'r')
    ax0.set_ylabel("v(s)")
    ax0.set_xlabel("Estados")
    ax0.set_title("Función de valores")

    ax1.pcolor(policy , edgecolors='k',linewidth=0.5)
    ax1.set_ylabel("Acción")
    ax1.set_xlabel("Estado")
    ax1.set_title("Politica obtenida")

    fig.tight_layout()
    plt.savefig(data_path + par_type + "_policy_value_func.png")
    plt.show()
    plt.close()

dir_name = os.path.dirname(os.path.realpath(__file__)).split("/")
dir_name = dir_name[:-1]
main_path = '/'.join(dir_name) + '/'

algo = "dyna_q/"
par_type = "big_solids"
data_path =  main_path + "result_data/" + algo 
last_batch = "_only_batch/"

ql_values = np.transpose( np.load(data_path + par_type + last_batch +"agent_ql_values.npy") )
ql_func = np.amax(ql_values, axis=0)

policy = get_policy_graph(ql_values)
save_policy_image(ql_func,policy,data_path,par_type)
