import numpy as np
import matplotlib.pyplot as plt
import os

from numpy.core.fromnumeric import shape

dir_name = os.path.dirname(os.path.realpath(__file__)).split("/")

algo = "expected_sarsa/"
batch_name = "small_solids"
batch_num = "_first_batch/"
main_path = "/home/repairs/myRepos/tfm_rl_uc3m/"
load_save_path =  main_path + "result_data/" + algo + batch_name + batch_num

ql_values = np.transpose( np.load(load_save_path + "agent_ql_values.npy") )
ql_func = np.amax(ql_values, axis=0)

reward_curve = np.load(load_save_path + "reward_curve.npy")
performance = np.load(load_save_path + "p_reached_per_episode.npy")

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


fig, (ax0,ax1) = plt.subplots(2,1)

ax0.plot(reward_curve)
ax0.set_ylabel("Recompensas por episodio")
ax0.set_xlabel("Episodios")
ax0.set_title("Prueba de liquidos")

ax1.plot(performance, 'g')
ax1.set_ylabel("Particulas")
ax1.set_xlabel("Episodios")
ax1.set_title("Rendimiento del agente")

fig.tight_layout()
plt.savefig(load_save_path + "p_in_goal_reward_curve.png")
plt.close()

fig, (ax0,ax1) = plt.subplots(2,1)

ax0.plot(ql_func,'r')
ax0.set_ylabel("Q(s,a)")
ax0.set_xlabel("Estados")
ax0.set_title("Función de valores")

ax1.pcolor(policy , edgecolors='k',linewidth=0.5)
ax1.set_ylabel("Acción")
ax1.set_xlabel("Estado")
ax1.set_title("Politica obtenida")

fig.tight_layout()
plt.savefig(load_save_path + "policy_value_func.png")
plt.close()




"""
plt.pcolormesh(ql_values)
plt.show()
"""
