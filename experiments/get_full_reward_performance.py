import numpy as np
import matplotlib.pyplot as plt
import os

from numpy.core.fromnumeric import shape

def join_batchs_data(load_path, par_type):

    fb_rew = np.load( load_path + par_type + '_first_batch/reward_curve.npy')
    fb_per = np.load( load_path + par_type + '_first_batch/p_reached_per_episode.npy')

    sb_rew = np.load( load_path + par_type + '_second_batch/reward_curve.npy')
    sb_per = np.load( load_path + par_type + '_second_batch/p_reached_per_episode.npy')

    tb_rew = np.load( load_path + par_type + '_third_batch/reward_curve.npy')
    tb_per = np.load( load_path + par_type + '_third_batch/p_reached_per_episode.npy')

    reward_curve = np.concatenate([fb_rew,sb_rew,tb_rew])
    performance_curve = np.concatenate([fb_per,sb_per,tb_per])

    return reward_curve, performance_curve

def get_batch_data(load_path, par_type, batch_name):

    """
    This function is designed to get just the data from one batch
    """

    rew_batch = np.load( load_path + par_type + batch_name + 'reward_curve.npy')
    per_batch = np.load( load_path + par_type + batch_name + 'p_reached_per_episode.npy')

    return rew_batch, per_batch


def save_reward_performance(reward_curve, performance_curve, data_path, par_type):

    fig, (ax0,ax1) = plt.subplots(2,1)

    ax0.plot(reward_curve)
    ax0.set_ylabel("Recompensas por episodio")
    ax0.set_xlabel("Episodios")
    ax0.set_title("Curva de entrenamiento")

    ax1.plot(performance_curve, 'g')
    ax1.set_ylabel("Partículas")
    ax1.set_xlabel("Episodios")
    ax1.set_title("Rendimiento del agente")

    fig.tight_layout()

    plt.savefig(data_path + par_type + '_' + "performance_reward_curve.png")
    plt.show()
    plt.close()

dir_name = os.path.dirname(os.path.realpath(__file__)).split("/")
dir_name = dir_name[:-1]
main_path = '/'.join(dir_name) + '/'

algo = "dyna_q/"
par_type = "big_solids"
#main_path = "/home/repairs/myRepos/tfm_rl_uc3m/"
data_path =  main_path + "result_data/" + algo 

if algo == "dyna_q/":
    reward_curve, performance_curve = get_batch_data(data_path, par_type, '_only_batch/')
else:
    reward_curve, performance_curve = join_batchs_data(data_path,par_type)

save_reward_performance(reward_curve,performance_curve,data_path,par_type)

