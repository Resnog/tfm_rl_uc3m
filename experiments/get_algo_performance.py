import numpy as np
import matplotlib.pyplot as plt
import os

def show_batchs_performance(data_path, par_type):

    fb_per = np.load( data_path + par_type + '_first_batch/p_reached_per_episode.npy')
    sb_per = np.load( data_path + par_type + '_second_batch/p_reached_per_episode.npy')
    tb_per = np.load( data_path + par_type + '_third_batch/p_reached_per_episode.npy')

    print("First batch: {}".format(np.average(fb_per)))
    print("Second batch: {}".format(np.average(sb_per)))
    print("Third batch: {}".format(np.average(tb_per)))

def show_batch_performance_dyna(data_path, par_type):

    per = np.load( data_path + par_type + '_only_batch/p_reached_per_episode.npy')
   
    print("Only batch: {}".format(np.average(per)))
    
dir_name = os.path.dirname(os.path.realpath(__file__)).split("/")
dir_name = dir_name[:-1]

algo = "dyna_q/"
par_type = "big_solids"
main_path = '/'.join(dir_name) + '/'
data_path =  main_path + "result_data/" + algo 

print("")
print("Algorithm: " + algo)
print("Particle type: " + par_type)
print("")
if algo == "dyna_q/":
    show_batch_performance_dyna(data_path,par_type)
else:
    show_batchs_performance(data_path,par_type)

print("")
