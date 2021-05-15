#© 2021. This work is licensed under a CC-BY-NC-SA license.
#"Thalamo-cortical spiking model of incremental learning combining perception, context and NREM-sleep-mediated noise-resilience"
#Authors: Bruno Golosio, Chiara De Luca, Cristiano Capone, Elena Pastorelli, Giovanni Stegel, Gianmarco Tiddia, Giulia De Bonis and Pier Stanislao Paolucci
#arxiv.2003.11859

import os
import sys
import time
import matplotlib.pyplot as plt
import nest
import numpy as np
import scipy.io as sio

#from sklearn.metrics import confusion_matrix

print 'start'

# Set save path
home = os.getcwd()
save_path = home +  '/NoiseSP_0502/ThOFF/set' + str(_isubset_) + '/'
#save_path = home +  '/PlotUTOT/Raster2Es/'

interval = _interval_

# checking path
if not os.path.exists(save_path):
    os.makedirs(save_path)

print _interval_
syn_T = 15.

# numero di classi
N_CLASSES = 10
# numero di esempi per classe
N_RANKS = 40
# N_RANKS_2 = 1
# numero di esempi mostratidurante il test
N_TEST = 500#250  # 250
# numero di colonne
NUM_COL = 1  # 9
INPUT_SIZE = 324

weight_dynamic = 'NO'#'YES'

Bernulli_InpToExc = 1.
Bernulli_ExcToInh = .75
Bernulli_InhToExc = .75
# numero di processi da avviare
n_threads = _nthreads_
startTime = time.time()

# loading delle immagini preprocessate
coding_number = 6
print 'coding number', coding_number

fn_train = 'Mnist_preprocessing_noise/S_P/Mean0.5_var0.2/mnist_training_features_3x3_Coding6_0.9_14.npy'
feat_arr_train0 = np.load(fn_train)
fn_test = 'Mnist_preprocessing_noise/S_P/Mean0.5_var0.2/mnist_test_features_3x3_Coding6_0.9_14.npy'
feat_arr_test = np.load(fn_test)


INPUT_SIZE = 3*3*9*coding_number

label_fn_train =  'Mnist_preprocessing_noise/S_P/Mean0.5_var0.2/mnist_training_labels_0.9_14.npy'
labels_train0 = np.load(label_fn_train)

label_fn_test = 'Mnist_preprocessing_noise/S_P/Mean0.5_var0.2/mnist_test_labels_0.9_14.npy'
labels_test = np.load(label_fn_test)
labels_test = labels_test.astype(int)

# creo feat_train_class: e' una lista di N_Classes elementi: ognuno contiene gli
# array di tutte le immagini in training appartenenti a quella classe.
feat_train_class = [[] for i in range(N_CLASSES)]  # empty 2d list (N_CLASSES x N)
label_train_class = [[] for i in range(N_CLASSES)]

for i in range(len(labels_train0)):
    i_class = labels_train0[i]
    feat_train_class[i_class].append(feat_arr_train0[i])
    label_train_class[i_class].append(i_class)

seed_rand =  _isubset_ # 28 7 53 88 100 69 8 18 38 48 90 108 44 4 65 55 42 23 95 102 172 238 41 10 13
            # Buoni: 48, 28, 100, 69, 108, 65,    23, 102, 172, 238, 41, 13,    59, 66, 99, 59, 55, 1, 80, 87
            # 2 3 4 5 6 7 8 9

#mi confronto con 69.0
print seed_rand
subset = seed_rand

feat_arr_train = []
labels_train = []

for num in range(0, N_RANKS, _interval_):
    # seleziono per ogni classe il numero di esempi da mostrare nel training,
    # salvo in feat_red le immagini, in labels_red i label corrispondente
    # feat_red e' lungo N_classes*N_ranks ed e' ordinato

    feat_red = [feat_train_class[i][j] for i in range(N_CLASSES)
                for j in range(subset * 1 + num, (subset + 1) * 1 + num)]
    labels_red = [label_train_class[i][j] for i in range(N_CLASSES)
                  for j in range(subset * 1 + num, (subset + 1 + num) * 1)]

    print 'inizio', subset*1 + num
    print 'fine', subset*1+num
    print 'labels red', labels_red

    sio.savemat('labels_red.mat', {'labels_red': labels_red})
    sio.savemat('feat_red.mat', {'feat_red': feat_red})

    rand = np.random.RandomState(num)

    # Shuffle data
    shuffle = rand.permutation(len(labels_red))

    # ora label_red ha i dati mischiati
    labels_red = [labels_red[i] for i in shuffle]

    feat_red = [feat_red[i] for i in shuffle]

    feat_arr_train.extend(np.asarray(feat_red))
    labels_train.extend(np.asarray(labels_red))


# labels_train = np.array(labels_train, dtype=np.float32)

labels_train = np.asarray(labels_train)
labels_train = labels_train.astype(int)

msd = 1234506 + subset
nest.SetKernelStatus({"local_num_threads": n_threads})
N_vp = nest.GetKernelStatus(['total_num_virtual_procs'])[0]
pyrngs = [np.random.RandomState(s) for s in range(msd, msd + N_vp)]
nest.SetKernelStatus({'grng_seed': msd + N_vp})
nest.SetKernelStatus({'rng_seeds': range(msd + N_vp + 1, msd + 2 * N_vp + 1)})

# nx=10
# ny=16
# input_size = nx*ny
input_size = INPUT_SIZE  # 324

train_digit = feat_arr_train  # array lungo 30 (numero di immagini) ciascun elemento lungo 324

n_classes = N_CLASSES
teach_out = labels_train

n_train = len(teach_out)

# qui mi creo gli analoghi per il test
test_digit = feat_arr_test[0 * N_TEST:(0 + 1) * N_TEST]
test_out = labels_test[0 * N_TEST:(0 + 1) * N_TEST]


n_test = len(test_out)
# n_test = 1

n_columns = NUM_COL


n_exc_example = 20 #numero neuroni per esempio #20, 15, 10, 5
k_exc = n_exc_example*_interval_ #N_RANKS
#n_exc = k_exc * n_train * n_columns  # numero neuroni eccitatori 1*numero elementi training (30)*num colonne 2
n_exc = k_exc * N_CLASSES
n_inh = n_exc_example*N_CLASSES*N_RANKS / 4  # numero neuroni inibitori

n_inh2 = 200


############################################## parameters ############################################

r_noise_inp = _rInp_
r_noise_exc = _rExc_
r_noise_inh = r_noise_exc
r_noise_out = r_noise_exc

r_noise_inp_recog = r_noise_inp

W_noise_inp = _W_noise_inp_#1600.
W_noise_exc = _W_noise_exc_ * 10.#310.#2250.
W_noise_out = 1100.
W_noise_inh = 320. #/2.5 /2.

Wmax_exc_exc = _WmaxEE_
Wmax_inp_exc = 10.5
Wmax_exc_inp = 0.#5.#0.333
Wmax_exc_out = 67500.
Wmax_out_out = 20.
Wmax_out_exc = 0.#5.

W0_exc_inp = 0.
W0_inp_exc = _W0_inp_exc_#0.3
W0_exc_out = 0.1
W0_exc_exc = _W0_exc_exc_ #0.
W0_out_out = 0.01
W0_out_exc = 0.

W_inh_inh = -1.

mu_minus = 1.
mu_plus = 1.
alpha = 1.

W_exc_inh = 200.0
W_inh_exc_retrieval = _Winhexcretrieval_#-64.
W_inh_exc = -64.

lambda_inp_exc = 0.03
lambda_exc_inp = 0.003
lambda_exc_out = .03
lambda_exc_exc = 0.1#0.05
lambda_out_out = 0.1
lambda_out_exc = 0.003

t_train = 1500.#500.
t_pause = 1500.0
t_check = 1000.0
t_sleep = 1500.

W_inp_inh2 = 3.0
W_inh2_inp = -1.0


######################################################################
n_inp = input_size  # 324, lunghezza dell'array che discrive una immagine

n_out = n_classes * n_exc_example

print 'initialization...'
# build input patterns
# ilpattern di train e' una lista bidimensionale lunga numero di casi per il training
# e dimensione dell'input

train_pattern = [[0 for i in range(n_inp)] for j in range(n_train)]

for i in range(input_size):
    for i_train in range(n_train):
        train_pattern[i_train][i] = train_digit[i_train][i]

test_pattern = [[0 for i in range(n_inp)] for j in range(n_test)]
for i in range(input_size):
    for i_test in range(n_test):
        test_pattern[i_test][i] = test_digit[i_test][i]

n_spikes_inp = [0] * n_inp
cum_spikes_inp = [0] * n_inp
n_spikes_exc = [0] * n_exc * int(N_RANKS/interval)
cum_spikes_exc = [0] * n_exc *int(N_RANKS/interval)
n_spikes_inh = [0] * n_inh
cum_spikes_inh = [0] * n_inh
n_spikes_out = [0] * n_out
cum_spikes_out = [0] * n_out
n_spikes_inh2 = [0] * n_inh2
cum_spikes_inh2 = [0] * n_inh2
times = [0] * n_out

neur_inp = nest.Create("aeif_cond_alpha", n_inp)  # 324 input neurons
neur_exc = nest.Create("aeif_cond_alpha", n_exc)  # excitatory neurons
neur_inh = nest.Create("aeif_cond_alpha", n_inh)  # inhibitory neurons
neur_out = nest.Create("aeif_cond_alpha", n_out)  # output neurons (classes)
neur_inh2 = nest.Create("aeif_cond_alpha", n_inh2)  # inhibitory neurons 2

nest.SetStatus(neur_inp, {"b": .01})
nest.SetStatus(neur_exc, {"b": .01})
nest.SetStatus(neur_inh, {"b": .01})
nest.SetStatus(neur_out, {"b": .01})
nest.SetStatus(neur_inh2, {"b": .01})

nest.SetStatus(neur_inp, {"t_ref": 2.0})
nest.SetStatus(neur_exc, {"t_ref": 2.0})
nest.SetStatus(neur_inh, {"t_ref": 2.0})
nest.SetStatus(neur_out, {"t_ref": 2.0})
nest.SetStatus(neur_inh2, {"t_ref": 2.0})

# random V to exc neurons

print 'noise connection...'
Vrest = -71.2
Vth = -70.

Vms = Vrest + (Vth - Vrest) * np.random.rand(len(neur_exc))
# nest.SetStatus(neur_exc, "V_m", Vms)

# input noise
noise_inp = nest.Create("poisson_generator", n_inp)
syn_dict_noise_inp = {"weight": W_noise_inp, "delay": 1.0}
nest.Connect(noise_inp, neur_inp, "one_to_one", syn_dict_noise_inp)

# training noise
noise_exc = nest.Create("poisson_generator", n_exc)
syn_dict_noise_exc = {"weight": W_noise_exc, "delay": 3.0}
conn_dict_noise_exc = {'rule': 'one_to_one'}
nest.Connect(noise_exc, neur_exc, conn_dict_noise_exc, syn_dict_noise_exc)

# build train exc neuron groups (numero di gruppi pari al numero di colonne)
# e' una lista lunga il numero di esempi da mostrare. Ogni elemento ha due elementi


train_target_exc = [[0 for i in range(n_exc_example)] for j in range(n_train)]
tgt=range(n_exc)

for i_train in range(n_train):
    for i_k in range(n_exc_example):
        i = i_train*n_exc_example + i_k
        train_target_exc[i_train][i_k]=i


# teaching output noise
noise_out = nest.Create("poisson_generator", n_out)
syn_dict_noise_out = {"weight": W_noise_out, "delay": 1.0}
nest.Connect(noise_out, neur_out, "one_to_one", syn_dict_noise_out)

# inhibitory noise
noise_inh = nest.Create("poisson_generator", n_inh)
syn_dict_noise_inh = {"weight": W_noise_inh, "delay": 1.0}
nest.Connect(noise_inh, neur_inh, "one_to_one", syn_dict_noise_inh)


# spike detectors for input layer
sd_inp = nest.Create("spike_detector", n_inp)
nest.SetStatus(sd_inp, {"withgid": True, "withtime": True})
nest.Connect(neur_inp, sd_inp, "one_to_one")


# spike detectors for excitatory neurons
sd_exc = nest.Create("spike_detector", n_exc_example*N_CLASSES*(N_RANKS+1))
nest.SetStatus(sd_exc, {"withgid": True, "withtime": True})
nest.Connect(neur_exc, sd_exc[0:_interval_*n_exc_example*N_CLASSES], "one_to_one")


# spike detectors for inhibitory neurons
sd_inh = nest.Create("spike_detector", n_inh)
nest.SetStatus(sd_inh, {"withgid": True, "withtime": True})
nest.Connect(neur_inh, sd_inh, "one_to_one")

# spike detectors for output layer
sd_out = nest.Create("spike_detector", n_out)
nest.SetStatus(sd_out, {"withgid": True, "withtime": True})
nest.Connect(neur_out, sd_out, "one_to_one")


# spike detectors for inhibitory neurons 2
sd_inh2 = nest.Create("spike_detector", n_inh2)
nest.SetStatus(sd_inh2, {"withgid": True, "withtime": True})
nest.Connect(neur_inh2, sd_inh2, "one_to_one")



#####################
print 'input to excitatory connections'
#####################

# from input to excitatory
syn_dict_inp_exc = {"model": "stdp_synapse", "lambda": lambda_inp_exc, "weight": W0_inp_exc, "Wmax": Wmax_inp_exc,
                    "delay": 1.0}
conn_dict = {'rule': 'pairwise_bernoulli', 'p': Bernulli_InpToExc}
# nel caso in cui io voglia 5/9 di input per parte
nest.Connect(neur_inp,
             neur_exc, conn_dict, syn_dict_inp_exc)

# connetti neur_inp[da 324(dimensione input)/2(num colonne)*k(colonna che considero) a 324/2*k+1(colonna che
# considero)] a neuroni eccitatori (stessi termini)


print 'from excitatory to input connections'
syn_dict_exc_inp = {"model": "stdp_synapse", "lambda": lambda_exc_inp, "weight": W0_exc_inp, "Wmax": Wmax_exc_inp,
                    "delay": 1.0}
# Nel caso in cui io voglia 5/9 di input per parte
nest.Connect(neur_exc,
             neur_inp,
             "all_to_all", syn_dict_exc_inp)



print ' excitatory to output connections'

syn_dict_exc_out = {"model": "stdp_synapse", "lambda": lambda_exc_out, "weight": W0_exc_out, "Wmax": Wmax_exc_out,
                    "delay": 1.0}
nest.Connect(neur_exc, neur_out, "all_to_all", syn_dict_exc_out)


print 'excitatory to excitatory connections intra'
w_min = 0.
w_max = W0_exc_exc + 0.000000000000001
# excitatory to excitatory connections
syn_dict_exc_exc = {"model": "stdp_synapse", "lambda": lambda_exc_exc, "weight": W0_exc_exc,
                          "Wmax": Wmax_exc_exc,
                          "delay":
                              {'distribution': 'exponential_clipped',  # esponenziale troncato
                               'lambda': 10.,
                               'low': 1.,
                               'high': 10.},

                          "weight": {"distribution": "uniform", "low": w_min, "high": w_max}
                          # "weight": {"distribution": "exponential", "lambda": .000000000001 }
                          }


nest.Connect(neur_exc, neur_exc,
             "all_to_all", syn_dict_exc_exc)

print ' output to output connections'

syn_dict_out_out = {"model": "stdp_synapse", "lambda": lambda_out_out, "weight": W0_out_out, "Wmax": Wmax_out_out,
                    "delay": 1.0}
nest.Connect(neur_out, neur_out, "all_to_all", syn_dict_out_out)

print ' output to excitatory connections'

syn_dict_out_exc = {"model": "stdp_synapse", "lambda": lambda_out_exc, "weight": W0_out_exc, "Wmax": Wmax_out_exc,
                    "delay": 1.0}
nest.Connect(neur_out, neur_exc, "all_to_all", syn_dict_out_exc)



print 'excitatory to inhibitory connections'

syn_dict_exc_inh = {"weight": W_exc_inh, "delay": 1.0}
# conn_dict = {'rule': 'pairwise_bernoulli', 'p': Bernulli_ExcToInh}
conn_dict = {'rule': 'all_to_all'}
nest.Connect(neur_exc[0:n_exc],
             neur_inh[0:n_inh], conn_dict, syn_dict_exc_inh)

# inhibitory to excitatory connections

syn_dict_inh_exc = {"weight": W_inh_exc, "delay": 1.0}
# conn_dict = {'rule': 'pairwise_bernoulli', 'p': Bernulli_InhToExc}

conn_dict = {'rule': 'all_to_all'}
nest.Connect(neur_inh[0:n_inh],
             neur_exc[0:n_exc], conn_dict, syn_dict_inh_exc)

# inh to inh
syn_dict_inh_inh = {"weight": W_inh_inh, "delay": 1.0}
nest.Connect(neur_inh[0:n_inh],
             neur_inh[0:n_inh], "all_to_all", syn_dict_inh_exc)


print 'input to inhibitory connections'
syn_dict_inp_inh2 = {"weight": W_inp_inh2, "delay": 1.0}
nest.Connect(neur_inp, neur_inh2, "all_to_all", syn_dict_inp_inh2)

# inhibitory to input connections
syn_dict_inh2_inp = {"weight": W_inh2_inp, "delay": 1.0}
nest.Connect(neur_inh2, neur_inp, "all_to_all", syn_dict_inh2_inp)

#VERIFICA CONTROLLA FUNZIONI FINO A QUI


#multimeter = nest.Create('multimeter', params={'record_from': ['V_m', 'g_ex', 'g_in', 'w']})

#nest.Connect(multimeter, [neur_exc[0]])
#nest.Connect(multimeter, neur_exc[0:20])
################################################
# training

nest.SetStatus(nest.GetConnections(source=neur_inp, target=neur_exc), {'mu_minus': mu_minus})
nest.SetStatus(nest.GetConnections(source=neur_exc, target=neur_out), {'mu_minus': mu_minus})
nest.SetStatus(nest.GetConnections(source=neur_exc, target=neur_exc), {'mu_minus': mu_minus})
nest.SetStatus(nest.GetConnections(source=neur_out, target=neur_out), {'mu_minus': mu_minus})
nest.SetStatus(nest.GetConnections(source=neur_out, target=neur_exc), {'mu_minus': mu_minus})

nest.SetStatus(nest.GetConnections(source=neur_inp, target=neur_exc), {'mu_minus': mu_plus})
nest.SetStatus(nest.GetConnections(source=neur_exc, target=neur_out), {'mu_minus': mu_plus})
nest.SetStatus(nest.GetConnections(source=neur_exc, target=neur_exc), {'mu_minus': mu_plus})
nest.SetStatus(nest.GetConnections(source=neur_out, target=neur_exc), {'mu_minus': mu_plus})
nest.SetStatus(nest.GetConnections(source=neur_out, target=neur_out), {'mu_minus': mu_plus})


nest.SetStatus(nest.GetConnections(source=neur_inp, target=neur_exc), {'alpha': alpha})
nest.SetStatus(nest.GetConnections(source=neur_exc, target=neur_out), {'alpha': alpha})
nest.SetStatus(nest.GetConnections(source=neur_exc, target=neur_exc), {'alpha': alpha})
nest.SetStatus(nest.GetConnections(source=neur_out, target=neur_exc), {'alpha': alpha})
nest.SetStatus(nest.GetConnections(source=neur_out, target=neur_out), {'alpha': alpha})


#####################################################################################################
#                                       STRUMENTI DI MISURA                                         #
#####################################################################################################

multimeter_100 = nest.Create('multimeter', params={'record_from': ['V_m', 'g_ex', 'g_in', 'w']})
nest.Connect(multimeter_100, [neur_exc[30]])

multimeter_300 = nest.Create('multimeter', params={'record_from': ['V_m', 'g_ex', 'g_in', 'w']})
nest.Connect(multimeter_300, [neur_exc[30]])

w_inp_exc = []
w_exc_exc = []

sd_exc_cnt_th = nest.Create("spike_detector", n_exc)
nest.SetStatus(sd_exc_cnt_th, {"withgid": True, "withtime": True})
nest.Connect(neur_exc, sd_exc_cnt_th, "one_to_one")
sd_th_cnt_th = nest.Create("spike_detector", n_inp)
nest.SetStatus(sd_th_cnt_th, {"withgid": True, "withtime": True})
nest.Connect(neur_inp, sd_th_cnt_th, "one_to_one")


########################################################################################################################
#                                           TRAINING PHASE                                                             #
########################################################################################################################



output = []



AccuracyArr = []
AccuracyArrUnGr = []
AccuracyArrUnCl = []

NumTrain = []

i_out_train = []

count = np.zeros((N_CLASSES))

step = 0
i_cumul = 0

dist_train = np.zeros((10, 10));

for t in range(0, 10):
    for t1 in range(0,10):

        temp = [(test_pattern[t][i] - test_pattern[t1][i]) ** 2 for i in range(0, input_size)]
        print sum(temp)
        dist_train[t, t1] = sum(temp);

name = save_path + 'Train' + str(step) + '.mat'
sio.savemat(name, {'train_distance': dist_train})

train_vector = teach_out[0:10];

for i_cumul in range(0,N_RANKS*N_CLASSES, _interval_*N_CLASSES):

    events_readout = []
    events_exc = []


    for i_train in range(i_cumul, i_cumul + _interval_ * N_CLASSES):
        start_train = time.time()
        i_out = teach_out[i_train]

        print 'i_train', i_train
        print 'train label', i_out

        print 'train target:', train_target_exc[i_train]

        Nup = 0
        for i in range(n_inp):
            if train_pattern[i_train][i] == 1:
                nest.SetStatus([noise_inp[i]], {"rate": r_noise_inp})
                Nup = Nup + 1
            else:
                nest.SetStatus([noise_inp[i]], {"rate": 0.0})
        print "Nup ", Nup

        for i in range(n_exc * (step + 1)):
            nest.SetStatus([noise_exc[i]], {"rate": 0.0})

        for i_k in range(n_exc_example):
            nest.SetStatus([noise_exc[train_target_exc[i_train][i_k]]],
                           {"rate": r_noise_exc})


        nest.SetStatus(nest.GetConnections(source=neur_exc, target=neur_out), {
            'lambda': -lambda_exc_out * .5})  # connetti NEGATIVAMENTE i neuroni eccitatori ai neuroni di read out

        for i in range(n_classes):  # per ciascun numero di classi
            if i == i_out:  # se e' la classe che sto guardando
                nest.SetStatus(noise_out[i*n_exc_example:(i+1)*n_exc_example], {"rate": r_noise_out})
                nest.SetStatus(nest.GetConnections(source=neur_exc, target=neur_out[i*n_exc_example:(i+1)*n_exc_example]), {
                    'lambda': lambda_exc_out})  # connetti i neuroni ecc giusti con quelli di read out corrispondenti

            else:

                nest.SetStatus(noise_out[i*n_exc_example:(i+1)*n_exc_example], {"rate": 0.0})  # altrimenti azzera il rumore

        for i in range(n_inh):
            nest.SetStatus([noise_inh[i]], {"rate": r_noise_inh})  # accendi il rumore su tutti ineuroni inibitori

        # unfreeze weights for training
        nest.SetStatus(nest.GetConnections(source=neur_inp, target=neur_exc),
                       {'lambda': lambda_inp_exc})
        nest.SetStatus(nest.GetConnections(source=neur_exc, target=neur_inp),
                       {'lambda': lambda_exc_inp})

        nest.SetStatus(nest.GetConnections(source=neur_exc, target=neur_out),
                       {'lambda': lambda_exc_out})
        nest.SetStatus(nest.GetConnections(source=neur_exc, target=neur_exc),
                       {'lambda': lambda_exc_exc})
        nest.SetStatus(nest.GetConnections(source=neur_out, target=neur_exc),
                       {'lambda': lambda_out_exc})
        nest.SetStatus(nest.GetConnections(source=neur_out, target=neur_out),
                       {'lambda': lambda_out_out})
        # simulation

        nest.Simulate(t_train)

        # freeze weights for running tests
        nest.SetStatus(nest.GetConnections(source=neur_inp, target=neur_exc),
                       {'lambda': 0.0})
        nest.SetStatus(nest.GetConnections(source=neur_exc, target=neur_inp),
                       {'lambda': 0.0})

        nest.SetStatus(nest.GetConnections(source=neur_exc, target=neur_out),
                       {'lambda': 0.0})
        nest.SetStatus(nest.GetConnections(source=neur_exc, target=neur_exc),
                       {'lambda': 0.0})
        nest.SetStatus(nest.GetConnections(source=neur_out, target=neur_exc),
                       {'lambda': 0.0})
        nest.SetStatus(nest.GetConnections(source=neur_out, target=neur_out),
                       {'lambda': 0.0})

        # roba per leggere lo status nel training
        evt_sd_inp = nest.GetStatus(sd_inp, keys="events")
        for i in range(n_inp):
            send_sd_inp = evt_sd_inp[i]["senders"]
            t_sd_inp = evt_sd_inp[i]["times"]
            n_spikes_inp[i] = len(t_sd_inp) - cum_spikes_inp[i]
            cum_spikes_inp[i] = len(t_sd_inp)

        t_out_list = []
        evt_sd_out = nest.GetStatus(sd_out, keys="events")
        for i in range(n_out):
            send_sd_out = evt_sd_out[i]["senders"]
            t_sd_out = evt_sd_out[i]["times"]
            t_out_list.append(t_sd_out)
            n_spikes_out[i] = len(t_sd_out) - cum_spikes_out[i]
            cum_spikes_out[i] = len(t_sd_out)


        evt_sd_exc = nest.GetStatus(sd_exc, keys="events")
        for i in range(n_exc):
            send_sd_exc = evt_sd_exc[i]["senders"]
            t_sd_exc = evt_sd_exc[i]["times"]
            n_spikes_exc[i] = len(t_sd_exc) - cum_spikes_exc[i]
            cum_spikes_exc[i] = len(t_sd_exc)



        evt_sd_inh = nest.GetStatus(sd_inh, keys="events")

        for i in range(n_inh):
            send_sd_inh = evt_sd_inh[i]["senders"]
            t_sd_inh = evt_sd_inh[i]["times"]
            n_spikes_inh[i] = len(t_sd_inh) - cum_spikes_inh[i]
            cum_spikes_inh[i] = len(t_sd_inh)



        evt_sd_inh2 = nest.GetStatus(sd_inh2, keys="events")

        for i in range(n_inh2):
            send_sd_inh2 = evt_sd_inh2[i]["senders"]
            t_sd_inh2 = evt_sd_inh2[i]["times"]
            n_spikes_inh2[i] = len(t_sd_inh2) - cum_spikes_inh2[i]
            cum_spikes_inh2[i] = len(t_sd_inh2)


        n_spikes_max = 0
        i_out_num = -1

        for i in range(n_classes):
            n_spikes = 0

            for neur in range(n_exc_example):
                n_spikes = n_spikes + n_spikes_out[i*n_exc_example + neur]

            if n_spikes > n_spikes_max:
                i_out_num = i
                n_spikes_max = n_spikes

        i_out_train.append(i_out_num)

        # switch off all teaching outputs
        for i in range(n_exc * (step + 1)):
            nest.SetStatus([noise_exc[i]], {"rate": 0.0})

        for i in range(n_out):
            nest.SetStatus([noise_out[i]], {"rate": 0.0})

        for i in range(n_inh):
            nest.SetStatus([noise_inh[i]], {"rate": r_noise_inh*0.})

        for i in range(n_inp):
            nest.SetStatus([noise_inp[i]], {"rate": 0.0})

        nest.SetStatus(neur_exc, "V_m", Vrest)

        # simulation
        nest.Simulate(t_pause)

        count[i_out] += 1

    print 'teach_out', teach_out[i_cumul: i_cumul + _interval_ * N_CLASSES]
    print 'inferred', i_out_train[i_cumul:i_cumul + _interval_ * N_CLASSES]

    if i_cumul == 110:

        ######################################################################
        # Test

        nest.SetStatus(nest.GetConnections(source=neur_exc, target=neur_out),
                       {'lambda': 0.})  # disconnetti i neuroni eccitatori con quelli di oputut

        nest.SetStatus(nest.GetConnections(source=neur_inh, target=neur_exc),
                       {"weight": W_inh_exc_retrieval})  # e' un peso parecchio negativo! #-0.09???

        # switch off all teaching outputs
        for i in range(n_exc*(step+1)):
            nest.SetStatus([noise_exc[i]], {"rate": 0.0})

        for i in range(n_out):
            nest.SetStatus([noise_out[i]], {"rate": 0.0})

        for i in range(n_inh):
            nest.SetStatus([noise_inh[i]], {"rate": r_noise_inh * 0.})

        for i in range(n_inp):
            nest.SetStatus([noise_inp[i]], {"rate": 0.0})

        nest.Simulate(t_pause * 20)  # 5

        # Cerco il tempo da cui inizia la simulazione
        end_time = 0
        evt_sd_out = nest.GetStatus(sd_out, keys="events")
        for i in range(n_out):
            t_sd_out = evt_sd_out[i]["times"]
            try:
                if max(t_sd_out) > end_time:
                    end_time = max(t_sd_out)
            except ValueError:
                pass

        print end_time


        print '######################################################################'
        print '# Test pre-sleep con context'
        print '######################################################################'
        print

        # test_seq = [2,0,1,3]

        dist = np.zeros((n_test, N_CLASSES))
        dotdist_W = np.zeros((n_test, N_CLASSES))
        dotdist_K = np.zeros((n_test, N_CLASSES))

        count_right_num = 0
        count_right_time = 0
        count_unsupervised_class_right = 0
        count_unsupervised_group_right = 0
        count_KNN_right = 0

        test_pred_num = np.zeros((n_test))
        # test_pred_time = np.zeros((n_test))
        times = np.zeros((n_out))

        for i_test in range(n_test):  # per ogni esempio del test set

            # switch off all teaching outputs

            #Calcolo la distanza
            for train in range(N_CLASSES):
                print train
                temp = [(test_pattern[i_test][i] - train_pattern[train][i])**2 for i in range(0,input_size)]
                dist[i_test, train] = sum(temp);


            for i in range(n_exc*(step+1)):
                nest.SetStatus([noise_exc[i]], {"rate": 0.0})

            for i in range(n_out):
                nest.SetStatus([noise_out[i]], {"rate": 0.0})

            for i in range(n_inh):
                nest.SetStatus([noise_inh[i]], {"rate": 0.0})

            # prepare input pattern
            Nup = 0
            for i in range(n_inp):  # per i tra 0 e 365 lunghezza dell'array per esempio
                if test_pattern[i_test][i] == 1:  # se il test pattern per tale esempio in corrispondenza del bit i e' 1
                    nest.SetStatus([noise_inp[i]], {"rate": r_noise_inp_recog})  # attiva il rumore
                    Nup = Nup + 1
                else:
                    nest.SetStatus([noise_inp[i]], {"rate": 0.0})  # altrimenti azzeralo
            print "Nup ", Nup  # conta il numero di canali attivi
            # per farla breve, per ogni immagine attiva solo i neuroni in input che corrispondono ad un bit 1

            # simulation
            # t_check=2000.0
            nest.Simulate(t_check)



            # acquisisci lo stato
            evt_sd_inp = nest.GetStatus(sd_inp, keys="events")
            for i in range(n_inp):
                send_sd_inp = evt_sd_inp[i]["senders"]
                t_sd_inp = evt_sd_inp[i]["times"]
                n_spikes_inp[i] = len(t_sd_inp) - cum_spikes_inp[i]
                cum_spikes_inp[i] = len(t_sd_inp)


            # t_out_list = [] #serve per il coalcolo della risposta secondo cui 'il primo che arriva vince'

            evt_sd_out = nest.GetStatus(sd_out, keys="events")
            for i in range(n_out):
                send_sd_out = evt_sd_out[i]["senders"]
                t_sd_out = evt_sd_out[i]["times"]
                # print t_sd_out
                # t_out_list.append(t_sd_out) #anche questo...
                n_spikes_out[i] = len(t_sd_out) - cum_spikes_out[i]
                cum_spikes_out[i] = len(t_sd_out)

            evt_sd_exc = nest.GetStatus(sd_exc, keys="events")
            for i in range(n_exc*(step+1)):
                send_sd_exc = evt_sd_exc[i]["senders"]
                t_sd_exc = evt_sd_exc[i]["times"]
                n_spikes_exc[i] = len(t_sd_exc) - cum_spikes_exc[i]
                cum_spikes_exc[i] = len(t_sd_exc)

            n_spikes_un = np.zeros(N_CLASSES)
            n_spikes_group = np.zeros(int(n_exc*(step+1)/n_exc_example))

            for group in range(0, n_exc*(step+1)/n_exc_example):
                # Per ogni gruppo da 20 neuroni
                # calcolo il numero di spike del gruppo

                cl = labels_train[group]
                for neur in range(group*n_exc_example, (group+1)*n_exc_example):
                    n_spikes_un[cl] += n_spikes_exc[neur]
                    n_spikes_group[group] += n_spikes_exc[neur]


            evt_sd_inh = nest.GetStatus(sd_inh, keys="events")
            for i in range(n_inh):
                send_sd_inh = evt_sd_inh[i]["senders"]
                t_sd_inh = evt_sd_inh[i]["times"]
                n_spikes_inh[i] = len(t_sd_inh) - cum_spikes_inh[i]
                cum_spikes_inh[i] = len(t_sd_inh)

            evt_sd_inh2 = nest.GetStatus(sd_inh2, keys="events")
            for i in range(n_inh2):
                send_sd_inh2 = evt_sd_inh2[i]["senders"]
                t_sd_inh2 = evt_sd_inh2[i]["times"]
                n_spikes_inh2[i] = len(t_sd_inh2) - cum_spikes_inh2[i]
                cum_spikes_inh2[i] = len(t_sd_inh2)


            # switch off all teaching outputs
            for i in range(n_exc*(step+1)):
                nest.SetStatus([noise_exc[i]], {"rate": 0.0})

            for i in range(n_out):
                nest.SetStatus([noise_out[i]], {"rate": 0.0})

            for i in range(n_inh):
                nest.SetStatus([noise_inh[i]], {"rate": r_noise_inh * 0.})

            for i in range(n_inp):
                nest.SetStatus([noise_inp[i]], {"rate": 0.0})

            nest.SetStatus(neur_exc, "V_m", Vrest)
            nest.Simulate(t_pause)

            # qui valuto la predizione in base alla posizione...

            n_spikes_max = 0
            i_out_num = -1

            for i in range(n_classes):
                n_spikes = 0

                for neur in range(n_exc_example):
                    n_spikes = n_spikes + n_spikes_out[i * n_exc_example + neur]

                if n_spikes > n_spikes_max:
                    i_out_num = i
                    n_spikes_max = n_spikes

                #events_readout.append(n_spikes/n_exc_example/t_check)

            test_pred_num[i_test] = i_out_num

            # ...fino a qui

            print 'Output class index num: ', i_out_num
            # print 'Output class index time: ', i_out_time[i_test]
            print 'Target class index: ', test_out[i_test]

            # if i_out_time[i_test] == test_out[i_test]:
            #    count_right_time = count_right_time + 1
            if i_out_num == test_out[i_test]:
                count_right_num = count_right_num + 1
            # print 'Corrects time: ', count_right_time, '/', i_test + 1
            print 'Corrects num: ', count_right_num, '/', i_test + 1
            # print 'Accuracy time = ', float(count_right_time) / float(i_test + 1.) * 100., '%'
            print 'Accuracy num = ', float(count_right_num) / float(i_test + 1.) * 100., '%'

            # questa roba non dovrebbe servire per ora...

            # qui valuto la posizione in base alla classe che spara di piu' ...

            unsupervised_output_class = -1
            maximum = -1;
            print n_spikes_un
            for cl in range(0, N_CLASSES):
                if n_spikes_un[cl] >= maximum:
                    maximum = n_spikes_un[cl]
                    unsupervised_output_class = cl

            if unsupervised_output_class == test_out[i_test]:
                count_unsupervised_class_right = count_unsupervised_class_right + 1

            print 'Corrects unsupervised class: ', count_unsupervised_class_right, '/', i_test+1
            print 'Accuracy unsupervised class = ', float(count_unsupervised_class_right)/float(i_test+1.)*100., '%'

            # ... fino a qui

            # qui valuto la posizione in base al gruppo che spara di piu' ...

            unsupervised_output_group = -1
            maximum = -1;

            print n_spikes_group
            for gr in range(0, int(n_exc*(step+1)/n_exc_example)):
                if n_spikes_group[gr] >= maximum:
                    maximum = n_spikes_group[gr]
                    unsupervised_output_group = labels_train[gr]

            #events_exc.extend(n_spikes_group/t_check/n_exc_example)

            output.append(unsupervised_output_group)
            if unsupervised_output_group == test_out[i_test]:
                count_unsupervised_group_right = count_unsupervised_group_right + 1

            print 'Corrects unsupervised group: ', count_unsupervised_group_right, '/', i_test + 1
            print 'Accuracy unsupervised group = ', float(count_unsupervised_group_right) / float(i_test + 1.) * 100., '%'

            # ... fino a qui



            endTime = time.time()
            print ('Simulation time: %.2f s') % (endTime - startTime)
            sys.stdout.flush()

            #Calcolo la dot distance tra questo esempio e il vincitore
            K = np.where(dist[i_test] == min(dist[i_test])); #elemento i train con minima distanza dallo stimolo
            W = np.where(train_vector[0:i_cumul+10] == unsupervised_output_class) #elemento i train della risposta


            #for train in range(N_CLASSES):
            #    temp_test_train = np.dot(np.asarray(test_pattern[i_test]), np.asarray(train_pattern[train]))
            #    temp_train_W = np.dot(temp_test_train, np.asarray(train_pattern[W[0][0]]))
            #    temp_train_K = np.dot(temp_test_train, np.asarray(train_pattern[K[0][0]]))
            #    dotdist_W[i_test, train] = sum(temp_train_W);
            #    dotdist_K[i_test, train] = sum(temp_train_K);


            if teach_out[K[0][0]] == test_out[i_test]:
                count_KNN_right = count_KNN_right +1

            print 'Output class index KNN: ', teach_out[K[0][0]]
            print 'Corrects KNN: ', count_KNN_right, '/', i_test+1
            print 'Accuracy KNN= ', float(count_KNN_right)/float(i_test+1.)*100., '%'

        # Accuracy_pre_sleep = float(count_right)/float(i_test+1.)*100.
        # Accuracy_pre_sleep_unsupervised = float(count_unsupervised_right)/float(i_test+1.)*100.

        # Alla fine il risultato e'...
        #Accuracy_pre_sleep_Context[index] = float(count_right_num) / float(i_test + 1.) * 100.
        print 'Accuracy pre sleep = ', float(count_right_num) / float(i_test + 1.) * 100., '%'
        print 'Accuracy_pre_sleep_unsupervised_class =', float(count_unsupervised_class_right) / float(i_test + 1.) * 100.
        print 'Accuracy_pre_sleep_unsupervised_group =', float(count_unsupervised_group_right) / float(i_test + 1.) * 100.
        print 'Accuracy KNN = ', float(count_KNN_right)/float(i_test+1.)*100., '%'

        AccuracyArr.append(float(count_right_num) / float(i_test + 1.) * 100.)
        AccuracyArrUnCl.append(float(count_unsupervised_class_right) / float(i_test + 1.) * 100.)
        AccuracyArrUnGr.append(float(count_unsupervised_group_right) / float(i_test + 1.) * 100.)
        NumTrain.append((step+1)*N_CLASSES*_interval_)


    ''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''

    #                                           CREO ALTRI NEURONI e ricomincio                                            #

    ''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''

    # Dopo questo creo altri neuroni eccitatori ed inibitori...

    neur_exc = neur_exc + nest.Create("aeif_cond_alpha", n_exc)  # excitatory neurons
    print len(neur_exc)

    nest.SetStatus(neur_exc[n_exc * (step + 1):n_exc * (step + 2)], {"b": .01})
    nest.SetStatus(neur_exc[n_exc * (step + 1):n_exc * (step + 2)], {"t_ref": 2.0})


    noise_exc = noise_exc + nest.Create("poisson_generator", n_exc)
    syn_dict_noise_exc = {"weight": W_noise_exc, "delay": 3.0}
    conn_dict_noise_exc = {'rule': 'one_to_one'}
    nest.Connect(noise_exc[n_exc * (step + 1):n_exc * (step + 2)], neur_exc[n_exc * (step + 1):n_exc * (step + 2)],
                 conn_dict_noise_exc, syn_dict_noise_exc)



    # ... e li connetto alla rete

    print 'input to excitatory connections'

    conn_dict = {'rule': 'pairwise_bernoulli', 'p': Bernulli_InpToExc}
    nest.Connect(neur_inp,
                 neur_exc[n_exc * (step + 1):n_exc * (step + 2)], conn_dict, syn_dict_inp_exc)

    print 'from excitatory to input connections'

    nest.Connect(neur_exc[n_exc * (step + 1):n_exc * (step + 2)],
                 neur_inp,
                 "all_to_all", syn_dict_exc_inp)

    print ' excitatory to output connections'

    nest.Connect(neur_exc[n_exc * (step + 1):n_exc * (step + 2)], neur_out, "all_to_all", syn_dict_exc_out)


    print '  output  to excitatoryconnections'

    nest.Connect(neur_out, neur_exc[n_exc * (step + 1):n_exc * (step + 2)], "all_to_all", syn_dict_out_exc)


    print 'excitatory to excitatory connections intra'

    nest.Connect(neur_exc[n_exc * (step + 1):n_exc * (step + 2)], neur_exc[n_exc * (step + 1):n_exc * (step + 2)],
                 "all_to_all", syn_dict_exc_exc)

    nest.Connect(neur_exc[n_exc * (step + 1):n_exc * (step + 2)], neur_exc[0:n_exc * (step + 1)], "all_to_all",
                 syn_dict_exc_exc)

    nest.Connect(neur_exc[0:n_exc * (step + 1)], neur_exc[n_exc * (step + 1):n_exc * (step + 2)], "all_to_all",
                 syn_dict_exc_exc)

    print 'excitatory to inhibitory connections'

    conn_dict = {'rule': 'all_to_all'}

    nest.Connect(neur_exc[n_exc * (step + 1):n_exc * (step + 2)],
                 neur_inh, conn_dict, syn_dict_exc_inh)

    # inhibitory to excitatory connections

    nest.Connect(neur_inh, neur_exc[n_exc * (step + 1):n_exc * (step + 2)],
                 conn_dict, syn_dict_inh_exc)


    #connetto il misuratore di eventi

    nest.Connect(neur_exc[n_exc*(step+1):n_exc*(step+2)], sd_exc[n_exc*(step+1) : n_exc*(step+2)], "one_to_one")

    nest.SetStatus(nest.GetConnections(source=neur_inp, target=neur_exc), {'mu_minus': mu_minus})
    nest.SetStatus(nest.GetConnections(source=neur_exc, target=neur_out), {'mu_minus': mu_minus})
    nest.SetStatus(nest.GetConnections(source=neur_exc, target=neur_exc), {'mu_minus': mu_minus})
    nest.SetStatus(nest.GetConnections(source=neur_out, target=neur_exc), {'mu_minus': mu_minus})
    nest.SetStatus(nest.GetConnections(source=neur_out, target=neur_out), {'mu_minus': mu_minus})

    nest.SetStatus(nest.GetConnections(source=neur_inp, target=neur_exc), {'mu_minus': mu_plus})
    nest.SetStatus(nest.GetConnections(source=neur_exc, target=neur_out), {'mu_minus': mu_plus})
    nest.SetStatus(nest.GetConnections(source=neur_exc, target=neur_exc), {'mu_minus': mu_plus})
    nest.SetStatus(nest.GetConnections(source=neur_out, target=neur_exc), {'mu_minus': mu_plus})
    nest.SetStatus(nest.GetConnections(source=neur_out, target=neur_out), {'mu_minus': mu_plus})

    nest.SetStatus(nest.GetConnections(source=neur_inp, target=neur_exc), {'alpha': alpha})
    nest.SetStatus(nest.GetConnections(source=neur_exc, target=neur_out), {'alpha': alpha})
    nest.SetStatus(nest.GetConnections(source=neur_exc, target=neur_exc), {'alpha': alpha})
    nest.SetStatus(nest.GetConnections(source=neur_out, target=neur_exc), {'alpha': alpha})
    nest.SetStatus(nest.GetConnections(source=neur_out, target=neur_out), {'alpha': alpha})

    for i in range(n_exc * (step + 1)):
        nest.SetStatus([noise_exc[i]], {"rate": 0.0})

    for i in range(n_out):
        nest.SetStatus([noise_out[i]], {"rate": 0.0})

    for i in range(n_inh):
        nest.SetStatus([noise_inh[i]], {"rate": 0.0})

    for i in range(n_inp):
        nest.SetStatus([noise_inp[i]], {"rate": 0.0})

    nest.Simulate(t_pause)

    print 'Status neur 100', nest.GetStatus([neur_exc[100]])
    print 'Status neur 300', nest.GetStatus([neur_exc[300]])


    #name = save_path + 'Outpus' + str(step) + '.mat'
    #sio.savemat(name, {'outputs_unsupervised': output, 'test_out': test_out, 'distance':dist, 'dotdist_K': dotdist_K, 'dotdist_W': dotdist_W})


    #name = save_path + 'EventsFr' + str(step) + '.mat'
    #sio.savemat(name, {'events_readout': events_readout, 'events_exc': events_exc})

    step = step + 1





########################################################################################################################
#PLOT



#Salvo i parametri
evt_exc_cnt_th = []
evt_th_cnt_th = []

for i in range(n_exc * (step)):
    evt_exc_cnt_th.append(nest.GetStatus(sd_exc, 'events')[i]['times'])

for i in range(n_inp):
    evt_th_cnt_th.append(nest.GetStatus(sd_inp, 'events')[i]['times'])

name = save_path + 'events_CNT_Th_' + str(step) + '.mat'
sio.savemat(name, {'evt_exc_cnt_th': evt_exc_cnt_th, 'evt_th_cnt_th': evt_th_cnt_th})

V_100 = nest.GetStatus(multimeter_100)[0]['events']
V_100 = V_100['V_m']

V_300 = nest.GetStatus(multimeter_300)[0]['events']
V_300 = V_300['V_m']

#sio.savemat(save_path + 'potential.mat', {'V_cnt': V_cnt, 'V_th': V_th, 'V_cnt_th': V_cnt_th})
sio.savemat(save_path + 'potential.mat', {'V_100': V_100, 'V_300': V_300})


timetot = ((t_train+t_pause)*n_train + t_pause*20 + (t_check + t_pause)*n_test + t_pause)*step


#timetot = t_train*4+t_check*4+t_train_trans*4

DT = 40.
NT = int(round(timetot/DT))

fr_exc=np.zeros((n_exc*step,NT),dtype=float)
fr_out=np.zeros((n_out,NT),dtype=float)
fr_inh=np.zeros((n_inh,NT),dtype=float)
fr_inp=np.zeros((n_inp,NT),dtype=float)
fr_inh2=np.zeros((n_inh2,NT),dtype=float)

evt_sd_exc = nest.GetStatus(sd_exc, keys="events")

for i in range(n_exc*step):
    times = []
    times_bin = []
    times = np.array(evt_sd_exc[i]["times"])

    times_bin = np.floor(times / DT)
    times_bin  = times_bin.astype(int)

    fr =  np.bincount(times_bin) / DT*1000
    fr_exc[i,0:len(fr)] = fr

evt_sd_inp = nest.GetStatus(sd_th_cnt_th, keys="events")

for i in range(n_inp):
    times = []
    times_bin = []
    times = np.array(evt_sd_inp[i]["times"])
    times_bin = np.floor(times / DT)
    times_bin  = times_bin.astype(int)

    fr =  np.bincount(times_bin) / DT*1000
    fr_inp[i,0:len(fr)] = fr

evt_sd_inh = nest.GetStatus(sd_inh, keys="events")

for i in range(n_inh):
    times = []
    times_bin = []
    times = np.array(evt_sd_inh[i]["times"])

    times_bin = np.floor(times / DT)
    times_bin  = times_bin.astype(int)

    fr =  np.bincount(times_bin) / DT*1000
    fr_inh[i,0:len(fr)] = fr

name = save_path + 'fr' + str(step) + '.mat'
sio.savemat(name, {'fr_inp': fr_inp, 'fr_exc': fr_exc, 'fr_inh': fr_inh})
#sio.savemat(save_path + 'weights_exc_.mat', {'w_inp_excOT': w_inp_exc, 'w_exc_excOT': w_exc_exc})

conn_par_inp_exc = nest.GetConnections(neur_inp, neur_exc[20:40])
w_inp_exc = np.mean(nest.GetStatus(conn_par_inp_exc, ["weight"]))
conn_par_exc_exc = nest.GetConnections(neur_exc[20:40], neur_exc[20:40])
w_exc_exc = np.mean(nest.GetStatus(conn_par_exc_exc, ["weight"]))
conn_par_exc_exc_altri = nest.GetConnections(neur_exc[20:40], neur_exc[0:20])
w_exc_exc_altri = np.mean(nest.GetStatus(conn_par_exc_exc_altri, ["weight"]))

print w_inp_exc*2.
print w_exc_exc
print w_exc_exc_altri


conn_par_inp_exc = nest.GetConnections(neur_inp, neur_exc)
w_inp_exc = nest.GetStatus(conn_par_inp_exc, ["source", "target","weight"])
conn_par_exc_exc = nest.GetConnections(neur_exc, neur_exc)
w_exc_exc = nest.GetStatus(conn_par_exc_exc, ["source", "target","weight"])
conn_par_inh_exc = nest.GetConnections(neur_inh, neur_exc)
w_inh_exc = nest.GetStatus(conn_par_inh_exc, ["source", "target","weight"])

name = save_path + 'weights' + str(step) + '.mat'

sio.savemat(name, {'w_exc_exc': w_exc_exc, 'w_inp_exc': w_inp_exc, 'w_inh_exc': w_inh_exc})
