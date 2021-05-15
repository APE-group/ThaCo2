#© 2021. This work is licensed under a CC-BY-NC-SA license.
#"Thalamo-cortical spiking model of incremental learning combining perception, context and NREM-sleep-mediated noise-resilience"
#Authors: Bruno Golosio, Chiara De Luca, Cristiano Capone, Elena Pastorelli, Giovanni Stegel, Gianmarco Tiddia, Giulia De Bonis and Pier Stanislao Paolucci
#arxiv.2003.11859


import os
import sys
import time
import nest
import numpy as np
import scipy.io as sio

print 'start'

# Set save path
home = os.getcwd()
save_path = home +  '/../SimulationOutput/ThaCo_SleepInterval_Ver2/set' + str(_isubset_) + '/'

interval = 1 #5

# checking path
if not os.path.exists(save_path):
    os.makedirs(save_path)

print interval
syn_T = 15.

# numero di classi
N_CLASSES = 10
# numero di esempi per classe
N_RANKS = 2 #40
# N_RANKS_2 = 1
# numero di esempi mostratidurante il test
N_TEST = 50 #250  # 250
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
INPUT_SIZE = 3*3*9*coding_number

fn_train = home+'/../Mnist_PreprocessingNew/mnist_training_features_3x3_Coding6_0.9_14.npy'
feat_arr_train0 = np.load(fn_train)
fn_test = home+ '/../Mnist_PreprocessingNew/mnist_test_features_3x3_Coding6_0.9_14.npy'
feat_arr_test = np.load(fn_test)

label_fn_train = home+ '/../Mnist_PreprocessingNew/mnist_training_labels_0.9_14.npy'
labels_train0 = np.load(label_fn_train)

label_fn_test = home+'/../Mnist_PreprocessingNew/mnist_test_labels_0.9_14.npy'
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

seed_rand =  _isubset_

print seed_rand
subset = seed_rand

feat_arr_train = []
labels_train = []

for num in range(0, N_RANKS, interval):
    # seleziono per ogni classe il numero di esempi da mostrare nel training,
    # salvo in feat_red le immagini, in labels_red i label corrispondente
    # feat_red e' lungo N_classes*N_ranks ed e' ordinato

    feat_red = [feat_train_class[i][j] for i in range(N_CLASSES)
                for j in range(subset * 1 + num, (subset + 1) * 1 + num)]
    labels_red = [label_train_class[i][j] for i in range(N_CLASSES)
                  for j in range(subset * 1 + num, (subset + 1 + num) * 1)]



    sio.savemat('labels_red.mat', {'labels_red': labels_red})
    sio.savemat('feat_red.mat', {'feat_red': feat_red})

    rand = np.random.RandomState(num)

    # Shuffle data
    #shuffle = rand.permutation(len(labels_red))

    # ora label_red ha i dati mischiati
    #labels_red = [labels_red[i] for i in shuffle]

    #feat_red = [feat_red[i] for i in shuffle]

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
k_exc = n_exc_example*interval #N_RANKS
#n_exc = k_exc * n_train * n_columns  # numero neuroni eccitatori 1*numero elementi training (30)*num colonne 2
n_exc = k_exc * N_CLASSES *N_RANKS
n_inh = 20*10*10/4 #2500 #n_exc / 4 #n_exc_example*N_CLASSES*N_RANKS / 4  # numero neuroni inibitori

n_inh2 = 200


############################################## parameters ############################################

r_noise_inp = 40.
r_noise_exc = 80.
r_noise_inh = r_noise_exc
r_noise_out = r_noise_exc
r_noise_inp_recog = r_noise_inp

W_noise_inp = 1600.
W_noise_exc = 1100.
W_noise_out = 1100. # diverso per leonardo 1920
W_noise_inh = 320.

Wmax_exc_exc = 150. # 150 o 20?
Wmax_inp_exc = 10.5
Wmax_exc_inp = 0. #100. # per leonardo, oppure 5.#0.333
Wmax_exc_out = 67500.
Wmax_out_out = 20.
Wmax_out_exc = 0.#5.

W0_exc_inp = 0.
W0_inp_exc = 0.3
W0_exc_out = 0.1
W0_exc_exc = 0.1 # oppure 0.5
W0_out_out = 0.01
W0_out_exc = 0.

W_inh_inh = -64. #-1.

mu_minus = 1.
mu_plus = 1.
alpha = 1.

W_exc_inh = 200.0
W_inh_exc_retrieval = -64.
W_inh_exc = -64.

lambda_inp_exc = 0.03
lambda_exc_inp = 0.03 #0.003
lambda_exc_out = .03
lambda_exc_exc = 0.1#0.05
lambda_out_out = 0.1
lambda_out_exc = 0.003

t_train = 1500.
t_pause = 1500.
t_check = 1000.

t_sleep = 60000.

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
sd_exc = nest.Create("spike_detector", n_exc)
nest.SetStatus(sd_exc, {"withgid": True, "withtime": True})
nest.Connect(neur_exc, sd_exc, "one_to_one")

# spike detectors for excitatory neurons
sd_inh = nest.Create("spike_detector", n_inh)
nest.SetStatus(sd_inh, {"withgid": True, "withtime": True})
nest.Connect(neur_inh, sd_inh, "one_to_one")


# spike detectors for output layer
sd_out = nest.Create("spike_detector", n_out)
nest.SetStatus(sd_out, {"withgid": True, "withtime": True})
nest.Connect(neur_out, sd_out, "one_to_one")


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

print 'excitatory to inhibitory connections'

syn_dict_exc_inh = {"weight": W_exc_inh, "delay": 1.0}
# conn_dict = {'rule': 'pairwise_bernoulli', 'p': Bernulli_ExcToInh}
conn_dict = {'rule': 'all_to_all'}
nest.Connect(neur_exc[0:n_exc],
             neur_inh[0:n_inh], conn_dict, syn_dict_exc_inh)

# inhibitory to excitatory connections

syn_dict_inh_exc = {"weight": W_inh_exc, "delay": 1.0}
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




multi = nest.Create('multimeter', params={'record_from': ['V_m', 'w'], 'interval' :50.,  'withgid': True})
nest.Connect(multi, neur_exc)


################################################
# training

nest.SetStatus(nest.GetConnections(source=neur_inp, target=neur_exc), {'mu_minus': mu_minus, 'mu_plus': mu_plus, 'alpha': alpha})
nest.SetStatus(nest.GetConnections(source=neur_exc, target=neur_exc), {'mu_minus': mu_minus, 'mu_plus': mu_plus, 'alpha': alpha})
nest.SetStatus(nest.GetConnections(source=neur_out, target=neur_exc), {'mu_minus': mu_minus, 'mu_plus': mu_plus, 'alpha': alpha})
nest.SetStatus(nest.GetConnections(source=neur_out, target=neur_out), {'mu_minus': mu_minus, 'mu_plus': mu_plus, 'alpha': alpha})

output = []


NumTrain = []

i_out_train = []

count = np.zeros((N_CLASSES))

step = 0
i_cumul = 0


ClassAccuracy_Arr = []
GroupAccuracy_Arr = []
ReadoutAccuracy_Arr = []
TimeSleep = []


for i_cumul in range(0, N_RANKS):
    for i_train in range(i_cumul*(interval*N_CLASSES), (i_cumul+1)*interval*N_CLASSES):
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


        media = np.random.normal(_Mean_, _Std_)

        for i_k in range(n_exc_example):
            rate = np.random.normal(media, _Std_/3)
            while rate <= 0:
                rate = np.random.normal(media, _Std_ / 3)
            nest.SetStatus([noise_exc[train_target_exc[i_train][i_k]]],
                           {"rate": rate})



        nest.SetStatus(nest.GetConnections(source=neur_exc, target=neur_out), {
            'lambda': -lambda_exc_out * .5})  # connetti NEGATIVAMENTE i neuroni eccitatori ai neuroni di read out

        for i in range(n_classes):  # per ciascun numero di classi
            if i == i_out:  # se e' la classe che sto guardando
                nest.SetStatus(noise_out[i*n_exc_example:(i+1)*n_exc_example], {"rate": r_noise_out})
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
        nest.SetStatus(nest.GetConnections(source=neur_out, target=neur_out),
                       {'lambda': 0.0})

        # roba per leggere lo status nel training

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


    #print 'teach_out', teach_out[i_cumul: i_cumul + interval * N_CLASSES]

    #Salvo i parametri
    evt_exc_cnt_th = []
    evt_th_cnt_th = []
    evt_inh = []

    for i in range(n_exc * (step+1)):
        evt_exc_cnt_th.append(nest.GetStatus(sd_exc, 'events')[i]['times'])

    for i in range(n_inp):
        evt_th_cnt_th.append(nest.GetStatus(sd_inp, 'events')[i]['times'])
    for i in range(n_inh):
            evt_inh.append(nest.GetStatus(sd_inh, 'events')[i]['times'])

    name = save_path + 'events' + '.mat'
    sio.savemat(name, {'evt_exc': evt_exc_cnt_th, 'evt_th': evt_th_cnt_th, 'evt_inh': evt_inh})
        
    name = save_path + 'multimeter' + '.mat'
    
    print('heeey', nest.GetStatus(multi)[0]['events'])
    V = nest.GetStatus(multi)[0]['events']['V_m']
    sender_m =nest.GetStatus(multi)[0]['events']['senders']
    time_m  =nest.GetStatus(multi)[0]['events']['times']
    w_m =nest.GetStatus(multi)[0]['events']['w']
    sio.savemat(name, {'V': V, 'senders': list(sender_m), 'times': list(time_m), 'w': list(w_m)})

    #sio.savemat(name, {'V': V, 'senders': list(sender_m), })

    nest.Simulate(t_pause)

    ######################################################################
    # Test

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



    print '######################################################################'
    print '# Test pre-sleep con context'
    print '######################################################################'
    print

    # test_seq = [2,0,1,3]

    #dist = np.zeros((n_test, N_CLASSES))
    #dotdist_W = np.zeros((n_test, N_CLASSES))
    #dotdist_K = np.zeros((n_test, N_CLASSES))

    count_right_num = 0
    count_right_time = 0
    count_unsupervised_class_right = 0
    count_unsupervised_group_right = 0

    #test_pred_num = np.zeros((n_test))
    # test_pred_time = np.zeros((n_test))
    #times = np.zeros((n_out))

    for i_test in range(n_test):  # per ogni esempio del test set

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

        #test_pred_num[i_test] = i_out_num
        #ReadoutPrediction[np.int32(float(cl)/N_RANKS), i_test] = i_out_num

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
        #print n_spikes_un
        for cl in range(0, N_CLASSES):
            if n_spikes_un[cl] >= maximum:
                maximum = n_spikes_un[cl]
                unsupervised_output_class = cl

        if unsupervised_output_class == test_out[i_test]:
            count_unsupervised_class_right = count_unsupervised_class_right + 1
        
        #ClassPrediction[np.int32(float(cl)/N_RANKS), i_test] = unsupervised_output_class

        #print 'Corrects unsupervised class: ', count_unsupervised_class_right, '/', i_test+1
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

        #print 'Corrects unsupervised group: ', count_unsupervised_group_right, '/', i_test + 1
        print 'Accuracy unsupervised group = ', float(count_unsupervised_group_right) / float(i_test + 1.) * 100., '%'
        
        #GroupPrediction[np.int32(float(cl)/N_RANKS), i_test] = unsupervised_output_group

        # ... fino a qui

        endTime = time.time()
        #print ('Simulation time: %.2f s') % (endTime - startTime)
        sys.stdout.flush()


    print 'Accuracy pre sleep = ', float(count_right_num) / float(i_test + 1.) * 100., '%'
    print 'Accuracy_pre_sleep_unsupervised_class =', float(count_unsupervised_class_right) / float(i_test + 1.) * 100.
    print 'Accuracy_pre_sleep_unsupervised_group =', float(count_unsupervised_group_right) / float(i_test + 1.) * 100.
    #print 'Accuracy KNN = ', float(count_KNN_right)/float(i_test+1.)*100., '%'


    ClassAccuracy_Arr.append(float(count_unsupervised_class_right) / float(i_test + 1.) * 100.)
    GroupAccuracy_Arr.append(float(count_unsupervised_group_right) / float(i_test + 1.) * 100.)
    ReadoutAccuracy_Arr.append(float(count_right_num) / float(i_test + 1.) * 100.)
    TimeSleep.append(0.)

    np.save(save_path + 'ClassAccuracy_Arr', ClassAccuracy_Arr)
    np.save(save_path + 'GroupAccuracy_Arr', GroupAccuracy_Arr)
    np.save(save_path + 'ReadoutAccuracy_Arr', ReadoutAccuracy_Arr)
    np.save(save_path + 'TimeSleep', TimeSleep)


    conn_par_exc_exc = nest.GetConnections(neur_exc, neur_exc)
    w_exc_exc = nest.GetStatus(conn_par_exc_exc, ["source", "target", "weight"])
    conn_par_inp_exc = nest.GetConnections(neur_inp, neur_exc)
    w_inp_exc = nest.GetStatus(conn_par_inp_exc, ["source", "target", "weight"])

    name = save_path + 'weights_PreSleep_' + str(i_cumul) +  '.mat'
    sio.savemat(name, {'w_exc_exc': w_exc_exc, 'w_inp_exc': w_inp_exc})

    
    name = save_path + 'multimeter' + '.mat'
    V = nest.GetStatus(multi)[0]['events']['V_m']
    sender_m =nest.GetStatus(multi)[0]['events']['senders']
    time_m  =nest.GetStatus(multi)[0]['events']['times']
    w_m =nest.GetStatus(multi)[0]['events']['w']
    sio.savemat(name, {'V': V, 'senders': list(sender_m), 'times': list(time_m), 'w': list(w_m)})
    
    nest.Simulate(t_pause*10.)


    nest.Simulate(t_pause*2.)



    print '######################################################################'
    print '# Sleep'
    print '######################################################################'
    print

    lambda_exc_exc_sleep = 0.0008 #0.0008
    b_sleep = 150. #150.#150
    alpha_sleep = 5. #5.
    r_noise_sleep = 650. #650. #650. # o 900
    Winh_sleep = -0.5 #-0.5 #-0.5*b_sleep/150. #-0.7
    W_noise_sleep = 18.


    # Rimetto bene il peso delle connessioni con il rumore
    conn = nest.GetConnections(source=noise_exc, target=neur_exc);
    nest.SetStatus(conn, {'weight': W_noise_sleep})
    nest.SetStatus(nest.GetConnections(source=neur_exc, target=neur_exc), {'alpha': alpha_sleep})  # era 5
    nest.SetStatus(neur_exc, {"b": b_sleep})  # 150.
    nest.SetStatus(nest.GetConnections(source=neur_inh, target=neur_exc),
                   {"weight": Winh_sleep})  # -0.5
    # switch off all teaching outputs
    for i in range(n_exc * (step + 1)):
        nest.SetStatus([noise_exc[i]], {"rate": r_noise_sleep})  # 650.

    for inp in range(0, n_inp):
        for ex in range(0, n_exc * (step + 1)):

            conn = nest.GetConnections(source=[neur_inp[inp]], target=[neur_exc[ex]])
            w = nest.GetStatus(conn, ['weight'])
            nest.SetStatus(conn, {'weight': w[0][0] * 0.5})
            w = nest.GetStatus(conn, ['weight'])

    nest.SetStatus(nest.GetConnections(source=neur_exc, target=neur_exc), {'lambda': 0.0})
    nest.Simulate(t_sleep)
    nest.SetStatus(nest.GetConnections(source=neur_exc, target=neur_exc), {'lambda': lambda_exc_exc_sleep})
    nest.Simulate(200000)

    nest.SetStatus(nest.GetConnections(source=neur_exc, target=neur_exc), {'lambda': 0.0})
    nest.Simulate(t_sleep)

    conn_par_exc_exc = nest.GetConnections(neur_exc, neur_exc)
    w_exc_exc = nest.GetStatus(conn_par_exc_exc, ["source", "target", "weight"])
    name = save_path + 'weights_PostSleep_' + str(int(i_cumul)) + '.mat'
    sio.savemat(name, {'w_exc_exc': w_exc_exc})

    #Salvo i parametri
    evt_exc_cnt_th = []
    evt_th_cnt_th = []
    evt_inh = []

    for i in range(n_exc * (step+1)):
        evt_exc_cnt_th.append(nest.GetStatus(sd_exc, 'events')[i]['times'])

    for i in range(n_inp):
        evt_th_cnt_th.append(nest.GetStatus(sd_inp, 'events')[i]['times'])
    for i in range(n_inh):
            evt_inh.append(nest.GetStatus(sd_inh, 'events')[i]['times'])

    name = save_path + 'events' + '.mat'
    sio.savemat(name, {'evt_exc': evt_exc_cnt_th, 'evt_th': evt_th_cnt_th, 'evt_inh': evt_inh})

    # Prepare for testing phase

    # Rimetto bene il peso delle connessioni con il rumore
    conn = nest.GetConnections(source=noise_exc, target=neur_exc);
    nest.SetStatus(conn, {'weight': W_noise_exc})
    nest.SetStatus(nest.GetConnections(source=neur_exc, target=neur_exc), {'alpha': 1.})
    nest.SetStatus(neur_exc, {"b": 0.01})
    nest.SetStatus(nest.GetConnections(source=neur_exc, target=neur_exc), {'lambda': 0.0})
    nest.SetStatus(nest.GetConnections(source=neur_inh, target=neur_exc),
                   {"weight": W_inh_exc})
    # switch off all teaching outputs
    for i in range(n_exc * (step + 1)):
        nest.SetStatus([noise_exc[i]], {"rate": 0.0})  # 650.
    for inp in range(0, n_inp):
        for ex in range(0, n_exc * (step + 1)):

            conn = nest.GetConnections(source=[neur_inp[inp]], target=[neur_exc[ex]])
            w = nest.GetStatus(conn, ['weight'])
            nest.SetStatus(conn, {'weight': w[0][0] / 0.5})
            w = nest.GetStatus(conn, ['weight'])

    count_right_num = 0
    count_right_time = 0
    count_unsupervised_class_right = 0
    count_unsupervised_group_right = 0

    # TESTING
    for i_test in range(n_test):  # per ogni esempio del test set

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

        #test_pred_num[i_test] = i_out_num
        #ReadoutPrediction[np.int32(float(cl)/N_RANKS), i_test] = i_out_num

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
        #print n_spikes_un
        for cl in range(0, N_CLASSES):
            if n_spikes_un[cl] >= maximum:
                maximum = n_spikes_un[cl]
                unsupervised_output_class = cl

        if unsupervised_output_class == test_out[i_test]:
            count_unsupervised_class_right = count_unsupervised_class_right + 1
        
        #ClassPrediction[np.int32(float(cl)/N_RANKS), i_test] = unsupervised_output_class

        #print 'Corrects unsupervised class: ', count_unsupervised_class_right, '/', i_test+1
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

        #print 'Corrects unsupervised group: ', count_unsupervised_group_right, '/', i_test + 1
        print 'Accuracy unsupervised group = ', float(count_unsupervised_group_right) / float(i_test + 1.) * 100., '%'
        
        #GroupPrediction[np.int32(float(cl)/N_RANKS), i_test] = unsupervised_output_group

        # ... fino a qui

        endTime = time.time()
        #print ('Simulation time: %.2f s') % (endTime - startTime)
        sys.stdout.flush()


    print 'Accuracy sleep = ', float(count_right_num) / float(i_test + 1.) * 100., '%'
    print 'Accuracy_sleep_unsupervised_class =', float(count_unsupervised_class_right) / float(i_test + 1.) * 100.
    print 'Accuracy_sleep_unsupervised_group =', float(count_unsupervised_group_right) / float(i_test + 1.) * 100.
    #print 'Accuracy KNN = ', float(count_KNN_right)/float(i_test+1.)*100., '%'


    ClassAccuracy_Arr.append(float(count_unsupervised_class_right) / float(i_test + 1.) * 100.)
    GroupAccuracy_Arr.append(float(count_unsupervised_group_right) / float(i_test + 1.) * 100.)
    ReadoutAccuracy_Arr.append(float(count_right_num) / float(i_test + 1.) * 100.)

    TimeSleep.append(TimeSleep[len(TimeSleep)-1] + t_sleep)

    np.save(save_path + 'ClassAccuracy_Arr', ClassAccuracy_Arr)
    np.save(save_path + 'GroupAccuracy_Arr', GroupAccuracy_Arr)
    np.save(save_path + 'ReadoutAccuracy_Arr', ReadoutAccuracy_Arr)
    np.save(save_path + 'TimeSleep', TimeSleep)

    
    #Salvo i parametri
    evt_exc_cnt_th = []
    evt_th_cnt_th = []
    evt_inh = []

    for i in range(n_exc * (step+1)):
        evt_exc_cnt_th.append(nest.GetStatus(sd_exc, 'events')[i]['times'])

    for i in range(n_inp):
        evt_th_cnt_th.append(nest.GetStatus(sd_inp, 'events')[i]['times'])
    for i in range(n_inh):
            evt_inh.append(nest.GetStatus(sd_inh, 'events')[i]['times'])

    name = save_path + 'events' + '.mat'
    sio.savemat(name, {'evt_exc': evt_exc_cnt_th, 'evt_th': evt_th_cnt_th, 'evt_inh': evt_inh})
    name = save_path + 'multimeter' + '.mat'
    
    V = nest.GetStatus(multi)[0]['events']['V_m']
    sender_m =nest.GetStatus(multi)[0]['events']['senders']
    time_m  =nest.GetStatus(multi)[0]['events']['times']
    w_m =nest.GetStatus(multi)[0]['events']['w']
    sio.savemat(name, {'V': V, 'senders': list(sender_m), 'times': list(time_m), 'w': list(w_m)})

    nest.Simulate(t_pause*10.)
