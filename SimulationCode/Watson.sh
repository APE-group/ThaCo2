#© 2021. This work is licensed under a CC-BY-NC-SA license.
#"Thalamo-cortical spiking model of incremental learning combining perception, context and NREM-sleep-mediated noise-resilience"
#Authors: Bruno Golosio, Chiara De Luca, Cristiano Capone, Elena Pastorelli, Giovanni Stegel, Gianmarco Tiddia, Giulia De Bonis and Pier Stanislao Paolucci
#arxiv.2003.11859

isubset=0
nthreads=36
data=1512020

interval=1

t_sleep=1000000
Mean=80.0
Std=40.0

rate=900.
alpha=2.
b=250.
tau=400.
inh=-0.7

source /opt/NEST-nesters/bin/nest_vars.sh
cat  Watson.templ.py | sed "s/_alpha_/$alpha/;s/_b_/$b/;s/_tau_/$tau/;s/_Winh_/$inh/;s/_rate_/$rate/;s/_Mean_/$Mean/;s/_Std_/$Std/;s/_t_sleep_/$t_sleep/;s/_nthreads_/$nthreads/;s/_data_/$data/;s/_isubset_/$isubset/g" > Watson10_a${alpha}_b${b}_t${tau}_inh${inh}_std${Std}_rate${rate}_sub${isubset}.py
python2 Watson10_a${alpha}_b${b}_t${tau}_inh${inh}_std${Std}_rate${rate}_sub${isubset}.py > Watson10_a${alpha}_b${b}_t${tau}_inh${inh}_std${Std}_rate${rate}_sub${isubset}.log




