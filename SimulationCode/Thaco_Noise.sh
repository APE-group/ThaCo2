#© 2021. This work is licensed under a CC-BY-NC-SA license.
#"Thalamo-cortical spiking model of incremental learning combining perception, context and NREM-sleep-mediated noise-resilience"
#Authors: Bruno Golosio, Chiara De Luca, Cristiano Capone, Elena Pastorelli, Giovanni Stegel, Gianmarco Tiddia, Giulia De Bonis and Pier Stanislao Paolucci
#arxiv.2003.11859

isubset=0
nthreads=36
data=1512020

interval=1

t_sleep=500000
Mean=80.0
Std=40.0

source /opt/NEST-nesters/bin/nest_vars.sh
cat  Thaco_Noise.templ.py | sed "s/_Mean_/$Mean/;s/_Std_/$Std/;s/_t_sleep_/$t_sleep/;s/_nthreads_/$nthreads/;s/_data_/$data/;s/_isubset_/$isubset/g" > Thaco_Noise_Cort_Mean${Mean}_Std${Std}_SleepStep_t${t_sleep}_sub${isubset}.py
python2 Thaco_Noise_Cort_Mean${Mean}_Std${Std}_SleepStep_t${t_sleep}_sub${isubset}.py > Thaco_Noise_Cort_Mean${Mean}_Std${Std}_SleepStep_t${t_sleep}_sub${isubset}.log




