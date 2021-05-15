#© 2021. This work is licensed under a CC-BY-NC-SA license.
#"Thalamo-cortical spiking model of incremental learning combining perception, context and NREM-sleep-mediated noise-resilience"
#Authors: Bruno Golosio, Chiara De Luca, Cristiano Capone, Elena Pastorelli, Giovanni Stegel, Gianmarco Tiddia, Giulia De Bonis and Pier Stanislao Paolucci
#arxiv.2003.11859


isubset=10
nthreads=36
data=12122019

interval=1


Mean=80.0
Std=30.0

source /opt/NEST-nesters/bin/nest_vars.sh
cat  ThaCo_SleepInterval.templ.py | sed "s/_interval_/$interval/;s/_nthreads_/$nthreads/;s/_Mean_/$Mean/;s/_Std_/$Std/;s/_isubset_/$isubset/g" >   ThaCo_SleepInterval_sub${isubset}.py
python2   ThaCo_SleepInterval_sub${isubset}.py >   ThaCo_SleepInterval_sub${isubset}.log




