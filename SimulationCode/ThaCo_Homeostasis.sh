#© 2021. This work is licensed under a CC-BY-NC-SA license.
#"Thalamo-cortical spiking model of incremental learning combining perception, context and NREM-sleep-mediated noise-resilience"
#Authors: Bruno Golosio, Chiara De Luca, Cristiano Capone, Elena Pastorelli, Giovanni Stegel, Gianmarco Tiddia, Giulia De Bonis and Pier Stanislao Paolucci
#arxiv.2003.11859


isubset=0
nthreads=36
data=12122019

interval=1


Mean=80.0
Std=30.0

alpha=$1

source /opt/NEST-nesters/bin/nest_vars.sh
cat  ThaCo_Homeostasis.templ.py | sed "s/_interval_/$interval/;s/_nthreads_/$nthreads/;s/_alpha_sleep_/$alpha/;s/_Mean_/$Mean/;s/_Std_/$Std/;s/_isubset_/$isubset/g" >   ThaCo_Homeostasis_alpha${alpha}_sub${isubset}.py
python2   ThaCo_Homeostasis_alpha${alpha}_sub${isubset}.py >   ThaCo_Homeostasis_alpha${alpha}_sub${isubset}.log




