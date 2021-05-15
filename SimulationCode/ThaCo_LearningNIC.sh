#© 2021. This work is licensed under a CC-BY-NC-SA license.
#"Thalamo-cortical spiking model of incremental learning combining perception, context and NREM-sleep-mediated noise-resilience"
#Authors: Bruno Golosio, Chiara De Luca, Cristiano Capone, Elena Pastorelli, Giovanni Stegel, Gianmarco Tiddia, Giulia De Bonis and Pier Stanislao Paolucci
#arxiv.2003.11859

isubset=10
nthreads=36

interval=1



source /opt/NEST-nesters/bin/nest_vars.sh
cat  ThaCo_LearningNIC.templ.py | sed "s/_interval_/$interval/;s/_nthreads_/$nthreads/;s/_isubset_/$isubset/g" >   ThaCo_LearningNIC_sub${isubset}.py
python2   ThaCo_LearningNIC_sub${isubset}.py >   ThaCo_LearningNIC_sub${isubset}.log




