#© 2021. This work is licensed under a CC-BY-NC-SA license.
#"Thalamo-cortical spiking model of incremental learning combining perception, context and NREM-sleep-mediated noise-resilience"
#Authors: Bruno Golosio, Chiara De Luca, Cristiano Capone, Elena Pastorelli, Giovanni Stegel, Gianmarco Tiddia, Giulia De Bonis and Pier Stanislao Paolucci
#arxiv.2003.11859

isubset=1
nthreads=36
data=16122019

interval=1
ops=4.
WmaxEE=20.

r_noise_inp=40.0
r_noise_exc=80.0

W_noise_inp=1600.
W_noise_exc=110.
Winhexcretrieval=-64.

W0_inp_exc=0.3
W0_exc_exc=0.5


source /opt/NEST-nesters/bin/nest_vars.sh
cat  Thaco_NoiseSP.templ.py | sed "s/_Winhexcretrieval_/$Winhexcretrieval/;s/_ops_/$ops/;s/_interval_/$interval/;s/_W0_exc_exc_/$W0_exc_exc/;s/_W0_inp_exc_/$W0_inp_exc/;s/_W_noise_inp_/$W_noise_inp/;s/_W_noise_exc_/$W_noise_exc/;s/_rExc_/$r_noise_exc/;s/_rInp_/$r_noise_inp/;s/_WmaxEE_/$WmaxEE/;s/_nthreads_/$nthreads/;s/_data_/$data/;s/_isubset_/$isubset/g" >   UTOT_SP0502_ThOff_sub${isubset}.py
python   UTOT_SP0502_ThOff_sub${isubset}.py >   UTOT_SP0502_ThOff_sub${isubset}.log




