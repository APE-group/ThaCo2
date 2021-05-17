%© 2021. This work is licensed under a CC-BY-NC-SA license.
%"Thalamo-cortical spiking model of incremental learning combining perception, context and NREM-sleep-mediated noise-resilience"
%Authors: Bruno Golosio, Chiara De Luca, Cristiano Capone, Elena Pastorelli, Giovanni Stegel, Gianmarco Tiddia, Giulia De Bonis and Pier Stanislao Paolucci
%arxiv.2003.11859

clear all
close all


%%

load('/Users/Mac/Desktop/Fig5/Data/ResultsNoisePrePost')
load('/Users/Mac/Desktop/Fig5/Data/Results')

figure
errorbar(Noisex, NoiseaccWTA, NoiseerrWTA/10, 'LineWidth', 2)
hold on
errorbar(xWTA/10, accWTA, errWTA/10, 'LineWidth', 2)
errorbar(Noisex, NoiseaccWTA_Post, NoiseerrWTA_Post/10, 'LineWidth', 2)

lg = legend('Noise', 'No Noise');
lg.FontSize = 20;
xlim([0, 10])

figure
errorbar(Noisex, NoiseaccWTAGr, NoiseerrWTAGr/10, 'LineWidth', 2)
hold on
errorbar(xWTA/10, accWTA_gr, errWTA_gr/10, 'LineWidth', 2)
errorbar(Noisex, NoiseaccWTAGr_Post, NoiseerrWTAGr_Post/10, 'LineWidth', 2)

lg = legend('Noise', 'No Noise');
lg.FontSize = 20;
xlim([0, 10])

