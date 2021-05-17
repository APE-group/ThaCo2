%© 2021. This work is licensed under a CC-BY-NC-SA license.
%"Thalamo-cortical spiking model of incremental learning combining perception, context and NREM-sleep-mediated noise-resilience"
%Authors: Bruno Golosio, Chiara De Luca, Cristiano Capone, Elena Pastorelli, Giovanni Stegel, Gianmarco Tiddia, Giulia De Bonis and Pier Stanislao Paolucci
%arxiv.2003.11859

clear all
close all

%%
load('/Users/Mac/Desktop/Fig4/Data/Results')

load('/Users/Mac/Desktop/Fig4/Data/ResultsNoise')
%%
figure
errorbar(x/10, accK1, errK1 , 'LineWidth', 2)
hold on
errorbar(x/10, accK3, errK3, 'LineWidth', 2)
errorbar(x/10, accK5, errK5, 'LineWidth', 2)
errorbar(xWTA/10, accWTA, errWTA, 'LineWidth', 2)
errorbar(xWTA/10, accWTA_gr, errWTA_gr, 'LineWidth', 2)


errorbar(Noisex/10, NoiseaccK1, NoiseerrK1 , 'LineWidth', 2)
hold on
errorbar(Noisex/10, NoiseaccK3, NoiseerrK3, 'LineWidth', 2)
errorbar(Noisex/10, NoiseaccK5, NoiseerrK5, 'LineWidth', 2)
errorbar(NoisexWTA/10, NoiseaccWTA, NoiseerrWTA, 'LineWidth', 2)


lg = legend('KNN-K1', 'KNN-K3', 'KNN-K5', 'WTA-Class', 'WTA-Group', 'Noise KNN-K1', 'Noise KNN-K3', 'NoiseKNN-K5', 'Noise WTA-Class');
lg.FontSize = 20;
ylabel('Accuracy [%]')
xlabel('#training examples per category')
title('Codifica 6 neuroni')
%xlim([0.5, 5.5])
ylim([10,100])
xticks([1 2 3 4 5])
xticklabels({'1','2','3','4','5'})
    
