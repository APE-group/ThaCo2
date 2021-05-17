%%
%© 2021. This work is licensed under a CC-BY-NC-SA license.
%"Thalamo-cortical spiking model of incremental learning combining perception, context and NREM-sleep-mediated noise-resilience"
%Authors: Bruno Golosio, Chiara De Luca, Cristiano Capone, Elena Pastorelli, Giovanni Stegel, Gianmarco Tiddia, Giulia De Bonis and Pier Stanislao Paolucci
%arxiv.2003.11859

n_exc = 200;
t_train = 1500.;
t_pause = 1500.;
n_train = 10.;
t_check = 1000.;
n_test = 2;
time_tot = (t_pause + t_train)*n_train;
step = 2;
t_sleep = 200000.;

time_tot = ((t_train+t_pause)*n_train + t_pause*20 + (t_check + t_pause)*n_test+t_pause)*step + (t_pause*10. + t_sleep)*(step -1);


y_exc = [];
x_exc = [];

cd = '/Users/Mac/Desktop/Fig6/Data/set3/'


%%
% Load pesi dopo 5 esempi...

n_exc = 200*5;

load(strcat(cd, 'weights_PostSleep_4.mat'));
w_post = w_exc_exc;
load(strcat(cd, 'weights_PreSleep_4.mat'));
w_pre = w_exc_exc;

uno = [2, 8, 4, 9, 1, 6, 7, 3, 0, 5, 2, 9, 6, 4, 0, 3, 1, 7, 8, 5, 4, 1, 5, 0, 7, 2, 3, 6, 9, 8, 5, 4, 1, 2, 9, 6, 7, 0, 3, 8, 3, 8, 4, 9, 2, 6, 0, 1, 5, 7];

ops = w_post;
ops1 = sortrows(ops);
ops_matrix_post= reshape(ops1(:,3), [n_exc, n_exc]);
ops = w_pre;
ops1 = sortrows(ops);
ops_matrix_pre= reshape(ops1(:,3), [n_exc, n_exc]);


%%
% Riordino
ops_matrix_nuovo = ops_matrix_post;

ops_matrix_nuovo_pre = ops_matrix_pre;



figure
imagesc(ops_matrix_nuovo)
colorbar()
%set(gca, 'clim', [6 8])
xlim([0, 200])
ylim([0, 200])
title('Post Sleep')
set(gca, 'clim', [10 130])
hold on
plot([0, 200],[100 100], 'Color', [0. 0. 0.])
plot([100, 100],[0 200], 'Color', [0. 0. 0.])
h = colorbar;
ylabel(h, 'Synaptic Weights')
xlabel('# Cortical neuron')
ylabel('# Cortical neuron')
ax = gca;
ax.FontSize = 20;

figure
imagesc(ops_matrix_nuovo_pre)
colorbar()
set(gca, 'clim', [10 130])
xlim([0, 200])
ylim([0, 200])
title('Pre Sleep')
hold on
plot([0, 200],[100 100], 'Color', [0. 0. 0.])
plot([100, 100],[0 200], 'Color', [0. 0. 0.])
h = colorbar;
ylabel(h, 'Synaptic Weights')
xlabel('# Cortical neuron')
ylabel('# Cortical neuron')
ax = gca;
ax.FontSize = 20;


%%

ops_matrix_nuovo_plot = ops_matrix_nuovo;


ops_matrix_nuovo_plot_pre = ops_matrix_nuovo_pre;

%%
% Plot post sleep dopo 5 esempi riordinato


figure
imagesc(log(ops_matrix_nuovo_plot))
hold on
plot([0, 1000],[100 100], 'Color', [0. 0. 0.])
plot([100, 100],[0 1000], 'Color', [0. 0. 0.])
plot([0, 1000],[200 200], 'Color', [0. 0. 0.])
plot([200, 200],[0 1000], 'Color', [0. 0. 0.])
plot([0, 1000],[300 300], 'Color', [0. 0. 0.])
plot([300, 300],[0 1000], 'Color', [0. 0. 0.])
plot([0, 1000],[400 400], 'Color', [0. 0. 0.])
plot([400, 400],[0 1000], 'Color', [0. 0. 0.])
plot([0, 1000],[500 500], 'Color', [0. 0. 0.])
plot([500, 500],[0 1000], 'Color', [0. 0. 0.])
plot([0, 1000],[600 600], 'Color', [0. 0. 0.])
plot([600, 600],[0 1000], 'Color', [0. 0. 0.])
plot([0, 1000],[700 700], 'Color', [0. 0. 0.])
plot([700, 700],[0 1000], 'Color', [0. 0. 0.])
plot([0, 1000],[800 800], 'Color', [0. 0. 0.])
plot([800, 800],[0 1000], 'Color', [0. 0. 0.])
plot([0, 1000],[900 900], 'Color', [0. 0. 0.])
plot([900, 900],[0 1000], 'Color', [0. 0. 0.])
plot([0, 1000],[1000 1000], 'Color', [0. 0. 0.])
plot([1000, 1000],[0 1000], 'Color', [0. 0. 0.])
h = colorbar;
ylabel(h, 'log(Synaptic Weights)')
xlabel('# Cortical neuron')
ylabel('# Cortical neuron')
title('Post Sleep')
set(gca, 'clim', [-17 2])
ax = gca;
ax.FontSize = 10;
%A4: 21cm x 29.7cm
set(gcf,'units','centimeters','position',[1,1,9.2,7])
%%
%Plot pre sleep dopo 5 esempi riordinato

figure
imagesc(log(ops_matrix_nuovo_pre))
hold on
plot([0, 1000],[100 100], 'Color', [0. 0. 0.])
plot([100, 100],[0 1000], 'Color', [0. 0. 0.])
plot([0, 1000],[200 200], 'Color', [0. 0. 0.])
plot([200, 200],[0 1000], 'Color', [0. 0. 0.])
plot([0, 1000],[300 300], 'Color', [0. 0. 0.])
plot([300, 300],[0 1000], 'Color', [0. 0. 0.])
plot([0, 1000],[400 400], 'Color', [0. 0. 0.])
plot([400, 400],[0 1000], 'Color', [0. 0. 0.])
plot([0, 1000],[500 500], 'Color', [0. 0. 0.])
plot([500, 500],[0 1000], 'Color', [0. 0. 0.])
plot([0, 1000],[600 600], 'Color', [0. 0. 0.])
plot([600, 600],[0 1000], 'Color', [0. 0. 0.])
plot([0, 1000],[700 700], 'Color', [0. 0. 0.])
plot([700, 700],[0 1000], 'Color', [0. 0. 0.])
plot([0, 1000],[800 800], 'Color', [0. 0. 0.])
plot([800, 800],[0 1000], 'Color', [0. 0. 0.])
plot([0, 1000],[900 900], 'Color', [0. 0. 0.])
plot([900, 900],[0 1000], 'Color', [0. 0. 0.])
plot([0, 1000],[1000 1000], 'Color', [0. 0. 0.])
plot([1000, 1000],[0 1000], 'Color', [0. 0. 0.])
%h = colorbar;
%ylabel(h, 'log(Synaptic Weights)')
xlabel('# Cortical neuron')
ylabel('# Cortical neuron')
title('Pre Sleep')
set(gca, 'clim', [-17 2])
ax = gca;
ax.FontSize = 10;

%A4: 21cm x 29.7cm
set(gcf,'units','centimeters','position',[1,1,7,7])

%%

combinato = zeros([100 10]);
for classe = 0:9
    for es = 0:4
        temp = ops_matrix_nuovo(classe*100 + es*20+1:classe*100 +(es+1)*20,classe*100 + es*20+1:classe*100 +(es+1)*20)
        combinato(es*20+1:(es+1)*20, classe*20+1: (1+ classe)*20) = temp;
    end
end

figure
%subplot(2,1,1)
imagesc(combinato)
hold on
plot([0, 200],[20 20], 'Color', [0. 0. 0.])
plot([0, 200],[40 40], 'Color', [0. 0. 0.])
plot([0, 200],[60 60], 'Color', [0. 0. 0.])
plot([0, 200],[80 80], 'Color', [0. 0. 0.])
plot([20, 20],[0 100], 'Color', [0. 0. 0.])
plot([40, 40],[0 100], 'Color', [0. 0. 0.])
plot([60, 60],[0 100], 'Color', [0. 0. 0.])
plot([80, 80],[0 100], 'Color', [0. 0. 0.])
plot([100, 100],[0 100], 'Color', [0. 0. 0.])
plot([120, 120],[0 100], 'Color', [0. 0. 0.])
plot([140, 140],[0 100], 'Color', [0. 0. 0.])
plot([160, 160],[0 100], 'Color', [0. 0. 0.])
plot([180, 180],[0 100], 'Color', [0. 0. 0.])
%title('Cortico-cortical synaptic weights')
colorbar()
set(gca, 'clim', [0 120])
%yyaxis left
%yticks([10, 30, 50, 70, 90])
xticks([10, 30, 50, 70, 90, 110, 130, 150, 170, 190])
xticklabels({'0', '1', '2', '3', '4', '5', '6', '7', '8', '9'})
%yticklabels({'example #0', 'example #1', 'example #2', 'example #3', 'example #4'})
%yyaxis right
%ylabel('Post-Sleep', 'Color', [0. 0. 0.])
set(gca,'ytick',[])
set(gca,'yticklabel',[])
%set(gca,'xtick',[])
%set(gca,'xticklabel',[])
xlabel('Class')
h = colorbar;
ylabel(h, 'Synaptic Weights')
ax = gca;
ax.FontSize = 10;
%A4: 21cm x 29.7cm
set(gcf,'units','centimeters','position',[1,1,8,7])

%%
combinato_pre = zeros([100 10]);
for classe = 0:9
    for es = 0:4
        temp = ops_matrix_nuovo_pre(classe*100 + es*20+1:classe*100 +(es+1)*20,classe*100 + es*20+1:classe*100 +(es+1)*20);
        combinato_pre(es*20+1:(es+1)*20, classe*20+1: (1+ classe)*20) = temp;
    end
end

figure
%subplot(2,1,2)
imagesc(combinato_pre)
hold on
plot([0, 200],[20 20], 'Color', [0. 0. 0.])
plot([0, 200],[40 40], 'Color', [0. 0. 0.])
plot([0, 200],[60 60], 'Color', [0. 0. 0.])
plot([0, 200],[80 80], 'Color', [0. 0. 0.])
plot([20, 20],[0 100], 'Color', [0. 0. 0.])
plot([40, 40],[0 100], 'Color', [0. 0. 0.])
plot([60, 60],[0 100], 'Color', [0. 0. 0.])
plot([80, 80],[0 100], 'Color', [0. 0. 0.])
plot([100, 100],[0 100], 'Color', [0. 0. 0.])
plot([120, 120],[0 100], 'Color', [0. 0. 0.])
plot([140, 140],[0 100], 'Color', [0. 0. 0.])
plot([160, 160],[0 100], 'Color', [0. 0. 0.])
plot([180, 180],[0 100], 'Color', [0. 0. 0.])
%yyaxis left
yticks([10, 30, 50, 70, 90])
xticks([10, 30, 50, 70, 90, 110, 130, 150, 170, 190])
xticklabels({'0', '1', '2', '3', '4', '5', '6', '7', '8', '9'})
yticklabels({'example #0', 'example #1', 'example #2', 'example #3', 'example #4'})
%yyaxis right
%ylabel('Pre-Sleep', 'Color', [0. 0. 0.])
%set(gca,'ytick',[])
%set(gca,'yticklabel',[])
%colorbar()
ax = gca;
ax.FontSize = 10;
xlabel('Class')

%A4: 21cm x 29.7cm
set(gcf,'units','centimeters','position',[1,1,7.5,7])

%%

