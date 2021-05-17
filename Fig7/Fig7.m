%© 2021. This work is licensed under a CC-BY-NC-SA license.
%"Thalamo-cortical spiking model of incremental learning combining perception, context and NREM-sleep-mediated noise-resilience"
%Authors: Bruno Golosio, Chiara De Luca, Cristiano Capone, Elena Pastorelli, Giovanni Stegel, Gianmarco Tiddia, Giulia De Bonis and Pier Stanislao Paolucci
%arxiv.2003.11859

%% VALUTO UT OT
%clear
cd = '/Users/Mac/Desktop/Golosio2/PlotUTOT/UT_7112019/';

name = strcat(cd, 'events_CNT_Th_0.mat');

load(name)

name = strcat(cd, 'fr0.mat');
load(name)

name = strcat(cd, 'weights0.mat');
load(name)

name = strcat(cd, 'potential.mat');
load(name)

n_exc = 200;
t_train = 1500.;
t_pause = 1500.;
n_train = 10.;
time_tot = (t_pause + t_train)*n_train;
dt = 20; %[ms]
N = n_exc; 
T = floor(time_tot/dt)+20;

%% Compute mean fr
fr_th_medio = zeros([1, numel(fr_inp(1,:))]);

for time = 1:numel(fr_inp(1,:))
    fr_th_medio(time) = mean(fr_inp(:, time)); 
end
fr_exc_medio = zeros([1, numel(fr_exc(1,:))]);

for time = 1:numel(fr_exc(1,:))
    fr_exc_medio(time) = mean(fr_exc(21:40, time)); 
end

%% PLOT fr
time_vector = linspace(-100, time_tot, numel(fr_exc(1,:)));
time_vector = time_vector+100

figure
plot(time_vector/1000., fr_exc_medio(:)*0.84,'LineWidth', 3, 'Color', [0. 0.298 0.6])
title('Thalamic and contextual signals')
xlabel('time[s]')
xlim([1,7])
ylim([0, 60])
ylabel('<fr> [Hz]')
hold on
plot([3, 3],[0, 60], '-.', 'Color', [34./255. 153./255. 84./255.], 'LineWidth', 2.5)
plot([4.8, 4.8],[0, 60], '-.', 'Color', [34./255. 153./255. 84./255.], 'LineWidth', 2.5)
plot([2, 2],[0, 60], '-.', 'Color', [169./255. 50./255. 38./255.], 'LineWidth', 2.5)
plot([6, 6],[0, 60], '-.', 'Color', [169./255. 50./255. 38./255.], 'LineWidth', 2.5)

ax = gca;
ax = gca;
ax.FontSize = 20; 
f=getframe(gcf);                                                         
[X, map] = frame2im(f);                                                  
imwrite(X,'Fig7B_13.bmp');                                            

saveas(gcf,'Fig7B_13','epsc')

%% Plot potential

time_vector = linspace(0, time_tot, numel(V_cnt_th));


Vth = -50.4;
th = ones([1, numel(time_vector)]);
th = th * Vth;

figure
plot(time_vector/1000., V_cnt_th, 'Color',  [0. 0.298 0.6])
hold on
plot(time_vector/1000., th, '-.', 'Color', [0. 0. 0.], 'LineWidth', 3.)
xlabel('time [s]')
ylabel('<V> [mV]')
title('Thalamic and contextual signals')
ylim([-85, -35])
xlim([1,7])
hold on
plot([3, 3],[-90, 60], '-.', 'Color', [34./255. 153./255. 84./255.], 'LineWidth', 2.5)
plot([4.8, 4.8],[-90, 60], '-.', 'Color', [34./255. 153./255. 84./255.], 'LineWidth', 2.5)
plot([1.8,1.8],[-90, 60], '-.', 'Color', [169./255. 50./255. 38./255.], 'LineWidth', 2.5)
plot([6, 6],[-90, 60], '-.', 'Color', [169./255. 50./255. 38./255.], 'LineWidth', 2.5)

ax = gca;
ax = gca;
ax.FontSize = 20; 

time_vector = linspace(0, time_tot, numel(fr_exc(1,:)));
f=getframe(gcf);                                                         
[X, map] = frame2im(f);                                                  
imwrite(X,'Fig7B_23.bmp');                                            

saveas(gcf,'Fig7B_23','epsc')



%% PLOT VALIDATION & TEST
cd = '/Users/Mac/Desktop/Golosio2/PlotUTOT/Att/set1/';

n_exc = 200;
t_train = 1500.;
t_pause = 1500.;
n_train = 10.;
t_check = 1000.;
name = strcat(cd, 'fr1.mat');
load(name)

for time = 1:numel(fr_exc(1,:))
    fr_exc_medio_0(time) = mean(fr_exc(1:21, time)); 
end

for time = 1:numel(fr_exc(1,:))
    fr_exc_medio_1(time) = mean(fr_exc(21:41, time)); 
end

for time = 1:numel(fr_exc(1,:))
    fr_exc_medio_2(time) = mean(fr_exc(41:61, time)); 
end

time_tot = (t_train+t_pause)*n_train + t_pause*10 + (t_check + t_pause)*3  + t_pause*10 + (t_check + t_pause)*3;
time_vector = linspace(0, time_tot, numel(fr_exc(1,:)));

idx_0 = find(time_vector/1000 < 9.65 & time_vector/1000 > 9.45);
xval_00 = ones([1,numel(idx_0)+1]) * mean(fr_exc_medio_0(idx_0));
xval_01 = ones([1,numel(idx_0)+1]) * mean(fr_exc_medio_1(idx_0));
xval_02 = ones([1,numel(idx_0)+1]) * mean(fr_exc_medio_2(idx_0));

idx_1 = find(time_vector/1000 < 10.2 & time_vector/1000 > 10.0);
xval_10 = ones([1,numel(idx_1)+1]) * mean(fr_exc_medio_0(idx_1));
xval_11 = ones([1,numel(idx_1)+1]) * mean(fr_exc_medio_1(idx_1));
xval_12 = ones([1,numel(idx_1)+1]) * mean(fr_exc_medio_2(idx_1));

idx_2 = find(time_vector/1000 < 10.8 & time_vector/1000 > 10.5);
xval_20 = ones([1,numel(idx_2)+1]) * mean(fr_exc_medio_0(idx_2));
xval_21 = ones([1,numel(idx_2)+1]) * mean(fr_exc_medio_1(idx_2));
xval_22 = ones([1,numel(idx_2)+1]) * mean(fr_exc_medio_2(idx_2));


figure
subplot(3,2,1)
plot(time_vector/1000.,fr_exc_medio_0,'LineWidth', 3, 'Color', [0 0.5 1.])
title('Retrieval phase')
xlim([7.0,9.0])
ylim([0,60])
ylabel({'Group 0','<fr> [Hz]'})
ax = gca;
ax.FontSize = 20; 

subplot(3,2,3)
plot(time_vector/1000.,fr_exc_medio_1,'LineWidth', 3, 'Color', [1 0.5 0.])
xlim([7.0,9.0])
ylim([0,60])
ylabel({'Group 1','<fr> [Hz]'})
ax = gca;
ax.FontSize = 20; 

subplot(3,2,5)
plot(time_vector/1000.,fr_exc_medio_2,'LineWidth', 3, 'Color', [0. 0.6 0.3])
xlim([7.0,9.0])
ylim([0,60])
ylabel({'Group 2','<fr> [Hz]'})
xlabel('time [s]')
ax = gca;
ax.FontSize = 20; 

subplot(3,2,2)
plot(time_vector/1000.,fr_exc_medio_0,'LineWidth', 3, 'Color', [0. 0.5 1.])
title('Testing phase')
xlim([9.0,11.0])
ylim([0,60])
ax = gca;
ax.FontSize = 20; 

subplot(3,2,4)
plot(time_vector/1000.,fr_exc_medio_1,'LineWidth', 3, 'Color', [1. 0.5 0.])
xlim([9.0,11.0])
ylim([0,60])
ax = gca;
ax.FontSize = 20; 

subplot(3,2,6)
plot(time_vector/1000.,fr_exc_medio_2,'LineWidth', 3, 'Color', [0. 0.6 0.3])
xlim([9.0,11.0])
ylim([0,60])
xlabel('time [s]')
ax = gca;
ax.FontSize = 20; 

f=getframe(gcf);                                                         
[X, map] = frame2im(f);                                                  
imwrite(X,'Fig7C.bmp');                                            

saveas(gcf,'Fig7C','epsc')

%%


%% VALUTO UT OT

cd = '/Users/Mac/Desktop/PlotUTOT_CntTh/';

name = strcat(cd, 'events0.mat');
load(name)

name = strcat(cd, 'potential.mat');
load(name)

n_exc = 200;
t_train = 1500.;
t_pause = 1500.;
n_train = 10.;
time_tot = (t_pause + t_train)*n_train;
dt = 20; %[ms]
N = n_exc; 
T = floor(time_tot/dt)+20;

%% CNT
%% Compute mean fr
dt = 100.
fr_cnt = []

for time = 0:dt:(16200-dt)
    fr_cnt = [fr_cnt numel(find(evt_cnt < time+dt & evt_cnt > time ))/(dt/1000)];
end

time_fr_cnt = 0:dt:(16200-dt);
time_fr_cnt = time_fr_cnt+0.10
numel(time_fr_cnt)
numel(fr_cnt)
fr_cnt(find(time_fr_cnt/1000. < 11)) = 0;

figure;
plot(time_fr_cnt/1000., fr_cnt, 'Color', [169./255. 50./255. 38./255.], 'LineWidth', 2.5)
xlim([9, 15])
ylim([0, 60])
hold on
plot([13, 13],[0, 60], '-.', 'Color', [34./255. 153./255. 84./255.], 'LineWidth', 2.5)
plot([11, 11],[0, 60], '-.', 'Color', [34./255. 153./255. 84./255.], 'LineWidth', 2.5)
plot([10, 10],[0, 60], '-.', 'Color', [169./255. 50./255. 38./255.], 'LineWidth', 2.5)
plot([14, 14],[0, 60], '-.', 'Color', [169./255. 50./255. 38./255.], 'LineWidth', 2.5)
ax = gca;
ax.FontSize = 20; 
f=getframe(gcf);                                                         
[X, map] = frame2im(f);                                                  
imwrite(X,'Fig7B_21.bmp');                                            

saveas(gcf,'Fig7B_21','epsc')

%% Plot potential

Vth = -50.4;
%th = ones([1, numel(time_vector)]);
%th = th * Vth;

figure
figure; plot(V_cnt_times/1000, V_cnt, 'Color', [169./255. 50./255. 38./255.])
%plot(time_vector/1000., V_100, 'Color',  [0. 0.298 0.6])
hold on
%plot(time_vector/1000., th, '-.', 'Color', [0. 0. 0.], 'LineWidth', 3.)
xlabel('time [s]')
ylabel('<V> [mV]')
title('Contextual signals')
plot([0, 20],[Vth, Vth], '-.', 'Color', [0. 0. 0.], 'LineWidth', 2.5)
hold on
plot([13, 13],[-90, 60], '-.', 'Color', [34./255. 153./255. 84./255.], 'LineWidth', 2.5)
plot([11, 11],[-90, 60], '-.', 'Color', [34./255. 153./255. 84./255.], 'LineWidth', 2.5)
plot([10, 10],[-90, 60], '-.', 'Color', [169./255. 50./255. 38./255.], 'LineWidth', 2.5)
plot([14, 14],[-90, 60], '-.', 'Color', [169./255. 50./255. 38./255.], 'LineWidth', 2.5)

ylim([-85, -35])
xlim([9,15])
ax = gca;
ax.FontSize = 20; 

%time_vector = linspace(0, time_tot, numel(fr_exc(1,:)));
f=getframe(gcf);                                                         
[X, map] = frame2im(f); 
f=getframe(gcf);                                                         
[X, map] = frame2im(f);                                                  
imwrite(X,'Fig7B_22.bmp');                                            

saveas(gcf,'Fig7B_22','epsc')
%imwrite(X,'Fig7B_23.bmp');                                            

%saveas(gcf,'Fig7B_23','epsc')


%% TH
%% Compute mean fr
dt = 130.
fr_th = []

for time = 0:dt:(16200-dt)
    fr_th = [fr_th numel(find(evt_th < time+dt & evt_th > time ))/(dt/1000)];
end

time_fr_th = 0:dt:(16200-dt);
time_fr_th = time_fr_th+0.10
numel(time_fr_cnt)
numel(fr_cnt)
%fr_th(find(time_fr_th/1000. < 11)) = 0;

figure;
plot(time_fr_th/1000., fr_th, 'Color', [31./255. 97./255. 41./255.], 'LineWidth', 2.5)
xlim([9, 15])
ylim([0, 60])
hold on
plot([13, 13],[0, 60], '-.', 'Color', [34./255. 153./255. 84./255.], 'LineWidth', 2.5)
plot([11, 11],[0, 60], '-.', 'Color', [34./255. 153./255. 84./255.], 'LineWidth', 2.5)
plot([10, 10],[0, 60], '-.', 'Color', [169./255. 50./255. 38./255.], 'LineWidth', 2.5)
plot([14, 14],[0, 60], '-.', 'Color', [169./255. 50./255. 38./255.], 'LineWidth', 2.5)
ax = gca;
ax.FontSize = 20; 
f=getframe(gcf);                                                         
[X, map] = frame2im(f);                                                  
imwrite(X,'Fig7B_11.bmp');                                            
saveas(gcf,'Fig7B_11','epsc')

%% Plot potential

Vth = -50.4;
%th = ones([1, numel(time_vector)]);
%th = th * Vth;

figure
figure; plot(V_th_times/1000, V_th, 'Color', [31./255. 97./255. 41./255.])
%plot(time_vector/1000., V_100, 'Color',  [0. 0.298 0.6])
hold on
%plot(time_vector/1000., th, '-.', 'Color', [0. 0. 0.], 'LineWidth', 3.)
xlabel('time [s]')
ylabel('<V> [mV]')
title('Thalamic signal only')
plot([0, 25],[Vth, Vth], '-.', 'Color', [0. 0. 0.], 'LineWidth', 2.5)
ylim([-85, -35])
xlim([16, 22])
hold on
plot([20, 20],[-90, 60], '-.', 'Color', [34./255. 153./255. 84./255.], 'LineWidth', 2.5)
plot([18, 18],[-90, 60], '-.', 'Color', [34./255. 153./255. 84./255.], 'LineWidth', 2.5)
plot([17, 17],[-90, 60], '-.', 'Color', [169./255. 50./255. 38./255.], 'LineWidth', 2.5)
plot([21, 21],[-90, 60], '-.', 'Color', [169./255. 50./255. 38./255.], 'LineWidth', 2.5)

ax = gca;
ax = gca;
ax.FontSize = 20; 

%time_vector = linspace(0, time_tot, numel(fr_exc(1,:)));
f=getframe(gcf);                                                         
[X, map] = frame2im(f);                                                  
imwrite(X,'Fig7B_21.bmp');                                            
saveas(gcf,'Fig7B_21','epsc')