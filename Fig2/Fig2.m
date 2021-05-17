%© 2021. This work is licensed under a CC-BY-NC-SA license.
%"Thalamo-cortical spiking model of incremental learning combining perception, context and NREM-sleep-mediated noise-resilience"
%Authors: Bruno Golosio, Chiara De Luca, Cristiano Capone, Elena Pastorelli, Giovanni Stegel, Gianmarco Tiddia, Giulia De Bonis and Pier Stanislao Paolucci
%arxiv.2003.11859
mean_fr_exc_test_pre = [];
mean_fr_exc_test_post = [];
mean_fr_exc_sleep = [];
mean_fr_exc_sleep_first = [];
mean_fr_exc_sleep_last = [];
mean_fr_exc_sleep_first20 = [];
    
set_tot = 20

cycle = 5
sleep_act_test = zeros([6, cycle+2, set_tot]);
errors_test = zeros([6, cycle+2, set_tot]);
%1,711; 2, 710; 

for trial = 1:set_tot
    mean_fr_exc_test_pre_set = [];

    cd = '/Users/Mac/Desktop/SleepPlot_09032020/Std_40.0/alpha2.0/b250.0/tau400.0/inh-0.7/Rate_900.0/set';
    cd = strcat(cd, num2str(trial-1));
    cd = strcat(cd, '/');


    %% Faccio il loading e costituisco il raster plot


    load(strcat(cd, 'events_CNT_Th_5.mat'));

    evt_exc = evt_exc_cnt_th;
    evt_exc = evt_exc(1:800);

    y_exc = [];
    x_exc = [];

    for neur = 1:numel(evt_exc(:))

        y_exc = [y_exc evt_exc{:,neur}/1000];
        bha = ones([1 numel(evt_exc{:,neur})]) * neur;
        x_exc = [x_exc bha];

    end


    %% misuro il fr

    ts = 0.2; % con 0.05, 650 va bene!
    fr_exc = zeros([max(x_exc) floor(max(y_exc)/ts)]);

    for neur = 1:max(x_exc)
        times = [];
        times_bin = [];
        idx_neur = find(x_exc == neur);
        times = y_exc(idx_neur);

        times_bin = floor(times / ts);
        [fr, edges] = histcounts(times, 'NumBins', floor(max(y_exc)/ts));%, 'BinWidth', ts);
        %fr =  np.bincount(times_bin) / DT*1000
        idx = floor(edges/ts)+1;
        fr_exc(neur,idx(1:end-1)) = fr/ts;
    end

    x_fr = 1:ts:max(y_exc);



    %% tempi

    min_time_test_pre = 186.0;
    max_time_test_pre = 436.0;


    min_time_sleep = 451.0;
    sleep_interval = 100.;
    cycle = 10;
    max_time_sleep = min_time_sleep + sleep_interval * cycle;

    min_time_test_post = 1466.0;
    max_time_test_post = 1761.0;


    time_idx_awake = [];
    for t=min_time_test_pre:2.5:max_time_test_pre
        time_idx_awake = [time_idx_awake t:ts:t+1.0];
    end
    time_idx_awake_post = [];
    for t=min_time_test_post:2.5:max_time_test_post
        time_idx_awake_post = [time_idx_awake_post t:ts:t+1.0];
    end
    

    %% mi salvo anche i time_idx


    time_idx_test_pre = find(x_fr > min_time_test_pre  & x_fr < max_time_test_pre);
    [bha, time_idx_test_pre] = intersect(x_fr, time_idx_awake,'stable');

    [bha, time_idx_test_post] = intersect(x_fr, time_idx_awake_post,'stable');

    time_idx_sleep = find(x_fr >= min_time_sleep+0 & x_fr <= max_time_sleep);%max_time_sleep);
    time_idx_sleep_first = find(x_fr >= min_time_sleep+0 & x_fr <= min_time_sleep+100);%max_time_sleep);
    time_idx_sleep_first20 = find(x_fr >= min_time_sleep+0 & x_fr <= min_time_sleep+20);%max_time_sleep);
    time_idx_sleep_last = find(x_fr >= max_time_sleep-100 & x_fr <= max_time_sleep);%max_time_sleep);


    %%  CALCOLO IL FR neuroni appesi consecutivi

    for neur = 1:numel(fr_exc(:,1))
        mean_fr_exc_test_pre_set = [mean_fr_exc_test_pre_set mean((fr_exc(neur,time_idx_test_pre)))];

        mean_fr_exc_test_pre = [mean_fr_exc_test_pre mean((fr_exc(neur,time_idx_test_pre)))];
        mean_fr_exc_sleep = [mean_fr_exc_sleep mean((fr_exc(neur,time_idx_sleep)))];
        mean_fr_exc_sleep_first = [mean_fr_exc_sleep_first mean((fr_exc(neur,time_idx_sleep_first)))];
        mean_fr_exc_sleep_first20 = [mean_fr_exc_sleep_first20 mean((fr_exc(neur,time_idx_sleep_first20)))];
        mean_fr_exc_test_post = [mean_fr_exc_test_post mean((fr_exc(neur,time_idx_test_post)))];
        mean_fr_exc_sleep_last = [mean_fr_exc_sleep_last mean((fr_exc(neur,time_idx_sleep_last)))];

    end

    idx_non_morti = numel(find(mean_fr_exc_test_pre_set == 0.))
    %mean_fr_exc_test_pre_set = mean_fr_exc_test_pre_set(idx_non_morti);

    %idx_non_morti = find(mean_fr_exc_test_pre > 0.01);
    %mean_fr_exc_test_pre = mean_fr_exc_test_pre(idx_non_morti);
    %mean_fr_exc_sleep = mean_fr_exc_sleep(idx_non_morti);
    %mean_fr_exc_sleep_first = mean_fr_exc_sleep_first(idx_non_morti);
    %mean_fr_exc_sleep_first20 = mean_fr_exc_sleep_first20(idx_non_morti);
    %mean_fr_exc_test_post = mean_fr_exc_test_post(idx_non_morti);
    %mean_fr_exc_sleep_last = mean_fr_exc_sleep_last(idx_non_morti);

    %% sixtile DALLA CUMULATIVA - Testing
    num_sixtile = floor(numel(mean_fr_exc_test_pre_set)/6);
    sixtile_idx_test = zeros([6, num_sixtile]);

    [sorted, idx] = sort(mean_fr_exc_test_pre_set);

    sixtile_idx_test(1,:) = idx(1:num_sixtile);
    sixtile_idx_test(2,:) = idx(num_sixtile+1: 2*num_sixtile);
    sixtile_idx_test(3,:) = idx(2*num_sixtile+1: 3*num_sixtile);
    sixtile_idx_test(4,:) = idx(3*num_sixtile+1: 4*num_sixtile);
    sixtile_idx_test(5,:) = idx(4*num_sixtile+1: 5*num_sixtile);
    sixtile_idx_test(6,:) = idx(5*num_sixtile+1: 6*num_sixtile);

    
    %% plot fr  del sixtile dalla cumulativa

    for c = 1:cycle
        time_idx = find(x_fr > min_time_sleep + (c-1)*sleep_interval & x_fr < min_time_sleep+c*sleep_interval);
        sleep_act_test(1, c+1, trial) = mean(mean(fr_exc(sixtile_idx_test(1,:),time_idx)));
        sleep_act_test(2, c+1, trial) = mean(mean(fr_exc(sixtile_idx_test(2,:),time_idx)));
        sleep_act_test(3, c+1, trial) = mean(mean(fr_exc(sixtile_idx_test(3,:),time_idx)));
        sleep_act_test(4, c+1, trial) = mean(mean(fr_exc(sixtile_idx_test(4,:),time_idx)));
        sleep_act_test(5, c+1, trial) = mean(mean(fr_exc(sixtile_idx_test(5,:),time_idx)));
        sleep_act_test(6, c+1, trial) = mean(mean(fr_exc(sixtile_idx_test(6,:),time_idx)));
    end

    sleep_act_test(1, 1, trial) = mean(mean(fr_exc(sixtile_idx_test(1,:),time_idx_test_pre)));
    sleep_act_test(2, 1, trial) = mean(mean(fr_exc(sixtile_idx_test(2,:),time_idx_test_pre)));
    sleep_act_test(3, 1, trial) = mean(mean(fr_exc(sixtile_idx_test(3,:),time_idx_test_pre)));
    sleep_act_test(4, 1, trial) = mean(mean(fr_exc(sixtile_idx_test(4,:),time_idx_test_pre)));
    sleep_act_test(5, 1, trial) = mean(mean(fr_exc(sixtile_idx_test(5,:),time_idx_test_pre)));
    sleep_act_test(6, 1, trial) = mean(mean(fr_exc(sixtile_idx_test(6,:),time_idx_test_pre)));


    sleep_act_test(1, cycle+2, trial) = mean(mean(fr_exc(sixtile_idx_test(1,:),time_idx_test_post)));
    sleep_act_test(2, cycle+2, trial) = mean(mean(fr_exc(sixtile_idx_test(2,:),time_idx_test_post)));
    sleep_act_test(3, cycle+2, trial) = mean(mean(fr_exc(sixtile_idx_test(3,:),time_idx_test_post)));
    sleep_act_test(4, cycle+2, trial) = mean(mean(fr_exc(sixtile_idx_test(4,:),time_idx_test_post)));
    sleep_act_test(5, cycle+2, trial) = mean(mean(fr_exc(sixtile_idx_test(5,:),time_idx_test_post)));
    sleep_act_test(6, cycle+2, trial) = mean(mean(fr_exc(sixtile_idx_test(6,:),time_idx_test_post)));


end
%%

idx_morti = find(mean_fr_exc_test_pre > 0.001);
mean_fr_exc_test_pre_reduced = mean_fr_exc_test_pre(idx_morti);
idx_morti = find(mean_fr_exc_test_post > 0.001);
mean_fr_exc_test_post_reduced = mean_fr_exc_test_post(idx_morti);

%% sixtile DALLA CUMULATIVA - Testing MEDIO su i set
num_sixtile = floor(numel(mean_fr_exc_test_pre)/6);
sixtile_idx_test = zeros([6, num_sixtile]);

[sorted, idx] = sort(mean_fr_exc_test_pre);

sixtile_idx_test(1,:) = idx(1:num_sixtile);
sixtile_idx_test(2,:) = idx(num_sixtile+1: 2*num_sixtile);
sixtile_idx_test(3,:) = idx(2*num_sixtile+1: 3*num_sixtile);
sixtile_idx_test(4,:) = idx(3*num_sixtile+1: 4*num_sixtile);
sixtile_idx_test(5,:) = idx(4*num_sixtile+1: 5*num_sixtile);
sixtile_idx_test(6,:) = idx(5*num_sixtile+1: 6*num_sixtile);


%% Plot Figure 2A

figure
ops = cdfplot(mean_fr_exc_test_pre_reduced)
hold on
set(ops, 'LineWidth', 2.5, 'Color', [0. 0. 0.]);
ops = cdfplot(mean_fr_exc_sleep_first20)
set( ops, 'LineWidth', 2.5, 'Color',[0. 0. 0.8039]);%
%ops = cdfplot(log10(mean_fr_exc_sleep_first75))
grid off
hold on
xlim([0.001, 10])
set(gca,'XScale','log')
%legend('Wake', 'nonREM', 'sextile1', 'sextile2', 'sextile3', 'sextile4', 'sextile5', 'sextile6')
plot([mean_fr_exc_test_pre(sixtile_idx_test(2,1)) mean_fr_exc_test_pre(sixtile_idx_test(2,1))],[0 1], 'Color', [0.1333 0.5451 0.1333], 'LineWidth', 2.)
plot([mean_fr_exc_test_pre(sixtile_idx_test(3,1)) mean_fr_exc_test_pre(sixtile_idx_test(3,1))],[0 1], 'Color', [0.1333 0.5451 0.1333], 'LineWidth', 2.)
plot([mean_fr_exc_test_pre(sixtile_idx_test(4,1)) mean_fr_exc_test_pre(sixtile_idx_test(4,1))],[0 1], 'Color', [0.1333 0.5451 0.1333], 'LineWidth', 2.)
plot([mean_fr_exc_test_pre(sixtile_idx_test(5,1)) mean_fr_exc_test_pre(sixtile_idx_test(5,1))],[0 1], 'Color', [0.1333 0.5451 0.1333], 'LineWidth', 2.)
plot([mean_fr_exc_test_pre(sixtile_idx_test(6,1)) mean_fr_exc_test_pre(sixtile_idx_test(6,1))],[0 1], 'Color',  [0.1333 0.5451 0.1333], 'LineWidth', 2.)
xlabel('f.r. [Hz]')
ylabel('Cumulative unit count')
xticks([0.001 0.01 0.1 1 10])
xticklabels({'0.001', '0.01', '0.1', '1', '10'})
%title('Cumulative distribution of the firing rates_awake post sleep')
title(' ')
legend('Wake', 'nonREM');
ax = gca;
ax.TickLength = [0.03, 0.03];
ax.YColor= [0. 0. 0.];
ax.FontSize = 10;
%A4: 21cm x 29.7cm
set(gcf,'units','centimeters','position',[1,1,7,7])


%% Plot figure 2B

%mean_fr_exc_test_pre(find(mean_fr_exc_test_pre == 0)) = 10^-2;
%mean_fr_exc_sleep(find(mean_fr_exc_sleep == 0)) = 10^-3;

A = (mean_fr_exc_test_pre);
B = (mean_fr_exc_sleep);

coef_fr_test_sleep = polyfit(log10(A),log10(B(1:numel(A))),1);

x_fit = -3:0.1:1;
y_fit = coef_fr_test_sleep(1)*x_fit + coef_fr_test_sleep(2);

figure
plot(log10(mean_fr_exc_test_pre), log10(mean_fr_exc_sleep(1:numel(A))), '.', 'Color',[0.0941 0.4157 0.2314])
hold on
xlim([-3, 1])
ylim([-3, 1])
plot([-3 1], [-3 1], 'Color', [0. 0. 0.])
hold on
plot(x_fit, y_fit, '--', 'Color', [0. 0.3922 0.], 'LineWidth', 2.5)
xlabel('Wake firing rate [log_1_0(Hz)]')
ylabel('nonREM firing rate [log_1_0(Hz)]')
title('fr medio pre-classification vs Sleep first')
%set(gca,'XScale','log')
%set(gca,'YScale','log')
ax = gca;
ax.FontSize = 10;
set(gcf,'units','centimeters','position',[1,1,7,7])

%% Plot figure 3D

%mean_fr_exc_sleep_first(find(mean_fr_exc_sleep_last == 0)) = 10^-2;
%mean_fr_exc_sleep_last(find(mean_fr_exc_sleep_first == 0)) = 10^-3;

A = (mean_fr_exc_sleep_first);
B = (mean_fr_exc_sleep_last);

coef_fr_test_sleep = polyfit(log10(A),log10(B(1:numel(A))),1);

x_fit = -3:0.1:1;
y_fit = coef_fr_test_sleep(1)*x_fit + coef_fr_test_sleep(2);

figure
plot(log10(mean_fr_exc_sleep_first), log10(mean_fr_exc_sleep_last), '.', 'Color', [0.1804 0.8 0.4431])
hold on
xlim([-3, 1])
ylim([-3, 1])
plot([-3 1], [-3 1], 'Color', [0. 0. 0.])
hold on
plot(x_fit, y_fit, '--', 'Color', [0.0941 0.4157 0.2314], 'LineWidth', 2.5)
xlabel('nonREM-First packet firing rate  [Hz]')
ylabel('nonREM-Last packet firing rate  [Hz]')
%xticks([-3 -2 -1 0 1])
%xticklabels({'0.001', '0.01', '0.1', '1', '10'})
%yticks([-3 -2 -1 0 1])
%yticklabels({'0.001', '0.01', '0.1', '1', '10'})
%title('fr medio Sleep first vs Sleep last')
%set(gca,'XScale','log')
%set(gca,'YScale','log')
ax = gca;
ax.TickLength = [0.03, 0.03];
ax.YColor= [0. 0. 0.];
ax.FontSize = 10;
set(gcf,'units','centimeters','position',[1,1,7,7])

%% Plot figure 3D

%mean_fr_exc_sleep_first(find(mean_fr_exc_sleep_last == 0)) = 10^-2;
%mean_fr_exc_sleep_last(find(mean_fr_exc_sleep_first == 0)) = 10^-3;

A = (mean_fr_exc_sleep_first);
B = (mean_fr_exc_sleep_last);

coef_fr_test_sleep = polyfit(log10(A),log10(B(1:numel(A))),1);

x_fit = -3:0.1:1;
y_fit = coef_fr_test_sleep(1)*x_fit + coef_fr_test_sleep(2);

figure
plot(mean_fr_exc_sleep_first, mean_fr_exc_sleep_last, '.', 'Color', [0.1804 0.8 0.4431])
hold on
xlim([0.001, 10])
ylim([0.001, 10])
plot([0.001 10], [0.001 10], 'Color', [0. 0. 0.])
hold on
plot(10.^(x_fit), 10.^(y_fit), '--', 'Color', [0.0941 0.4157 0.2314], 'LineWidth', 2.5)
xlabel('nonREM-First packet firing rate  [Hz]')
ylabel('nonREM-Last packet firing rate  [Hz]')
xticks([0.001 0.01 0.1 1 10])
%xticklabels({'0.001', '0.01', '0.1', '1', '10'})
%yticks([-3 -2 -1 0 1])
%yticklabels({'0.001', '0.01', '0.1', '1', '10'})
%title('fr medio Sleep first vs Sleep last')
set(gca,'XScale','log')
set(gca,'YScale','log')
ax = gca;
ax.TickLength = [0.03, 0.03];
ax.FontSize = 10;
set(gcf,'units','centimeters','position',[1,1,7,7])

%%
minimum = min(log10(mean_fr_exc_test_pre));
minimum2 = min(log10(mean_fr_exc_test_post));
maximum = max(log10(mean_fr_exc_test_pre));
maximum2 = max(log10(mean_fr_exc_test_post));

figure
subplot(2,1,1)
histogram(log10(mean_fr_exc_test_pre),20)
title('Awake pre-sleep')
hold on
plot([median(log10(mean_fr_exc_test_pre)) median(log10(mean_fr_exc_test_pre))], [0 3000], 'LineWidth', 2.)
plot([median(log10(mean_fr_exc_test_pre))-std(mean_fr_exc_test_pre)/sqrt(numel(mean_fr_exc_test_pre)) median(log10(mean_fr_exc_test_pre))+std(mean_fr_exc_test_post)/sqrt(numel(mean_fr_exc_test_pre))], [900 900], 'LineWidth', 2.)
xlim([-2, 1])
ax = gca;
ax.FontSize = 20;
subplot(2,1,2)
histogram(log10(mean_fr_exc_test_post),20)
title('Awake post-sleep')
hold on
plot([median(log10(mean_fr_exc_test_post)) median(log10(mean_fr_exc_test_post))], [0 3000], 'LineWidth', 2.)
plot([median(log10(mean_fr_exc_test_post))-std(mean_fr_exc_test_post)/sqrt(numel(mean_fr_exc_test_post)) median(log10(mean_fr_exc_test_post))+std(mean_fr_exc_test_post)/sqrt(numel(mean_fr_exc_test_post))], [900 900], 'LineWidth', 2.)
xlim([-2, 1])
ax = gca;
ax.FontSize = 20;
%%
%%

sleep_act_test_medio = zeros([6, cycle+2]);
errors_test_medio = zeros([6, cycle+2]);

for int = 1:cycle+2
    sleep_act_test_medio(1,int) = mean(sleep_act_test(1,int,:));
    sleep_act_test_medio(2,int) = mean(sleep_act_test(2,int,:));
    sleep_act_test_medio(3,int) = mean(sleep_act_test(3,int,:));
    sleep_act_test_medio(4,int) = mean(sleep_act_test(4,int,:));
    sleep_act_test_medio(5,int) = mean(sleep_act_test(5,int,:));
    sleep_act_test_medio(6,int) = mean(sleep_act_test(6,int,:));

end



x = 0:cycle+1;
%x = x*(200.)-100.;
x = x*(100.)-50.;


%% Plot  Figure 3A



fig = figure
set(fig,'defaultAxesColorOrder',[[0. 0.3922 0.]; [0.4863 0.9882 0.]]);
yyaxis left
errorbar(x(2:end-1),sleep_act_test_medio(1,2:end-1), errors_test_medio(1,2:end -1),'Marker', 'o', 'LineWidth', 2.5, 'Color', [0. 0.3922 0.])
hold on
ylim([0.27, 0.3])
ylabel('f.r. lowest sextile [Hz]')
yyaxis right
errorbar(x(2:end-1),sleep_act_test_medio(6,2:end-1), errors_test_medio(6,2:end-1),'Marker', 'o', 'LineWidth', 2.5, 'Color', [0.4863 0.9882 0.])
legend(strcat('Sextile 1 (lowest)'), strcat('Sextile 6 (highest)'))
%plot([0 0], [0 2.5], 'LineWidth', 1.5, 'Color',[0. 0. 0.])
%plot([1000 1000], [0.001 2.5], 'LineWidth', 1.5, 'Color', [0. 0. 0.])
%xlim([0, 500])
ylim([0.2, 0.75])

xlabel('time sleep [s]')
ylabel('f.r. highest sextile [Hz]')
%set(gca,'YScale','log')
%xlim([0, 900])
title('Sextile (AWAKE-TEST) mean firing rates during sleep')
ax = gca;
ax.FontSize = 20;

%% Plot  Figure 3A
 
fig = figure
set(fig,'defaultAxesColorOrder',[[0. 0.3922 0.]; [0.4863 0.9882 0.]]);
errorbar(x(2:end-1),log(sleep_act_test_medio(1,2:end-1)), errors_test_medio(1,2:end -1),'Marker', 'o', 'LineWidth', 2.5, 'Color', [0. 0.3922 0.])
hold on
errorbar(x(2:end-1),log(sleep_act_test_medio(6,2:end-1)), errors_test_medio(6,2:end-1),'Marker', 'o', 'LineWidth', 2.5, 'Color', [0.4863 0.9882 0.])
legend(strcat('Sextile 1 (lowest)'), strcat('Sextile 6 (highest)'))
%plot([0 0], [0 2.5], 'LineWidth', 1.5, 'Color',[0. 0. 0.])
%plot([1000 1000], [0.001 2.5], 'LineWidth', 1.5, 'Color', [0. 0. 0.])
%xlim([0, 500])
xlabel('time sleep [s]')
set(gca,'YScale','log')
%xlim([0, 900])
title('Sextile (AWAKE-TEST) mean firing rates during sleep')
ax = gca;
ax.FontSize = 20;


%% Plot  Figure 3B_ sotto

figure
errorbar(x,sleep_act_test_medio(1,:), errors_test_medio(1,:),'Marker', 'o', 'LineWidth', 2.5, 'Color', [0. 0.3922 0.])
hold on
errorbar(x,sleep_act_test_medio(2,:), errors_test_medio(2,:),'Marker', 'o', 'LineWidth', 0.5, 'Color', [0. 0.5020 0.])
errorbar(x,sleep_act_test_medio(3,:), errors_test_medio(3,:),'Marker', 'o', 'LineWidth', 0.5, 'Color', [0.1333 0.5451 0.1333])
errorbar(x,sleep_act_test_medio(4,:), errors_test_medio(4,:),'Marker', 'o', 'LineWidth', 0.5, 'Color', [0.1961 0.8039 0.1961])
errorbar(x,sleep_act_test_medio(5,:), errors_test_medio(5,:),'Marker', 'o', 'LineWidth', 0.5, 'Color', [0.4980 1. 0.])
errorbar(x,sleep_act_test_medio(6,:), errors_test_medio(6,:),'Marker', 'o', 'LineWidth', 2.5, 'Color', [0.4863 0.9882 0.])
legend(strcat('Sextile 1 (lowest)'), strcat('Sextile 2'), strcat('Sextile 3'), strcat('Sextile 4'), strcat('Sextile 5'), strcat('Sextile 6 (highest)'))
%plot([0 0], [0 2.], 'LineWidth', 1.5, 'Color',[0. 0. 0.])
%plot([1000 1000], [0.001 2.], 'LineWidth', 1.5, 'Color', [0. 0. 0.])
xlim([-100, 1100])
ylim([0.05, 10])
xlabel('time sleep [s]')
ylabel('f.r. [Hz]')
set(gca,'YScale','log')

%xlim([0, 900])
%title('Sextile (AWAKE-TEST) mean firing rates during sleep')
ax = gca;
ax.TickLength = [0.03, 0.03];
ax.YColor= [0. 0. 0.];
ax.FontSize = 10;
set(gcf,'units','centimeters','position',[1,1,12,7])

%% Plot figure 3B sopra

mean_fr = []

for time = max_time_test_pre - 200: 20: max_time_test_pre- 20
   time_idx = find(x_fr > time & x_fr < time + 100);
   [time_idx, bha] = intersect(time_idx, time_idx_test_pre);
   m = mean(fr_exc(:, time_idx));
   mean_fr = [mean_fr mean(m)]
end
mean_fr(1) = mean_fr(1)
mean_fr(2) = mean_fr(2)
mean_fr(3) = mean_fr(3)
mean_fr(4) = mean_fr(4)
mean_fr(9) = mean_fr(9)

mean_fr = mean_fr-0.1

for time = min_time_sleep: 20: max_time_sleep-20
   time_idx = find(x_fr > time & x_fr < time + 20);
   m = mean(fr_exc(:, time_idx));
   mean_fr = [mean_fr mean(m)];
end


for time = min_time_test_post+20: 20: min_time_test_post+ 200
   time_idx = find(x_fr > time & x_fr < time + 50);
   [time_idx, bha] = intersect(time_idx, time_idx_test_post);
   m = mean(fr_exc(:, time_idx));
   mean_fr = [mean_fr mean(m)]
end


x_fr_sleep = 1:numel(mean_fr);
%x_fr_sleep = [max_time_test_pre-200:20:max_time_test_pre min_time_sleep:20:max_time_sleep];

inizio = numel(max_time_test_pre-200:20:max_time_test_pre-20);

sleep_interval = 5;
figure
plot(x_fr_sleep, mean_fr, 'LineWidth', 2.5, 'Color',   [0.1333 0.5451 0.1333])
%xlim([min_time_sleep-40, max_time_sleep+40])
%xticks([min_time_sleep min_time_sleep+sleep_interval  min_time_sleep+2*sleep_interval  min_time_sleep+3*sleep_interval  min_time_sleep+4*sleep_interval min_time_sleep+5*sleep_interval min_time_sleep+sleep_interval*6 min_time_sleep+sleep_interval*7 min_time_sleep+sleep_interval*8 min_time_sleep+sleep_interval*9 min_time_sleep+sleep_interval*10])
xticks([inizio inizio+sleep_interval  inizio+2*sleep_interval  inizio+3*sleep_interval  inizio+4*sleep_interval inizio+5*sleep_interval inizio+sleep_interval*6 inizio+sleep_interval*7 inizio+sleep_interval*8 inizio+sleep_interval*9 inizio+sleep_interval*10])
xticklabels({'0', '100', '200', '300', '400', '500', '600', '700', '800', '900', '1000'})
hold on
plot([10 10], [0 1.5], 'LineWidth', 1.5, 'Color',[0. 0. 0.])
plot([inizio+sleep_interval*10 inizio+sleep_interval*10], [0 1.5], 'LineWidth', 1.5, 'Color', [0. 0. 0.])
plot([0 inizio+sleep_interval*10], [mean(mean_fr_exc_test_pre) mean(mean_fr_exc_test_pre)], 'LineWidth', 2.5, 'Color', [1. 0.4980 0.3137])
plot([10 inizio+sleep_interval*10 + 10], [mean(mean_fr_exc_test_post) mean(mean_fr_exc_test_post)], 'LineWidth', 2.5, 'Color', [0.5451 0. 0.5451])
ylim([0.4, 1.4])
%legend('mean fr', 'inizio sleep', 'fine sleep', 'mean fr test pre', 'mean fr test post')
%set(gca,'YScale','log')
xlabel('time sleep [s]')
ylabel('f.r. [Hz]')
%title('cx population mean fr during nonREM')
ax = gca;
ax.FontSize = 10;
set(gcf,'units','centimeters','position',[1,1,12,7])



%% CALCOLO LE CORRELAZIONI

[correlation_test_sleep, pval_test_sleep] = corr(mean_fr_exc_test_pre.', mean_fr_exc_sleep.')


mean_fr = []
for time = min_time_sleep: 20: max_time_sleep-20
   time_idx = find(x_fr > time & x_fr < time + 20);
   m = mean(fr_exc(:, time_idx));
   mean_fr = [mean_fr mean(m)];
end


x_fr_sleep = 1:numel(mean_fr);
x_fr_sleep = x_fr_sleep*20-20;
x_fr_sleep = x_fr_sleep/max(x_fr_sleep)

[correlation_sleep_time, pval_sleep_time] = corr(mean_fr.', x_fr_sleep.')


x(2:end-1) = x(2:end-1)/max(x(2:end-1));
[correlation_sleep_time_s1, pval_sleep_time_s1] = corr(x(2:end-1).', sleep_act_test_medio(1,2:end-1).')
[correlation_sleep_time_s2, pval_sleep_time_s2] = corr(x(2:end-1).', sleep_act_test_medio(2,2:end-1).')
[correlation_sleep_time_s3, pval_sleep_time_s3] = corr(x(2:end-1).', sleep_act_test_medio(3,2:end-1).')
[correlation_sleep_time_s4, pval_sleep_time_s4] = corr(x(2:end-1).', sleep_act_test_medio(4,2:end-1).')
[correlation_sleep_time_s5, pval_sleep_time_s5] = corr(x(2:end-1).', sleep_act_test_medio(5,2:end-1).')
[correlation_sleep_time_s6, pval_sleep_time_s6] = corr(x(2:end-1).', sleep_act_test_medio(6,2:end-1).')


%% slope regression

A = (mean_fr_exc_sleep_first);
B = (mean_fr_exc_sleep_last);

[coef_fr_test_sleep, S] = polyfit(log10(A),log10(B(1:numel(A))),1)
polyparci(coef_fr_test_sleep, S)

%% slope regression

A = (mean_fr_exc_test_pre);
B = (mean_fr_exc_sleep);

[coef_fr_test_sleep, S] = polyfit(log10(A),log10(B(1:numel(A))),1)
polyparci(coef_fr_test_sleep, S)

%%

figure
subplot(2,1,1)
h = histogram(log10(mean_fr_exc_test_pre_reduced),'BinWidth', 0.1, 'EdgeColor', [1. 0.4980 0.3137], 'FaceColor', [1. 0.4980 0.3137])
hold on
[ops ,moda_idx] = max(h.Values);
moda = h.BinEdges(moda_idx) + h.BinWidth/2.;
plot([median(log10(mean_fr_exc_test_pre_reduced)) median(log10(mean_fr_exc_test_pre_reduced))], [0 3000], 'LineWidth', 2.)
plot([moda moda], [0 3000], 'LineWidth', 2.)
plot([mean(log10(mean_fr_exc_test_pre_reduced)) mean(log10(mean_fr_exc_test_pre_reduced))], [0 3000], 'LineWidth', 2.)
plot([mean(log10(mean_fr_exc_test_pre_reduced))-std(log10(mean_fr_exc_test_pre_reduced)) mean(log10(mean_fr_exc_test_pre_reduced))+std(log10(mean_fr_exc_test_post_reduced))], [2000 2000], 'LineWidth', 2.)
[Q,IQR] = quartile(log10(mean_fr_exc_test_pre_reduced))
plot([Q(1) Q(1)], [0 3000], 'LineWidth', 2.)
plot([Q(2) Q(2)], [0 3000], 'LineWidth', 2.)
plot([Q(3) Q(3)], [0 3000], 'LineWidth', 2.)
%plot([IQR IQR], [0 3000], 'LineWidth', 2.)

xlim([-2, 1])
%title('Awake - pre sleep')
ax = gca;
ax.FontSize = 20;


subplot(2,1,2)
h = histogram(log10(mean_fr_exc_test_post_reduced),'BinWidth', 0.1, 'EdgeColor', [0.5451 0. 0.5451], 'FaceColor', [0.5451 0. 0.5451])
hold on
[ops ,moda_idx] = max(h.Values);
moda = h.BinEdges(moda_idx) + h.BinWidth/2.;
xticks([-2 -1 0 1.])
xticklabels({'0.01', '0.1', '1', '10'})
plot([median(log10(mean_fr_exc_test_post_reduced)) median(log10(mean_fr_exc_test_post_reduced))], [0 3000], 'LineWidth', 2.)
plot([moda moda], [0 3000], 'LineWidth', 2.)
plot([mean(log10(mean_fr_exc_test_post_reduced)) mean(log10(mean_fr_exc_test_post_reduced))], [0 3000], 'LineWidth', 2.)
plot([mean(log10(mean_fr_exc_test_post_reduced))-std(log10(mean_fr_exc_test_post_reduced)) mean(log10(mean_fr_exc_test_post_reduced))+std(log10(mean_fr_exc_test_post_reduced))], [2000 2000], 'LineWidth', 2.)
[Q,IQR] = quartile(log10(mean_fr_exc_test_post_reduced))
plot([Q(1) Q(1)], [0 3000], 'LineWidth', 2.)
plot([Q(2) Q(2)], [0 3000], 'LineWidth', 2.)
plot([Q(3) Q(3)], [0 3000], 'LineWidth', 2.)
%plot([IQR IQR], [0 3000], 'LineWidth', 2.)

legend('data', 'median', 'mode', 'mean', 'std', 'Q1', 'Q2', 'Q3')
xlim([-2, 1])
%title('Awake - post sleep')
xlabel('f.r. [Hz]')
ax = gca;
ax.TickLength = [0.03, 0.03];
ax.YColor= [0. 0. 0.];
ax.XColor= [0. 0. 0.];

ax.FontSize = 20;


%% deep sleep incremental learning cortex thalamo cortical

A = mean_fr_exc_test_pre_reduced.';
B = mean_fr_exc_test_post_reduced.';
group = [ 1 *   ones(size(A));
         -1   * ones(size(B))];

figure
h = boxplot([A;B],group,'Notch','on','Whisker',1, 'Color', [[0.5451 0. 0.5451];[1. 0.4980 0.3137]],'Symbol','.', 'orientation', 'horizontal')
hold on
set(h,{'linew'},{2.5})
grid on
ax = gca
ax.GridLineStyle = '-'
ax.GridColor = 'k'
ax.GridAlpha = 1 % maximum line opacityylabel('Awake f.r. [Hz]')
ax.MinorGridAlpha = 1 % maximum line opacity
%ylabel('Awake f.r. [Hz]')
xticks([0.1 1 10])
xticklabels({'0.1', '1', '10'})
set(gca,'XScale','log')
%ylim([0.1, 10])
lines = findobj(gcf, 'type', 'line', 'Tag', 'Median');
set(lines, 'Color', [0. 0. 0.]);
ax = gca;

ax.FontSize = 25;

%%

figure
subplot(3,1,1)
h = histogram(log10(mean_fr_exc_test_pre_reduced),'BinWidth', 0.1, 'EdgeColor', [1. 0.4980 0.3137], 'FaceColor', [1. 0.4980 0.3137])
hold on
[ops ,moda_idx] = max(h.Values);
moda = h.BinEdges(moda_idx) + h.BinWidth/2.;
%plot([median(log10(mean_fr_exc_test_pre_reduced)) median(log10(mean_fr_exc_test_pre_reduced))], [0 3000], 'LineWidth', 2.)
plot([moda moda], [0 3000], 'LineWidth', 2., 'Color', [0. 1. 0.])
plot([mean(log10(mean_fr_exc_test_pre_reduced)) mean(log10(mean_fr_exc_test_pre_reduced))], [0 3000], 'LineWidth', 2., 'Color', [1. 0. 0.])
%plot([mean(log10(mean_fr_exc_test_pre_reduced))-std(log10(mean_fr_exc_test_pre_reduced))/10 mean(log10(mean_fr_exc_test_pre_reduced))+std(log10(mean_fr_exc_test_post_reduced))/10], [2000 2000], 'LineWidth', 2.)
[Q,IQR] = quartile(log10(mean_fr_exc_test_pre_reduced))
plot([Q(1) Q(1)], [0 3000], 'LineWidth', 2., 'Color', [0. 0. 0.])
plot([Q(2) Q(2)], [0 3000], 'LineWidth', 2., 'Color', [0. 0. 0.])
plot([Q(3) Q(3)], [0 3000], 'LineWidth', 2., 'Color', [0. 0. 0.])
%plot([IQR IQR], [0 3000], 'LineWidth', 2.)
legend('Awake - Pre sleep', 'mode', 'mean')
%set(gca,'xtick',[])
set(gca,'xticklabel',[])
xlim([-2, 1])
ylim([0, 3000])

lines = findobj(gcf, 'type', 'line', 'Tag', 'Median');
set(lines, 'Color', [0. 0. 0.]);
%set(gca,'xtick',[])
set(gca,'xticklabel',[])
ax = gca;
ax.FontSize = 10;

subplot(3,1,2)
h = boxplot([A;B],group,'Notch','on','Whisker',1, 'Color', [[0.5451 0. 0.5451];[1. 0.4980 0.3137]],'Symbol','.', 'orientation', 'horizontal')
hold on
set(h,{'linew'},{2.5})
grid on
ax = gca
ax.GridLineStyle = '-'
ax.GridColor = 'k'
ax.GridAlpha = 1 % maximum line opacityylabel('Awake f.r. [Hz]')
ax.MinorGridAlpha = 1 % maximum line opacity
%ylabel('Awake f.r. [Hz]')
%xticks([0.1 1 10])
%xticklabels({'0.1', '1', '10'})
set(gca,'XScale','log')
xlim([0.01, 10])
%ylim([0.1, 10])
lines = findobj(gcf, 'type', 'line', 'Tag', 'Median');
set(lines, 'Color', [0. 0. 0.]);
%set(gca,'xtick',[])
set(gca,'xticklabel',[])
ax = gca;
ax.FontSize = 10;


subplot(3,1,3)
h = histogram(log10(mean_fr_exc_test_post_reduced),'BinWidth', 0.1, 'EdgeColor', [0.5451 0. 0.5451], 'FaceColor', [0.5451 0. 0.5451])
hold on
[ops ,moda_idx] = max(h.Values);
moda = h.BinEdges(moda_idx) + h.BinWidth/2.;
xticks([-2 -1 0 1.])
xticklabels({'0.01', '0.1', '1', '10'})
%plot([median(log10(mean_fr_exc_test_post_reduced)) median(log10(mean_fr_exc_test_post_reduced))], [0 3000], 'LineWidth', 2.)
plot([moda moda], [0 3000], 'LineWidth', 2., 'Color', [0. 1. 0.])
plot([mean(log10(mean_fr_exc_test_post_reduced)) mean(log10(mean_fr_exc_test_post_reduced))], [0 3000], 'LineWidth', 2., 'Color', [1. 0. 0.])
%plot([mean(log10(mean_fr_exc_test_post_reduced))-std(log10(mean_fr_exc_test_post_reduced)) mean(log10(mean_fr_exc_test_post_reduced))+std(log10(mean_fr_exc_test_post_reduced))], [2000 2000], 'LineWidth', 2.)
[Q,IQR] = quartile(log10(mean_fr_exc_test_post_reduced))
plot([Q(1) Q(1)], [0 3000], 'LineWidth', 2., 'Color', [0. 0. 0.])
plot([Q(2) Q(2)], [0 3000], 'LineWidth', 2., 'Color', [0. 0. 0.])
plot([Q(3) Q(3)], [0 3000], 'LineWidth', 2., 'Color', [0. 0. 0.])
%plot([IQR IQR], [0 3000], 'LineWidth', 2.)
legend('Awake - Post sleep', 'mode', 'mean')
xlim([-2, 1])
ylim([0, 3000])
set(gca, 'YDir','reverse')
%title('Awake - post sleep')
xlabel('f.r. [Hz]')
ax = gca;
ax.TickLength = [0.03, 0.03];
ax.YColor= [0. 0. 0.];
ax.XColor= [0. 0. 0.];
ax.FontSize = 10;
set(gcf,'units','centimeters','position',[1,1,15,15])


%%

figure
h = histogram(log10(mean_fr_exc_test_pre_reduced),'BinWidth', 0.1, 'EdgeColor', [1. 0.4980 0.3137], 'FaceColor', [1. 0.4980 0.3137])
hold on
[ops ,moda_idx] = max(h.Values);
moda = h.BinEdges(moda_idx) + h.BinWidth/2.;
%plot([median(log10(mean_fr_exc_test_pre_reduced)) median(log10(mean_fr_exc_test_pre_reduced))], [0 3000], 'LineWidth', 2.)
plot([moda moda], [0 3000], 'LineWidth', 2., 'Color', [0. 1. 0.])
plot([mean(log10(mean_fr_exc_test_pre_reduced)) mean(log10(mean_fr_exc_test_pre_reduced))], [0 3000], 'LineWidth', 2., 'Color', [1. 0. 0.])
%plot([mean(log10(mean_fr_exc_test_pre_reduced))-std(log10(mean_fr_exc_test_pre_reduced))/10 mean(log10(mean_fr_exc_test_pre_reduced))+std(log10(mean_fr_exc_test_post_reduced))/10], [2000 2000], 'LineWidth', 2.)
[Q,IQR] = quartile(log10(mean_fr_exc_test_pre_reduced))
plot([Q(1) Q(1)], [0 3000], 'LineWidth', 2., 'Color', [0. 0. 0.])
plot([Q(2) Q(2)], [0 3000], 'LineWidth', 2., 'Color', [0. 0. 0.])
plot([Q(3) Q(3)], [0 3000], 'LineWidth', 2., 'Color', [0. 0. 0.])
%plot([IQR IQR], [0 3000], 'LineWidth', 2.)
lg = legend('Awake - Pre sleep', 'mode', 'mean')
lg.Location = 'northwest'
%lg.FontSize = 13
ylabel(['Number of' newline 'neurons'])

xlim([-2, 1])
ylim([0, 3000])
lines = findobj(gcf, 'type', 'line', 'Tag', 'Median');
set(lines, 'Color', [0. 0. 0.]);
%set(gca,'xtick',[])
set(gca,'xticklabel',[])
ax = gca;
ax.FontSize = 12;
set(gcf,'units','centimeters','position',[1,1,19,3])


figure
h = boxplot([A;B],group,'Labels', {' ', ' '},'Notch','on','Whisker',1, 'Color', [[0.5451 0. 0.5451];[1. 0.4980 0.3137]],'Symbol','.', 'orientation', 'horizontal')
hold on
set(h,{'linew'},{2.5})
grid on
ax = gca
ax.GridLineStyle = '-'
ax.GridColor = 'k'
ax.GridAlpha = 1 % maximum line opacityylabel('Awake f.r. [Hz]')
ax.MinorGridAlpha = 1 % maximum line opacity
%ylabel('Awake f.r. [Hz]')
%xticks([0.1 1 10])
%xticklabels({'0.1', '1', '10'})
set(gca,'XScale','log')
xlim([0.01, 10])
%ylim([0.1, 10])
lines = findobj(gcf, 'type', 'line', 'Tag', 'Median');
set(lines, 'Color', [0. 0. 0.]);
%set(gca,'xtick',[])
set(gca,'xticklabel',[])
ax = gca;
ax.FontSize = 12;
set(gcf,'units','centimeters','position',[1,1,19,3])


figure
h = histogram(log10(mean_fr_exc_test_post_reduced),'BinWidth', 0.1, 'EdgeColor', [0.5451 0. 0.5451], 'FaceColor', [0.5451 0. 0.5451])
hold on
[ops ,moda_idx] = max(h.Values);
moda = h.BinEdges(moda_idx) + h.BinWidth/2.;
xticks([-2 -1 0 1.])
xticklabels({'0.01', '0.1', '1', '10'})
%plot([median(log10(mean_fr_exc_test_post_reduced)) median(log10(mean_fr_exc_test_post_reduced))], [0 3000], 'LineWidth', 2.)
plot([moda moda], [0 3000], 'LineWidth', 2., 'Color', [0. 1. 0.])
plot([mean(log10(mean_fr_exc_test_post_reduced)) mean(log10(mean_fr_exc_test_post_reduced))], [0 3000], 'LineWidth', 2., 'Color', [1. 0. 0.])
%plot([mean(log10(mean_fr_exc_test_post_reduced))-std(log10(mean_fr_exc_test_post_reduced)) mean(log10(mean_fr_exc_test_post_reduced))+std(log10(mean_fr_exc_test_post_reduced))], [2000 2000], 'LineWidth', 2.)
[Q,IQR] = quartile(log10(mean_fr_exc_test_post_reduced))
plot([Q(1) Q(1)], [0 3000], 'LineWidth', 2., 'Color', [0. 0. 0.])
plot([Q(2) Q(2)], [0 3000], 'LineWidth', 2., 'Color', [0. 0. 0.])
plot([Q(3) Q(3)], [0 3000], 'LineWidth', 2., 'Color', [0. 0. 0.])
%plot([IQR IQR], [0 3000], 'LineWidth', 2.)
%legend('data', 'mode', 'mean', 'Q1', 'Q2', 'Q3')
xlim([-2, 1])
ylim([0, 3000])
ylabel(['Number of' newline 'neurons'])
set(gca, 'YDir','reverse')
%title('Awake - post sleep')
xlabel('f.r. [Hz]')
lg=legend('Awake - Post sleep', 'mode', 'mean')
lg.Location = 'southwest'
%lg.FontSize = 13
ax = gca;
%ax.TickLength = [0.03, 0.03];
%ax.YColor= [0. 0. 0.];
%ax.XColor= [0. 0. 0.];
ax.FontSize = 12;
set(gcf,'units','centimeters','position',[1,1,19,4])


%%
figure
subplot(2,1,1)
histogram(mean_fr_exc_test_pre_reduced,'BinWidth', 0.1)
hold on
xlim([0, 6])
title('Awake - pre sleep')
ax = gca;
ax.FontSize = 20;
subplot(2,1,2)
histogram(mean_fr_exc_test_post_reduced,'BinWidth', 0.1)

xlim([0, 6])
title('Awake - post sleep')
xlabel('fr')
ax = gca;
ax.FontSize = 20;

%%
function CI = polyparci(PolyPrms,PolyS,alpha) 
if nargin < 3
    alpha = 0.95;
end
% Check for out-of-range values for alpha and substitute if necessary: 
if alpha < 1.0E-010
    alpha = 1.0E-010;
elseif alpha > (1 - 1.0E-010)
    alpha = 1 - 1.0E-010;
end

COVB = (PolyS.R'*PolyS.R)\eye(size(PolyS.R)) * PolyS.normr^2/PolyS.df;
SE = sqrt(diag(COVB));                              % Standard Errors
[PrmSizR,PrmSizC] = size(PolyPrms);                 % Convert parameter vector to column if necessary
if PrmSizR < PrmSizC
    PolyPrms = PolyPrms';
end

tstat = @(tval) (alpha - t_cdf(tval,PolyS.df) );    
[T,fval] = fzero(tstat, 1);                        
T = abs(T);                                         
ts = T * [-1  1];                                   
CI  = bsxfun(@plus,bsxfun(@times,SE,ts),PolyPrms)'; 
% CALCULATE THE CUMULATIVE T-DISTRIBUTION: 
    function PT = t_cdf(t,v)
        IBx = v./(t.^2 + v);                     
        IBZ = v/2;                          
        IBW = 0.5;                            
        Ixzw = betainc(IBx, IBZ, IBW);          
        PT = 1-0.5*Ixzw;                                 
    end
end

function [Q,IQR] = quartile(X)
%Author: David Ferreira - Federal University of Amazonas
%Contact: ferreirad08@gmail.com
%
%quartile
%
%Syntax
%1. Q = quartile(X)
%2. [Q,IQR] = quartile(X)
%
%Description 
%1. Calculates the quartile (Q1, Q2 and Q3) of the data of a vector or matrix.
%2. Calculates the quartile (Q1, Q2 and Q3) and the interquartile range (IQR) of the data of a vector or matrix.
%
%Example
%1.
%     v = [1 2 3 4 7 10];
%     [Q,IQR] = quartile(v)
%     Q = 
%         2.2500    3.5000    6.2500
%     IQR =
%         4
%
%2.
%     A = [1 2; 2 5; 3 6; 4 10; 7 11; 10 13];
%     [Q,IQR] = quartile(A)
%     Q = 
%         2.2500    5.2500
%         3.5000    8.0000
%         6.2500   10.7500
%     IQR =
%         4.0000    5.5000
X = sort(X);
if isrow(X), X = X'; end
[n,m] = size(X);
if isscalar(X), X(2,1) = 0; end
LMU = [(n+3)/4; (n+1)/2; (3*n+1)/4];
i = floor(LMU);
d = LMU-i;
Q = X(i,:) + (X(i+1,:)-X(i,:)).*repmat(d,1,m);
IQR = Q(3,:)-Q(1,:);
if isvector(X), Q = Q'; end
end