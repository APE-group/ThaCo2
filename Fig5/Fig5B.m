%© 2021. This work is licensed under a CC-BY-NC-SA license.
%"Thalamo-cortical spiking model of incremental learning combining perception, context and NREM-sleep-mediated noise-resilience"
%Authors: Bruno Golosio, Chiara De Luca, Cristiano Capone, Elena Pastorelli, Giovanni Stegel, Gianmarco Tiddia, Giulia De Bonis and Pier Stanislao Paolucci
%arxiv.2003.11859

%%

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
pesi = [];
pesi_pre = [];

for ops = 0:49
    pesi = [pesi ops_matrix_nuovo(ops*20+1:(ops+1)*20,ops*20+1:(ops+1)*20)];
    pesi_pre = [pesi_pre ops_matrix_nuovo_pre(ops*20+1:(ops+1)*20,ops*20+1:(ops+1)*20)];
end

pesi_line = [];
pesi_line_pre = [];

for ops = 1:20
    pesi_line = [pesi_line pesi(ops,:)];
    pesi_line_pre = [pesi_line_pre pesi_pre(ops,:)];
end


%%
% Riordino
ops_matrix_nuovo = ops_matrix_post;

ops_matrix_nuovo_pre = ops_matrix_pre;

ops_matrix_nuovo_plot = ops_matrix_nuovo;
ops_matrix_nuovo_plot_pre = ops_matrix_nuovo_pre;

combinato = zeros([100 10]);
for classe = 0:9
    for es = 0:4
        temp = ops_matrix_nuovo(classe*100 + es*20+1:classe*100 +(es+1)*20,classe*100 + es*20+1:classe*100 +(es+1)*20)
        combinato(es*20+1:(es+1)*20, classe*20+1: (1+ classe)*20) = temp;
    end
end

combinato_pre = zeros([100 10]);
for classe = 0:9
    for es = 0:4
        temp = ops_matrix_nuovo_pre(classe*100 + es*20+1:classe*100 +(es+1)*20,classe*100 + es*20+1:classe*100 +(es+1)*20);
        combinato_pre(es*20+1:(es+1)*20, classe*20+1: (1+ classe)*20) = temp;
    end
end

%%

%%

%Calcolo per istogrammi

for ops = 0:49
    ops_matrix_nuovo_plot(ops*20+1:(ops+1)*20,ops*20+1:(ops+1)*20) = 0.;
    ops_matrix_nuovo_plot_pre(ops*20+1:(ops+1)*20,ops*20+1:(ops+1)*20) = 0.;
end

weights_dentro = [];
weights_dentro_pre = [];



for ops = 0:9
    for ops2 = 0:4
        if numel(ops_matrix_nuovo_plot(ops*100+1:ops*100+ops2*20,ops*100+1 +ops2*20:ops*100 + (ops2+1)*20)) ~= 0
            temp = ops_matrix_nuovo_plot(ops*100+1:ops*100 + ops2*20,ops*100+1 +ops2*20:ops*100 + (ops2+1)*20);
        
            for line = 1:numel(temp(1,:))
                for line2 = 1:numel(temp(:, 1))
                    t = temp(line2,line);
                    weights_dentro(end+1) = t;
                end
            end
        end
        if numel(ops_matrix_nuovo_plot(ops*100+1 +(ops2+1)*20:(ops+1)*100,ops*100+1 +ops2*20:ops*100 + (ops2+1)*20)) ~= 0
            temp = ops_matrix_nuovo_plot(ops*100+1 +(ops2+1)*20:(ops+1)*100,ops*100+1 +ops2*20:ops*100 + (ops2+1)*20);
        
            for line = 1:numel(temp(1,:))
                for line2 = 1:numel(temp(:,1))
                    t = temp(line2,line);
                    weights_dentro(end+1) = t;
                end
            end
        
        end
        
        if numel(ops_matrix_nuovo_plot_pre(ops*100+1:ops*100+ops2*20,ops*100+1 +ops2*20:ops*100 + (ops2+1)*20)) ~= 0
            temp = ops_matrix_nuovo_plot_pre(ops*100+1:ops*100 + ops2*20,ops*100+1 +ops2*20:ops*100 + (ops2+1)*20);
        
            for line = 1:numel(temp(1,:))
                for line2 = 1:numel(temp(:, 1))
                    t = temp(line2,line);
                    weights_dentro_pre(end+1) = t;
                end
            end
        end
        if numel(ops_matrix_nuovo_plot_pre(ops*100+1 +(ops2+1)*20:(ops+1)*100,ops*100+1 +ops2*20:ops*100 + (ops2+1)*20)) ~= 0
            temp = ops_matrix_nuovo_plot_pre(ops*100+1 +(ops2+1)*20:(ops+1)*100,ops*100+1 +ops2*20:ops*100 + (ops2+1)*20);
        
            for line = 1:numel(temp(1,:))
                for line2 = 1:numel(temp(:,1))
                    t = temp(line2,line);
                    weights_dentro_pre(end+1) = t;
                end
            end
        
        end
    end
end

weights_fuori = [];
weights_fuori_pre = [];


for ops = 0:9
    if numel(ops_matrix_nuovo_plot(1:ops*100,ops*100+1:(ops+1)*100)) ~= 0
        temp = ops_matrix_nuovo_plot(1:ops*100,ops*100+1:(ops+1)*100);  
        for line = 1:numel(temp(1,:))
            for line2 = 1:numel(temp(:, 1))
                t = temp(line2,line);
                weights_fuori(end+1) = t;
            end
        end
    end
            
    if numel(ops_matrix_nuovo_plot((ops+1)*100+1:1000,ops*100+1:(ops+1)*100)) ~= 0
        temp = ops_matrix_nuovo_plot((ops+1)*100+1:1000,ops*100+1:(ops+1)*100);
        
        for line = 1:numel(temp(1,:))
            for line2 = 1:numel(temp(:,1))
                t = temp(line2,line);
                weights_fuori(end+1) = t;
            end
        end
        
    end
    if numel(ops_matrix_nuovo_plot_pre(1:ops*100,ops*100+1:(ops+1)*100)) ~= 0
        temp = ops_matrix_nuovo_plot_pre(1:ops*100,ops*100+1:(ops+1)*100);  
        for line = 1:numel(temp(1,:))
            for line2 = 1:numel(temp(:, 1))
                t = temp(line2,line);
                weights_fuori_pre(end+1) = t;
            end
        end
    end
            
    if numel(ops_matrix_nuovo_plot_pre((ops+1)*100+1:1000,ops*100+1:(ops+1)*100)) ~= 0
        temp = ops_matrix_nuovo_plot_pre((ops+1)*100+1:1000,ops*100+1:(ops+1)*100);
        
        for line = 1:numel(temp(1,:))
            for line2 = 1:numel(temp(:,1))
                t = temp(line2,line);
                weights_fuori_pre(end+1) = t;
            end
        end
        
    end
    
end


%%
figure
subplot(1,3,3)

media = mean(weights_fuori);
media_pre = mean(weights_fuori_pre);
dev = std(weights_fuori);
dev_pre = std(weights_fuori_pre);

histogram(weights_fuori, 200, 'Normalization', 'probability', 'FaceColor', [0.2078 0.6 0.], 'EdgeColor', 'none')
hold on
histogram(weights_fuori_pre*10, 200, 'Normalization', 'probability', 'BinLimits',[0. 0.4], 'FaceColor',  [0.7294 0.4078 0.7843], 'EdgeColor', 'none')
plot([media, media],[0.000001 1],  'LineWidth', 2.5, 'Color', [0.2078 0.6 0.])
plot([media_pre, media_pre],[0.000001 1], 'LineWidth', 2.5, 'Color',  [0.6 0.2 0.8])
plot([media-dev, media+dev],[0.06 0.06], 'LineWidth', 2.5, 'Color', [0.2078 0.6 0.])
plot([media_pre-dev_pre, media_pre+dev_pre],[0.035 0.035],'LineWidth', 2.5, 'Color',  [0.6 0.2 0.8])
title(['Synapses trained over' newline 'different classes'])
lg = legend('Post-Sleep', 'Pre-Sleep', ['Mean' newline 'Post-Sleep'], ['Mean' newline 'Pre-Sleep'])
lg.Location = 'southeast'
set(gca,'YScale','log')
%set(gca,'XScale','log')
yticks([10^(-5), 10^(-4), 10^(-3), 10^(-2), 10^(-1), 10^(-0)])
ylim([10^(-5),1])
xlim([-0.5,6.5])
xlabel('Synaptic weight')
xticks([0 1 2 3 4 5 6])
%lg = legend({'Post-Sleep', 'Pre-Sleep', 'Median Pre-Sleep', 'Median Post-Sleep', 'Standard deviation Pre-Sleep', 'Standard deviation Post-Sleep'})
ax = gca;
ax.FontSize = 10;
hold on
% create smaller axes in top right, and plot on it
axes('Position',[0.77 0.62  .12 .2])
box on
histogram(weights_fuori, 200, 'Normalization', 'probability', 'FaceColor', [0.2078 0.6 0.], 'EdgeColor', 'none')
hold on
histogram(weights_fuori_pre*10, 200, 'Normalization', 'probability', 'BinLimits',[0. 0.4], 'FaceColor',  [0.7294 0.4078 0.7843], 'EdgeColor', 'none')
plot([media_pre, media_pre],[0.000001 1], 'LineWidth', 2.5, 'Color',  [0.6 0.2 0.8])
plot([media, media],[0.000001 1],  'LineWidth', 2.5, 'Color', [0.2078 0.6 0.])

ax = gca;
ax.TickLength = [0.1, 0.05];
%set(gca,'XMinorTick','on','YMinorTick','on')
a = gca;
% set box property to off and remove background color
set(a,'box','off','color','none')
% create new, empty axes with box but without ticks
b = axes('Position',get(a,'Position'),'box','on','xtick',[],'ytick',[]);
% set original axes as active
axes(a)
% link axes in case of zooming
linkaxes([a b])
xlim([-0.05, 0.3])
%ylim([0., 0.1])
ylim([0., 0.03])
xticks([0 0.05 0.1 0.15 0.2 0.25])
xticklabels({'0', ' ', '0.1', ' ', '0.2', ' '})

subplot(1,3,2)
media = mean(weights_dentro);
media_pre = mean(weights_dentro_pre);
dev = std(weights_dentro);
dev_pre = 0.1;

histogram(weights_dentro,200, 'Normalization', 'probability', 'FaceColor', [0.2078 0.6 0.], 'EdgeColor', 'none')
hold on
histogram(weights_dentro_pre*10, 200, 'Normalization', 'probability', 'BinLimits',[0. 0.4], 'FaceColor', [0.7294 0.4078 0.7843], 'EdgeColor','none')
plot([media, media],[0.00001 1],  'LineWidth', 2.5, 'Color', [0.2078 0.6 0.])
plot([media_pre, media_pre],[0.00001 1], 'LineWidth', 2.5, 'Color', [0.6 0.2 0.8])
plot([media-dev, media+dev],[0.06 0.06], 'LineWidth', 2.5, 'Color', [0.2078 0.6 0.])
plot([media_pre-dev_pre, media_pre+dev_pre],[0.035 0.035],'LineWidth', 2.5, 'Color',  [0.6 0.2 0.8])
title(['Synapses trained over the ' newline 'same class -different example'])
%legend('Post-Sleep', 'Pre-Sleep', 'Mean Pre-Sleep', 'Mean Post-Sleep', 'Standard deviation Pre-Sleep', 'Standard deviation Post-Sleep')
set(gca,'YScale','log')
%set(gca,'XScale','log')
%ylim([0,0.05])
yticks([10^(-5), 10^(-4), 10^(-3), 10^(-2), 10^(-1), 10^(-0)])
xlabel('Synaptic weight')
xlim([-0.5,6.5])
xticks([0 1 2 3 4 5 6])
%legend('Post-Sleep', 'Pre-Sleep', 'Mean Pre-Sleep', 'Mean Post-Sleep', 'Standard deviation Pre-Sleep', 'Standard deviation Post-Sleep')
ax = gca;
ax.FontSize = 10;
hold on
% create smaller axes in top right, and plot on it
axes('Position',[.49 .62 .12 .2])
box on
histogram(weights_dentro,200, 'Normalization', 'probability', 'FaceColor', [0.2078 0.6 0.], 'EdgeColor', 'none')
hold on
histogram(weights_dentro_pre*10, 200, 'Normalization', 'probability', 'BinLimits',[0. 0.4], 'FaceColor', [0.7294 0.4078 0.7843], 'EdgeColor','none')
plot([media, media],[0.00001 1],  'LineWidth', 2.5, 'Color', [0.2078 0.6 0.])
plot([media_pre, media_pre],[0.00001 0.03], 'LineWidth', 2.5, 'Color', [0.6 0.2 0.8])

ax = gca;
ax.TickLength = [0.1, 0.05];
a = gca;
% set box property to off and remove background color
set(a,'box','off','color','none')
% create new, empty axes with box but without ticks
b = axes('Position',get(a,'Position'),'box','on','xtick',[],'ytick',[]);
% set original axes as active
axes(a)
% link axes in case of zooming
linkaxes([a b])
%set(gca,'XMinorTick','on','YMinorTick','on')
%xlim([-0.05, 0.15])
xlim([-0.05, 0.3])
ylim([0., 0.03])
xticks([0 0.05 0.1 0.15 0.2 0.25])
xticklabels({'0', ' ', '0.1', ' ', '0.2', ' '})


subplot(1,3,1)
media = mean(pesi_line);
media_pre = mean(pesi_line_pre);
dev = std(pesi_line);
dev_pre = std(pesi_line_pre);

histogram(pesi_line,200, 'Normalization', 'probability', 'FaceColor', [0.2078 0.6 0.], 'EdgeColor', 'none')
hold on
histogram(pesi_line_pre, 200, 'Normalization', 'probability', 'FaceColor', [0.7294 0.4078 0.7843], 'EdgeColor','none')
plot([media, media],[0.00001 1],  'LineWidth', 2.5, 'Color', [0.2078 0.6 0.])
plot([media_pre, media_pre],[0.00001 1], 'LineWidth', 2.5, 'Color', [0.6 0.2 0.8])
plot([media-dev, media+dev],[0.03 0.03], 'LineWidth', 2.5, 'Color', [0.2078 0.6 0.])
plot([media_pre-dev_pre, media_pre+dev_pre],[0.04 0.04],'LineWidth', 2.5, 'Color', [0.6 0.2 0.8])
title(['Synapses trained over' newline 'the same example'])
xlabel('Synaptic weight')
%set(gca,'YScale','log')
ylim([0,0.05])
ylabel('Number of synapses')
ax = gca;
ax.FontSize = 10;
%A4: 21cm x 29.7cm
set(gcf,'units','centimeters','position',[1,1,21,9])


%%

media = mean(pesi_line);
media_pre = mean(pesi_line_pre);
dev = std(pesi_line)/1.1;
dev_pre = std(pesi_line_pre);


figure
histogram(pesi_line,200, 'Normalization', 'probability')
hold on
histogram(pesi_line_pre, 100, 'Normalization', 'probability')
title('Neurons trained over the same example')
hold on
plot([media, media],[0 0.05],  'LineWidth', 2.5, 'Color', [0.0977 0.0977 0.4375])
plot([media_pre, media_pre],[0 0.05], 'LineWidth', 2.5, 'Color', [0.8594 0.0781 0.2344])
plot([media-dev, media+dev],[0.04 0.04], 'LineWidth', 2.5, 'Color', [0.0977 0.0977 0.4375])
plot([media_pre-dev_pre, media_pre+dev_pre],[0.045 0.045],'LineWidth', 2.5, 'Color', [0.8594 0.0781 0.2344])
xlabel('Synaptic weight')
legend('Post-Sleep', 'Pre-Sleep', 'Mean Pre-Sleep', 'Mean Post-Sleep', 'Standard deviation Pre-Sleep', 'Standard deviation Post-Sleep')
ax = gca;
ax.FontSize = 15;
%set(gca,'YScale','log')

%%
figure

x_vector = linspace(0, 400, 400/20+1);

subplot(1,2,1)
imagesc(log(ops_matrix_pre))
yticks([10, 30, 50, 70, 90, 110, 130, 150, 170, 190, 210, 230, 250, 270, 290, 310, 330, 350, 370, 390])
xticks([10, 30, 50, 70, 90, 110, 130, 150, 170, 190, 210, 230, 250, 270, 290, 310, 330, 350, 370, 390])
yticklabels({'2', '8', '4', '9', '1', '6', '7', '3', '0', '5', '2', '9', '6', '4', '0', '3', '1', '7', '8', '5'})
xticklabels({'2', '8', '4', '9', '1', '6', '7', '3', '0', '5', '2', '9', '6', '4', '0', '3', '1', '7', '8', '5'})
xlabel('Trained class')
ylabel('Trained class')
title('PreSleep')
colorbar()
set(gca, 'clim', [-16 2])
%caxis([-16, 3])
subplot(1,2,2)
imagesc(log(ops_matrix_post))
yticks([10, 30, 50, 70, 90, 110, 130, 150, 170, 190, 210, 230, 250, 270, 290, 310, 330, 350, 370, 390])
xticks([10, 30, 50, 70, 90, 110, 130, 150, 170, 190, 210, 230, 250, 270, 290, 310, 330, 350, 370, 390])
yticklabels({'2', '8', '4', '9', '1', '6', '3', '7', '0', '5', '2', '9', '6', '4', '0', '3', '1', '7', '8', '5'})
xticklabels({'2', '8', '4', '9', '1', '6', '3', '7', '0', '5', '2', '9', '6', '4', '0', '3', '1', '7', '8', '5'})
xlabel('Trained class')
ylabel('Trained class')
title('PostSleep')
colorbar()
%caxis([-16, 3])

%%
%idx = find(ops_matrix_post > 1.5026 & ops_matrix_post < 2.3026^1.5);
%ops_matrix_post(idx) = ops_matrix_post(idx) * 100;
figure
subplot(1,2,1)
imagesc(log(ops_matrix_pre))
xlabel('#number cortical neuron')
ylabel('#number cortical neuron')
title('PreSleep')
colorbar()
%caxis([-16, 3])
subplot(1,2,2)
imagesc(log(ops_matrix_post))
hold on
plot([0, 1000],[200 200], 'Color', [0. 0. 0.])
plot([200, 200],[0 1000], 'Color', [0. 0. 0.])
plot([0, 1000],[400 400], 'Color', [0. 0. 0.])
plot([400, 400],[0 1000], 'Color', [0. 0. 0.])
plot([0, 1000],[600 600], 'Color', [0. 0. 0.])
plot([600, 600],[0 1000], 'Color', [0. 0. 0.])
plot([0, 1000],[800 800], 'Color', [0. 0. 0.])
plot([800, 800],[0 1000], 'Color', [0. 0. 0.])
plot([0, 1000],[1000 1000], 'Color', [0. 0. 0.])
plot([1000, 1000],[0 1000], 'Color', [0. 0. 0.])
xlabel('#number cortical neuron')
ylabel('#number cortical neuron')
title('PostSleep')
colorbar()
%caxis([-16, 3])


%%
figure
subplot(1,2,1)
imagesc((ops_matrix_pre))
xlabel('#number cortical neuron')
ylabel('#number cortical neuron')
title('PreSleep')
colorbar()
%caxis([-16, 3])
subplot(1,2,2)
imagesc((ops_matrix_post))
xlabel('#number cortical neuron')
ylabel('#number cortical neuron')
title('PostSleep')
colorbar()
%caxis([-16, 3])

figure
histogram(w_post(:,3))
hold on
histogram(w_pre(:,3))
ylim([0,800])
xlabel('Synaptic cx-cx weight')
legend('Post sleep', 'Pre sleep')
ax = gca;
ax.FontSize = 20;


%%
x_vector = linspace(0, 400, 400/20+1);

figure
imagesc(log(ops_matrix_post))
xlabel('#number cortical neuron')
ylabel('#number cortical neuron')
title('PostSleep')
colorbar()
xticks(x_vector)
yticks(x_vector)

%caxis([-18, 5])

%%



