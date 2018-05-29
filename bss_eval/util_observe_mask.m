
clear;
clc;

data_path = '../../result/tmp/';

y1 = dlmread([data_path, 'y1.txt']);
y2 = dlmread([data_path, 'y2.txt']);
pred1 = dlmread([data_path, 'pred1.txt']);
pred2 = dlmread([data_path, 'pred2.txt']);
pred_aj_1 = dlmread([data_path, 'pred_aj_1.txt']);
pred_aj_2 = dlmread([data_path, 'pred_aj_2.txt']);
X_mag = dlmread([data_path, 'X_mag.txt']);

%%
figure;
idx = 1 : 626;
subplot(321); imagesc(y1(:,idx));
subplot(322); imagesc(y2(:,idx));
subplot(323); imagesc(pred1(:,idx));
subplot(324); imagesc(pred2(:,idx));
subplot(325); imagesc(pred_aj_1(:,idx));
subplot(326); imagesc(pred_aj_2(:,idx));


%%
figure; imagesc(X_mag);



