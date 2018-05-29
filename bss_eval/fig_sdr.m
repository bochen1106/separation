%%
clear;
clc;

path_result = '../../result/model_03/';
filenames = dir([path_result, '*.wav']);
n = length(filenames);
% SDR = zeros(n, 2);
SDR = [];

fid = fopen([path_result, '_SDR.txt'], 'r');
tline = fgetl(fid);
while tline ~= -1 
    tmp = split(tline);
    a = str2double(tmp{2});
    b = str2double(tmp{3});
    SDR = cat(1, SDR, [a, b]);
    tline = fgetl(fid);
end

fclose(fid);


% SDR = SDR(1:2,:);
figure;
set(gcf, 'position', [100,100,250,400]);
boxplot(SDR, 'color', 'k', 'notch', 'on');
set(gca, 'ylim', [-15, 30]);
set(gca, 'ygrid', 'on');
set(gca, 'fontsize', 16);
ylabel('Delta-SDR')
% title('vn-vn');
