
%%
clear;
clc;
SDR_all = [];

path_result = '../../result/model_03/';
filenames = dir([path_result, '*.wav']);
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
SDR_all = cat(2, SDR_all, SDR);

path_result = '../../result/model_03_adjust/';
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
SDR_all = cat(2, SDR_all, SDR);

path_result = '../../result/model_03_ibm/';
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
SDR_all = cat(2, SDR_all, SDR);