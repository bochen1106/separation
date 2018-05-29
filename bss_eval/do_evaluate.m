

path_result = '../../result/model_03_adjust_med55/';
path_single = '../../data/audio/single/test/';
filenames = dir([path_result, '*.wav']);

SDR = zeros(100, 2);
SDR0 = zeros(100, 2);

fid = fopen([path_result, '_SDR.txt'], 'wb');
for i = 1 : length(filenames)/2
    %%
    fprintf('%d\n', i);
    filename1 = filenames((i-1)*2+1).name;
    filename2 = filenames((i-1)*2+2).name;
    fprintf(fid, '%s\t', [filename1(1:end-6), '.wav']);
    
    wav1 = audioread([path_result, filename1]);    
    wav2 = audioread([path_result, filename2]);
    se = [wav1'; wav2'];
    
    name_indv_1 = filename1(1 : 6); instr1 = name_indv_1(1:2);
    name_indv_2 = filename2(8 : 13);instr2 = name_indv_2(1:2);
    
    filename1 = dir( [path_single, instr1, '/', name_indv_1, '*.wav'] ); 
    filename1 = [path_single, instr1, '/', filename1(1).name];
    filename2 = dir( [path_single, instr2, '/', name_indv_2, '*.wav'] ); 
    filename2 = [path_single, instr2, '/', filename2(1).name];
    
    wav1 = audioread(filename1);    
    wav2 = audioread(filename2);
    s = [wav1'; wav2'];
    se0 = [(wav1+wav2)'; (wav1+wav2)'];
    [sdr,~,~,perm] = bss_eval_sources(se,s);
    SDR(i, 1) = sdr(perm(1));   SDR(i, 2) = sdr(perm(2));
    [sdr,~,~,perm] = bss_eval_sources(se0,s);
    SDR0(i, 1) = sdr(perm(1));  SDR0(i, 2) = sdr(perm(2));
    delta = SDR(i,:) - SDR0(i,:);
    fprintf( fid, '%.2f\t%.2f\n', delta(1), delta(2) );
    
end

%%
% dlmwrite([path_result, '_SDR.txt'], SDR-SDR0, '\t');

