%% CODING EXPERIMENTS
clear;
addpath(genpath('RA-GFT'));

dataset = 'MVUB';
sequence =  'andrew9';
sequence =  'david9';
sequence =  'phil9';
sequence =  'ricardo9';
sequence =  'sarah9';

%dataset='8iVFBv2';
%sequence = 'redandblack';
%sequence = 'soldier';
%sequence = 'longdress';
%sequence = 'loot';

%colorStep = [1 2 4 8 16 32 64];
colorStep = [1 2 4 8 12 16 20 24 32 64];
%
%% Experiment 1: level L has blocksize = 2, 4, 8, 16. Other levels 1,2,...,L-1, have blocksize =2
%for sarah
exp_list.bsize{1} = [2 2 2 2 2 2 2 2 2]; %first experiment, first level(L) bsize=2, all remaining levels bsize2
exp_list.bsize{2} = [2 2 2 2 2 2 2 4]; %first experiment, first level(L) bsize=4, all remaining levels bsize2
exp_list.bsize{3} = [2 2 2 2 2 2 8];
exp_list.bsize{4} = [2 2 2 2 2 16];

exp_list.exp_name{1} = 'exp1_b2';
exp_list.exp_name{2} = 'exp1_b4';
exp_list.exp_name{3} = 'exp1_b8';
exp_list.exp_name{4} = 'exp1_b16';

exp_list.sequence{1} = 'sarah9';
exp_list.sequence{2} = 'sarah9';
exp_list.sequence{3} = 'sarah9';
exp_list.sequence{4} = 'sarah9';

%for ricardo
exp_list.bsize{5} = [2 2 2 2 2 2 2 2 2]; %first experiment, first level(L) bsize=2, all remaining levels bsize2
exp_list.bsize{6} = [2 2 2 2 2 2 2 4]; %first experiment, first level(L) bsize=4, all remaining levels bsize2
exp_list.bsize{7} = [2 2 2 2 2 2 8];
exp_list.bsize{8} = [2 2 2 2 2 16];

exp_list.exp_name{5} = 'exp1_b2';
exp_list.exp_name{6} = 'exp1_b4';
exp_list.exp_name{7} = 'exp1_b8';
exp_list.exp_name{8} = 'exp1_b16';

exp_list.sequence{5} = 'ricardo9';
exp_list.sequence{6} = 'ricardo9';
exp_list.sequence{7} = 'ricardo9';
exp_list.sequence{8} = 'ricardo9';

%for phil
exp_list.bsize{9} = [2 2 2 2 2 2 2 2 2]; %first experiment, first level(L) bsize=2, all remaining levels bsize2
exp_list.bsize{10} = [2 2 2 2 2 2 2 4]; %first experiment, first level(L) bsize=4, all remaining levels bsize2
exp_list.bsize{11} = [2 2 2 2 2 2 8];
exp_list.bsize{12} = [2 2 2 2 2 16];

exp_list.exp_name{9} = 'exp1_b2';
exp_list.exp_name{10} = 'exp1_b4';
exp_list.exp_name{11} = 'exp1_b8';
exp_list.exp_name{12} = 'exp1_b16';

exp_list.sequence{9} = 'phil9';
exp_list.sequence{10} = 'phil9';
exp_list.sequence{11} = 'phil9';
exp_list.sequence{12} = 'phil9';

%for david
exp_list.bsize{13} = [2 2 2 2 2 2 2 2 2]; %first experiment, first level(L) bsize=2, all remaining levels bsize2
exp_list.bsize{14} = [2 2 2 2 2 2 2 4]; %first experiment, first level(L) bsize=4, all remaining levels bsize2
exp_list.bsize{15} = [2 2 2 2 2 2 8];
exp_list.bsize{16} = [2 2 2 2 2 16];

exp_list.exp_name{13} = 'exp1_b2';
exp_list.exp_name{14} = 'exp1_b4';
exp_list.exp_name{15} = 'exp1_b8';
exp_list.exp_name{16} = 'exp1_b16';

exp_list.sequence{13} = 'david9';
exp_list.sequence{14} = 'david9';
exp_list.sequence{15} = 'david9';
exp_list.sequence{16} = 'david9';

%for andrew
exp_list.bsize{17} = [2 2 2 2 2 2 2 2 2]; %first experiment, first level(L) bsize=2, all remaining levels bsize2
exp_list.bsize{18} = [2 2 2 2 2 2 2 4]; %first experiment, first level(L) bsize=4, all remaining levels bsize2
exp_list.bsize{19} = [2 2 2 2 2 2 8];
exp_list.bsize{20} = [2 2 2 2 2 16];

exp_list.exp_name{17} = 'exp1_b2';
exp_list.exp_name{18} = 'exp1_b4';
exp_list.exp_name{19} = 'exp1_b8';
exp_list.exp_name{20} = 'exp1_b16';

exp_list.sequence{17} = 'andrew9';
exp_list.sequence{18} = 'andrew9';
exp_list.sequence{19} = 'andrew9';
exp_list.sequence{20} = 'andrew9';

parfor i=1:length(exp_list.bsize)
    b = exp_list.bsize{i};
    experiment = exp_list.exp_name{i};
    sequence = exp_list.sequence{i};
    encoder_RAGFT_RLGR_large_blocks(dataset,sequence,b,colorStep,experiment);
end
%% Experiment 1: 8iVFB data:
clear;
addpath(genpath('RA-GFT'));

dataset='8iVFBv2';
sequence = 'redandblack';
sequence = 'soldier';
sequence = 'longdress';
sequence = 'loot';

%colorStep = [1 2 4 8 16 32 64];
colorStep = [1 2 4 8 12 16 20 24 32 64];

%for redandblack
exp_list.bsize{1} = [2 2 2 2 2 2 2 2 2 2]; %first experiment, first level(L) bsize=2, all remaining levels bsize2
exp_list.bsize{2} = [2 2 2 2 2 2 2 2 4]; %first experiment, first level(L) bsize=4, all remaining levels bsize2
exp_list.bsize{3} = [2 2 2 2 2 2 2 8];
exp_list.bsize{4} = [2 2 2 2 2 2 16];

exp_list.exp_name{1} = 'exp1_b2';
exp_list.exp_name{2} = 'exp1_b4';
exp_list.exp_name{3} = 'exp1_b8';
exp_list.exp_name{4} = 'exp1_b16';

exp_list.sequence{1} = 'redandblack';
exp_list.sequence{2} = 'redandblack';
exp_list.sequence{3} = 'redandblack';
exp_list.sequence{4} = 'redandblack';

%for longdress
exp_list.bsize{5} = [2 2 2 2 2 2 2 2 2 2]; %first experiment, first level(L) bsize=2, all remaining levels bsize2
exp_list.bsize{6} = [2 2 2 2 2 2 2 2 4]; %first experiment, first level(L) bsize=4, all remaining levels bsize2
exp_list.bsize{7} = [2 2 2 2 2 2 2 8];
exp_list.bsize{8} = [2 2 2 2 2 2 16];

exp_list.exp_name{5} = 'exp1_b2';
exp_list.exp_name{6} = 'exp1_b4';
exp_list.exp_name{7} = 'exp1_b8';
exp_list.exp_name{8} = 'exp1_b16';

exp_list.sequence{5} = 'longdress';
exp_list.sequence{6} = 'longdress';
exp_list.sequence{7} = 'longdress';
exp_list.sequence{8} = 'longdress';

%for soldier
exp_list.bsize{9} = [2 2 2 2 2 2 2 2 2 2]; %first experiment, first level(L) bsize=2, all remaining levels bsize2
exp_list.bsize{10} = [2 2 2 2 2 2 2 2 4]; %first experiment, first level(L) bsize=4, all remaining levels bsize2
exp_list.bsize{11} = [2 2 2 2 2 2 2 8];
exp_list.bsize{12} = [2 2 2 2 2 2 16];

exp_list.exp_name{9} = 'exp1_b2';
exp_list.exp_name{10} = 'exp1_b4';
exp_list.exp_name{11} = 'exp1_b8';
exp_list.exp_name{12} = 'exp1_b16';

exp_list.sequence{9} = 'soldier';
exp_list.sequence{10} = 'soldier';
exp_list.sequence{11} = 'soldier';
exp_list.sequence{12} = 'soldier';

%for loot
exp_list.bsize{13} = [2 2 2 2 2 2 2 2 2 2]; %first experiment, first level(L) bsize=2, all remaining levels bsize2
exp_list.bsize{14} = [2 2 2 2 2 2 2 2 4]; %first experiment, first level(L) bsize=4, all remaining levels bsize2
exp_list.bsize{15} = [2 2 2 2 2 2 2 8];
exp_list.bsize{16} = [2 2 2 2 2 2 16];

exp_list.exp_name{13} = 'exp1_b2';
exp_list.exp_name{14} = 'exp1_b4';
exp_list.exp_name{15} = 'exp1_b8';
exp_list.exp_name{16} = 'exp1_b16';

exp_list.sequence{13} = 'loot';
exp_list.sequence{14} = 'loot';
exp_list.sequence{15} = 'loot';
exp_list.sequence{16} = 'loot';

%additional experiment with bsize=32 at level L
%for all 8ivfb sequences
exp_list.bsize{17} = [2 2 2 2 2 32]; %
exp_list.bsize{18} = [2 2 2 2 2 32]; %
exp_list.bsize{19} = [2 2 2 2 2 32];
exp_list.bsize{20} = [2 2 2 2 2 32];

exp_list.exp_name{17} = 'exp1_b32';
exp_list.exp_name{18} = 'exp1_b32';
exp_list.exp_name{19} = 'exp1_b32';
exp_list.exp_name{20} = 'exp1_b32';

exp_list.sequence{17} = 'longdress';
exp_list.sequence{18} = 'soldier';
exp_list.sequence{19} = 'redandblack';
exp_list.sequence{20} = 'loot';
parfor i=17:20
    b = exp_list.bsize{i};
    experiment = exp_list.exp_name{i};
    sequence = exp_list.sequence{i};
    encoder_RAGFT_RLGR_large_blocks(dataset,sequence,b,colorStep,experiment);
end
%% Experiment 2: effect of having block size 4 at lower resolution levels
%for sarah
exp_list.bsize{1} = [2 2 2 2 2 4 4]; %first experiment, first level(L) bsize=2, all remaining levels bsize2
exp_list.bsize{2} = [ 2 2 2 4 4 4]; %first experiment, first level(L) bsize=4, all remaining levels bsize2
exp_list.bsize{3} = [2  4 4 4 4];

exp_list.exp_name{1} = 'exp2_b44';
exp_list.exp_name{2} = 'exp2_b444';
exp_list.exp_name{3} = 'exp2_b4444';

exp_list.sequence{1} = 'sarah9';
exp_list.sequence{2} = 'sarah9';
exp_list.sequence{3} = 'sarah9';
%for andrew
exp_list.bsize{4} = [2 2 2 2 2 4 4]; %first experiment, first level(L) bsize=2, all remaining levels bsize2
exp_list.bsize{5} = [ 2 2 2 4 4 4]; %first experiment, first level(L) bsize=4, all remaining levels bsize2
exp_list.bsize{6} = [2  4 4 4 4];

exp_list.exp_name{4} = 'exp2_b44';
exp_list.exp_name{5} = 'exp2_b444';
exp_list.exp_name{6} = 'exp2_b4444';

exp_list.sequence{4} = 'andrew9';
exp_list.sequence{5} = 'andrew9';
exp_list.sequence{6} = 'andrew9';
%for david9
exp_list.bsize{7} = [2 2 2 2 2 4 4]; %first experiment, first level(L) bsize=2, all remaining levels bsize2
exp_list.bsize{8} = [ 2 2 2 4 4 4]; %first experiment, first level(L) bsize=4, all remaining levels bsize2
exp_list.bsize{9} = [2  4 4 4 4];

exp_list.exp_name{7} = 'exp2_b44';
exp_list.exp_name{8} = 'exp2_b444';
exp_list.exp_name{9} = 'exp2_b4444';

exp_list.sequence{7} = 'david9';
exp_list.sequence{8} = 'david9';
exp_list.sequence{9} = 'david9';
%for phil9
exp_list.bsize{10} = [2 2 2 2 2 4 4]; %first experiment, first level(L) bsize=2, all remaining levels bsize2
exp_list.bsize{11} = [ 2 2 2 4 4 4]; %first experiment, first level(L) bsize=4, all remaining levels bsize2
exp_list.bsize{12} = [2  4 4 4 4];

exp_list.exp_name{10} = 'exp2_b44';
exp_list.exp_name{11} = 'exp2_b444';
exp_list.exp_name{12} = 'exp2_b4444';

exp_list.sequence{10} = 'phil9';
exp_list.sequence{11} = 'phil9';
exp_list.sequence{12} = 'phil9';

%for phil9
exp_list.bsize{13} = [2 2 2 2 2 4 4]; %first experiment, first level(L) bsize=2, all remaining levels bsize2
exp_list.bsize{14} = [ 2 2 2 4 4 4]; %first experiment, first level(L) bsize=4, all remaining levels bsize2
exp_list.bsize{15} = [2  4 4 4 4];

exp_list.exp_name{13} = 'exp2_b44';
exp_list.exp_name{14} = 'exp2_b444';
exp_list.exp_name{15} = 'exp2_b4444';

exp_list.sequence{13} = 'ricardo9';
exp_list.sequence{14} = 'ricardo9';
exp_list.sequence{15} = 'ricardo9';


parfor i=1:length(exp_list.bsize)
    b = exp_list.bsize{i};
    experiment = exp_list.exp_name{i};
    sequence = exp_list.sequence{i};
    encoder_RAGFT_RLGR_large_blocks(dataset,sequence,b,colorStep,experiment);
end

%% 8iVFBv2 experiment for Zhang's approach (1 level with fixed block size)
%experiment is called
clear;
experiment = 'exp_zhang';

addpath(genpath('RA-GFT'));

dataset='8iVFBv2';
sequence = 'redandblack';
sequence = 'soldier';
sequence = 'longdress';
sequence = 'loot';

%colorStep = [1 2 4 8 16 32 64];
colorStep = [1 2 4 8 12 16 20 24 32 64];

%for redandblack
exp_list.bsize{1} = 2; %first experiment, first level(L) bsize=2, all remaining levels bsize2
exp_list.bsize{2} =  4; %first experiment, first level(L) bsize=4, all remaining levels bsize2
exp_list.bsize{3} =  8;
exp_list.bsize{4} = 16;

exp_list.exp_name{1} = [experiment,'_b2'];
exp_list.exp_name{2} = [experiment,'_b4'];
exp_list.exp_name{3} = [experiment,'_b8'];
exp_list.exp_name{4} = [experiment,'_b16'];

exp_list.sequence{1} = 'redandblack';
exp_list.sequence{2} = 'redandblack';
exp_list.sequence{3} = 'redandblack';
exp_list.sequence{4} = 'redandblack';

%for longdress
exp_list.bsize{5} =  2; %first experiment, first level(L) bsize=2, all remaining levels bsize2
exp_list.bsize{6} =  4; %first experiment, first level(L) bsize=4, all remaining levels bsize2
exp_list.bsize{7} =  8;
exp_list.bsize{8} =  16;

exp_list.exp_name{5} = [experiment,'_b2'];
exp_list.exp_name{6} = [experiment,'_b4'];
exp_list.exp_name{7} = [experiment,'_b8'];
exp_list.exp_name{8} = [experiment,'_b16'];

exp_list.sequence{5} = 'longdress';
exp_list.sequence{6} = 'longdress';
exp_list.sequence{7} = 'longdress';
exp_list.sequence{8} = 'longdress';

%for soldier
exp_list.bsize{9} =  2; %first experiment, first level(L) bsize=2, all remaining levels bsize2
exp_list.bsize{10} =  4; %first experiment, first level(L) bsize=4, all remaining levels bsize2
exp_list.bsize{11} =  8;
exp_list.bsize{12} =  16;

exp_list.exp_name{9} = [experiment,'_b2'];
exp_list.exp_name{10} = [experiment,'_b4'];
exp_list.exp_name{11} = [experiment,'_b8'];
exp_list.exp_name{12} = [experiment,'_b16'];

exp_list.sequence{9} = 'soldier';
exp_list.sequence{10} = 'soldier';
exp_list.sequence{11} = 'soldier';
exp_list.sequence{12} = 'soldier';

%for loot
exp_list.bsize{13} =  2; %first experiment, first level(L) bsize=2, all remaining levels bsize2
exp_list.bsize{14} = 4; %first experiment, first level(L) bsize=4, all remaining levels bsize2
exp_list.bsize{15} =  8;
exp_list.bsize{16} = 16;

exp_list.exp_name{13} = [experiment,'_b2'];
exp_list.exp_name{14} = [experiment,'_b4'];
exp_list.exp_name{15} = [experiment,'_b8'];
exp_list.exp_name{16} = [experiment,'_b16'];

exp_list.sequence{13} = 'loot';
exp_list.sequence{14} = 'loot';
exp_list.sequence{15} = 'loot';
exp_list.sequence{16} = 'loot';


parfor i=1:16
    b = exp_list.bsize{i};
    experiment = exp_list.exp_name{i};
    sequence = exp_list.sequence{i};
    encoder_RAGFT_RLGR_large_blocks(dataset,sequence,b,colorStep,experiment);
end


%% MVUB experiment for Zhang's approach (1 level with fixed block size)
%experiment is called
clear;
experiment = 'exp_zhang';

addpath(genpath('RA-GFT'));

dataset = 'MVUB';
sequence =  'andrew9';
sequence =  'david9';
sequence =  'phil9';
sequence =  'ricardo9';
sequence =  'sarah9';

%colorStep = [1 2 4 8 16 32 64];
colorStep = [1 2 4 8 12 16 20 24 32 64];

%for andrew9
exp_list.bsize{1} = 2; %first experiment, first level(L) bsize=2, all remaining levels bsize2
exp_list.bsize{2} =  4; %first experiment, first level(L) bsize=4, all remaining levels bsize2
exp_list.bsize{3} =  8;
exp_list.bsize{4} = 16;

exp_list.exp_name{1} = [experiment,'_b2'];
exp_list.exp_name{2} = [experiment,'_b4'];
exp_list.exp_name{3} = [experiment,'_b8'];
exp_list.exp_name{4} = [experiment,'_b16'];

exp_list.sequence{1} = 'andrew9';
exp_list.sequence{2} = 'andrew9';
exp_list.sequence{3} = 'andrew9';
exp_list.sequence{4} = 'andrew9';

%for david9
exp_list.bsize{5} =  2; %first experiment, first level(L) bsize=2, all remaining levels bsize2
exp_list.bsize{6} =  4; %first experiment, first level(L) bsize=4, all remaining levels bsize2
exp_list.bsize{7} =  8;
exp_list.bsize{8} =  16;

exp_list.exp_name{5} = [experiment,'_b2'];
exp_list.exp_name{6} = [experiment,'_b4'];
exp_list.exp_name{7} = [experiment,'_b8'];
exp_list.exp_name{8} = [experiment,'_b16'];

exp_list.sequence{5} = 'david9';
exp_list.sequence{6} = 'david9';
exp_list.sequence{7} = 'david9';
exp_list.sequence{8} = 'david9';

%for phil9
exp_list.bsize{9} =  2; %first experiment, first level(L) bsize=2, all remaining levels bsize2
exp_list.bsize{10} =  4; %first experiment, first level(L) bsize=4, all remaining levels bsize2
exp_list.bsize{11} =  8;
exp_list.bsize{12} =  16;

exp_list.exp_name{9} = [experiment,'_b2'];
exp_list.exp_name{10} = [experiment,'_b4'];
exp_list.exp_name{11} = [experiment,'_b8'];
exp_list.exp_name{12} = [experiment,'_b16'];

exp_list.sequence{9} = 'phil9';
exp_list.sequence{10} = 'phil9';
exp_list.sequence{11} = 'phil9';
exp_list.sequence{12} = 'phil9';

%for ricardo9
exp_list.bsize{13} =  2; %first experiment, first level(L) bsize=2, all remaining levels bsize2
exp_list.bsize{14} = 4; %first experiment, first level(L) bsize=4, all remaining levels bsize2
exp_list.bsize{15} =  8;
exp_list.bsize{16} = 16;

exp_list.exp_name{13} = [experiment,'_b2'];
exp_list.exp_name{14} = [experiment,'_b4'];
exp_list.exp_name{15} = [experiment,'_b8'];
exp_list.exp_name{16} = [experiment,'_b16'];

exp_list.sequence{13} = 'ricardo9';
exp_list.sequence{14} = 'ricardo9';
exp_list.sequence{15} = 'ricardo9';
exp_list.sequence{16} = 'ricardo9';

%for sarah9
exp_list.bsize{17} =  2; %first experiment, first level(L) bsize=2, all remaining levels bsize2
exp_list.bsize{18} = 4; %first experiment, first level(L) bsize=4, all remaining levels bsize2
exp_list.bsize{19} =  8;
exp_list.bsize{20} = 16;

exp_list.exp_name{17} = [experiment,'_b2'];
exp_list.exp_name{18} = [experiment,'_b4'];
exp_list.exp_name{19} = [experiment,'_b8'];
exp_list.exp_name{20} = [experiment,'_b16'];

exp_list.sequence{17} = 'sarah9';
exp_list.sequence{18} = 'sarah9';
exp_list.sequence{19} = 'sarah9';
exp_list.sequence{20} = 'sarah9';


parfor i=1:20
    b = exp_list.bsize{i};
    experiment = exp_list.exp_name{i};
    sequence = exp_list.sequence{i};
    encoder_RAGFT_RLGR_large_blocks(dataset,sequence,b,colorStep,experiment);
end


%% Experiment 2: effect of having block size 4 at lower resolution levels
dataset='8iVFBv2';

%for loot
exp_list.bsize{1} = [2 2 2 2 2 4 4]; %first experiment, first level(L) bsize=2, all remaining levels bsize2
exp_list.bsize{2} = [ 2 2 2 4 4 4]; %first experiment, first level(L) bsize=4, all remaining levels bsize2
exp_list.bsize{3} = [2  4 4 4 4];

exp_list.exp_name{1} = 'exp2_b44';
exp_list.exp_name{2} = 'exp2_b444';
exp_list.exp_name{3} = 'exp2_b4444';

exp_list.sequence{1} = 'loot';
exp_list.sequence{2} = 'loot';
exp_list.sequence{3} = 'loot';
%for soldier
exp_list.bsize{4} = [2 2 2 2 2 4 4]; %first experiment, first level(L) bsize=2, all remaining levels bsize2
exp_list.bsize{5} = [ 2 2 2 4 4 4]; %first experiment, first level(L) bsize=4, all remaining levels bsize2
exp_list.bsize{6} = [2  4 4 4 4];

exp_list.exp_name{4} = 'exp2_b44';
exp_list.exp_name{5} = 'exp2_b444';
exp_list.exp_name{6} = 'exp2_b4444';

exp_list.sequence{4} = 'soldier';
exp_list.sequence{5} = 'soldier';
exp_list.sequence{6} = 'soldier';
%for redandblack
exp_list.bsize{7} = [2 2 2 2 2 4 4]; %first experiment, first level(L) bsize=2, all remaining levels bsize2
exp_list.bsize{8} = [ 2 2 2 4 4 4]; %first experiment, first level(L) bsize=4, all remaining levels bsize2
exp_list.bsize{9} = [2  4 4 4 4];

exp_list.exp_name{7} = 'exp2_b44';
exp_list.exp_name{8} = 'exp2_b444';
exp_list.exp_name{9} = 'exp2_b4444';

exp_list.sequence{7} = 'redandblack';
exp_list.sequence{8} = 'redandblack';
exp_list.sequence{9} = 'redandblack';
%for longdress
exp_list.bsize{10} = [2 2 2 2 2 4 4]; %first experiment, first level(L) bsize=2, all remaining levels bsize2
exp_list.bsize{11} = [ 2 2 2 4 4 4]; %first experiment, first level(L) bsize=4, all remaining levels bsize2
exp_list.bsize{12} = [2  4 4 4 4];

exp_list.exp_name{10} = 'exp2_b44';
exp_list.exp_name{11} = 'exp2_b444';
exp_list.exp_name{12} = 'exp2_b4444';

exp_list.sequence{10} = 'longdress';
exp_list.sequence{11} = 'longdress';
exp_list.sequence{12} = 'longdress';


parfor i=1:length(exp_list.bsize)
    b = exp_list.bsize{i};
    experiment = exp_list.exp_name{i};
    sequence = exp_list.sequence{i};
    encoder_RAGFT_RLGR_large_blocks(dataset,sequence,b,colorStep,experiment);
end


%% another experiment with varying block size
% Experiment 2: effect of having block size 8 at first level, then 4 and 2 at lower resolutions
dataset='8iVFBv2';

%for loot
exp_list.bsize{1} = [2 2 2 2 2 4 8]; %first experiment, first level(L) bsize=2, all remaining levels bsize2
exp_list.bsize{2} = [ 2 2 2 4 4 8]; %first experiment, first level(L) bsize=4, all remaining levels bsize2
exp_list.bsize{3} = [2  2 4 8 8];
exp_list.bsize{4} = [2   8 8 8];

exp_list.exp_name{1} = 'exp2_b48';
exp_list.exp_name{2} = 'exp2_b448';
exp_list.exp_name{3} = 'exp2_b488';
exp_list.exp_name{4} = 'exp2_b888';

exp_list.sequence{1} = 'loot';
exp_list.sequence{2} = 'loot';
exp_list.sequence{3} = 'loot';
exp_list.sequence{4} = 'loot';

%for soldier
exp_list.bsize{5} = [2 2 2 2 2 4 8]; %first experiment, first level(L) bsize=2, all remaining levels bsize2
exp_list.bsize{6} = [ 2 2 2 4 4 8]; %first experiment, first level(L) bsize=4, all remaining levels bsize2
exp_list.bsize{7} = [2  2 4 8 8];
exp_list.bsize{8} = [2   8 8 8];

exp_list.exp_name{5} = 'exp2_b48';
exp_list.exp_name{6} = 'exp2_b448';
exp_list.exp_name{7} = 'exp2_b488';
exp_list.exp_name{8} = 'exp2_b888';


exp_list.sequence{5} = 'soldier';
exp_list.sequence{6} = 'soldier';
exp_list.sequence{7} = 'soldier';
exp_list.sequence{8} = 'soldier';

%for redandblack
exp_list.bsize{9} = [2 2 2 2 2 4 8]; %first experiment, first level(L) bsize=2, all remaining levels bsize2
exp_list.bsize{10} = [ 2 2 2 4 4 8]; %first experiment, first level(L) bsize=4, all remaining levels bsize2
exp_list.bsize{11} = [2  2 4 8 8];
exp_list.bsize{12} = [2   8 8 8];

exp_list.exp_name{9} = 'exp2_b48';
exp_list.exp_name{10} = 'exp2_b448';
exp_list.exp_name{11} = 'exp2_b488';
exp_list.exp_name{12} = 'exp2_b888';


exp_list.sequence{9} = 'redandblack';
exp_list.sequence{10} = 'redandblack';
exp_list.sequence{11} = 'redandblack';
exp_list.sequence{12} = 'redandblack';

%for longdress
exp_list.bsize{13} = [2 2 2 2 2 4 8]; %first experiment, first level(L) bsize=2, all remaining levels bsize2
exp_list.bsize{14} = [ 2 2 2 4 4 8]; %first experiment, first level(L) bsize=4, all remaining levels bsize2
exp_list.bsize{15} = [2  2 4 8 8];
exp_list.bsize{16} = [2   8 8 8];

exp_list.exp_name{13} = 'exp2_b48';
exp_list.exp_name{14} = 'exp2_b448';
exp_list.exp_name{15} = 'exp2_b488';
exp_list.exp_name{16} = 'exp2_b888';

exp_list.sequence{13} = 'longdress';
exp_list.sequence{14} = 'longdress';
exp_list.sequence{15} = 'longdress';
exp_list.sequence{16} = 'longdress';


parfor i=1:length(exp_list.bsize)
    b = exp_list.bsize{i};
    experiment = exp_list.exp_name{i};
    sequence = exp_list.sequence{i};
    encoder_RAGFT_RLGR_large_blocks(dataset,sequence,b,colorStep,experiment);
end



%% experiment 4, post icip, try new degree normalized graph with gaussian weights

clear;
addpath(genpath('RA-GFT'));

dataset='8iVFBv2';
sequence = 'redandblack';
sequence = 'soldier';
sequence = 'longdress';
sequence = 'loot';

%colorStep = [1 2 4 8 16 32 64];
colorStep = [1 2 4 8 12 16 20 24 32 64];

%for redandblack
exp_list.bsize{1} = [2 2 2 2 2 2 2 2 2 2]; %first experiment, first level(L) bsize=2, all remaining levels bsize2
exp_list.bsize{2} = [2 2 2 2 2 2 2 2 4]; %first experiment, first level(L) bsize=4, all remaining levels bsize2
exp_list.bsize{3} = [2 2 2 2 2 2 2 8];
exp_list.bsize{4} = [2 2 2 2 2 2 16];

exp_list.exp_name{1} = 'exp4_b2';
exp_list.exp_name{2} = 'exp4_b4';
exp_list.exp_name{3} = 'exp4_b8';
exp_list.exp_name{4} = 'exp4_b16';

exp_list.sequence{1} = 'redandblack';
exp_list.sequence{2} = 'redandblack';
exp_list.sequence{3} = 'redandblack';
exp_list.sequence{4} = 'redandblack';

%for longdress
exp_list.bsize{5} = [2 2 2 2 2 2 2 2 2 2]; %first experiment, first level(L) bsize=2, all remaining levels bsize2
exp_list.bsize{6} = [2 2 2 2 2 2 2 2 4]; %first experiment, first level(L) bsize=4, all remaining levels bsize2
exp_list.bsize{7} = [2 2 2 2 2 2 2 8];
exp_list.bsize{8} = [2 2 2 2 2 2 16];

exp_list.exp_name{5} = 'exp4_b2';
exp_list.exp_name{6} = 'exp4_b4';
exp_list.exp_name{7} = 'exp4_b8';
exp_list.exp_name{8} = 'exp4_b16';

exp_list.sequence{5} = 'longdress';
exp_list.sequence{6} = 'longdress';
exp_list.sequence{7} = 'longdress';
exp_list.sequence{8} = 'longdress';

%for soldier
exp_list.bsize{9} = [2 2 2 2 2 2 2 2 2 2]; %first experiment, first level(L) bsize=2, all remaining levels bsize2
exp_list.bsize{10} = [2 2 2 2 2 2 2 2 4]; %first experiment, first level(L) bsize=4, all remaining levels bsize2
exp_list.bsize{11} = [2 2 2 2 2 2 2 8];
exp_list.bsize{12} = [2 2 2 2 2 2 16];

exp_list.exp_name{9} = 'exp4_b2';
exp_list.exp_name{10} = 'exp4_b4';
exp_list.exp_name{11} = 'exp4_b8';
exp_list.exp_name{12} = 'exp4_b16';

exp_list.sequence{9} = 'soldier';
exp_list.sequence{10} = 'soldier';
exp_list.sequence{11} = 'soldier';
exp_list.sequence{12} = 'soldier';

%for loot
exp_list.bsize{13} = [2 2 2 2 2 2 2 2 2 2]; %first experiment, first level(L) bsize=2, all remaining levels bsize2
exp_list.bsize{14} = [2 2 2 2 2 2 2 2 4]; %first experiment, first level(L) bsize=4, all remaining levels bsize2
exp_list.bsize{15} = [2 2 2 2 2 2 2 8];
exp_list.bsize{16} = [2 2 2 2 2 2 16];

exp_list.exp_name{13} = 'exp4_b2';
exp_list.exp_name{14} = 'exp4_b4';
exp_list.exp_name{15} = 'exp4_b8';
exp_list.exp_name{16} = 'exp4_b16';

exp_list.sequence{13} = 'loot';
exp_list.sequence{14} = 'loot';
exp_list.sequence{15} = 'loot';
exp_list.sequence{16} = 'loot';

parfor i=1:16
    b = exp_list.bsize{i};
    experiment = exp_list.exp_name{i};
    sequence = exp_list.sequence{i};
    encoder_RAGFT_RLGR_large_blocks(dataset,sequence,b,colorStep,experiment);
end