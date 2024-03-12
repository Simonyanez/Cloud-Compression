%
% clear;
% addpath(genpath('RA-GFT'));
function [] = encoder_RAGFT_RLGR_large_blocks(dataset,sequence,b,colorStep,experiment)

T = get_pointCloud_nFrames(dataset, sequence );
nSteps = length(colorStep);
bytes = zeros(T,nSteps);
MSE  = bytes;
Nvox = zeros(T,1);
time = Nvox;
%


param.bsize = b;


for frame =1:T
    tic;
    [ V,Crgb,J ] = get_pointCloud(dataset, sequence,  frame );
    N = size(V,1);
    Nvox(frame) = N;
    C = RGBtoYUV(Crgb); %transform to YUV
    
    %
    %[Coeff,w]=RAHT(C,ListC,FlagsC,weightsC);
    param.V=V;
    param.J=J;
    last_idx = min(length(experiment),9);
    if(strcmp('exp_zhang',experiment(1:last_idx)))
        param.isMultiLevel=0;
    else
        param.isMultiLevel=1;
    end
    [Coeff, Gfreq, weights]  = Region_Adaptive_GFT( C, param );
    Y = Coeff(:,1);
    
    for i=1:nSteps
        %quantize coeffs
        step = colorStep(i);
        Coeff_enc = round(Coeff/step);
        Y_hat = Coeff_enc(:,1)*step;
        %U_hat = Coeff_enc(:,2)*step;
        %V_hat = Coeff_enc(:,3)*step;
        
        %comptue squared error
        MSE(frame,i) = (norm(Y-Y_hat)^2/(N*255^2));
        
        %encode coeffs using RLGR
        [nbytesY,~]=RLGR_encoder(Coeff_enc(:,1));
        [nbytesU,~]=RLGR_encoder(Coeff_enc(:,2));
        [nbytesV,~]=RLGR_encoder(Coeff_enc(:,3));
        bytes(frame,i) = nbytesY + nbytesU + nbytesV;
        
    end
    time(frame)=toc;
    
    fprintf('%s/%s/\t %s\t %f\t %d\t %d\n',dataset,sequence, experiment, time(frame),  frame, T);
    
    %disp(disp_str);
end

%
folder = sprintf('RA-GFT/results/%s/%s/',dataset,sequence);
mkdir(folder);
filename  = sprintf('%s_RA-GFT_%s.mat',folder,experiment);
save(filename,'MSE','bytes','Nvox','b','colorStep','time');