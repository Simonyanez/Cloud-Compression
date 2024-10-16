% TODO: HACER ESTEEEEE PARA MI TRANSFORMADA

% Authors - Eduardo Pavez <eduardo.pavez.carvelli@gmail.com, pavezcar@usc.edu>
% Copyright Eduardo Pavez,  University of Southern California
%Los Angeles, USA, 05/30/2020
% E. Pavez, B. Girault, A. Ortega, and P. A. Chou. 
%"Region adaptive graph Fourier transform for 3D point clouds". 
%IEEE International Conference on Image Processing (ICIP), 2020
%https://arxiv.org/abs/2003.01866
%% script to test RA-GFT and its inverse
%encoder RA-GFT
clear;
filename = 'longdress_vox10_1051.ply';
[V,Crgb,J] = ply_read8i(filename);              % V lista de puntos, Crgb lista de colores, J resolución de voxel 2^10 se dividio 10 veces en cada cubo ahí un punto en cada voxel
N = size(V,1);
C = RGBtoYUV(Crgb); %transform to YUV

%%
bsize=[ 2 2 2 2 2 2  2 2 2 2]; % Tamaño de cada nivel en caso multilevel = 0. Tamaño 2 en que se dividen los bloques a lo más 2^3 8 puntos máximo. (8, 16)
param.V=V;
param.J=J;
param.bsize = bsize;
param.isMultiLevel=0;           % Solo una vez, setearlo a cero
tic;

step = 64;
%C = ones(N,1);
[Coeff, Gfreq, weights]  = Region_Adaptive_GFT( C, param ); % param posee la información definida
toc;
Y = Coeff(:,1);
Coeff_quant = round(Coeff/step)*step;
%%
tic;
[ start_indices, end_indices, V_MR, Crec ] = iRegion_Adaptive_GFT( Coeff_quant, param );
toc;

Crgb_rec = double(YUVtoRGB(Crec));

psnr_Y = -10*log10(norm(Y - Coeff_quant(:,1))^2/(N*255^2));

%%
 ply_write('PC_original.ply',V,Crgb,[]);
 ply_write('PC_coded.ply',V,Crgb_rec,[]);

