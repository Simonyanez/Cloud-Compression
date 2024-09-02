%CONFIDENTIAL (C) Mitsubishi Electric Research Labs (MERL) 2017 Eduardo
%Pavez 08/17/2017
function [ nbits,bytes ] = octreeByteCount( V, maxDepth )
%
%   This function computes number of bits used by an octree representation
%   of a voxelized point cloud, without actually finding the octree.
% Inspired in formula from Queiroz, Chou, Region adaptive hierarchical
% transform, IEE TIP.

width=2^(-maxDepth);

%first make sure points are voxelized
Vint=floor(V/width)*width;
nbits=0;
%the code basically, goes from leaves of octree to the root, counting
%number of points at each level, and adding up bytes
bytes=zeros(maxDepth,1);
for j=1:maxDepth
   
    Vj=floor(Vint/(2^j))*(2^j);
    [ Mj ] = get_morton_code( Vj,maxDepth);
    bytes(j)=size(unique(Mj),1);
    
end
nbits=8*sum(bytes);



end

