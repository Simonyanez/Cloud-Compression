%compute distance based graph from 
%Zhang, Cha, Dinei Flor�ncio, and Charles Loop. 
%"Point cloud attribute compression with graph transform." 
%Image Processing (ICIP), 2014 IEEE International Conference on. IEEE, 2014.
function [W,edge] = compute_graph_MSR(V, th)
% V: nx3. n points
% th: threshold to construct the graph
  N = size(V,1);
  %

  if(nargin==1)
      
      th =sqrt(3)+0.00001;
      
  end
  %compute EDM 
  squared_norms = sum(V.^2,2);
  D = sqrt(repmat(squared_norms,1,N) + repmat(squared_norms',N,1) - 2*(V*V'));
  % D = squareform(pdist(coords, 'euclidean')); % pairwise distances, n-by-n
  % matrix% only use pdist if have the statistics/ML toolbox
  % Formula tan conocida. 
  iD = D.^(-1);                     % Inverso escalar de los elementos de la matriz, pesos es el inverso de la distancia
  iD(find(D > th)) = 0;             % Encontrar todas las distancias que sean mayores al umbral y evaluarlas en 0
  iD(find(D==0))   =0;              % Adem�s encontrar aquellas que sean nulas, es decir, self-connections
  W=iD' + iD;                       % Matriz de adyacencia, agrega la transpuesta para hacer la matriz sim�trica

  idx = find(iD~=0);                

  [I, J] = ind2sub( size(D), idx ); % Se identifica cuales son los nodos conectados

   edge = [I, J];

end