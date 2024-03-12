function [W,edge] = bf_graph(V, C)

  N = size(V,1);
  
  squared_norms = sum(V.^2,2);   %Distancia cuadrada de cada fila de V
  D = sqrt(repmat(squared_norms,1,N) + repmat(squared_norms',N,1) - 2*(V*V'));  % Distancias cuadradas entre todos los puntos
  
      
  th =sqrt(3)+0.00001;              % Umbral de distancia máxima de los puntos
      
   
  
  iD = D.^(-1);                     % Inverso escalar de los elementos de la matriz, pesos es el inverso de la distancia
  iD(find(D > th)) = 0;             % Encontrar todas las distancias que sean mayores al umbral y evaluarlas en 0
  iD(find(D==0))   =0;              % Además encontrar aquellas que sean nulas, es decir, self-connections
  
  idx = find(iD~=0);                 
  [I, J] = ind2sub( size(D), idx ); % Se identifica cuales son los nodos conectados
  
  edge = [I,J];
  W_aux = D;                        % Conjunto auxiliar para poder ponderar las distancias
                                    % Matriz cuadrada de distancias entre los nodos
  %D_aux(find(D > th)) = 0;      
  %D_aux(find(D==0)) = 0;
  
  YUV_block_double = double(C);   % Bloque en formato YUV
  YUV_block_normed = YUV_block_double/256;           % Normalización
  
  size(J,1);
  size(YUV_block_normed(1,1));
  size(D);
  size(std(D(:)));
  size(std(YUV_block_normed(:,1)));
  for id = 1:size(J,1)
      i = I(id);
      j = J(id);
      W_aux(i,j) = exp(-(D(i,j)^2)/(2*std(D(:))^2))*exp(-(YUV_block_normed(i,1) - YUV_block_normed(j,1))^2)/(2*std(YUV_block_normed(:,1))^2);  % Utilizamos solo la información de luminscencia. Ponderamos la diferencia                                    %entre nodos como parámetro para los pesos, 
                                                                                                              %disminuye la distancia, aumenta la relevancia si el cambio de color es más brusco.
  end
  
  W=W_aux' + W_aux;    
  
end