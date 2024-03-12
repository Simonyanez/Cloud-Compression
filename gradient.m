function [G_vec,W,edge] = gradient(V, C)

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
  D_aux = D;                        % Conjunto auxiliar para poder ponderar las distancias
                                    % Matriz cuadrada de distancias entre los nodos
  %D_aux(find(D > th)) = 0;      
  %D_aux(find(D==0)) = 0;
  
  YUV_block_double = double(C);   % Bloque en formato YUV
  YUV_block_normed = YUV_block_double/256;           % Normalización
  
  c_len = size(C(:,1));
  G_vec = zeros(c_len(1), c_len(1));
  size(J,1);
  for id = 1:size(J,1)
      i = I(id);
      j = J(id);
      D_aux(i,j) = D_aux(i,j)*(1+abs(YUV_block_normed(i,1) - YUV_block_normed(j,1)));  % Utilizamos solo la información de luminscencia. Ponderamos la diferencia
      G_vec(i,j) = abs(YUV_block_normed(j,1) - YUV_block_normed(i,1));                                        %entre nodos como parámetro para los pesos, 
                                                                                                              %disminuye la distancia, aumenta la relevancia si el cambio de color es más brusco.
  end
  
  iD_aux = D_aux.^(-1);         % Si la distancia es grande (< similitud) ponderado por si el cambio de color es grande (> relevancia)
  iD_aux(find(D > th))=0;
  iD_aux(find(D == 0))=0;
  W=iD_aux' + iD_aux;    
  
end