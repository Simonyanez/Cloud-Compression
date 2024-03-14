function [G_vec,edge,distance_vectors,weights] = direction(V, C, aspect_ratio)

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
                                    % Conjunto auxiliar para poder ponderar las distancias
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
      j = J(id);                                                                              % Utilizamos solo la información de luminscencia. Ponderamos la diferencia
      G_vec(i,j) = YUV_block_normed(j,1) - YUV_block_normed(i,1);                                        %entre nodos como parámetro para los pesos, 
                                                                                                              %disminuye la distancia, aumenta la relevancia si el cambio de color es más brusco.
  end
  
  distance_vectors = zeros(size(G_vec,1),3);
  distance_indexes = zeros(size(G_vec,1),2);
  weights = zeros(size(G_vec,1),1);
  for iter = 1:size(G_vec,1)
    [min_val, min_index] = min(G_vec(:,iter));    % Se busca el mínimo cambio de color para el nodo j = iter (es decir, el más negativo)
    % Hay mínimos que son ceros, esto indicaría que no hay un cambio de
    % color signifactivo
    if abs(min_val) > 0.02 % Diferencia de color mayor 5.1 (normalizado)
        distance_indexes(iter,:) = ([min_index,iter]);
        weights(iter) = abs(G_vec(min_index,iter));
        dis_vec = (V(iter,:) - V(min_index,:));
    else
        dis_vec = [0 0 0]; % Distancia entre los nodos que presentan el cambio más abrupto de color
    end  
    if norm(dis_vec) ~= 0
      dis_vec = double(dis_vec)/norm(dis_vec);
      distance_vectors(iter,:) = dis_vec;
    else
      distance_vectors(iter,:) = double(dis_vec);  
    end
    
  %distance_vectors( abs(distance_vectors)<0.2) = [0. ,0. ,0.]
  end
  
  % Sample vectors (replace these with your actual vectors)
x = V(:,1); % X-coordinates of vector starting points
y = V(:,2); % Y-coordinates of vector starting points
z = V(:,3); % Z-coordinates of vector starting points

% Sample vector components (replace these with your actual vector components)
u = distance_vectors(:,1); % X-components of vectors
v = distance_vectors(:,2); % Y-components of vectors
w = distance_vectors(:,3); % Z-components of vectors

% Calculate magnitudes of vectors
magnitudes = sqrt(u.^2 + v.^2 + w.^2);

% Normalize vectors to unit length (divide components by magnitudes)
u_unit = u ./ magnitudes;
v_unit = v ./ magnitudes;
w_unit = w ./ magnitudes;

% Calculate super vector direction
r = mean(distance_vectors);
r = r/norm(r);
% Create a figure for the 3D plot
if exist('aspect_ratio', 'var')
figure;
hold on;

% Plot the unit vectors as arrows using quiver3
quiver3(x, y, z, u_unit, v_unit, w_unit, 0); % The last argument (0) specifies no automatic scaling

x_mult = max(x)-min(x) -1;
y_mult = max(y)-min(y) -1;
z_mult = max(z)-min(z) -1;
if r(1) < 0
 x_ax = max(x)  ;
else
 x_ax = min(x);
end
if r(2) < 0
 y_ax = max(y);
else
 y_ax = min(y);
end
if r(3) < 0
 z_ax = max(z);  
else
 z_ax = min(z);
end
h = quiver3(x_ax, y_ax,z_ax,r(1)*x_mult,r(2)*y_mult,r(3)*z_mult,'Color','r','LineWidth',6)
set(h, 'MaxHeadSize', 12);
% Set labels and title
xlabel('X-axis');
ylabel('Y-axis');
zlabel('Z-axis');
title('Unit Vectors in 3D');

% Get the current axes handle
ax = gca;

% Set plot properties (if needed)
grid on;
axis equal; % Set equal scaling for all axes

% Set the view to a 3D perspective
view(3);
view(60, 30)
% Set the aspect ratio to match scatter3 plot
daspect(ax, aspect_ratio); % Set the aspect ratio of the plot

% Adjust the axes scaling for a larger plot
axis([min(x)-1 max(x)+1 min(y)-1 max(y)+1 min(z)-1 max(z)+1]); % Customize the limits as per your requirement

% Show plot
hold off;
else
    disp('No aspect ratio entry. Skipping plotting...');
end
  
  %iD_aux = D_aux.^(-1);         % Si la distancia es grande (< similitud) ponderado por si el cambio de color es grande (> relevancia)
  %iD_aux(find(D > th))=0;
  %iD_aux(find(D == 0))=0;
  %W=iD_aux' + iD_aux;    
  
end