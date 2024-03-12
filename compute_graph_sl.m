function [W,edge,degrees,iD,idx_closest] = compute_graph_sl(V,distance_vector,weights, th)
% V: nx3. n points
% th: threshold to construct the graph
  N = size(V,1);
  %

  if(nargin==3)
      
      th =sqrt(3)+0.00001;
      
  end
  
% Calculate the mean along each dimension (X, Y, Z)
mean_X = mean(V(:, 1));
mean_Y = mean(V(:, 2));
mean_Z = mean(V(:, 3));

% Calculate the center (mean) of the point cloud
mean_point_cloud = [mean_X, mean_Y, mean_Z];

% Subtract the mean values from each corresponding dimension to center the point cloud
centered_V = V - mean_point_cloud;
  % Calculate dot products of points with the mean direction vector
mean_direction = sum(distance_vector .* weights) / sum(weights);  % Promedio ponderado según la intensidad del cambio de color
mean_direction = mean_direction/norm(mean_direction);     


%size(mean_direction)

% Count the number of values below the threshold
%below_threshold = round(0.2 * numel(sort(dot_products)))
%[~, sorted_indices] = sort(dot_products);
%idx_closest = sorted_indices(1:below_threshold);  % Segundo argumento threshold
% Find the index of the point closest and furthest along the mean direction   %Índices dentro del vector de Volumen del bloque
%[~, idx_furthest] = max(dot_products);

% Retrieve the points closest and furthest along the mean direction
  
  %compute EDM 
  squared_norms = sum(V.^2,2);
  D = sqrt(repmat(squared_norms,1,N) + repmat(squared_norms',N,1) - 2*(V*V'));
  % D = squareform(pdist(coords, 'euclidean')); % pairwise distances, n-by-n
  % matrix% only use pdist if have the statistics/ML toolbox
  % Formula tan conocida.
  iD = D.^(-1);                     % Inverso escalar de los elementos de la matriz, pesos es el inverso de la distancia
  iD(find(D > th)) = 0;             % Encontrar todas las distancias que sean mayores al umbral y evaluarlas en 0
  iD(find(D==0))   =0;              % Además encontrar aquellas que sean nulas, es decir, self-connections
  
  degrees = zeros(size(iD,1),1);
  for i = 1:size(iD,1)
      degrees(i) = sum(iD(:,i));
  end
  
  degrees = degrees/norm(degrees);  
  %dot_products_degreed = dot_products.*degrees;
  % Count the number of values below the threshold
  %below_threshold = sum(degrees < mean(degrees));
  %[sorted_degrees, sorted_indices] = sort(degrees);
  %below_threshold = round(0.2 * numel(sorted_degrees));
  
  [sorted_degrees, sorted_indices] = sort(degrees);
  below_threshold = round(0.2 * numel(sorted_degrees));

  idx_closest_original = sorted_indices(1:below_threshold);
 
% Select the corresponding vectors from centered_V using idx_closest_original
  selected_vectors = centered_V(idx_closest_original, :);

% Calculate dot products between selected vectors and the mean_direction
  dot_products_degreed = selected_vectors * mean_direction';

% Find the new indices of the closest points along the mean direction
  [~, idx_closest] = sort(dot_products_degreed);

% Use the new indices to reorder the original indices
  below_threshold = round(0.2 * numel(sort(dot_products_degreed)));
  if below_threshold > 0
    idx_closest = idx_closest_original(idx_closest(1));
  end  
  W=iD' + iD;                       % Matriz de adyacencia, agrega la transpuesta para hacer la matriz simétrica

  idx = find(iD~=0);                

  [I, J] = ind2sub( size(D), idx ); % Se identifica cuales son los nodos conectados

   edge = [I, J];
   



end