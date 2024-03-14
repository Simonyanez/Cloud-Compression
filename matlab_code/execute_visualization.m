clear;
filename = 'longdress_vox10_1051.ply';
[V,Crgb,J] = ply_read8i(filename);              % V lista de puntos, Crgb lista de colores, J resolución de voxel 2^10 se dividio 10 veces en cada cubo ahí un punto en cada voxel
N = size(V,1);
C = RGBtoYUV(Crgb); %transform to YUV
bsize=[16]; % Tamaño de cada nivel en caso multilevel = 0. Tamaño 2 en que se dividen los bloques a lo más 2^3 8 puntos máximo. (8, 16)
param.V=V;
param.J=J;
param.bsize = bsize;
param.isMultiLevel=0;           % Solo una vez, setearlo a cero
tic;

step = 64;
%C = ones(N,1);
%%
[Coeff, Gfreq, weights, Vblock, Ablock, Sorted_Blocks]  = block_visualization( C, param ); % param posee la información definida, C está el color
% Solo tenemos el bloque de la última ejecución
%toc;
%Y = Coeff(:,1);
%Coeff_quant = round(Coeff/step)*step;
%%
% Extrae el 5 y 6 valor de los bloques,
means = cellfun(@(x) x(6), Sorted_Blocks);
stds = cellfun(@(x) x(7), Sorted_Blocks);
% Encuentra los índices en orden para reordenar los bloques
[~, sortedIndices] = sort(cell2mat(means));
[~, sortedIndices2] = sort(cell2mat(stds));
% Sort the original cell array based on the sorted indices
Sorted_Blocks_1 = Sorted_Blocks(sortedIndices);
Sorted_Blocks_2 = Sorted_Blocks(sortedIndices2);
%%
stds = cell2mat(stds);
means = cell2mat(means);
%%
% Hacer Histograma
figure(1)
    histogram(stds)
    title(['Standard Deviation of Simple Gradient from ' 'Block - Size = ' num2str(bsize)])
    xlabel('Standard Deviation');
    ylabel('Count');
figure(2)
    histogram(means)
    title(['Mean of Simple Gradient from ' 'Block - Size = ' num2str(bsize)])
    xlabel('Mean');
    ylabel('Count');
%%
T_orig_all = {};
T_mod_all = {};
%%
% Veamos la transformada gráfica de fourier para cada bloque
for i = 1:numel(Sorted_Blocks)
    Adj = Sorted_Blocks{i}{3}; %SAD
    C_aux = Sorted_Blocks{i}{2};
    [GFT,Gfreq, Ahat] = compute_GFT_noQ(Adj,C_aux);
    T_mod_all{i} = {GFT,Gfreq, Ahat};
end
%%

for i = 1:numel(Sorted_Blocks)
    Vblock = Sorted_Blocks{i}{1}; %Volumenes
    Adj = Sorted_Blocks{i}{4}; %Distances
    C_aux = Sorted_Blocks{i}{2};
    [GFT,Gfreq, Ahat] = compute_GFT_noQ(Adj,C_aux);
    T_orig_all{i} = {GFT,Gfreq, Ahat, Vblock};
end
%% Bilateral (por implementar)
%for i = 1:numel(Sorted_Blocks)
%    Adj = Sorted_Blocks{i}{10}; %BF
%    C = Sorted_Blocks{i}{2};
%    [GFT,Gfreq, Ahat] = compute_GFT_noQ(Adj,C);
%    T_mod_2_all{i} = {GFT,Gfreq, Ahat};
%end

%%
% Clustering
means_var = cell2mat(cellfun(@(x) x(6), Sorted_Blocks))';
stds_var = cell2mat(cellfun(@(x) x(7), Sorted_Blocks))';
means_color = cell2mat(cellfun(@(x) x(8), Sorted_Blocks))';
stds_color = cell2mat(cellfun(@(x) x(9), Sorted_Blocks))';

non_nan_index = ~isnan(means_var) &  ~isnan(stds_var);

means_var = normalize_vector(means_var(non_nan_index));
stds_var = normalize_vector(stds_var(non_nan_index));
means_color = normalize_vector(means_color(non_nan_index));
stds_color = normalize_vector(stds_color(non_nan_index));

%X = [means_var, stds_var, means_color, stds_color];
X = [means_var, stds_var, stds_color];
k = 3;
[idx, centroids, ~, D] = kmeans(X, k);
%%
% Visualización componentes DC y AC
A_hat_orig = T_orig_all{604}{3};
A_hat_mod = T_mod_all{604}{3};
std = Sorted_Blocks_2{604}{7};
coeff_visualization(A_hat_orig, A_hat_mod, D(604,1),1)



%%
Cluster_1 = Sorted_Blocks(idx==1);
Cluster_2 = Sorted_Blocks(idx==2);
Cluster_3 = Sorted_Blocks(idx==3);

%%
% Calculate distances between data points and centroids
distances = zeros(size(X, 1), size(centroids, 1)); % Preallocate matrix
for i = 1:size(centroids, 1)
    distances(:, i) = sqrt(sum((X - centroids(i, :)).^2, 2)); % Euclidean distance
end

%%
num_smallest = 3;

% Initialize variables to store the indices of the smallest distances
indices_smallest = zeros(num_smallest, size(distances, 2));

% Find indices of the two smallest distances for each centroid
for i = 1:size(distances, 2) % For each centroid (column)
    [~, sorted_idx] = sort(distances(:, i)); % Sort distances for the current centroid
    indices_smallest(:, i) = sorted_idx(1:num_smallest); % Store indices of the smallest distances
end
%%
% Find the closest centroid for each data point
[~, minIdx] = min(distances, [],2);

% Visualize distances of data points to centroids
figure;

% Plotting individual clusters
cluster1 = minIdx == 1;
cluster2 = minIdx == 2;
cluster3 = minIdx == 3;

scatter3(X(cluster1,1), X(cluster1,2), X(cluster1,3), 20, [0.3010 0.7450 0.9330], 'filled');
hold on;
scatter3(X(cluster2,1), X(cluster2,2), X(cluster2,3), 20, [0.4660 0.6740 0.1880], 'filled');
scatter3(X(cluster3,1), X(cluster3,2), X(cluster3,3), 20, [0.9290 0.6940 0.1250], 'filled');

% Plotting centroids with different markers and labeling them in the legend
scatter3(centroids(:,1), centroids(:,2), centroids(:,3), 100, 'k', 'filled');

% Counting elements in each cluster
num_elems_cluster1 = sum(cluster1);
num_elems_cluster2 = sum(cluster2);
num_elems_cluster3 = sum(cluster3);

% Displaying the number of elements in each cluster as text annotations
text(centroids(1,1)+0.02, centroids(1,2), centroids(1,3), sprintf('Cluster 1: %d', num_elems_cluster1), 'Color', [0.3010 0.7450 0.9330], 'FontSize', 10);
text(centroids(2,1)+0.02, centroids(2,2), centroids(2,3), sprintf('Cluster 2: %d', num_elems_cluster2), 'Color', [0.4660 0.6740 0.1880], 'FontSize', 10);
text(centroids(3,1)+0.02, centroids(3,2), centroids(3,3), sprintf('Cluster 3: %d', num_elems_cluster3), 'Color', [0.9290 0.6940 0.1250], 'FontSize', 10);

legend('Cluster 1', 'Cluster 2', 'Cluster 3', 'Data Points', 'Centroids');
title('Cluster Elements by Distance to Centroids');
xlabel('Variation Mean');
ylabel('Variation STD');
zlabel('Color STD');
hold off;

%%
for j = 1:length(indices_smallest(1,:))
    for i = 1:length(indices_smallest)
        aux_id = indices_smallest(i,j);
        aspect_ratio = visualization(Sorted_Blocks{aux_id}{1},Sorted_Blocks{aux_id}{2},D(aux_id,j), ['Cluster ' num2str(aux_id)], Sorted_Blocks{aux_id}{12});
        direction(Sorted_Blocks{aux_id}{1},Sorted_Blocks{aux_id}{2},aspect_ratio)
        
        %bin_edges = [-10.5 , -9.5, -8.5, -7.5, -6.5, -5.5, -4.5, -3.5, -2.5, -1.5, -0.5, 0.5, 1.5, 2.5,3.5,4.5,5.5,6.5,7.5,8.5,9.5,10.5];
        figure;
        histogram(Sorted_Blocks{aux_id}{10})%,'BinEdges',bin_edges); % Adjust BinWidth as needed for better visualization

        %Label the axes and title
        xlabel('Values');
        ylabel('Frequency');
        title('Histogram of Degrees');
        
    end
end
%%
% Probaremos los self-loops con el segundo resultado más cercano del
% Cluster 2
T_sl_all = {};
for i = 1:numel(Sorted_Blocks)
    Vblock = Sorted_Blocks{i}{1}; %Volumenes
    Adj = Sorted_Blocks{i}{11}; %Distances
    idx_closest = Sorted_Blocks{i}{12};
    C_aux = Sorted_Blocks{i}{2};
    [GFT,Gfreq, Ahat] = compute_GFT_noQ(Adj,C_aux,idx_closest);
    T_sl_all{i} = {GFT,Gfreq, Ahat,Vblock};
end

%%
good_example = [2363 1487 2586];

for example = 1:length(good_example)
    current = good_example(example);
    
    C_block = T_orig_all{current}{2};
    %Original processing
    A_hat_orig = T_orig_all{current}{3};
    GFT_og = T_orig_all{current}{1};
    Vblock_og = T_orig_all{current}{4};
    
    %Modified processing
    A_hat_sl = T_sl_all{current}{3};
    GFT_sl = T_sl_all{current}{1};
    Vblock_sl = T_sl_all{current}{4};
    
    coeff_visualization(A_hat_orig, A_hat_sl, current,1)
    
    visualization(Vblock_og, C_block, current, 'none');
    visualization(Vblock_og, C_block, current, 'og', 'GFT', GFT_og(:, 1));
    visualization(Vblock_sl, C_block, current, 'sl', 'GFT', GFT_sl(:, 1));
end
%%
good_example = [2363 1487 2586];
% Calculate the magnitudes of the eigenvector (you can adjust this based on your requirement)
for example = 1:length(good_example)
    current = good_example(example);
    
    % Eigenvector and magnitudes for SL method
    eigen_vector_sl = T_sl_all{current}{1}(:, 1);
    magnitudes_sl = abs(eigen_vector_sl);

    % Eigenvector and magnitudes for original method
    eigen_vector_og = T_orig_all{current}{1}(:, 1);
    magnitudes_og = abs(eigen_vector_og);

    % Create a figure
    figure;
    hold on;

    % Plot eigenvector for SL method
    h1 = plot(1:numel(eigen_vector_sl), eigen_vector_sl, 'LineWidth', 2, 'Color', 'r');

    % Plot eigenvector for original method
    h2 = plot(1:numel(eigen_vector_og), eigen_vector_og, 'LineWidth', 2, 'Color', 'b');

    xlabel('Index');
    ylabel('Value');
    title(['First Eigenvector as a Signal for Example '  num2str(current)] );
    legend([h1, h2], 'SL Method', 'Original Method');  % Specify plot handles in legend
    grid on;
    hold off;
end

    %%
%[W,edge] = compute_graph_MSR(Vblock);            % Método de consideración solo geométrica
%[G_vec,W_mod,edge] = gradient(Vblock, Ablock);   % Método de consideración geométrica y de color

%W_comparison = W_mod - W;                        % Comparamos los pesos de los nodos entre los dos métodos
%%
%tic;
%[ start_indices, end_indices, V_MR, Crec ] = iRegion_Adaptive_GFT( Coeff_quant, param );
%toc;

%Crgb_rec = double(YUVtoRGB(Crec));

%psnr_Y = -10*log10(norm(Y - Coeff_quant(:,1))^2/(N*255^2));

%%
 %ply_write('PC_original.ply',V,Crgb,[]);
 %ply_write('PC_coded.ply',V,Crgb_rec,[]);
 
 function normalized_vector = normalize_vector(vector)
    % Calculate the magnitude of the vector
    magnitude = norm(vector);
    
    % Normalize the vector
    normalized_vector = vector / magnitude;
end