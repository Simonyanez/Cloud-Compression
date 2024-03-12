

function [ Ahat, freqs, weights, Vblock, Ablock, Sorted_Blocks ] = block_visualization( A, params ) % A de atributos, podría ser la matriz de color
%This function implements the Region adaptive graph fourier transform
%(RA-GFT) for point cloud attributes of voxelized point clouds
%A: attribute
%params.V pointcloud coordinates
%params.bsize = block size
%params.J, depth of octree
V           = params.V;
b           = params.bsize; %b = [b_1, b_2,...,b_L] if multilevel, or b scalar. b_1*b_2*...*b_L = 2^J, 
J           = params.J;
isMultiLevel = params.isMultiLevel;
N = size(V,1); %number of points

%% Check consistency of block sizes, resolution levels, and octree depth
if(length(b)==1)
    
    if(isMultiLevel)%basically all levels have the same block size
        
        base_bsize = log2(b);  % Sea potencia de dos
        if(floor(base_bsize)~= base_bsize)%make sure that block size is power of 2
            error('block size bsize should be a power of 2');
        end
        L = J/base_bsize;
        
        if( L ~= floor(L))%make sure number of levels is an integer
            error('block size do not match number of levels');
        end
        bsize = ones(L,1)*b; %block size at each level is the same
        
    else
        base_bsize = log2(b);
        if(floor(base_bsize)~= base_bsize)%make sure that block size is power of 2
            error('block size bsize should be a power of 2');
        end
        L=1;
        bsize = b;
        
    end
else
    bsize =b;
    L = length(bsize);
    
    %check all entries of bsize are powers of 2
    base_bsize = log2(b);
    if(sum(base_bsize ~= floor(base_bsize)))
        error('entries of block size should be a power of 2');
    end
    %check if block sizes are consistent with octree depth
    if(sum(base_bsize)>J)
        error('block sizes do not match octree depth J');
    end
    
end
%Antes de esto es check del ´input
%%
Ahat = [];
Vcurr = V;
Acurr = A;
Qin = ones(N,1);
Gfreq_curr = zeros(N,1);
freqs =[];
weights=[];
for level=L:-1:1   % Esto será un loop de una iteración
    
    %%% block level processing
    %get block indices
    start_indices = block_indices(Vcurr,bsize(level)); %start index of blocks %Puntos enteros, se pueden ordenar consecutivos
    % Todos los bloques tienen distinta cantidad de puntos, el offset
    % siempre es distinto. Morton Order - Code. Es facil saber donde
    % empieza y donde termina cada bloque. Te da el índice en que empieza
    %cada bloque. Primer bloque start_indices[0]
    Nlevel = size(Vcurr,1);                   %number of points at curr level
    end_indices = [start_indices(2:end)-1;Nlevel]; %Último punto de cada bloque.
    %get blocks with more than 1 point
    ni = end_indices - start_indices +1;  %Cantidad de puntos por bloque
    %unchanged =  find(ni==1);%indices of blocks with single point
    to_change = find(ni ~=1); %indices of blocks that have more than 1 point
    disp("Cantidad de bloques con más de un punto")
    tc_size = size(to_change);
    Acurr_hat = Acurr;
    Qout=Qin;                       % Para la primera iteración es la identidad para todos los Q
    Gfreq_curr = zeros(size(Qin));
    Sorted_Blocks = cell(1,tc_size(1));  % No ordenados todavía
    %
    for currblock = 1:tc_size(1)         % Cuántos bloques se iterarán
        
    first_point = start_indices(currblock);
    last_point  = end_indices(currblock);
    Vblock = Vcurr(first_point:last_point,:);  % Toma todas las columnas del bloque
    Qin_block = Qin(first_point:last_point);
    Ablock =Acurr(first_point:last_point,:);
    
    %Clustering
    [W_orig,~] = compute_graph_MSR(Vblock);
    [G_vec, W_mod, edge] = gradient(Vblock, Ablock);
    [~, ~, distance_vectors,weights] = direction(Vblock, Ablock);
    [W_sl, ~, dot_products, ~, idx_closest] = compute_graph_sl(Vblock,distance_vectors,weights);
    
    
    %[W_mod_2,edge_2] = bf_graph(Vblock, Ablock);           % Bilateral
                                                            % filter (intentar después)
    G_vec_values = G_vec(G_vec ~= 0);
    Metric_1 = mean(G_vec_values);             % Media del vector de diferencias
    Metric_2 = std(G_vec_values);              % Desviación estandar del vector de diferencias
    Metric_3 = mean(Ablock(:,1));              % Media del vector Y del color
    Metric_4 = std(Ablock(:,1));               % Desviación estandar del vector Y del color
    Metric_5 = dot_products;                   % Producto interno entre distancias y volumen
                                               % ¿Alguna métrica de
                                               % detección de bordes?
                                               
    
    
    % Bloque: 
    %Geometría             (V)
    %Atributos             (A)
    %Pesos modificados     (W_mod)
    %Pesos originales      (W_orig)
    %Conexiones            (edge)
    %Metricas              (Metric)
    block_data = {Vblock, Ablock,W_mod,W_orig,edge, Metric_1, Metric_2, Metric_3, Metric_4 ,Metric_5,W_sl,idx_closest};
    Sorted_Blocks{currblock} = block_data;
    
    
    %Visualización
        
    X_block = Vblock(:,1);
    Y_block = Vblock(:,2);
    Z_block = Vblock(:,3);
        
    RGB_block = YUVtoRGB(Ablock);
    
    RGB_block;
    RGB_block_double = double(RGB_block);
    RGB_block_double/256;
  
    %%fig = figure;
    %%set(fig, 'Color', [0.7, 0.7, 0.7]);
    %%axis equal;
    %%grid on;
        
    %%scatter3(X_block, Y_block, Z_block,10,RGB_block_double/256)
        
    %%set(gca, 'Color', [0.7, 0.7, 0.7]);
    %%xlabel('X-axis');
    %%ylabel('Y-axis');
    %%zlabel('Z-axis');
    %%title('3D Point Cloud Plot with Custom Colors');
    %%view(60, 30); % Adjust the view angle as needed
       
        %plot3d scatter3d de matlab. Para point clouds pequeñas funciona
        %bien}
       
        % Se puede setear el currentblock para ir viendo bloques por
        % separado.
        
        %[Ahatblock, Gfreq_block,weights_block] = block_coeffs(Vblock,Ablock,Qin_block,bsize(level));
        %block_coeffs hacer la transformada, construir el grafo. copiarlo a
        %parte y hacer una versión aparte
        
        %Acurr_hat(first_point:last_point,:) = Ahatblock;
        %Qout(first_point:last_point) = weights_block;
        %Gfreq_curr(first_point:last_point) = Gfreq_block;
    end
end
end

