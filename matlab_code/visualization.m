
function [aspect_ratio] = visualization(Vblock, Ablock, im_num, method, varargin)
    % Visualización
    X_block = Vblock(:, 1);
    Y_block = Vblock(:, 2);
    Z_block = Vblock(:, 3);
        
    X_mean = mean(Vblock(:, 1));
    Y_mean = mean(Vblock(:, 2));
    Z_mean = mean(Vblock(:, 3));
    
    RGB_block = YUVtoRGB(Ablock);
    RGB_block_double = double(RGB_block);
    %RGB_block_double = RGB_block_double / 256;

    fig = figure;
    set(fig, 'Color', [0.7, 0.7, 0.7]);
    axis equal;
    grid on;

    % Check if 'GFT' is passed as an argument and if GFT data is provided
    if any(strcmpi(varargin, 'GFT')) && numel(varargin) > 1
        idx = find(strcmpi(varargin, 'GFT'));

        if idx < numel(varargin)
            GFT = varargin{idx + 1};

            scatter3(X_block, Y_block, Z_block, 20, GFT, 'filled');  % Use GFT as the colormap
            colormap(hot);
            colorbar;
        else
            warning('No value provided for ''GFT'' property.');
        end
    else
        scatter3(X_block, Y_block, Z_block, 20, RGB_block_double, 'filled');  % Use default colormap
    end
    
    hold on;  % To overlay additional points
    
    % Highlight specific points of interest with different marker or color
    scatter3(X_mean, Y_mean, Z_mean, 100, 'black') % Modify marker size and color as needed
    
    hold on 
    
    scatter3(X_mean, Y_mean, Z_mean, 100, 'b')
    set(gca, 'Color', [0.7, 0.7, 0.7]);
    xlabel('X-axis');
    ylabel('Y-axis');
    zlabel('Z-axis');
    title(['3D Point Cloud Plot with Custom Colors - Image Number: ' num2str(im_num) ' Method: ' method]);
    view(60, 30)
    
    ax = gca;
    aspect_ratio = daspect(ax);
end
