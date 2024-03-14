function [] = coeff_visualization(Ahat_orig, Ahat_mod, distance, cluster) %A_hat = T{3300}{3}
    %Visualización coeficientes
        
    Y_og = abs(Ahat_orig(:,1));
    U_og = abs(Ahat_orig(:,2));
    V_og = abs(Ahat_orig(:,3));
    
    Y_mod = abs(Ahat_mod(:,1));
    U_mod = abs(Ahat_mod(:,2));
    V_mod = abs(Ahat_mod(:,3));
    
  
    fig = figure;
    grid on;
    
    hold on
    scatter(1:length(Y_og), Y_og,'s')
    scatter(1:length(Y_mod), Y_mod,'d')
    
    legend('Original', 'Modified')
    
    xlabel('Index');
    ylabel('Magnitude');
    title(['Transform coefficients Y '  '- Distance = '  num2str(distance) ' Cluster ' num2str(cluster)]);
    xlim([0.5, length(Y_og) + 0.5]) 
    ylim([min(Y_og) - 1, max(Y_og) + 1])
    axis tight
    hold off
    
   
    fig = figure;
    grid on;
    
    hold on
    scatter(1:length(U_og), U_og,'s')
    scatter(1:length(U_mod), U_mod, 'd')
    
    legend('Original', 'Modified')
    
    xlabel('Index');
    ylabel('Magnitude');
    title(['Transform coefficients U '  '- Distance = '  num2str(distance) ' Cluster ' num2str(cluster)]);
    xlim([0.5, length(U_og) + 0.5]) 
    ylim([min(U_og) - 1, max(U_og) + 1])
    axis tight
    hold off
    
    fig = figure;   
    grid on;
    
    hold on
    scatter(1:length(V_og), V_og,'s')
    scatter(1:length(V_mod), V_mod,'d')
    
    legend('Original', 'Modified')
    
    xlabel('Index');
    ylabel('Magnitude');
    title(['Transform coefficients V '  '- Distance = '  num2str(distance) ' Cluster ' num2str(cluster)]);
    xlim([0.5, length(V_og) + 0.5]) 
    ylim([min(V_og) - 1, max(V_og) + 1])
    axis tight
    hold off
end