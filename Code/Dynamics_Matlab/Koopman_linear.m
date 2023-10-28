%% Koopman identification using casadi as a optimization framework
clc, clear all, close all;

%% Load Data System
load("Data_System_2.mat");
[Data_1_X_k, Data_1_X_1, Data_1_U_1] = get_data_simple_v(h, hp, T);


%% Rearrange data in order to develp DMD ext
%% State K
X1 = [Data_1_X_1];

%% State K+1
X2 = [Data_1_X_k];
n_normal = size(X1,1);
%% Input K
Gamma = [Data_1_U_1];
% Gamma = Gamma./max(abs(Gamma), [], 2);
l = 1.1;
liftFun = @(xx)([xx]);
             

%% Lifdted space system
X1 = liftFun(X1);

%% State K+1
X2 = liftFun(X2);

%% Size system 
n = size(X2, 1);
m = size(Gamma, 1);
%% Optimization  variables 
alpha = 1;
beta = 1;
% 
%% Optimization Problem
[A_a, B_a] = funcion_costo_koopman_csadi_linear(X1, X2, Gamma, alpha, beta, n, m, n_normal, h);
C_a = [eye(n_normal,n_normal), zeros(n_normal, n-n_normal)];

%% Load Data System
load("Data_System_3.mat");
[Data_1_X_k, Data_1_X_1, Data_1_U_1] = get_data_simple_v(h, hp, T);

%% Rearrange data in order to develp DMD ext
%% State K
X1 = [Data_1_X_1];

%% State K+1
X2 = [Data_1_X_k];
n_normal = size(X1,1);
%% Input K
Gamma = [Data_1_U_1];

%% Lifdted space system
X1 = liftFun(X1);

%% State K+1
X2 = liftFun(X2);

%% Compute Proyection
options.graphic = 0;
options.posdef = 10e-12;
options.maxiter = 200000;
X1_aux = transform(X1, h);
[A_SUB, B_SUB, ~, ~] = learnSOCmodel_withControl(X1_aux,X2, Gamma, options);
%%
v_estimate(:, 1) = C_a*(X1(:, 1));
v_estimate_2(:, 1) = C_a*(X1(:, 1));
for k= 1:length(X1)-1
    %% Output of the system
    salida_es(:, k) = v_estimate(:, k);
    salida_es_2(:, k) = v_estimate_2(:, k);
    
    Gamma_real = (X1(:, k));
    salida_real(:, k) = C_a*Gamma_real;
    
    %% Error of the estimation
    error(:, k) = salida_real(:,k) - salida_es(:, k);
    norm_error(k) = norm(error(:, k), 2);
    
    error_2(:, k) = salida_real(:,k) - salida_es_2(:, k);
    norm_error_2(k) = norm(error_2(:, k), 2);
    
    Rotacion = [1, 0, cos(h(10, k)), -sin(h(10, k));...
                0, 1, sin(h(10, k)), cos(h(10, k));...
                0, 0, 1, 0;...
                0, 0, 0, 1];
            
    %% Evolution of the system
    v_estimate(:, k+1) = C_a*(A_a*Rotacion*liftFun(v_estimate(:, k)) + B_a*Gamma(:, k));
    v_estimate_2(:, k+1) = C_a*(A_SUB*Rotacion*liftFun(v_estimate_2(:, k)) + B_SUB*Gamma(:, k));
    
end

figure
set(gcf, 'PaperUnits', 'inches');
set(gcf, 'PaperSize', [4 2]);
set(gcf, 'PaperPositionMode', 'manual');
set(gcf, 'PaperPosition', [0 0 10 4]);

subplot(3,1,1)
plot(salida_real(1,1:length(X2)-1),'-','Color',[226,76,44]/255,'linewidth',1); hold on
grid on;
plot(salida_es(1,1:length(X2)-1),'--','Color',[100,76,10]/255,'linewidth',1); hold on
plot(salida_es_2(1,1:length(X2)-1),'-.','Color',[50,76,10]/255,'linewidth',1); hold on
legend({'${x}$','$\hat{x}$', '$\hat{x_2}$'},'Interpreter','latex','FontSize',11,'Orientation','horizontal');
legend('boxoff')
ylabel('$[rad/s]$','Interpreter','latex','FontSize',9);
set(gcf, 'Color', 'w'); % Sets axes background

subplot(3,1,2)
plot(salida_real(2,1:length(X2)-1),'-','Color',[226,76,44]/255,'linewidth',1); hold on
grid on;
plot(salida_es(2,1:length(X2)-1),'--','Color',[100,76,10]/255,'linewidth',1); hold on
plot(salida_es_2(2,1:length(X2)-1),'-.','Color',[50,76,10]/255,'linewidth',1); hold on

legend({'${y}$','$\hat{y}$','$\hat{y_2}$'},'Interpreter','latex','FontSize',11,'Orientation','horizontal');
legend('boxoff')
ylabel('$[rad]$','Interpreter','latex','FontSize',9);

subplot(3,1,3)
plot(salida_real(3,1:length(X2)-1),'-','Color',[226,76,44]/255,'linewidth',1); hold on
grid on;
plot(salida_es(3,1:length(X2)-1),'--','Color',[100,76,10]/255,'linewidth',1); hold on
plot(salida_es_2(3,1:length(X2)-1),'-.','Color',[50,76,10]/255,'linewidth',1); hold on
legend({'${\mu_l}$','$\hat{\mu_l}$' ,'$\hat{\mu_{l2}}$'},'Interpreter','latex','FontSize',11,'Orientation','horizontal');
legend('boxoff')
ylabel('$[rad]$','Interpreter','latex','FontSize',9);

figure
set(gcf, 'PaperUnits', 'inches');
set(gcf, 'PaperSize', [4 2]);
set(gcf, 'PaperPositionMode', 'manual');
set(gcf, 'PaperPosition', [0 0 10 4]);
subplot(1,1,1)
plot(norm_error(1,1:length(X2)-1),'-','Color',[226,76,44]/255,'linewidth',1); hold on
plot(norm_error_2(1,1:length(X2)-1),'--','Color',[100,76,44]/255,'linewidth',1); hold on
grid on;
legend({'$||e_{estimation}||$', '$||e_{estimation2}||$'},'Interpreter','latex','FontSize',11,'Orientation','horizontal');
legend('boxoff')
title('$\textrm{Error estimation}$','Interpreter','latex','FontSize',9);

figure
imagesc(A_a);
eig(A_a)
figure
imagesc(B_a);

figure
imagesc(A_SUB);
eig(A_SUB)
figure
imagesc(B_SUB);
