%% Koopman identification using casadi as a optimization framework
clc, clear all, close all;

%% Load Data System
load("Data_System_2.mat");
[Data_1_X_k, Data_1_X_1, Data_1_U_1] = get_data_simple(h, hp, T);

load("Data_System_1.mat");
[Data_2_X_k, Data_2_X_1, Data_2_U_1] = get_data_simple(h, hp, T);

load("Data_System_3.mat");
[Data_3_X_k, Data_3_X_1, Data_3_U_1] = get_data_simple(h, hp, T);

load("Data_System_4.mat");
[Data_4_X_k, Data_4_X_1, Data_4_U_1] = get_data_simple(h, hp, T);

load("Data_System_5.mat");
[Data_5_X_k, Data_5_X_1, Data_5_U_1] = get_data_simple(h, hp, T);
%% Data order
%% X = [r_11; r_21; r_12; r_22; wz; alpha; vx; vy; hx; hy]
%% Rearrange data in order to develp DMD ext
%% State K
X1 = [Data_1_X_1, Data_2_X_1, Data_3_X_1, Data_4_X_1, Data_5_X_1];

%% State K+1
X2 = [Data_1_X_k, Data_2_X_k, Data_3_X_k, Data_4_X_k, Data_5_X_k];
n_normal = size(X1,1);
%% Input K
Gamma = [Data_1_U_1, Data_2_U_1, Data_3_U_1, Data_4_U_1, Data_5_U_1];

%% Lifdted space system
X1 = liftFun(X1);

%% State K+1
X2 = liftFun(X2);

%% Size system 
n = size(X2, 1);
m = size(Gamma, 1);
%% Optimization  variables 
alpha = 0.00;
beta = 1;
% 
%% Optimization Problem
[A_a, B_a] = funcion_costo_koopman_csadi(X1, X2, Gamma, alpha, beta, n, m, n_normal);
C_a = [eye(n_normal,n_normal), zeros(n_normal, n-n_normal)];


%% Initial Conditions System
load("Data_System_1.mat");
[Data_2_X_k, Data_2_X_1, Data_2_U_1] = get_data_simple(h, hp, T);

X1 = [Data_2_X_1];
X2 = [Data_2_X_k];
n_normal = size(X1,1);
Gamma = [Data_2_U_1];

X1 = liftFun(X1);
X2 = liftFun(X2);

n = size(X2, 1);
m = size(Gamma, 1);

C_a = [eye(n_normal,n_normal), zeros(n_normal, n-n_normal)];
v_estimate(:, 1) = C_a*(X1(:, 1));
for k= 1:length(X1)
    %% Output of the system
    salida_es(:, k) = v_estimate(:, k);
    rot_es(:, :, k) = reshape(v_estimate(1:4, k), 2, 2);
    det(rot_es(:, :, k))
    angles_est(: , k) =  get_angles(rot_es(:, :, k));
    
    Gamma_real = (X1(:, k));
    salida_real(:, k) = C_a*Gamma_real;
    rot_real(:, :, k) = reshape(salida_real(1:4, k), 2, 2);
    angles_real(: , k) =  get_angles(rot_real(:, :, k));
    
    %% Error of the estimation
    error(:, k) = salida_real(:,k) - salida_es(:, k);
    norm_error(k) = norm(error(:, k), 2);

    %% Evolution of the system
    v_estimate(:, k+1) = C_a*(A_a*liftFun_v(v_estimate(:, k)) + B_a*Gamma(:, k));
    
end

figure
set(gcf, 'PaperUnits', 'inches');
set(gcf, 'PaperSize', [4 2]);
set(gcf, 'PaperPositionMode', 'manual');
set(gcf, 'PaperPosition', [0 0 10 4]);

subplot(2,1,1)
plot(salida_real(5,1:length(X2)-1),'-','Color',[226,76,44]/255,'linewidth',1); hold on
grid on;
plot(salida_es(5,1:length(X2)-1),'--','Color',[100,76,10]/255,'linewidth',1); hold on
legend({'${r}$','$\hat{r}$'},'Interpreter','latex','FontSize',11,'Orientation','horizontal');
legend('boxoff')
ylabel('$[rad/s]$','Interpreter','latex','FontSize',9);
set(gcf, 'Color', 'w'); % Sets axes background


subplot(2,1,2)
plot(angles_real(1,1:length(X2)-1),'-','Color',[226,76,44]/255,'linewidth',1); hold on
grid on;
plot(angles_est(1,1:length(X2)-1),'--','Color',[100,76,10]/255,'linewidth',1); hold on
legend({'${\psi}$','$\hat{\psi}$'},'Interpreter','latex','FontSize',11,'Orientation','horizontal');
legend('boxoff')
ylabel('$[rad]$','Interpreter','latex','FontSize',9);


figure
set(gcf, 'PaperUnits', 'inches');
set(gcf, 'PaperSize', [4 2]);
set(gcf, 'PaperPositionMode', 'manual');
set(gcf, 'PaperPosition', [0 0 10 4]);

subplot(1,1,1)
plot(salida_real(6,1:length(X2)-1),'-','Color',[226,76,44]/255,'linewidth',1); hold on
grid on;
plot(salida_es(6, 1:length(X2)-1),'--','Color',[100,76,10]/255,'linewidth',1); hold on
legend({'${\alpha}$','$\hat{\alpha}$'},'Interpreter','latex','FontSize',11,'Orientation','horizontal');
legend('boxoff')
ylabel('$[rad]$','Interpreter','latex','FontSize',9);
set(gcf, 'Color', 'w'); % Sets axes background

figure
set(gcf, 'PaperUnits', 'inches');
set(gcf, 'PaperSize', [4 2]);
set(gcf, 'PaperPositionMode', 'manual');
set(gcf, 'PaperPosition', [0 0 10 4]);

subplot(1,1,1)
plot(salida_real(7,1:length(X2)-1),'-','Color',[226,76,44]/255,'linewidth',1); hold on
grid on;
plot(salida_es(7,1:length(X2)-1),'--','Color',[100,76,10]/255,'linewidth',1); hold on
legend({'${v_x}$','$\hat{v_x}$'},'Interpreter','latex','FontSize',11,'Orientation','horizontal');
legend('boxoff')
ylabel('$[m/s]$','Interpreter','latex','FontSize',9);
set(gcf, 'Color', 'w'); % Sets axes background

figure
set(gcf, 'PaperUnits', 'inches');
set(gcf, 'PaperSize', [4 2]);
set(gcf, 'PaperPositionMode', 'manual');
set(gcf, 'PaperPosition', [0 0 10 4]);

subplot(2,1,1)
plot(salida_real(9,1:length(X2)-1),'-','Color',[226,76,44]/255,'linewidth',1); hold on
grid on;
plot(salida_es(9,1:length(X2)-1),'--','Color',[100,76,10]/255,'linewidth',1); hold on
legend({'${x}$','$\hat{x}$'},'Interpreter','latex','FontSize',11,'Orientation','horizontal');
legend('boxoff')
ylabel('$[rad/s]$','Interpreter','latex','FontSize',9);
set(gcf, 'Color', 'w'); % Sets axes background


subplot(2,1,2)
plot(salida_real(10,1:length(X2)-1),'-','Color',[226,76,44]/255,'linewidth',1); hold on
grid on;
plot(salida_es(10,1:length(X2)-1),'--','Color',[100,76,10]/255,'linewidth',1); hold on
legend({'${y}$','$\hat{y}$'},'Interpreter','latex','FontSize',11,'Orientation','horizontal');
legend('boxoff')
ylabel('$[rad]$','Interpreter','latex','FontSize',9);

figure
set(gcf, 'PaperUnits', 'inches');
set(gcf, 'PaperSize', [4 2]);
set(gcf, 'PaperPositionMode', 'manual');
set(gcf, 'PaperPosition', [0 0 10 4]);

subplot(1,1,1)
plot(salida_real(8,1:length(X2)-1),'-','Color',[226,76,44]/255,'linewidth',1); hold on
grid on;
plot(salida_es(8,1:length(X2)-1),'--','Color',[100,76,10]/255,'linewidth',1); hold on
legend({'${v_y}$','$\hat{v_y}$'},'Interpreter','latex','FontSize',11,'Orientation','horizontal');
legend('boxoff')
ylabel('$[m/s]$','Interpreter','latex','FontSize',9);
set(gcf, 'Color', 'w'); % Sets axes background

figure
set(gcf, 'PaperUnits', 'inches');
set(gcf, 'PaperSize', [4 2]);
set(gcf, 'PaperPositionMode', 'manual');
set(gcf, 'PaperPosition', [0 0 10 4]);
subplot(1,1,1)
plot(norm_error(1,1:length(X2)-1),'-','Color',[226,76,44]/255,'linewidth',1); hold on
grid on;
legend({'$||e_{estimation}||$'},'Interpreter','latex','FontSize',11,'Orientation','horizontal');
legend('boxoff')
title('$\textrm{Error estimation}$','Interpreter','latex','FontSize',9);

figure
imagesc(A_a);
eig_v = eig(A_a)
det(A_a)
figure
imagesc(B_a);


disp('Error normal estimation')
norm(error)


save('matrices.mat', 'A_a', 'B_a');
