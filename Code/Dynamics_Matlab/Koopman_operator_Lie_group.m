%% Koopman identification using casadi as a optimization framework
clc, clear all, close all;

%% Load Data System
load("Data_System_2.mat");
[Data_1_X_k, Data_1_X_1, Data_1_U_1] = get_data_full(h, hp, T);

%% Rearrange data in order to develp DMD ext
%% State K
X1 = [Data_1_X_1];
%% State K+1
X2 = [Data_1_X_k];
n_normal = size(X1,1);
%% Input K
Gamma = [Data_1_U_1];

%% Lifdted space system
X1 = liftFun_aux(X1);

%% State K+1
X2 = liftFun_aux(X2);

%% Size system 
n = size(X2, 1);
m = size(Gamma, 1);
%% Optimization  variables 
alpha = 1;
beta = 1;
% 
%% Optimization Problem
[A_a, B_a, P] = funcion_costo_koopman_restriction(X1, X2, Gamma, alpha, beta, n, m, n_normal);
C_a = [eye(n_normal,n_normal), zeros(n_normal, n-n_normal)];
%%  Simulation
v_estimate(:, 1) = C_a*(X1(:, 1));
for k= 1:length(X1)
    %% Output of the system
    salida_es(:, k) = v_estimate(:, k);
    quaternion_es(:, k) = v_estimate(1:4, k);
    norm(quaternion_es(:, k));
    rot_es(:, :, k) = QuatToRot(quaternion_es(:, k));
    rot_es(:, :, k);
    det(rot_es(:, :, k));
    angles_est(: , k) =  get_angles(rot_es(:, :, k));
    
    Gamma_real = (X1(:, k));
    salida_real(:, k) = C_a*Gamma_real;
    quaternion_real(:, k) = salida_real(1:4, k);
    rot_real(:, :, k) = QuatToRot(quaternion_real(:, k));
    angles_real(: , k) =  get_angles(rot_real(:, :, k));
    
    %% Error of the estimation
    error(:, k) = salida_real(:,k) - salida_es(:, k);
    norm_error(k) = norm(error(:, k), 2);

    %% Evolution of the system
    v_estimate(:, k+1) = C_a*(A_a*liftFun_v_aux(v_estimate(:, k)) + B_a*Gamma(:, k));
    q = v_estimate(1:4,  k+1);
    q = q/norm(q, 2);
    v_estimate(1:4, k+1) = q;

end

figure
set(gcf, 'PaperUnits', 'inches');
set(gcf, 'PaperSize', [4 2]);
set(gcf, 'PaperPositionMode', 'manual');
set(gcf, 'PaperPosition', [0 0 10 4]);

subplot(2,1,1)
plot(salida_real(7,1:length(X2)),'-','Color',[226,76,44]/255,'linewidth',1); hold on
grid on;
plot(salida_es(7,1:length(X2)),'--','Color',[100,76,10]/255,'linewidth',1); hold on
legend({'${r}$','$\hat{r}$'},'Interpreter','latex','FontSize',11,'Orientation','horizontal');
legend('boxoff')
ylabel('$[rad/s]$','Interpreter','latex','FontSize',9);
set(gcf, 'Color', 'w'); % Sets axes background


subplot(2,1,2)
plot(angles_real(3,1:length(X2)-1),'-','Color',[226,76,44]/255,'linewidth',1); hold on
grid on;
plot(angles_est(3,1:length(X2)-1),'--','Color',[100,76,10]/255,'linewidth',1); hold on
legend({'${\psi}$','$\hat{\psi}$'},'Interpreter','latex','FontSize',11,'Orientation','horizontal');
legend('boxoff')
ylabel('$[rad]$','Interpreter','latex','FontSize',9);

figure
set(gcf, 'PaperUnits', 'inches');
set(gcf, 'PaperSize', [4 2]);
set(gcf, 'PaperPositionMode', 'manual');
set(gcf, 'PaperPosition', [0 0 10 4]);

subplot(2,1,1)
plot(salida_real(5,1:length(X2)-1),'-','Color',[226,76,44]/255,'linewidth',1); hold on
grid on;
plot(salida_es(5,1:length(X2)-1),'--','Color',[100,76,10]/255,'linewidth',1); hold on
legend({'${p}$','$\hat{p}$'},'Interpreter','latex','FontSize',11,'Orientation','horizontal');
legend('boxoff')
ylabel('$[rad/s]$','Interpreter','latex','FontSize',9);
set(gcf, 'Color', 'w'); % Sets axes background


subplot(2,1,2)
plot(angles_real(1,1:length(X2)-1),'-','Color',[226,76,44]/255,'linewidth',1); hold on
grid on;
plot(angles_est(1,1:length(X2)-1),'--','Color',[100,76,10]/255,'linewidth',1); hold on
legend({'${\phi}$','$\hat{\phi}$'},'Interpreter','latex','FontSize',11,'Orientation','horizontal');
legend('boxoff')
ylabel('$[rad]$','Interpreter','latex','FontSize',9);

figure
set(gcf, 'PaperUnits', 'inches');
set(gcf, 'PaperSize', [4 2]);
set(gcf, 'PaperPositionMode', 'manual');
set(gcf, 'PaperPosition', [0 0 10 4]);

subplot(2,1,1)
plot(salida_real(6,1:length(X2)-1),'-','Color',[226,76,44]/255,'linewidth',1); hold on
grid on;
plot(salida_es(6,1:length(X2)-1),'--','Color',[100,76,10]/255,'linewidth',1); hold on
legend({'${q}$','$\hat{q}$'},'Interpreter','latex','FontSize',11,'Orientation','horizontal');
legend('boxoff')
ylabel('$[rad/s]$','Interpreter','latex','FontSize',9);
set(gcf, 'Color', 'w'); % Sets axes background


subplot(2,1,2)
plot(angles_real(2,1:length(X2)-1),'-','Color',[226,76,44]/255,'linewidth',1); hold on
grid on;
plot(angles_est(2,1:length(X2)-1),'--','Color',[100,76,10]/255,'linewidth',1); hold on
legend({'${\theta}$','$\hat{\theta}$'},'Interpreter','latex','FontSize',11,'Orientation','horizontal');
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
plot(salida_real(9,1:length(X2)-1),'-','Color',[226,76,44]/255,'linewidth',1); hold on
grid on;
plot(salida_es(9,1:length(X2)-1),'--','Color',[100,76,10]/255,'linewidth',1); hold on
legend({'${v_x}$','$\hat{v_x}$'},'Interpreter','latex','FontSize',11,'Orientation','horizontal');
legend('boxoff')
ylabel('$[m/s]$','Interpreter','latex','FontSize',9);
set(gcf, 'Color', 'w'); % Sets axes background


figure
set(gcf, 'PaperUnits', 'inches');
set(gcf, 'PaperSize', [4 2]);
set(gcf, 'PaperPositionMode', 'manual');
set(gcf, 'PaperPosition', [0 0 10 4]);

subplot(1,1,1)
plot(salida_real(10,1:length(X2)-1),'-','Color',[226,76,44]/255,'linewidth',1); hold on
grid on;
plot(salida_es(10,1:length(X2)-1),'--','Color',[100,76,10]/255,'linewidth',1); hold on
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
eig_v = eig(A_a);
figure
imagesc(B_a);


disp('Error normal estimation')
norm(error)

