%% Load Data System
clc, clear all, close all;

load("Data_System_2.mat");
load("A_data.mat");
load("B_data.mat");
[Data_1_X_k, Data_1_X_1, Data_1_U_1] = get_data_simple(h, hp, T);

%% Rearrange data in order to develp DMD ext
%% State K
X1 = [Data_1_X_1(:, 1:100)];

%% State K+1
X2 = [Data_1_X_k(:, 1:100)];
n_normal = size(X1,1);
%% Input K
Gamma = [Data_1_U_1(:, 1:100)];

liftFun = @(xx)([
                 xx;...
                  tan(xx(3, :));...
                  tan(xx(3, :)).*xx(4, :);...
                  
                  sin(xx(3, :)).*xx(4, :);...
                  cos(xx(3, :)).*xx(4, :);...
                  
                  sin(xx(1, :)).*xx(4, :);...
                  cos(xx(1, :)).*xx(4, :);...
                  ]);

%% Lifdted space system
X1 = liftFun(X1);

%% State K+1
X2 = liftFun(X2);

%% Size system 
n = size(X2, 1);
m = size(Gamma, 1);

%% Output Matrix
C_a = [eye(n_normal,n_normal), zeros(n_normal, n-n_normal)];

%% Initial Conditions System
v_estimate(:, 1) = C_a*(X1(:, 1));
for k= 1:100-1
    %% Output of the system
    salida_es(:, k) = v_estimate(:, k);
    
    Gamma_real = (X1(:, k));
    salida_real(:, k) = C_a*Gamma_real;
    
    %% Error of the estimation
    error(:, k) = salida_real(:,k) - salida_es(:, k);
    norm_error(k) = norm(error(:, k), 2);

    %% Evolution of the system
    v_estimate(:, k+1) = C_a*(A_a*liftFun(v_estimate(:, k)) + B_a*Gamma(:, k));
    
end

figure
set(gcf, 'PaperUnits', 'inches');
set(gcf, 'PaperSize', [4 2]);
set(gcf, 'PaperPositionMode', 'manual');
set(gcf, 'PaperPosition', [0 0 10 4]);

subplot(3,1,1)
plot(salida_real(2,1:length(X2)-1),'-','Color',[226,76,44]/255,'linewidth',1); hold on
grid on;
plot(salida_es(2,1:length(X2)-1),'--','Color',[100,76,10]/255,'linewidth',1); hold on
legend({'${\omega}$','$\hat{\omega}$'},'Interpreter','latex','FontSize',11,'Orientation','horizontal');
legend('boxoff')
ylabel('$[rad/s]$','Interpreter','latex','FontSize',9);
set(gcf, 'Color', 'w'); % Sets axes background

subplot(3,1,2)
plot(salida_real(3,1:length(X2)-1),'-','Color',[226,76,44]/255,'linewidth',1); hold on
grid on;
plot(salida_es(3,1:length(X2)-1),'--','Color',[100,76,10]/255,'linewidth',1); hold on
legend({'${\alpha}$','$\hat{\alpha}$'},'Interpreter','latex','FontSize',11,'Orientation','horizontal');
legend('boxoff')
ylabel('$[rad]$','Interpreter','latex','FontSize',9);

subplot(3,1,3)
plot(salida_real(1,1:length(X2)-1),'-','Color',[226,76,44]/255,'linewidth',1); hold on
grid on;
plot(salida_es(1,1:length(X2)-1),'--','Color',[100,76,10]/255,'linewidth',1); hold on
legend({'${\psi}$','$\hat{\psi}$'},'Interpreter','latex','FontSize',11,'Orientation','horizontal');
legend('boxoff')
ylabel('$[rad]$','Interpreter','latex','FontSize',9);


figure
set(gcf, 'PaperUnits', 'inches');
set(gcf, 'PaperSize', [4 2]);
set(gcf, 'PaperPositionMode', 'manual');
set(gcf, 'PaperPosition', [0 0 10 4]);
subplot(2,1,1)
plot(salida_real(4,1:length(X2)-1),'-','Color',[226,76,44]/255,'linewidth',1); hold on
plot(salida_es(4,1:length(X2)-1),'--','Color',[100,76,10]/255,'linewidth',1); hold on
grid on;
legend({'${{\mu_l}}$','${\hat{\mu_l}}$'},'Interpreter','latex','FontSize',11,'Orientation','horizontal');
legend('boxoff')
title('$\textrm{Estimation}$','Interpreter','latex','FontSize',9);
ylabel('$[m/]$','Interpreter','latex','FontSize',9);



figure
set(gcf, 'PaperUnits', 'inches');
set(gcf, 'PaperSize', [4 2]);
set(gcf, 'PaperPositionMode', 'manual');
set(gcf, 'PaperPosition', [0 0 10 4]);
subplot(2,1,1)
plot(salida_real(5,1:length(X2)-1),'-','Color',[226,76,44]/255,'linewidth',1); hold on
plot(salida_es(5,1:length(X2)-1),'--','Color',[100,76,10]/255,'linewidth',1); hold on
grid on;
legend({'${{x}}$','${\hat{x}}$'},'Interpreter','latex','FontSize',11,'Orientation','horizontal');
legend('boxoff')
title('$\textrm{Estimation}$','Interpreter','latex','FontSize',9);
ylabel('$[m]$','Interpreter','latex','FontSize',9);

subplot(2,1,2)
plot(salida_real(6,1:length(X2)-1),'-','Color',[226,76,44]/255,'linewidth',1); hold on
grid on;
plot(salida_es(6,1:length(X2)-1),'--','Color',[100,76,10]/255,'linewidth',1); hold on
legend({'${y}$','${\hat{y}}$'},'Interpreter','latex','FontSize',11,'Orientation','horizontal');
legend('boxoff')
ylabel('$[m]$','Interpreter','latex','FontSize',9);
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
figure
imagesc(B_a);

%% Error
disp('Error normal estimation')
norm(error)
