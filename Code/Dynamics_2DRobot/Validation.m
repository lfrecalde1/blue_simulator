%% Load Data System
clc, clear all, close all;

%% Initial Conditions System
%% The identification works with Data 1 and 2 since we identidied with theses values
load("Data_System_1.mat");
load("matrices.mat")
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

%% get Covatiance Matrix
P = inv(X1(:,:)*X1(:,:)');
[U,S,V] = svd(P);
condition_number = max(S(1, 1))/min(S(end, end))

for k= 1:length(X1)
    %% Output of the system
    salida_es(:, k) = v_estimate(:, k);
    rot_es(:, :, k) = reshape(v_estimate(1:4, k), 2, 2);
    angles_est(: , k) =  get_angles(rot_es(:, :, k));
    det(rot_es(:, :, k));
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
eig_v = eig(A_a);

figure
imagesc(B_a);


figure
imagesc(P);

disp('Error normal estimation')
norm(error)


save('matrices.mat', 'A_a', 'B_a');