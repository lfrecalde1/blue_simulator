%% Code to plot the measure values from the system %%
%% Clean variable
clc, clear all, close all;

%% Load data from the specified file
load("blue_data_p4.mat");
size_data = 580;
%% Odometry of the system
[h, hp, T] = get_odometry(data_odom_blue', steering_real', vel_real', vy', vz', wx', wy', wz', vel_control', steering_control', size_data);

%% Angular Velocities of the sytem
p = hp(4, :); 
q = hp(5, :);
r = hp(6, :);

%% Verify Quaternions norm
quaternions = h(4:7, :);
for k = 1:length(quaternions)
   norm_1(k) = norm(quaternions(:, k)); 
end

%% Get rotational matrices
[R_1] = get_Rot_m(h);
[R_2] = get_Rot_c(h);

%% Get angles from the rotational matrices
[angles_1] = get_angles(R_1);
[angles_2] = get_angles(R_2);

%% Vector Rotational matricz
R_v = vectorize_R(R_1);
%% Control Signals
%% Plot Euler Angles of the system
figure
set(gcf, 'PaperUnits', 'inches');
set(gcf, 'PaperSize', [4 2]);
set(gcf, 'PaperPositionMode', 'manual');
set(gcf, 'PaperPosition', [0 0 10 4]);
subplot(3,1,1)
plot(h(8,:),'-','Color',[226,76,44]/255,'linewidth',1); hold on
grid on;
legend({'${{\phi}}$'},'Interpreter','latex','FontSize',11,'Orientation','horizontal');
legend('boxoff')
title('$\textrm{Angles System}$','Interpreter','latex','FontSize',9);
ylabel('$[rad]$','Interpreter','latex','FontSize',9);

subplot(3,1,2)
plot(h(9,:),'-','Color',[46,188,89]/255,'linewidth',1); hold on
grid on;
legend({'${\theta}$'},'Interpreter','latex','FontSize',11,'Orientation','horizontal');
legend('boxoff')
ylabel('$[rad]$','Interpreter','latex','FontSize',9);
set(gcf, 'Color', 'w'); % Sets axes background

subplot(3,1,3)
plot(h(10,:),'-','Color',[26,115,160]/255,'linewidth',1); hold on
plot(angles_1(3,:),'--','Color',[26,20,20]/255,'linewidth',1); hold on
grid on;
legend({'${\psi}$', '${\psi}_e$'},'Interpreter','latex','FontSize',11,'Orientation','horizontal');
legend('boxoff')
ylabel('$[rad]$','Interpreter','latex','FontSize',9);

%% Quaternions PLot
figure
set(gcf, 'PaperUnits', 'inches');
set(gcf, 'PaperSize', [4 2]);
set(gcf, 'PaperPositionMode', 'manual');
set(gcf, 'PaperPosition', [0 0 10 4]);
subplot(4,1,1)
plot(h(4,:),'-','Color',[226,76,44]/255,'linewidth',1); hold on
grid on;
legend({'${q_w}$'},'Interpreter','latex','FontSize',11,'Orientation','horizontal');
legend('boxoff')
title('$\textrm{Quaternions System}$','Interpreter','latex','FontSize',9);

subplot(4,1,2)
plot(h(5,:),'-','Color',[226,76,44]/255,'linewidth',1); hold on
grid on;
legend({'${q_x}$'},'Interpreter','latex','FontSize',11,'Orientation','horizontal');
legend('boxoff')
set(gcf, 'Color', 'w'); % Sets axes background

subplot(4,1,3)
plot(h(6,:),'-','Color',[226,76,44]/255,'linewidth',1); hold on
grid on;
legend({'${q_y}$'},'Interpreter','latex','FontSize',11,'Orientation','horizontal');
legend('boxoff')
set(gcf, 'Color', 'w'); % Sets axes background

subplot(4,1,4)
plot(h(7,:),'-','Color',[226,76,44]/255,'linewidth',1); hold on
grid on;
legend({'${q_z}$'},'Interpreter','latex','FontSize',11,'Orientation','horizontal');
legend('boxoff')

%% PLot Position system
figure
set(gcf, 'PaperUnits', 'inches');
set(gcf, 'PaperSize', [4 2]);
set(gcf, 'PaperPositionMode', 'manual');
set(gcf, 'PaperPosition', [0 0 10 4]);
subplot(3,1,1)
plot(h(1, :),'Color',[226,76,44]/255,'linewidth',1); hold on
grid on;
legend({'$x$'},'Interpreter','latex','FontSize',11,'Orientation','horizontal');
legend('boxoff')
title('$\textrm{Position system}$','Interpreter','latex','FontSize',9);
ylabel('$[m]$','Interpreter','latex','FontSize',9);

subplot(3,1,2)
plot(h(2, :),'Color',[46,188,89]/255,'linewidth',1); hold on
grid on;
legend({'$y$'},'Interpreter','latex','FontSize',11,'Orientation','horizontal');
legend('boxoff')
ylabel('$[m]$','Interpreter','latex','FontSize',9);

subplot(3,1,3)
plot(h(3, :),'Color',[26,115,160]/255,'linewidth',1); hold on
grid on;
legend({'$z$',},'Interpreter','latex','FontSize',11,'Orientation','horizontal');
legend('boxoff')
ylabel('$[m]$','Interpreter','latex','FontSize',9);



figure
set(gcf, 'PaperUnits', 'inches');
set(gcf, 'PaperSize', [4 2]);
set(gcf, 'PaperPositionMode', 'manual');
set(gcf, 'PaperPosition', [0 0 10 4]);
subplot(3,1,1)
plot(hp(1,:),'--','Color',[226,76,44]/255,'linewidth',1); hold on
plot(T(1,:),'-','Color',[100,76,44]/255,'linewidth',1); hold on
grid on;
legend({'$\mu_{l}$', '$\mu_{lc}$'},'Interpreter','latex','FontSize',11,'Orientation','horizontal');
legend('boxoff')
title('$\textrm{Linear Velocities System}$','Interpreter','latex','FontSize',9);
ylabel('$[m/s]$','Interpreter','latex','FontSize',9);

subplot(3,1,2)
plot(hp(2,:),'-','Color',[46,188,89]/255,'linewidth',1); hold on
grid on;
legend({'$\mu_{m}$'},'Interpreter','latex','FontSize',11,'Orientation','horizontal');
legend('boxoff')
ylabel('$[m/s]$','Interpreter','latex','FontSize',9);

subplot(3,1,3)
plot(hp(3, :),'-','Color',[26,115,160]/255,'linewidth',1); hold on
grid on;
legend({'$\mu_{n}$'},'Interpreter','latex','FontSize',11,'Orientation','horizontal');
legend('boxoff')
ylabel('$[m/s]$','Interpreter','latex','FontSize',9);


figure
set(gcf, 'PaperUnits', 'inches');
set(gcf, 'PaperSize', [4 2]);
set(gcf, 'PaperPositionMode', 'manual');
set(gcf, 'PaperPosition', [0 0 10 4]);
subplot(3,1,1)
plot(q(1,:),'-','Color',[226,76,44]/255,'linewidth',1); hold on
grid on;
legend({'$q$'},'Interpreter','latex','FontSize',11,'Orientation','horizontal');
legend('boxoff')
title('$\textrm{Angular Velocities System}$','Interpreter','latex','FontSize',9);
ylabel('$[rad/s]$','Interpreter','latex','FontSize',9);

subplot(3,1,2)
plot(p(1,:),'-','Color',[46,188,89]/255,'linewidth',1); hold on
grid on;
legend({'$p$'},'Interpreter','latex','FontSize',11,'Orientation','horizontal');
legend('boxoff')
ylabel('$[rad/s]$','Interpreter','latex','FontSize',9);

subplot(3,1,3)
plot(r(1,:),'-','Color',[83,57,217]/255,'linewidth',1); hold on
plot(h(11,:),'--','Color',[226,76,44]/255,'linewidth',1); hold on
grid on;
legend({'$r$', '$\alpha$'},'Interpreter','latex','FontSize',11,'Orientation','horizontal');
legend('boxoff')
ylabel('$[rad/s]$','Interpreter','latex','FontSize',9);
xlabel('$\textrm{Time}[s]$','Interpreter','latex','FontSize',9);

figure
set(gcf, 'PaperUnits', 'inches');
set(gcf, 'PaperSize', [4 2]);
set(gcf, 'PaperPositionMode', 'manual');
set(gcf, 'PaperPosition', [0 0 10 4]);
plot(h(11,:),'--','Color',[226,76,44]/255,'linewidth',1); hold on
plot(T(2,:),'-','Color',[100,76,44]/255,'linewidth',1); hold on
grid on;
legend({'$\alpha$', '$\alpha_d$'},'Interpreter','latex','FontSize',11,'Orientation','horizontal');
legend('boxoff')
title('$\textrm{Angular Velocities System}$','Interpreter','latex','FontSize',9);
ylabel('$[rad]$','Interpreter','latex','FontSize',9);

%% Save Data of the system
save("Data_System_3.mat","T", "h", "hp")