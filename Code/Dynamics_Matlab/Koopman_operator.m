%% Koopman identification using casadi as a optimization framework
clc, clear all, close all;

%% Load Data System
load("Data_System_2.mat");
[Data_1_X_k, Data_1_X_1, Data_1_U_1] = get_data_simple(h, hp, T);
load("Data_System_3.mat");
[Data_2_X_k, Data_2_X_1, Data_2_U_1] = get_data_simple(h, hp, T);

%% Rearrange data in order to develp DMD ext
%% State K
X1 = [Data_1_X_1, Data_2_X_1];

%% State K+1
X2 = [Data_1_X_k, Data_2_X_k];
n_normal = size(X1,1);
%% Input K
Gamma = [Data_1_U_1, Data_2_U_1];
% Gamma = Gamma./max(abs(Gamma), [], 2);

liftFun = @(xx)([
                 xx;...
                  
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
%% Optimization  variables 
alpha = 1;
beta = 1;
% 
%% Optimization Problem
[A_a, B_a, P] = funcion_costo_koopman_csadi(X1, X2, Gamma, alpha, beta, n, m, n_normal);
C_a = [eye(n_normal,n_normal), zeros(n_normal, n-n_normal)];


%% Compute Proyection
options.graphic = 0;
options.posdef = 10e-12;
options.maxiter = 200000;

X1 = X1*P;
X2 = X2*P;
Gamma = Gamma*P;

[A_SUB, B_SUB, ~, ~] = learnSOCmodel_withControl(X1,X2, Gamma, options);

%% New Initial Conditions
save('matrices.mat', 'A_SUB', 'B_SUB', 'A_a', 'B_a');
