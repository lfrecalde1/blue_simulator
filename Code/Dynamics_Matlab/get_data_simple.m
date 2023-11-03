function [X2, X1, Gamma] = get_data_simple(h, hp, u_ref)
%UNTITLED6 Summary of this function goes here
%   Detailed explanation goes here
%% Load Data experiment 1
des = 1;

%% Load Data System Pose
h = h(:, des:end);

%% Load Data Velocities
hp = hp(:, des:end);

%% Get Rotational Matrices
[R_1] = get_Rot_m(h);
[R_2] = get_Rot_c(h);

%% get Vector Rotational Matrix
R_v = vectorize_R(R_1);

%% Angular Values
p = hp(4, :);
q = hp(5, :);
r = hp(6, :);

omega = [p;q;r];
quaternion = h(4:7, :);

%% Real Angles System
phi = h(8, :);
theta = h(9,:);
psi = h(10, :);

%% Euler Angles
euler = [phi;...
         theta;...
          psi];
%% Control Values
ul_d = u_ref(1,:);
alpha_d = u_ref(2,:);

%% generalized Data system
X = [R_v(:, :);...
    omega(:,:);...
    h(11, :);...
    hp(1:3, :);];

%% Control Signal
U_ref = [ul_d;...
         alpha_d];
     
%% Rearrange data in order to develp DMD ext

X1 = [X(:,1:end-1)];
  
X2 = [X(:,2:end)];
  
Gamma = U_ref(:,1:end);
end
