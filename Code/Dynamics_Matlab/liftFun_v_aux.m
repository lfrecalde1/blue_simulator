function [xlift] = liftFun_v_aux(x)
%UNTITLED2 Summary of this function goes here
%   Detailed explanation goes here

xlift_1 = x;

%% Angular movement
quaternion = x(1:4, 1);
rot = QuatToRot(quaternion);
w = x(5:7, 1);

w_matrix = hat_map(w);
z = rot*w_matrix;
z_vector = vectorize(z);
xlift_2 = z_vector;
 

%% Steer angle
steer_aux = (1/(0.6 + tan(x(8, 1))*tan(x(8, 1))));
angular_aux = tan(x(8, 1))*x(9, 1);
angular_aux_1 = (-1/(cos(x(8, 1))^2))*x(9, 1);

%% Linear Velocities Frame Robot
 v_b = x(9:11, 1);
 aux_alpha = [1, 0, 0;...
              0, tan(x(8, 1)), 0;...
              0, 0, 1];
 v_i = aux_alpha*rot*v_b;

%% Complete Vector
xlift = [xlift_1; xlift_2; v_i; angular_aux; angular_aux_1];
%xlift = [xlift_1; v_i; angular_aux; angular_aux_1;1];
end