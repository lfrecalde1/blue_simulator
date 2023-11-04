function [xlift] = liftFun_v(x)
%UNTITLED2 Summary of this function goes here
%   Detailed explanation goes here

xlift_1 = x;

%% Angular movement
rot = reshape(x(1:9, 1), 3, 3);
w = x(10:12, 1);
w_matrix = hat_map(w);
z = rot*w_matrix;
z_vector = vectorize(z);
xlift_2 = z_vector;

%% Steer angle
steer_aux = (1/(0.6 + tan(x(13, 1))*tan(x(13, 1))));
angular_aux = tan(x(13, 1))*x(14, 1);
angular_aux_1 = (-1/(cos(x(13, 1))^2))*x(14, 1);

%% Linear Velocities Frame Robot
v_b = x(14:16, 1);
 aux_alpha = [1, 0, 0;...
              0, tan(x(13, 1)), 0;...
              0, 0, 1];
v_i = aux_alpha*rot*v_b;

%% Complete Vector
xlift = [xlift_1; xlift_2; v_i; angular_aux; angular_aux_1];
%xlift = [xlift_1; v_i; angular_aux; angular_aux_1;1];
end