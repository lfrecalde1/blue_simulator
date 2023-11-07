function [xlift] = liftFun_aux(x)
%UNTITLED2 Summary of this function goes here
%   Detailed explanation goes here
for k = 1:length(x)
 xlift_1 = x(:, k);  
 
 %% Angular
 quaternion = x(1:4, k);
 rot = QuatToRot(quaternion);
 w = x(5:7, k);
 
 w_matrix = hat_map(w);
 z = rot*w_matrix;
 z_vector = vectorize(z);
 xlift_2 = z_vector;
 
 %% Steer angle
 steer_aux = (1/(0.6 + tan(x(8, k))*tan(x(8, k))));
 angular_aux = tan(x(8, k))*x(9, k);
 angular_aux_1 = (-1/(cos(x(8, k))^2))*x(9, k);
 
 %% Linear Velocities Frame Robot
 v_b = x(9:11, k);
 aux_alpha = [1, 0, 0;...
              0, tan(x(8, k)), 0;...
              0, 0, 1];
 v_i = aux_alpha*rot*v_b;
 
 %% Complete vector
 xlift(:, k) = [xlift_1; xlift_2; v_i; angular_aux; angular_aux_1];
 %xlift(:, k) = [xlift_1; v_i; angular_aux; angular_aux_1;1];

end
end
