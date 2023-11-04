function [xlift] = liftFun(x)
%UNTITLED2 Summary of this function goes here
%   Detailed explanation goes here
for k = 1:length(x)
 xlift_1 = x(:, k);  
 
 %% Angular
 rot = reshape(x(1:9, k), 3, 3);
 w = x(10:12, k);
 w_matrix = hat_map(w);
 z = rot*w_matrix;
 z_vector = vectorize(z);
 xlift_2 = z_vector;
 
 %% Steer angle
 steer_aux = (1/(0.6 + tan(x(13, k))*tan(x(13, k))));
 angular_aux = tan(x(13, k))*x(14, k);
 angular_aux_1 = (-1/(cos(x(13, k))^2))*x(14, k);
 
 %% Linear Velocities Frame Robot
 v_b = x(14:16, k);
 aux_alpha = [1, 0, 0;...
              0, tan(x(13, k)), 0;...
              0, 0, 1];
 v_i = aux_alpha*rot*v_b;
 %% Complete vector
 xlift(:, k) = [xlift_1; xlift_2; v_i; angular_aux];
 %xlift(:, k) = [xlift_1; v_i; angular_aux; angular_aux_1;1];

end
end

