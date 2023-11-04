function [xlift] = liftFun(x)
%UNTITLED2 Summary of this function goes here
%   Detailed explanation goes here
for k = 1:length(x)
 xlift_1 = x(:, k);  
 
 %% Angular
 rot = reshape(x(1:9, k), 3, 3);
 T = [rot, x(17:19, k);...
      0, 0, 0, 1];
  
 w = x(10:12, k);
 vel = x(14:16, k);
 differential_system = differential(w, vel)*0.05;
 z = T*expm(differential_system);
 R_vector = vectorize(z(1:3, 1:3));
 P_vector = z(1:3, 4);
 xlift_2 = R_vector;
 xlift_3 = P_vector;
 %% Steer angle
 steer_aux = (1/(0.6 + tan(x(13, k))*tan(x(13, k))));
 angular_aux = tan(x(13, k))*x(14, k);
 angular_aux_1 = (-1/(cos(x(13, k))^2))*x(14, k);
 
 %% Linear Velocities Frame Robot
%  v_b = x(14:16, k);
%  aux_alpha = [1, 0, 0;...
%               0, tan(x(13, k)), 0;...
%               0, 0, 1];
%  v_i = aux_alpha*rot*v_b;
 %% Complete vector
 xlift(:, k) = [xlift_1; xlift_2; xlift_3; angular_aux];
 %xlift(:, k) = [xlift_1; v_i; angular_aux; angular_aux_1;1];

end
end

