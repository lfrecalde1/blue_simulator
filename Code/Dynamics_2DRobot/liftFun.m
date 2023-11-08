function [xlift] = liftFun(x)
%UNTITLED2 Summary of this function goes here
%   Detailed explanation goes here
for k = 1:length(x)
 xlift_1 = x(:, k);  
 
 %% Angular
 rot = reshape(x(1:4, k), 2, 2);
 T = [rot, x(9:10, k);...
      0, 0, 1];
  
 w = x(5, k);
 vel = x(7:8, k);
 differential_system = differential(w, vel);
 z = T*expm(differential_system);
 R_vector = vectorize(z(1:2, 1:2));
 P_vector = z(1:2, 3);
 xlift_2 = R_vector;
 xlift_3 = P_vector;
 %% Steer angle
 steer_aux = (1/(0.6 + tan(x(6, k))*tan(x(6, k))));
 angular_aux = tan(x(6, k))*x(7, k);
 angular_aux_1 = (-1/(cos(x(6, k))^2))*x(7, k);
 
 %% Complete vector
 xlift(:, k) = [xlift_1; xlift_2; xlift_3; steer_aux; angular_aux; angular_aux_1];
 %xlift(:, k) = [xlift_1; v_i; angular_aux; angular_aux_1;1];

end
end

