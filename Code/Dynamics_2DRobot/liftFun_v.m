function [xlift] = liftFun_v(x)
%UNTITLED2 Summary of this function goes here
%   Detailed explanation goes here

xlift_1 = x(:, 1);  
 
 %% Angular
 rot = reshape(x(1:4, 1), 2, 2);
 T = [rot, x(9:10, 1);...
      0, 0, 1];
  
 w = x(5, 1);
 vel = x(7:8, 1);
 differential_system = differential(w, vel);
 z = T*expm(differential_system);
 R_vector = vectorize(z(1:2, 1:2));
 P_vector = z(1:2, 3);
 xlift_2 = R_vector;
 xlift_3 = P_vector;
 %% Steer angle
 steer_aux = (1/(0.6 + tan(x(6, 1))*tan(x(6, 1))));
 angular_aux = tan(x(6, 1))*x(7, 1);
 angular_aux_1 = (-1/(cos(x(6, 1))^2))*x(7, 1);
 
 %% Complete vector
 xlift(:, 1) = [xlift_1; xlift_2; xlift_3; steer_aux; angular_aux; angular_aux_1];
 %xlift(:, k) = [xlift_1; v_i; angular_aux; angular_aux_1;1];
end