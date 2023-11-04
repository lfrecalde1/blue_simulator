function [xlift] = liftFun_v(x)
%UNTITLED2 Summary of this function goes here
%   Detailed explanation goes here

 xlift_1 = x;  
 
 %% Angular
 rot = reshape(x(1:9, 1), 3, 3);
 T = [rot, x(17:19, 1);...
      0, 0, 0, 1];
  
 w = x(10:12, 1);
 vel = x(14:16, 1);
 differential_system = differential(w, vel)*0.05;
 z = T*expm(differential_system);
 R_vector = vectorize(z(1:3, 1:3));
 P_vector = z(1:3, 4);
 xlift_2 = R_vector;
 xlift_3 = P_vector;
 %% Steer angle
 steer_aux = (1/(0.6 + tan(x(13, 1))*tan(x(13, 1))));
 angular_aux = tan(x(13, 1))*x(14, 1);
 angular_aux_1 = (-1/(cos(x(13, 1))^2))*x(14, 1);
 
 %% Linear Velocities Frame Robot
%  v_b = x(14:16, k);
%  aux_alpha = [1, 0, 0;...
%               0, tan(x(13, k)), 0;...
%               0, 0, 1];
%  v_i = aux_alpha*rot*v_b;
 %% Complete vector
 xlift = [xlift_1; xlift_2; xlift_3; angular_aux];
 %xlift(:, k) = [xlift_1; v_i; angular_aux; angular_aux_1;1];
end