function [q_dot] =quat_dot(quat, omega)
%Derivative of quaternions
%   This function gets the derivate of the quaternions
%% Split que quaternions
qw = quat(1);
qx = quat(2);
qy = quat(3);
qz = quat(4);

%% Split angular velocities
p = omega(1);
q = omega(2);
r = omega(3);

%% Aux values to avoid numerical problems
K_quat = 2;
quat_error = 1 - (qw^2 + qx^2 + qy^2 + qz^2);

%% Create Skew symetric matrix
S = [0, -p, -q, -r;...
     p, 0, r, -q;...
     q, -r, 0, p;...
     r, q, -p, 0];
 
q_dot = (1/2)* S*quat + K_quat*quat_error*quat;
end