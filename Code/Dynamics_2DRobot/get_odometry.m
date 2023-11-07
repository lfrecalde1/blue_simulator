function [h, hp, T] = get_odometry(values_odometry, steering_real, vx, vy, vz, wx, wy, wz, vel_control, steering_control, size)
%Get_odometry Function
%   Function developed to get all the odometry values
% values [x y qx qy qz qw]
for k = 1:length(values_odometry)
    %% Get translation
    x(k) = values_odometry(1, k);
    y(k) = values_odometry(2, k);
    z(k) = 0;
    
    %% Get quaternions
    quaternion_values(:, k) = [values_odometry(6, k); values_odometry(3, k); values_odometry(4, k); values_odometry(5, k)];
    
    %% Get Orientation Euler Angles
    orientacion_aux(:, k) = (quat2eul(quaternion_values(:, k)','ZYX'))';
end
%% General Vector includying position, quaternions and euler angles
h = [x;y;z;quaternion_values;orientacion_aux(3, :);orientacion_aux(2, :);orientacion_aux(1, :)];

h(11, :) = (steering_real)*(pi/180);

%% Veloities vector of the system

hp = [vx; vy; vz; wx; wy; wz];


%% New Size system
h = h(:, 1:size+1);
hp = hp(:, 1:size+1);

%% Control vectors
T = [vel_control;(steering_control)*(pi/180)];

T = T(:, 1:size);
end

