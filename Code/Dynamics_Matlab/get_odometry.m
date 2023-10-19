function [h] = get_odometry(values_odometry)
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

end

