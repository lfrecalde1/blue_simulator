function [V] = V_theta(angle, axis)
%UNTITLED2 Summary of this function goes here
%   Detailed explanation goes here
I = eye(3, 3);
axis_cross = hat_map(axis);
V = I + sin(angle)*axis_cross+ (1-cos(angle))*axis_cross^2;
V = inv(V);
end

