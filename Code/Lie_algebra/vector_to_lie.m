function [T_init] = vector_to_lie(angle, pose)
%UNTITLED4 Summary of this function goes here
%   Detailed explanation goes here
hat_angle = hat_map([0, 0, angle]);
hat_exp = expm(hat_angle);

T_init = [hat_exp, pose;...
          0, 0, 0, 1];
end

