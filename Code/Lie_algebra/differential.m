function [T] = differential(w, v)
%UNTITLED2 Summary of this function goes here
%   Detailed explanation goes here
hat_w = hat_map(w);
T = [hat_w, v(1:3);...
     0, 0, 0, 0];
end

