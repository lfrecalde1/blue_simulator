function [T] = differential(w, v)
%UNTITLED2 Summary of this function goes here
%   Detailed explanation goes here
T = [0, -w, v(1);...
     w, 0, v(2);...
     0, 0, 0];
end

