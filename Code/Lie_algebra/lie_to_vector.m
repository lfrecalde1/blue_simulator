function [P, angle] = lie_to_vector(T)
%UNTITLED6 Summary of this function goes here
%   Detailed explanation goes here
R = T(1:3, 1:3);
p = T(1:3, 4);
R_log = logm(R);
angle = vee_map(R_log);



P = p;
end

