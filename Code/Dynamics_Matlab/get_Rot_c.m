function [R] = get_Rot_c(h)
%UNTITLED2 Summary of this function goes here
%   Detailed explanation goes here
R = zeros(3, 3, length(h));
for k = 1:length(h)
    quat = h(4:7, k);
    R(:, :, k) = QuatToRot(quat);
end
end