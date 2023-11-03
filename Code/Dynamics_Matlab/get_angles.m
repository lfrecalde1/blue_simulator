function [angles] = get_angles(R)
%UNTITLED5 Summary of this function goes here
%   Detailed explanation goes here
for k = 1:size(R,3)
    log_R = logm(R(:, :, k));
    angles(:, k) = vee_map(log_R);
end

