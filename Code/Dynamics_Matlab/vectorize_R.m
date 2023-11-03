function [R_v] = vectorize_R(R)
%UNTITLED7 Summary of this function goes here
%   Detailed explanation goes here
for k = 1:size(R,3)
   R_v(:, k) = vectorize(R(:, :, k)); 
end
end

