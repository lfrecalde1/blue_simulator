function [X_new] = transform(X1, h)
%UNTITLED5 Summary of this function goes here
%   Detailed explanation goes here
for k = 1:length(X1)
     Rotacion = [1, 0, cos(h(10, k)), -sin(h(10, k));...
        0, 1, sin(h(10, k)), cos(h(10, k));...
        0, 0, 1, 0;...
        0, 0, 0, 1];
    
 X_new(:, k) = Rotacion*X1(:, k);  
end
end

