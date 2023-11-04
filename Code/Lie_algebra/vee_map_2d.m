function vector = vee_map_2d(matrix)
% Inputs
% matrix:   n x n skew-symmetric matrix 
% Outputs
% vector:   3D vector extracted from the skew-symmetric matrix


vector(1) = -matrix(1,2);

end