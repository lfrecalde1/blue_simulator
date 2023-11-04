function matrix = hat_map(theta)
% Inputs
% vector:   3D vector with elements [v1 v2 v3]
% Outputs
% matrix:   skew summetrix matrix formed with the given vector

matrix=[ 0, theta;...
         -theta, 0];

end