function matrix = hat_map_2(vector)
% Inputs
% vector:   3D vector with elements [v1 v2 v3]
% Outputs
% matrix:   skew summetrix matrix formed with the given vector

matrix=[0, -vector(1);...
        vector(1) , 0 ];

end