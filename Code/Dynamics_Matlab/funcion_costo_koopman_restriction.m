function [A_final, B_final,P] = funcion_costo_koopman_restriction(X_1, X_K, U, alpha, beta, n, m, n_normal)
%% Load Casadi to the path of matlab 
addpath('/home/fer/casadi-linux-matlabR2014b-v3.4.5');
import casadi.*;


%% Empty vector for the optimization process
he_koop = [];
he_prediction = [];
%% Matrix Casadi Generarion
A = MX.sym('A', n, n);
B = MX.sym('B', n, m);
C_a = [eye(n_normal,n_normal), zeros(n_normal, n - n_normal)];

p = 1;

g = [];

for k = 1:length(U)

    %% Minimization Koopman
    % %% Lifdted space system
    x_1 = C_a*X_1(:,k);
    
    x_k = C_a*X_K(:,k);

    Gamma_k = (X_K(:,k));
    Gamma_1 = (X_1(:,k));

    
    error_koop = Gamma_k  -A*Gamma_1 - B*U(:,k);
    error_prediction = x_k - C_a*(A*Gamma_1 + B*U(:, k)) ;
    aux_p(1, k) = p^(length(U)-k);
    
    
    
    %obj = obj + (p)^(length(U)-k)*norm(error_koop);
    %% Error Vector
    he_koop = [he_koop; error_koop];
    he_prediction = [he_prediction; error_prediction];
    
end
v_init = (X_1(:, 1));
for k = 1:length(U)
    %% Rotational matrix
    quat = v_init(1:4, 1);
    norma = sqrt(quat'*quat);
    g = [g; norma - 1];
    v_estimate = (A*(v_init) + B*U(:, k));
    v_init = v_estimate;
end
%% Optimization Problem
P = diag(aux_p);

he_koop = reshape(he_koop, size(X_1, 1), size(X_1, 2));
he_koop = he_koop*P;

obj = 1*norm(he_koop, 'fro');
%obj = beta*norm(he_koop, 'fro');

%% General Vector Optimziation Variables
OPT_variables = [reshape(A,size(A,1)*size(A,2),1);reshape(B,size(B,1)*size(B,2),1)];

nlprob = struct('f', obj, 'x', OPT_variables);
% 
opts = struct;
opts.ipopt.max_iter = 20;
opts.ipopt.acceptable_tol =1e-8;
opts.ipopt.acceptable_obj_change_tol = 1e-6;

%% Solver of the problem
solver = nlpsol('solver', 'ipopt', nlprob, opts);
%% Initial Conditions System
A_0 = eye(n, n)*0.1;
B_0 = eye(n, m)*0.1;

args = struct;
args.x0 = [reshape(A_0,size(A_0,1)*size(A_0,2),1);reshape(B_0,size(B_0,1)*size(B_0,2),1)]; % initial value of the optimization variables

args.lbg((length(U))) = -0.5;  %Equality constraints
args.ubg((length(U))) = 0.5;   %Equality constraints

sol = solver('x0', args.x0);
Solution = full(sol.x);

%% Full matriz system
Full_matrix = reshape(Solution, size(A,1), size(A, 2) + size(B, 2));

%% Get matrices system
A_final = Full_matrix(1:size(A,1), 1:size(A,2));
B_final = Full_matrix(1:size(B, 1), size(A,2)+1:size(A,2)+size(B,2));
%G_final = Full_matrix(1:size(G, 1), size(A,2)+size(B,2)+1:size(A,2)+size(B,2));

end