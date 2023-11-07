clc, clear all, close all

x_init = 0;
y_init = 0;
z_init = 0;
angle_init = pi/2;

%% Inital Conditions
T_initial = vector_to_lie(angle_init, [x_init; y_init; z_init]);
%% Sample Time Definition
dt = 0.1;  
total_time = 20; 
t = (0: dt:total_time);

%% Initial Condition
T_current = T_initial;

%% Control actions
vx = 1*ones(length(t));  
vy = 0*ones(length(t));
vz = 1*ones(length(t));

r = 2*sin(2*t);
p = 0*sin(2*t);
q = 0*sin(2*t);

%% Initial Contidion vector space
x = x_init;
y = y_init;
theta = angle_init;
h = [x; y; theta];

%% Iterate for each time step
for i = 1:length(t)
    h(3) = Angulo(h(3));
    pos_x_e(i) = h(1);
    pos_y_e(i) = h(2);
    pos_theta_e(i) = h(3);
    
    
    %% Extract Location system
    
    
    [pose, ang] = lie_to_vector(T_current);
    
    pos_x(i) = pose(1, 1);
    pos_y(i) = pose(2, 1);
    pos_z(i) = pose(3, 1);
    angle(i) = ang(3);
    T_current_aux = T_current(1:2,1:2)
    Aux_matrix = [cos(angle(i)), -sin(angle(i));sin(angle(i)), cos(angle(i))]
    %% Define Lie algebra element xi for this time step
    xi = differential([p(i); q(i); r(i)], [vx(i); vy(i); vz(i)])*dt;
    
    %% Exponential map to obtain transformation matrix for small motion
    T_step = expm(xi);
    
    %% Update current pose
    T_current = T_current * T_step;
    
    J = [cos(h(3)), 0;...
        sin(h(3)), 0;...
        0, 1];
    hp = J*[vx(i); r(i)];
    
    h = h + hp*dt;
    

end

% Extract final position and orientation from T_current
final_x = T_current(1, 3);
final_y = T_current(2, 3);
final_theta = atan2(T_current(2, 1), T_current(1, 1));

% Print final pose
fprintf('Final Position Lie(x, y): (%f, %f)\n', pos_x(i), pos_y(i));
fprintf('Final Orientation  Lie(theta): %f\n', angle(i));

fprintf('Final Position (x, y): (%f, %f)\n', pos_x_e(i), pos_y_e(i));
fprintf('Final Orientation (theta): %f\n', pos_theta_e(i));

figure
plot(angle)
hold on
plot(pos_theta_e,'--')

figure
plot(pos_x, pos_y)
hold on 
plot(pos_x_e, pos_y_e, '--')