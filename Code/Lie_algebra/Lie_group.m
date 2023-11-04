clc, clear all, close all

x_init = 0;
y_init = 0;
angle_init = 0;
%% Inital Conditions
T_initial = [cos(angle_init), -sin(angle_init), x_init;
             sin(angle_init), cos(angle_init), y_init;
             0, 0, 1];
%% Sample Time Definition
dt = 0.1;  
total_time = 10; 
t = (0: dt:total_time);

%% Initial Condition
T_current = T_initial;

%% Control actions
v = 1*ones(length(t));  
omega = 1*sin(1*t);

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
    T_currec_aux = logm(T_current);
    pose = vee_map(T_currec_aux);
    
    pos_x(i) = T_current(1, 3);
    pos_y(i) = T_current(2, 3);
    angle(i) = pose(3);
    
    %% Define Lie algebra element xi for this time step
    xi = [0, -omega(i), v(i);
        omega(i), 0, 0;
        0, 0, 0]*dt;
    
    %% Exponential map to obtain transformation matrix for small motion
    T_step = expm(xi);
    
    %% Update current pose
    T_current = T_current * T_step;
    
    J = [cos(h(3)), 0;...
        sin(h(3)), 0;...
        0, 1];
    hp = J*[v(i); omega(i)];
    
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