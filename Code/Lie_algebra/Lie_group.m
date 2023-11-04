clc, clear all, close all

%% Inital Conditions
T_initial = [cos(pi/4), -sin(pi/4), 1;
             sin(pi/4), cos(pi/4), 0;
             0, 0, 1];



v = 1.0;  
omega = 0.0;  


dt = 0.05;  
total_time = 1.5; 


T_current = T_initial;

% Iterate for each time step
for i = 1:(total_time / dt)
    T_currec_aux = logm(T_current);
    pose = vee_map(T_currec_aux);
    pos_x(i) = T_current(1, 3);
    pos_y(i) = T_current(2, 3);
    angle(i) = pose(3);
    % Define Lie algebra element xi for this time step
    xi = [0, -omega, v*dt;
          omega, 0, 0;
          0, 0, 0];
    
    % Exponential map to obtain transformation matrix for small motion
    T_step = expm(xi);
    
    % Update current pose
    T_current = T_current * T_step;

end

% Extract final position and orientation from T_current
final_x = T_current(1, 3);
final_y = T_current(2, 3);
final_theta = atan2(T_current(2, 1), T_current(1, 1));

% Print final pose
fprintf('Final Position (x, y): (%f, %f)\n', final_x, final_y);
fprintf('Final Orientation (theta): %f\n', final_theta);

figure
plot(angle)

figure
plot(pos_x, pos_y)