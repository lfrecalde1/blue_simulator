#!/usr/bin/env python3
import time
from controller import Supervisor
import rospy
import numpy as np
from blue_simulator.blue_robot import BlueRobot
from nav_msgs.msg import Odometry
from geometry_msgs.msg import Twist

# Global Variable desired Initial Position
x_i = 0.481201
y_i = -0.0684486
z_i = 1.3859978

qx_i = 0.0005
qy_i = 0.0
qz_i = 0.0
qw_i = 1.0

vx_i = 0.0
vy_i = 0.0
vz_i = 0.0

wx_i = 0.0
wy_i = 0.0
wz_i = 0.0

# Desired and control velocities.
# Linear Velocity and steer angle
vxd = 0.0
wzd = 0.0

def velocity_call_back(velocity_message):
    global vxd, wzd
    # Read the linear Velocities
    vxd = velocity_message.linear.x
    wzd = velocity_message.angular.z
    return None

def main(robot, odom_pub):
    # Get time step Simulation
    time_step = int(robot.getBasicTimeStep()) 

    # Sample Time Defintion
    sample_time = 0.01

    # Time defintion aux variable
    t = 0

    # Frequency of the simulation
    hz = int(1/sample_time)
    loop_rate = rospy.Rate(hz)

    # Message Ros
    rospy.loginfo_once("Blue Robot Simulation")

    # Define Initial Conditions of the System
    initial_pose = np.array([x_i, y_i, z_i, qx_i, qy_i, qz_i, qw_i], dtype=np.double)
    initial_velocity = np.array([vx_i, vy_i, vz_i, wx_i, wy_i, wz_i], dtype=np.double)

    # Variables of the Robot, wheel radious and more
    # Distance between axis
    D = 1.1 
    # Control point
    a = -0.3
    # Frontal wheel radious
    r_f = 0.15
    # rear wheel radious
    r_r = 0.20
    L = [D, a, r_f, r_r]

    # Names of the actuator (verify Webots world in order to add more actuator)
    right_rear_wheel = "right_rear_wheel"
    left_rear_wheel = "left_rear_wheel"
    left_steer = "left_steer"
    right_steer = "right_steer"
    actuator_names = [right_rear_wheel, left_rear_wheel, right_steer, left_steer]


    # Create Blue Robot instance
    blue_1 = BlueRobot(L, initial_velocity, initial_pose, sample_time, odom_pub, robot, actuator_names, "Blue")



    message_ros = "Blue Robot Simulation Webots"
    # Simulation Loop
    while robot.step(time_step) != -1:
        tic = rospy.get_time()
        # Move robot based on desired frontal velocity and steer angle
        blue_1.set_frontal_velocity(vxd)
        blue_1.set_steer_angle(wzd)


        # Time restriction Correct
        loop_rate.sleep()
        # Send Odometry
        blue_1.send_odometry(odom_pub)

        # Print Time Verification
        toc = rospy.get_time()
        delta = toc - tic
        rospy.loginfo(message_ros + str(delta))
        t = t + delta
    return None

if __name__ == '__main__':
    try:
        # Initialization Robot
        robot = Supervisor()
        # Initialization Node
        rospy.init_node("blue_robot_webots",disable_signals=True, anonymous=True)

        # Publisher Info
        odomety_topic = "/blue_robot/odom"
        odometry_publisher = rospy.Publisher(odomety_topic, Odometry, queue_size = 10)

        # Subscribe Info
        velocity_topic = "/blue_robot/cmd_vel"
        velocity_subscriber = rospy.Subscriber(velocity_topic, Twist, velocity_call_back)

        main(robot, odometry_publisher)


    except(rospy.ROSInterruptException, KeyboardInterrupt):
        print("Error System")
        pass
    else:
        print("Complete Execution")
        pass