#!/usr/bin/env python3
import time
from controller import Supervisor
import rospy
import numpy as np
from blue_simulator.blue_robot import BlueRobot
from nav_msgs.msg import Odometry

# Global Variable desired Initial Position
xd = 0.481201
yd = -0.0684486
zd = 1.3859978

qxd = 0.0005
qyd = 0.0
qzd = 0.0
qwd = 1.0

vxd = 0.0
vyd = 0.0
vzd = 0.0

wxd = 0.0
wyd = 0.0
wzd = 0.0


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
    initial_pose = np.array([xd, yd, zd, qxd, qyd, qzd, qwd], dtype=np.double)
    initial_velocity = np.array([vxd, vyd, vzd, wxd, wyd, wzd], dtype=np.double)

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
        signal = 0.45*np.sin(4*t)
        tic = rospy.get_time()
        blue_1.set_motors_velocity(0.5, 0.5)
        blue_1.set_steer_angle(signal)



        # Time restriction Correct
        loop_rate.sleep()
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


        main(robot, odometry_publisher)


    except(rospy.ROSInterruptException, KeyboardInterrupt):
        print("Error System")
        pass
    else:
        print("Complete Execution")
        pass