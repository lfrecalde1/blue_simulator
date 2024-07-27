#!/usr/bin/env python3     
import time
import math
import numpy as np
from controller import *
import rospy
from sensor_msgs.msg import JointState
from std_msgs.msg import Header

# Global Variables for ecah joint
q_1 = 0.5
q_2 = -0.5
q_3 = 0.0
q_4 = 0.0
q_5 = 0.0
q_6 = 0.0

def main(robot):
    # Get time step Simulation
    time_step = int(robot.getBasicTimeStep()) 

    # Sample Time Defintion
    sample_time = 0.05

    # Message Ros
    rospy.loginfo_once("CR3 ROBOT SIMULATION")

    # Cr3 Robot Joints
    names_m = ['joint1', 'joint2', 'joint3', 'joint4', 'joint5', 'joint6']

    # Definition of the names of rotational sensors
    names_s = ['joint1_sensor', 'joint2_sensor', 'joint3_sensor', 'joint4_sensor', 'joint5_sensor', 'joint6_sensor']

    # Set and activate motors
    motors = get_motor(robot, names_m)

    # Set and activate rotational sensors
    sensors = get_sensor_pos(names_s, time_step)

    # Joint Message
    joint_message = set_joint_message()

    # Init System
    init_system(robot, motors, [0.5, -0.8, -0.0, -0.0, 0.0, 0.0], time_step, sample_time)

    # get initial conditions of the system
    q = get_angular_position(robot, sensors, time_step)

    # Simulation Loop
    while (robot.step(time_step) != -1) and (not rospy.is_shutdown()):
        tic = time.time()
        # Compute desired values
        print("Angular displacement")
        print(q)

        # Compute desired values
        q_c = [q_1, q_2, q_3, q_4, q_5, q_6]

        # actuate the rotational motors of the system
        set_motor_pos(motors, q_c)

        # Get current states of the system
        q = get_angular_position(robot, sensors, time_step)

        # Time restriction Correct
        while (time.time() - tic <= sample_time):
            None
        toc = time.time() - tic 
        #rospy.loginfo(message_ros + str(toc))
    return None

def set_joint_message():
     # Create a new JointState message
    joint_state = JointState()
    joint_state.header = Header()
    return joint_state

def send_joint_message(message, pub, states, names):
    message.header.stamp = rospy.Time.now()
    message.name = ['joint1', 'joint2']
    message.position = [1.0, 1.5]
    message.velocity = [0.0, 0.0]
    message.effort = []

    # Publish the message
    pub.publish(message)

def get_motor(robot, name):
    # Motor Configuration
    # INPUTS
    # robot                                             - robot class
    # name                                              - device name  
    # OUTPUT
    # motors                                             - motor devices        
    motor = []
    for k in name:
        motor.append(robot.getDevice(k))
    return motor

def set_motor_pos(motor, pos):
    # Set motor with position control
    # INPUT
    # motor                                             - device motor
    # pos                                               - desired position for each motor
    # OUTPUT
    # None
    size = len(motor)
    for k in range(0, size):
        motor[k].setPosition(pos[k])
    return None

def set_motor_vel(motor, vel):
    # Set motor with velocity control
    # INPUT
    # motor                                             - device motor
    # vel                                               - desired velocity for each motor
    # OUTPUT
    # None
    size = len(motor)
    for k in range(0, size):
        motor[k].setPosition(float('inf'))
        motor[k].setVelocity(vel[k])
    return None

def get_sensor_pos(names, time_step):
    # Configuration of the rotational motor sensors
    # INPUT 
    # names                                              - list of the names for each sensor
    # time_step                                          - sample time simulation
    # OUTPUTS 
    # sensors                                            - A list with different objects for positional sensing
    sensors = []
    for k in names:
        instance = PositionSensor(k)
        instance.enable(time_step)
        sensors.append(instance)
    return sensors

def get_angular_position(robot, sensors, time_step):
    # A Function that enables the acquisition of the angular displacement from each rotational sensor
    # INPUT 
    # robot                                                                 - object instance 
    # sensors                                                               - A list with sensor objects            
    # OUTPUT                                                                
    # q                                                                     - A vector array with the respective information        
    q = np.zeros((len(sensors), ), dtype=np.double)
    size = len(sensors)
    if robot.step(time_step) != -1:
        for k in range(0, size):
            data = sensors[k].getValue()
            q[k] = data
    return q

def init_system(robot, motors, q_c, time_step, t_s):
    # Function that moves the robot to an initial configuration
    # INPUT 
    # robot                                        - A robot object that contains all the required information
    # motors                                       - A list with the required motor objects
    # q_c                                          - A vector of desired initial angular values
    # time_step                                    - sample time of the simulation webots
    # ts                                           - sample time of the simulation
    # OUTPUT
    # None
    for k in range(0, 100):
         if robot.step(time_step) != -1:
            tic = time.time()
            print("Init System")
            set_motor_pos(motors, q_c)
            # Sample time saturation
            while (time.time() - tic <= t_s):
                None
            toc = time.time() - tic 
    return None

if __name__ == '__main__':
    try:
        # Node Initialization
        rospy.init_node("cr3_robot_simulation",disable_signals=True, anonymous=True)

        # Conection Webots External
        robot_a = Supervisor()

        # Publisher robot joint states
        joint_states_publisher = rospy.Publisher('/cr3_robot/joint_states', JointState, queue_size=10)

        # Simulation 
        main(robot_a, joint_states_publisher)

    except(rospy.ROSInterruptException, KeyboardInterrupt):
        print("Error System")
        pass
    else:
        print("Complete Execution")
        pass