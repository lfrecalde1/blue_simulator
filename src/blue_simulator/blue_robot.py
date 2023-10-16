import rospy
import numpy as np
from scipy.spatial.transform import Rotation as R


class BlueRobot:
    def __init__(self, L : list, v: np.ndarray, pose: np.ndarray, ts: float, odom_publi: rospy.topics.Publisher, Robot, actuators: list, robot_name: str):
        super().__init__()
        # Sample time
        self.ts = ts

        # Variables robot
        self.D = L[0]
        self.a = L[1]
        self.r_f = L[2]
        self.r_r = L[3]

        # Actuator names
        self.right_rear_wheel_name = actuators[0]
        self.left_rear_wheel_name = actuators[1]
        self.right_steer_name = actuators[2]
        self.left_steer_name = actuators[3]

        # Define Pose of the vehicle position and orientation
        self.x = np.array([pose[0], pose[1], pose[2], pose[3], pose[4], pose[5], pose[6]], dtype=np.double)
        self.xp = np.array([v[0], v[1], v[2], v[3], v[4], v[5]], dtype=np.double)  

        # Comunication Odometry
        self.odom_publisher = odom_publi

        # Robot instance 
        self.robot = Robot

        # Supervisor Node Webots
        self.supervisor_node = self.robot.getFromDef(robot_name)
        self.translation_field = self.supervisor_node.getField('translation')
        self.rotation_field = self.supervisor_node.getField('rotation')

        # Initial Position and Orientation system
        displacement = [pose[0], pose[1], pose[2]]
        quaternion_d = [pose[3], pose[4], pose[5], pose[6]] # x y z w
        r_quat = R.from_quat(quaternion_d)
        r_axis = r_quat.as_rotvec()
        r_axis_angle = np.linalg.norm([r_axis]) 
        r_axis_normalized = r_axis/r_axis_angle
        angles = [r_axis_normalized[0], r_axis_normalized[1], r_axis_normalized[2], r_axis_angle] 
        self.translation_field.setSFVec3f(displacement)
        self.translation_field.setSFRotation(angles)

        # Set motors
        self.right_rear_wheel = self.robot.getDevice(self.right_rear_wheel_name)
        self.left_rear_wheel = self.robot.getDevice(self.left_rear_wheel_name)

        self.right_rear_wheel.setPosition(float('inf'))
        self.right_rear_wheel.setVelocity(0.0)

        self.left_rear_wheel.setPosition(float('inf'))
        self.left_rear_wheel.setVelocity(0.0)


        self.right_steer = self.robot.getDevice(self.right_steer_name)
        self.left_steer = self.robot.getDevice(self.left_steer_name)
        
        self.right_steer.setPosition(0.0)
        self.left_steer.setPosition(0.0)




    def set_motors_velocity(self, right: float, left: float)-> None:
        self.right_rear_wheel.setVelocity(right)
        self.left_rear_wheel.setVelocity(left)
        return None

    def set_steer_angle(self, steer)-> None:
        self.right_steer.setPosition(steer)
        self.left_steer.setPosition(steer)
        return None
    



