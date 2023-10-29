import rospy
import numpy as np
from scipy.spatial.transform import Rotation as R
from nav_msgs.msg import Odometry

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
        self.rotation_field.setSFRotation(angles)

        # Set motors
        self.right_rear_wheel = self.robot.getDevice(self.right_rear_wheel_name)
        self.left_rear_wheel = self.robot.getDevice(self.left_rear_wheel_name)

        self.right_rear_wheel.setPosition(float('inf'))
        self.right_rear_wheel.setVelocity(0.0)

        self.left_rear_wheel.setPosition(float('inf'))
        self.left_rear_wheel.setVelocity(0.0)

        # Set steer angle motor
        self.right_steer = self.robot.getDevice(self.right_steer_name)
        self.left_steer = self.robot.getDevice(self.left_steer_name)
        self.right_steer.setPosition(0.0)
        self.left_steer.setPosition(0.0)

        # Set Odometry message
        self.odom_msg = Odometry()

    def set_motors_velocity(self, right: float, left: float)-> None:
        self.right_rear_wheel.setVelocity(right)
        self.left_rear_wheel.setVelocity(left)
        return None

    def set_steer_angle(self, steer)-> None:
        self.right_steer.setPosition(steer)
        self.left_steer.setPosition(steer)
        return None

    def set_frontal_velocity(self, velocity: float)-> None:
        R = self.r_r
        wr = velocity/R
        wl = velocity/R
        self.set_motors_velocity(wr, wl)
        return None

    def get_odometry(self):
        # Get system states position, rotation, velocity
        position_traslation = self.translation_field.getSFVec3f()
        angles_rotation = self.rotation_field.getSFRotation()
        velocity_system = self.supervisor_node.getVelocity()

        # Get axix representation
        r = R.from_rotvec(angles_rotation[3] * np.array([angles_rotation[0], angles_rotation[1], angles_rotation[2]]))
        quaternion = r.as_quat()
        Rotational_matrix = r.as_matrix()

        # Traform Velocities to Robot frame 
        velocity_linear = np.array([[velocity_system[0]], [velocity_system[1]], [velocity_system[2]]])
        velocity_angular = np.array([[velocity_system[3]], [velocity_system[4]], [velocity_system[5]]])

        velocity_linear_B = Rotational_matrix.T@velocity_linear
        velocity_angular_B = Rotational_matrix.T@velocity_angular

        self.odom_msg.header.stamp = rospy.Time.now()
        self.odom_msg.header.frame_id = "world"
        self.odom_msg.child_frame_id = "blue_robot_base"

        self.odom_msg.pose.pose.position.x = position_traslation[0]
        self.odom_msg.pose.pose.position.y = position_traslation[1]
        self.odom_msg.pose.pose.position.z = position_traslation[2]

        self.odom_msg.pose.pose.orientation.x = quaternion[0]
        self.odom_msg.pose.pose.orientation.y = quaternion[1]
        self.odom_msg.pose.pose.orientation.z = quaternion[2]
        self.odom_msg.pose.pose.orientation.w = quaternion[3]

        self.odom_msg.twist.twist.linear.x = velocity_linear_B[0, 0]
        self.odom_msg.twist.twist.linear.y = velocity_linear_B[1, 0]
        self.odom_msg.twist.twist.linear.z = velocity_linear_B[2, 0]

        self.odom_msg.twist.twist.angular.x = velocity_angular_B[0, 0]
        self.odom_msg.twist.twist.angular.y = velocity_angular_B[1, 0]
        self.odom_msg.twist.twist.angular.z = velocity_angular_B[2, 0]
        return None

    def send_odometry(self, odom_pu):
        # Update Odometry
        self.get_odometry()
        # Send Odometry
        odom_pu.publish(self.odom_msg)
        return None

