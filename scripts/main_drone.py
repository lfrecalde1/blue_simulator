#!/usr/bin/env python3     
import time
import math
import numpy as np
from controller import *
import rospy
import cv2
from cv_bridge import CvBridge, CvBridgeError
from sensor_msgs.msg import Image
from nav_msgs.msg import Odometry
from geometry_msgs.msg import Twist
from scipy.spatial.transform import Rotation as R
from tf import TransformBroadcaster
import tf.transformations as tft

from std_msgs.msg import Header
from sensor_msgs.msg import Image


xd = 0.0
yd = 0.0
zd = 5.16
vxd = 0.0
vyd = 0.0
vzd = 0.0

# Angular velocities
qx = 0.0005
qy = 0.0
qz = 0.0
qw = 1.0
wxd = 0.0
wyd = 0.0
wzd = 0.0

vx_c = 0
vy_c = 0
vz_c = 0

# Angular velocities
wx_c = 0
wy_c = 0
wz_c = 0


def velocity_call_back(velocity_message):
    global vx_c, vy_c, vz_c, wx_c, wy_c, wz_c
    # Read desired linear velocities from node
    vx_c = velocity_message.linear.x
    vy_c = velocity_message.linear.y
    vz_c = velocity_message.linear.z

    # Read desired angular velocities from node
    wx_c = velocity_message.angular.x
    wy_c = velocity_message.angular.y
    wz_c = velocity_message.angular.z
    return None
# Time message Header in the topic
time_message = 0.0
def odometry_call_back(odom_msg):
    global xd, yd, zd, qx, qy, qz, qw, time_message, vxd, vyd, vzd, wxd, wyd, wzd

    # Read desired linear velocities from node
    aux_p = np.random.uniform(low = -0.001, high= 0.001, size = (3,))
    aux_q = np.random.uniform(low = -0.005, high= 0.005, size = (4,))
    aux_v = np.random.uniform(low = -0.005, high= 0.005, size = (3,))
    aux_w = np.random.uniform(low = -0.005, high= 0.005, size = (3,))

    time_message = odom_msg.header.stamp

    xd = odom_msg.pose.pose.position.x + aux_p[0]
    yd = odom_msg.pose.pose.position.y + aux_p[1]
    zd = odom_msg.pose.pose.position.z + aux_p[2]

    vxd = odom_msg.twist.twist.linear.x + aux_v[0]
    vyd = odom_msg.twist.twist.linear.y + aux_v[1]
    vzd = odom_msg.twist.twist.linear.z + aux_v[2]


    qx = odom_msg.pose.pose.orientation.x  + aux_q[0]
    qy = odom_msg.pose.pose.orientation.y+  aux_q[1] 
    qz = odom_msg.pose.pose.orientation.z + aux_q[2]
    qw = odom_msg.pose.pose.orientation.w + aux_q[3]

    wxd = odom_msg.twist.twist.angular.x + aux_w[0]
    wyd = odom_msg.twist.twist.angular.y + aux_w[1]
    wzd = odom_msg.twist.twist.angular.z + aux_w[2]
    return None

def camera_system(robot, name, timestep):
    # System configuration for camera
    camera = robot.getDevice(name)
    camera.enable(timestep)
    return camera

def get_image(camera):
    # Adquisición de información de la cámara
    data = np.frombuffer(camera.getImage(), dtype=np.uint8)

    # Cambio de tamaño de la imagen a las dimensiones respectivas
    frame = data.reshape((camera.getHeight(), camera.getWidth(), 4))[:, :, :3]

    # Convert image to the respective type of open cv
    frame = cv2.cvtColor(frame, cv2.COLOR_BGRA2BGR)

    return frame

def send_image(bridge, imglr_pub, imgr):
    # Create the message header
    header = Header()
    header.stamp = rospy.Time.now()
    header.frame_id = "camera_link"

    # Convert the image to the appropriate message type
    img_msg = bridge.cv2_to_imgmsg(imgr, encoding='bgr8')
    img_msg.header = header

    # Publish the image message
    imglr_pub.publish(img_msg)





def send_image_depth(bridge, imglr_pub, imgr):
    # Concatenate images L and R
    # Decode Image left
    msglr = bridge.cv2_to_imgmsg(imgr, '16UC1')
    msglr.header.stamp = rospy.Time.now();
    msglr.header.frame_id = "camera_link"

    imglr_pub.publish(msglr)
    return None

def get_range_image(camera):
    # Adquisition of camera information
    data = camera.getRangeImageArray()
    #import pdb; pdb.set_trace()
    img = np.array(data)
    img[img == np.inf] = camera.getMaxRange()
    img_aux = img*(65536/8.0)
    img_normalized = np.array(img_aux, np.uint16)
    return img_normalized

def get_odometry(translation, rotation, supervisor, odom_msg):
        # Get system states position, rotation, velocity
        position_traslation = translation.getSFVec3f()
        angles_rotation = rotation.getSFRotation()
        velocity_system =supervisor.getVelocity()

        # Get axix representation
        r = R.from_rotvec(angles_rotation[3] * np.array([angles_rotation[0], angles_rotation[1], angles_rotation[2]]))
        quaternion = r.as_quat()


        odom_msg.header.stamp = rospy.Time.now()
        odom_msg.header.frame_id = "world"
        odom_msg.child_frame_id = "drone_link"

       
	
        odom_msg.pose.pose.position.x = position_traslation[0]  
        odom_msg.pose.pose.position.y = position_traslation[1]  
        odom_msg.pose.pose.position.z = position_traslation[2]  

        odom_msg.pose.pose.orientation.x = quaternion[0] 
        odom_msg.pose.pose.orientation.y = quaternion[1] 
        odom_msg.pose.pose.orientation.z = quaternion[2] 
        odom_msg.pose.pose.orientation.w = quaternion[3] 

        odom_msg.twist.twist.linear.x = velocity_system[0] 
        odom_msg.twist.twist.linear.y = velocity_system[1] 
        odom_msg.twist.twist.linear.z = velocity_system[2] 

        odom_msg.twist.twist.angular.x = velocity_system[3] 
        odom_msg.twist.twist.angular.y = velocity_system[4] 
        odom_msg.twist.twist.angular.z = velocity_system[5] 
        return odom_msg

def send_drone_tf(drone_tf, odom_msg):
    position = (odom_msg.pose.pose.position.x, odom_msg.pose.pose.position.y, odom_msg.pose.pose.position.z)
    quaternion = (odom_msg.pose.pose.orientation.x, odom_msg.pose.pose.orientation.y, odom_msg.pose.pose.orientation.z, odom_msg.pose.pose.orientation.w)   
    drone_tf.sendTransform(position, quaternion, rospy.Time.now(), "drone_link", "world")
    return None

def move_velocity(supervisor, rotation):

    angles_rotation = rotation.getSFRotation()
    velocity_system =supervisor.getVelocity()

    # Get axix representation
    r = R.from_rotvec(angles_rotation[3] * np.array([angles_rotation[0], angles_rotation[1], angles_rotation[2]]))
    quaternion = r.as_quat()
    Rotational_matrix = r.as_matrix()
    velocity_linear = np.array([[vx_c], [vy_c], [vz_c]])
    velocity_angular = np.array([[wx_c], [wy_c], [wz_c]])

    velocity_linear_world = Rotational_matrix@velocity_linear
    velocity_angular_world = Rotational_matrix@velocity_angular

    velocity = [velocity_linear_world[0], velocity_linear_world[1], velocity_linear_world[2], velocity_angular_world[0], velocity_angular_world[1], velocity_angular_world[2]]
    #velocity = [velocity_linear[0], velocity_linear[1], velocity_linear[2], velocity_angular[0], velocity_angular[1], velocity_angular[2]]

    supervisor.setVelocity(velocity)
    return None

def rotate_object(rotation_Prop1, angle):
    # Calcula los componentes del nuevo cuaternión de rotación
    cos_half_angle = math.cos(math.radians(angle) / 2)
    sin_half_angle = math.sin(math.radians(angle) / 2)
    new_rotation = [0, sin_half_angle, 0, cos_half_angle]  # Rotación en el eje Y

    # Establece el nuevo cuaternión de rotación
    rotation_Prop1.setSFRotation(new_rotation)

def send_camera_tf(camera_tf):
    # Parameters camera traslation and rotation
    position = (0.0, 0.0, 0.18)
    Rotation_matrix = R.from_matrix([[0, 0, 1],
                                    [-1, 0, 0],
                                    [0, -1, 0]])
    
    quaterni_aux = Rotation_matrix.as_quat()
    quaternion = (quaterni_aux[0], quaterni_aux[1], quaterni_aux[2], quaterni_aux[3])
    
    camera_tf.sendTransform(position, quaternion, rospy.Time.now(), "camera_link", "drone_link")
    return None

def set_robot(translation, rotation, h, angle):
    if translation != [0, 0, 0]:
        translation.setSFVec3f(h)
        rotation.setSFRotation(angle)
    return None


def send_odometry(odom_msg, odom_pu):
    odom_pu.publish(odom_msg)
    return None

def main(robot, image_pu_rgb, image_pu_d, odometry_pu):
    # Get time Step
    time_step = int(robot.getBasicTimeStep()) 

    # Sample Time Defintion
    sample_time = 0.01

    # Frequency of the simulation
    hz = int(1/sample_time)

    # Move propellers
    Prop1_node = robot.getFromDef('PROP1')
    Prop2_node = robot.getFromDef('PROP2')
    Prop3_node = robot.getFromDef('PROP3')
    Prop4_node = robot.getFromDef('PROP4')
 
    rotation_Prop1 = Prop1_node.getField('rotation')
    rotation_Prop2 = Prop2_node.getField('rotation')
    rotation_Prop3 = Prop3_node.getField('rotation')
    rotation_Prop4 = Prop4_node.getField('rotation')
    
    # Robot get Node using supervisor information
    UAV_node = robot.getFromDef('matrice')
    translation_field = UAV_node.getField('translation')
    rotation_field = UAV_node.getField('rotation')

    # Camera Definitions
    camera_d = camera_system(robot, "range-finder", time_step)
    camera_rgb = camera_system(robot, "camera_l", time_step)

    # Bridge Openc Cv
    bridge = CvBridge()

    # Time defintion
    t = 0

    # Odometry Message
    odom_drone = Odometry()

    # Tf Drone
    drone_tf = TransformBroadcaster()
    camera_tf = TransformBroadcaster()

    displacement = [xd, yd, zd]
    quaternion_d = [qx, qy, qz, qw] # x y z w
    r_quat = R.from_quat(quaternion_d)
    r_axis = r_quat.as_rotvec()
    r_axis_angle = np.linalg.norm([r_axis]) 
    r_axis_normalized = r_axis/r_axis_angle
    angles = [r_axis_normalized[0], r_axis_normalized[1], r_axis_normalized[2], r_axis_angle] 

    # Set Displacement and rotation
    set_robot(translation_field, rotation_field, displacement, angles)


    #####################################################3
    # Ángulo inicial de rotación en el eje Z
    angle = 0.0

    ###############################################################
    rate = rospy.Rate(15)
    # Initial Rotation system
    while robot.step(time_step) != -1:
        tic = time.time()
        
        # Position and anles from the callback
        displacement = [xd, yd, zd]
        quaternion_d = [qx, qy, qz, qw] # x y z w
        r_quat = R.from_quat(quaternion_d)
        r_axis = r_quat.as_rotvec()
        r_axis_angle = np.linalg.norm([r_axis]) 
        r_axis_normalized = r_axis/r_axis_angle
        angles = [r_axis_normalized[0], r_axis_normalized[1], r_axis_normalized[2], r_axis_angle] 

        # Set Displacement and rotation
        set_robot(translation_field, rotation_field, displacement, angles)
        #move_velocity(UAV_node, rotation_field)

        ############################################################3
        angle += 100 # Aumenta el ángulo en 1 grado por cada iteración (ajusta según sea necesario)

        
        rotate_object(rotation_Prop1, -angle)
        rotate_object(rotation_Prop2, angle)
        rotate_object(rotation_Prop3, -angle)
        rotate_object(rotation_Prop4, angle)

        ####################################################################################
        # Get Odometry
        odom_drone = get_odometry(translation_field, rotation_field, UAV_node, odom_drone)

        # Get image
        img_rgb = get_image(camera_rgb)
        img_d = get_range_image(camera_d)


        # Wait Ros Node and update times
        #loop_rate.sleep()
        #delta = time.time()- tic
        
        # Send Images
        send_image(bridge, image_pu_rgb, img_rgb)
        send_image_depth(bridge, image_pu_d, img_d)
        send_drone_tf(drone_tf, odom_drone)
        send_camera_tf(camera_tf)
        send_odometry(odom_drone, odometry_pu)

        rate.sleep()
        toc = time.time() - tic 
    
        print("FPS: {:.2f} segundos".format(1/toc), end='\r')
    
        
        
    return None

if __name__ == '__main__':
    try:
        # Node Initialization
        rospy.init_node("vision_system",disable_signals=True, anonymous=True)

        # Conection Webots External
        robot_a = Supervisor()

        # Vision Topic 
        image_topic_rbg = "/sim/color/image_raw"
        image_topic_d = "/camera/aligned_depth_to_color/image_raw"
        image_publisher_rgb = rospy.Publisher(image_topic_rbg, Image, queue_size=1)
        image_publisher_d = rospy.Publisher(image_topic_d, Image, queue_size=20)

        odometry_topic = "/dji_sdk/odometry"
        odometry_subscriber = rospy.Subscriber(odometry_topic, Odometry, odometry_call_back)

        velocity_topic = "/cmd_vel"
        velocity_subscriber = rospy.Subscriber(velocity_topic, Twist, velocity_call_back)

        odometry_webots = "/drone/odometry"
        odometry_publisher = rospy.Publisher(odometry_webots, Odometry, queue_size=10)

        # Simulation 
        main(robot_a, image_publisher_rgb, image_publisher_d, odometry_publisher)

    except(rospy.ROSInterruptException, KeyboardInterrupt):
        print("Error System")
        pass
    else:
        print("Complete Execution")
        pass
