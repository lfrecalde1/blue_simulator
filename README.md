# blue_simulator

<p float="left">
    <img src="Blue.gif" width="500"  />
 </p>

## Dependencies

This simulator was developed using Webots [2023a](https://github.com/cyberbotics/webots/releases/download/R2023a/webots_2023a_amd64.deb). Additionally, the ROS version is Noetic, which can be downloaded from [Noetic](https://wiki.ros.org/noetic/Installation/Ubuntu).

The developed package has the following dependencies, which must be included in the ROS working space: [webots_ros](https://github.com/cyberbotics/webots_ros.git) and [ackermann_msgs](https://github.com/ros-drivers/ackermann_msgs.git).

Finally, the project includes an installation file to include all necessary dependencies for Python and ROS, defined as follows:

```bash
chmod +x install_python.sh
./install_python.sh
```

The WEBOTS_HOME environment variable must be set to the installation folder of Webots.

```bash
export WEBOTS_HOME=/usr/local/webots
```

## Ros Workspace
Create the following ROS workspace

```bash
mkdir -p ~/catkin_ws/src
cd ~/catkin_ws/
```
Inside the ROS workspace /src must be included:
[webots_ros](https://github.com/cyberbotics/webots_ros.git) and [ackermann_msgs](https://github.com/ros-drivers/ackermann_msgs.git).

Due to the python3 is necessary to configure the catkin workspace as follows:


```bash
catkin_make -DPYTHON_EXECUTABLE=/usr/bin/python3
```
 Before continuing source your new setup.*sh file:

```bash
source devel/setup.bash
```

## Use
To run the simulator, execute the following command:

```bash
roslaunch blue_simulator simulator_webots.launch
```