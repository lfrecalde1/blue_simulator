ARG BASE_IMAGE=cyberbotics/webots:R2023a-ubuntu20.04
FROM $BASE_IMAGE AS downloader

# Determine Webots version to be used and set default argument
ARG WEBOTS_VERSION=R2023a

RUN apt update && apt install --yes unzip && \
  wget https://github.com/cyberbotics/webots/releases/download/$WEBOTS_VERSION/assets-$WEBOTS_VERSION.zip && \
  mkdir assets && cp assets-$WEBOTS_VERSION.zip assets/ && \
  cd assets && unzip assets-$WEBOTS_VERSION.zip && rm assets-$WEBOTS_VERSION.zip

FROM $BASE_IMAGE
ARG WEBOTS_VERSION=R2023b

#COPY Webots-R2023b.conf /root/.config/Cyberbotics/Webots-$WEBOTS_VERSION.conf
#COPY --from=downloader /usr/local/assets /root/.cache/Cyberbotics/Webots/assets/


ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update && apt-get install -y apt-utils curl wget git bash-completion build-essential sudo && rm -rf /var/lib/apt/lists/*

# Now create the user
ARG UID=1000
ARG GID=1000
RUN addgroup --gid ${GID} docker
RUN adduser --gecos "ROS User" --disabled-password --uid ${UID} --gid ${GID} docker
RUN usermod -a -G dialout docker
RUN mkdir config && echo "ros ALL=(ALL) NOPASSWD: ALL" > config/99_aptget
RUN cp config/99_aptget /etc/sudoers.d/99_aptget
RUN chmod 0440 /etc/sudoers.d/99_aptget && chown root:root /etc/sudoers.d/99_aptget

## ROS NOETIC INSTALL
# Minimal setup
RUN apt-get update \
 && apt-get install -y locales lsb-release
RUN dpkg-reconfigure locales
 
# Install ROS Noetic
RUN sh -c 'echo "deb http://packages.ros.org/ros/ubuntu $(lsb_release -sc) main" > /etc/apt/sources.list.d/ros-latest.list'
RUN apt-key adv --keyserver 'hkp://keyserver.ubuntu.com:80' --recv-key C1CF6E31E6BADE8868B172B4F42ED6FBAB17C654
RUN apt-get update \
 && apt-get install -y --no-install-recommends ros-noetic-desktop-full
RUN apt-get install -y --no-install-recommends python3-rosdep
RUN rosdep init \
 && rosdep fix-permissions \
 && rosdep update

# Change HOME environment variable
ENV HOME /home/webots
RUN mkdir -p ${HOME}/webots_ros/src

# INSTALL GIT
RUN apt-get update && \
    apt-get upgrade -y && \
    apt-get install -y git

# INSTALL DEPENDENS
RUN apt-get install -y python3-pip

# CLONE REPOSITORY
RUN cd ${HOME}/webots_ros/src/ \
	&& git clone https://github.com/lfrecalde1/blue_simulator.git \
	&& git clone https://github.com/cyberbotics/webots_ros.git \
	&& cd blue_simulator \
	&& chmod +x install_python.sh \
	&& ./install_python.sh

# INSTALL DEPENDENS
#RUN apt-get install -y python3-pip
#RUN pip3 install catkin_pkg numpy scipy==1.7.2 pyyaml empy
#RUN pip3 install -U rospkg
#RUN apt install -y ros-noetic-moveit

# Initialize the workspace
RUN pip3 install empy catkin_pkg
RUN /bin/bash -c 'source /opt/ros/noetic/setup.bash &&\
    cd ${HOME}/webots_ros/ &&\
    catkin_make'

# set up environment
COPY ./update_bashrc /sbin/update_bashrc
RUN sudo chmod +x /sbin/update_bashrc ; sudo chown ros /sbin/update_bashrc ; sync ; /bin/bash -c /sbin/update_bashrc ; sudo rm /sbin/update_bashrc

#INSTALL VIM y NANO
RUN apt-get install vim nano -y
 
# Enable OpenGL capabilities
#ENV NVIDIA_DRIVER_CAPABILITIES graphics,compute,utility

# Set the locales
RUN locale-gen en_US.UTF-8
ENV LANG='en_US.UTF-8' LANGUAGE='en_US:en' LC_ALL='en_US.UTF-8'

# Finally open a bash command to let the user interact
#CMD ["/bin/bash"]
CMD cd ${HOME}/webots_ros/ && /bin/bash -c "source devel/setup.bash; roslaunch blue_simulator simulator_webots.launch"
