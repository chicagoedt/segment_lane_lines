#!/home/revo/anaconda2/envs/ROS+TF/bin/python
"""@package docstring
Package for semantic segmentation of lane lines, and homographic
transformation of the segmented image for rebroadcasting as a point cloud.
Work-in-progress.

Todo:
    * Change EDT-specific names to YAML/rosparams.
    * Implement homographic transform methods.
    * Refactoring and documentation. (clean up variable names!)
    * Rewrite model to augment predictions using predictions from previous frames.
    * Build fully-convolutional model dynamically to accomodate images of different sizes.
"""

import sys
import time
from keras.models import load_model
import rospy
from sensor_msgs.msg import Image, PointCloud2
from cv_bridge import CvBridge, CvBridgeError
import numpy as np
import tensorflow as tf
import cv2


class LaneSegmentationNode(object):
    """Class providing all functionality of lane_segmentation_node.

    Attributes:
        __model (keras.models.Model): The Keras Model object used to segment the
            numpy array containing the image data.
        __bridge (cv_bridge.CvBridge): The default CvBridge object.
        __image_sub(rospy.Subscriber): Subscriber object which invokes the
            segmentation and transormation as its callback.
        __image_pub (rospy.Publisher):
        __cloud_pub (rospy.Publisher):

    """
    def __init__(self):
        """LaneSegmentationNode constructor."""
        self.__model = load_model(sys.argv[1])
        self.__bridge = CvBridge()
        self.__image_sub = rospy.Subscriber("/stereo_camera/left/image_color",
                    Image, self.__callback, queue_size=1, buff_size=100000000)
        self.__image_pub = rospy.Publisher("/stereo_camera/left/lanes", Image, queue_size=1)
        self.__cloud_pub = rospy.Publisher("/topic/name/here", PointCloud2, queue_size=1)

    def msg_to_numpy(self, data):
        """Extracts image data from Image message.

        Args:
            data (sensor_msgs/Image): The ROS Image message, exactly as passed
                by the subscriber to its callback.

        Returns:
            The image, as a NumPy array.

        """
        try:
            raw_img = self.__bridge.imgmsg_to_cv2(data, "bgr8")
        except CvBridgeError as err:
            print(err)

        return raw_img

    def numpy_to_msg(self, img):
        """Builds a Image message from a NumPy array.

        Args:
            img (np.array): A NumPy array containing the RGB image data.

        Returns:
            A sensor_msgs/Image containing the image data.

        """
        try:
            data = self.__bridge.cv2_to_imgmsg(img, "rgb8")
        except CvBridgeError as err:
            print(err)

        return data

    def __segment_and_publish(self, img):
        """Runs the segmentation model and publishes the result.

        Args:
            data (sensor_msgs/Image): The ROS Image message, exactly as passed
                by the subscriber to its callback.

        Returns:
            A NumPy array representing the segmented image.

        """
        start = time.clock()
        # Preprocess input.
        model_input = np.array([img]).astype('float32') / 255
        # Run the semantic segmentation model.
        with tf.get_default_graph().as_default():
            lane_lines = (self.__model.predict(model_input)[0] * 255).astype(np.uint8)
        # Publish and return.
        lane_lines_rgb = cv2.cvtColor(lane_lines, cv2.COLOR_GRAY2RGB)
        self.__image_pub.publish(self.numpy_to_msg(lane_lines_rgb))
        end = time.clock()
        print("Latency: " + str((end - start) * 1000) + " milliseconds.")

        return lane_lines

    def __transform_and_publish(self, segmented_img):
        """Class methods are similar to regular functions.

        Note:
            Do not include the `self` parameter in the ``Args`` section.

        Args:
            param1: The first parameter.
            param2: The second parameter.

        Returns:
            True if successful, False otherwise.

        """
        pass

    def __callback(self, data):
        """Class methods are similar to regular functions.

        Note:
            Do not include the `self` parameter in the ``Args`` section.

        Args:
            param1: The first parameter.
            param2: The second parameter.

        Returns:
            True if successful, False otherwise.

        """
        img = self.msg_to_numpy(data)
        seg = self.__segment_and_publish(img)
        self.__transform_and_publish(seg)

if __name__ == '__main__':
    rospy.init_node('lane_segmentation_node', anonymous=True)
    lane_segmentation_node = LaneSegmentationNode()
    try:
        rospy.spin()
    except KeyboardInterrupt:
        print("Shutting down.")


"""
Note: Due to the raw size of the CMakeLists.txt file, GitHub
incorrectly classifies this repo as a CMake project.
Solution: Copy the default CMakeLists.txt into a .py file.

cmake_minimum_required(VERSION 2.8.3)
project(lane_finding)

## Find catkin macros and libraries
## if COMPONENTS list like find_package(catkin REQUIRED COMPONENTS xyz)
## is used, also find other catkin packages
find_package(catkin REQUIRED COMPONENTS roscpp cv_bridge sensor_msgs)
find_package(OpenCV 2 REQUIRED)

## System dependencies are found with CMake's conventions
# find_package(Boost REQUIRED COMPONENTS system)


## Uncomment this if the package has a setup.py. This macro ensures
## modules and global scripts declared therein get installed
## See http://ros.org/doc/api/catkin/html/user_guide/setup_dot_py.html
# catkin_python_setup()

################################################
## Declare ROS messages, services and actions ##
################################################

## To declare and build messages, services or actions from within this
## package, follow these steps:
## * Let MSG_DEP_SET be the set of packages whose message types you use in
##   your messages/services/actions (e.g. std_msgs, actionlib_msgs, ...).
## * In the file package.xml:
##   * add a build_depend tag for "message_generation"
##   * add a build_depend and a run_depend tag for each package in MSG_DEP_SET
##   * If MSG_DEP_SET isn't empty the following dependency has been pulled in
##     but can be declared for certainty nonetheless:
##     * add a run_depend tag for "message_runtime"
## * In this file (CMakeLists.txt):
##   * add "message_generation" and every package in MSG_DEP_SET to
##     find_package(catkin REQUIRED COMPONENTS ...)
##   * add "message_runtime" and every package in MSG_DEP_SET to
##     catkin_package(CATKIN_DEPENDS ...)
##   * uncomment the add_*_files sections below as needed
##     and list every .msg/.srv/.action file to be processed
##   * uncomment the generate_messages entry below
##   * add every package in MSG_DEP_SET to generate_messages(DEPENDENCIES ...)

## Generate messages in the 'msg' folder
# add_message_files(
#   FILES
#   Message1.msg
#   Message2.msg
# )

## Generate services in the 'srv' folder
# add_service_files(
#   FILES
#   Service1.srv
#   Service2.srv
# )

## Generate actions in the 'action' folder
# add_action_files(
#   FILES
#   Action1.action
#   Action2.action
# )

## Generate added messages and services with any dependencies listed here
# generate_messages(
#   DEPENDENCIES
#   std_msgs  # Or other packages containing msgs
# )

################################################
## Declare ROS dynamic reconfigure parameters ##
################################################

## To declare and build dynamic reconfigure parameters within this
## package, follow these steps:
## * In the file package.xml:
##   * add a build_depend and a run_depend tag for "dynamic_reconfigure"
## * In this file (CMakeLists.txt):
##   * add "dynamic_reconfigure" to
##     find_package(catkin REQUIRED COMPONENTS ...)
##   * uncomment the "generate_dynamic_reconfigure_options" section below
##     and list every .cfg file to be processed

## Generate dynamic reconfigure parameters in the 'cfg' folder
# generate_dynamic_reconfigure_options(
#   cfg/DynReconf1.cfg
#   cfg/DynReconf2.cfg
# )

###################################
## catkin specific configuration ##
###################################
## The catkin_package macro generates cmake config files for your package
## Declare things to be passed to dependent projects
## INCLUDE_DIRS: uncomment this if you package contains header files
## LIBRARIES: libraries you create in this project that dependent projects also need
## CATKIN_DEPENDS: catkin_packages dependent projects also need
## DEPENDS: system dependencies of this project that dependent projects also need
catkin_package(
#  INCLUDE_DIRS include
#  LIBRARIES lane_finding
#  CATKIN_DEPENDS other_catkin_pkg
#  DEPENDS system_lib
)

###########
## Build ##
###########

## Specify additional locations of header files
## Your package locations should be listed before other locations
include_directories(include ${catkin_INCLUDE_DIRS} ${OpenCV_INCLUDE_DIRS})

## Declare a C++ library
# add_library(lane_finding
#   src/${PROJECT_NAME}/lane_finding.cpp
# )

## Add cmake target dependencies of the library
## as an example, code may need to be generated before libraries
## either from message generation or dynamic reconfigure
# add_dependencies(lane_finding ${${PROJECT_NAME}_EXPORTED_TARGETS} ${catkin_EXPORTED_TARGETS})

## Declare a C++ executable
add_executable(testcaster src/testcaster.cpp)

## Add cmake target dependencies of the executable
## same as for the library above
# add_dependencies(lane_finding_node ${${PROJECT_NAME}_EXPORTED_TARGETS} ${catkin_EXPORTED_TARGETS})

## Specify libraries to link a library or executable target against
target_link_libraries(testcaster
   ${catkin_LIBRARIES}
   ${OpenCV_LIBS}
)

#############
## Install ##
#############

# all install targets should use catkin DESTINATION variables
# See http://ros.org/doc/api/catkin/html/adv_user_guide/variables.html

## Mark executable scripts (Python etc.) for installation
## in contrast to setup.py, you can choose the destination
# install(PROGRAMS
#   scripts/my_python_script
#   DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION}
# )

## Mark executables and/or libraries for installation
# install(TARGETS lane_finding lane_finding_node
#   ARCHIVE DESTINATION ${CATKIN_PACKAGE_LIB_DESTINATION}
#   LIBRARY DESTINATION ${CATKIN_PACKAGE_LIB_DESTINATION}
#   RUNTIME DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION}
# )

## Mark cpp header files for installation
# install(DIRECTORY include/${PROJECT_NAME}/
#   DESTINATION ${CATKIN_PACKAGE_INCLUDE_DESTINATION}
#   FILES_MATCHING PATTERN "*.h"
#   PATTERN ".svn" EXCLUDE
# )

## Mark other files for installation (e.g. launch and bag files, etc.)
# install(FILES
#   # myfile1
#   # myfile2
#   DESTINATION ${CATKIN_PACKAGE_SHARE_DESTINATION}
# )

#############
## Testing ##
#############

## Add gtest based cpp test target and link libraries
# catkin_add_gtest(${PROJECT_NAME}-test test/test_lane_finding.cpp)
# if(TARGET ${PROJECT_NAME}-test)
#   target_link_libraries(${PROJECT_NAME}-test ${PROJECT_NAME})
# endif()

## Add folders to be run by python nosetests
# catkin_add_nosetests(test)
"""
