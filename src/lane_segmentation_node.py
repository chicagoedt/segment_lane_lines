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
