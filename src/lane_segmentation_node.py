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
"""

import sys
import time
from keras.models import load_model
import rospy
from sensor_msgs.msg import Image
from cv_bridge import CvBridge, CvBridgeError
import numpy as np
import tensorflow as tf
import cv2


class LaneSegmentationNode:
    """Documentation for the lane_segmentation_node class.

    More details.
    """
    def __init__(self):
        """The constructor."""
        self.model = load_model(sys.argv[1])
        self.graph = tf.get_default_graph()
        self.image_pub = rospy.Publisher("/stereo_camera/left/lanes", Image, queue_size=1)
        self.bridge = CvBridge()
        self.image_sub = rospy.Subscriber("/stereo_camera/left/image_color", Image, self.callback, queue_size=1, buff_size=100000000)

    def segment_and_publish(self, data):
        """Segments and publishes the image.

        More details.
        """
        start = time.clock()
        # sensor_msgs/Image -> NumPy array
        try:
            raw_img = self.bridge.imgmsg_to_cv2(data, "bgr8")
        except CvBridgeError as e:
            print(e)
        # Preprocess input.
        model_input = np.array([raw_img]).astype('float32') / 255
        # Run the semantic segmentation model.
        with self.graph.as_default():
            model_output = self.model.predict(model_input)[0] * 255
            lane_lines = model_output.astype(np.uint8)
            lane_lines_rgb = cv2.cvtColor(lane_lines, cv2.COLOR_GRAY2RGB)
        # NumPy array -> sensor_msgs/Image
        try:
            segmented_image = self.bridge.cv2_to_imgmsg(lane_lines_rgb, "rgb8")
        except CvBridgeError as e:
            print(e)
        # Publish and return.
        self.image_pub.publish(segmented_image)
        end = time.clock()
        print("Latency: " + str((end - start) * 1000) + " milliseconds.")

        return lane_lines

    def transform_and_publish(self, segmented_image):
        """Publishes the white pixels in the segmented image as a point cloud.

        More details.
        """
        pass

    def callback(self, data):
        """Callback for the image_sub Subscriber.

        More details.
        """
        self.transform_and_publish(self.segment_and_publish(data))

if __name__ == '__main__':
    rospy.init_node('lane_segmentation_node', anonymous=True)
    lane_segmentation_node = LaneSegmentationNode()
    try:
        rospy.spin()
    except KeyboardInterrupt:
        print("Shutting down.")
