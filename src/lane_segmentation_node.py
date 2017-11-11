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
    """Class containing all attributes of lane_segmentation_node.

    If the class has public attributes, they may be documented here
    in an ``Attributes`` section and follow the same formatting as a
    function's ``Args`` section. Alternatively, attributes may be documented
    inline with the attribute's declaration (see __init__ method below).

    Properties created with the ``@property`` decorator should be documented
    in the property's getter method.

    Attributes:
        attr1 (str): Description of `attr1`.
        attr2 (:obj:`int`, optional): Description of `attr2`.

    """
    def __init__(self):
        """LaneSegmentationNode constructor.

        The __init__ method may be documented in either the class level
        docstring, or as a docstring on the __init__ method itself.

        Either form is acceptable, but the two should not be mixed. Choose one
        convention to document the __init__ method and be consistent with it.

        Note:
            Do not include the `self` parameter in the ``Args`` section.

        Args:
            param1 (str): Description of `param1`.
            param2 (:obj:`int`, optional): Description of `param2`. Multiple
                lines are supported.
            param3 (:obj:`list` of :obj:`str`): Description of `param3`.

        """
        self.model = load_model(sys.argv[1])
        self.graph = tf.get_default_graph()
        self.image_pub = rospy.Publisher("/stereo_camera/left/lanes", Image, queue_size=1)
        self.bridge = CvBridge()
        self.image_sub = rospy.Subscriber("/stereo_camera/left/image_color", Image, self.callback, queue_size=1, buff_size=100000000)

    def segment_and_publish(self, data):
        """Class methods are similar to regular functions.

        Note:
            Do not include the `self` parameter in the ``Args`` section.

        Args:
            param1: The first parameter.
            param2: The second parameter.

        Returns:
            True if successful, False otherwise.

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
