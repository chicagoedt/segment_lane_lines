Package for semantic segmentation of lane lines, and homographic transformation of the segmented image for rebroadcasting as a point cloud. Work-in-progress.

# Overview

# Sample Usage

# Nodes
## lane_segmentation_node
### Subscribed Topics
`lane_camera_feed` [(sensor_msgs/Image)](http://docs.ros.org/api/sensor_msgs/html/msg/Image.html)

 Feed from the camera.

### Published Topics
`lane_lines_image` [(sensor_msgs/Image)](http://docs.ros.org/api/sensor_msgs/html/msg/Image.html)

 The most recently-segmented image in the feed, untransformed.

`lane_lines_cloud` [(sensor_msgs/Image)](http://docs.ros.org/api/sensor_msgs/html/msg/PointCloud2.html)

 A point cloud representing the pixels in the segmented image, homographically transformed into the ground plane.

### Parameters
