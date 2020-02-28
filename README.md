# 2D-3D pose tracking
## Monocular Camera Localization in Prior LiDAR Maps with 2D-3D Line Correspondences

2D-3D pose tracking is a real-time camera localization framework with prior LiDAR maps. It detects geometric 3D lines offline from LiDAR maps and use AFM to detect 2D lines from video sequences online. It efficiently obtains 2D-3D line correspondences based on the camera motion prediction from VINS-Mono. The camera poses and 2D-3D correspondences are iteratively optimized in the sliding window. The matching of visual features to a prior map in 3D does not embed color information.Therefore, it is robust to lighting or texture changes.
Based on geometric 2D-3D line correspondences, our method is suitable for localization in urban environment. This code runs on **Linux**. The video demos can be seen:



