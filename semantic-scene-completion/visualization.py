import numpy as np
import open3d as o3d
# Create a new visualizer
visualizer = o3d.visualization.Visualizer()

# Add a pointcloud to the visualizer
points = (np.random.rand(1000, 3) - 0.5) / 4
colors = np.random.rand(1000, 3)
pcd = o3d.geometry.PointCloud()
pcd.points = o3d.utility.Vector3dVector(points)
pcd.colors = o3d.utility.Vector3dVector(colors)
visualizer.add_geometry(pcd)

# Capture the screen image
# Visualize the point cloud
o3d.visualization.draw_plotly([point_cloud])

# Save the image