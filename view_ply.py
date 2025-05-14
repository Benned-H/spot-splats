import open3d as o3d
import argparse

parser = argparse.ArgumentParser(description="Open3D PLY Viewer")
parser.add_argument("--path", type=str, help="Path to the .ply file")
args = parser.parse_args()
ply_path = args.path

try:
    pcd = o3d.io.read_point_cloud(ply_path)
    o3d.visualization.draw_geometries([pcd])
except Exception as e:
    print(f"Error: Could not open or visualize the .ply file. {e}")


