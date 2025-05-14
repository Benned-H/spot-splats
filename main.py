import argparse
import torch
import os
import math
import numpy as np
import imageio
from gsplat.rendering import rasterization
from gsplat.utils import depth_to_points # Removed normalized_quat_to_rotmat as it's not used with COLMAP poses
import torch.nn.functional as F
from datasets.colmap import Parser
import open3d as o3d
from gsplat.exporter import splat2ply_bytes # Added for exporting splat parameters
from io import BytesIO # splat2ply_bytes uses BytesIO, ensure it's available
import pickle # Added for loading .pkl files
import sys # Added to exit after inspection

#Project Steps 
    #1 collect sequence of images of a scene from robot
    #2 run colmap to extract sparse reconstruction
    #3 train a gsplat model on a dataset with simple_trainer.py
    #4 run this script with the ckpt to render the images and save the point cloud as a .ply file

def parse_arguments():
    """Parses command-line arguments for the pose viewer script."""
    parser = argparse.ArgumentParser(description="Load Gaussian Splatting checkpoint and render images from COLMAP poses.")
    # Arguments for COLMAP data
    parser.add_argument("--colmap_data_dir", type=str, required=True, help="Path to the COLMAP data directory (e.g., data/spot_room).")
    parser.add_argument("--colmap_factor", type=int, default=1, help="Downsample factor for images when parsing COLMAP data.")
    parser.add_argument("--colmap_normalize", type=bool, default=True, help="Normalize COLMAP poses and points to a unit sphere (True/False).")
    
    parser.add_argument("--ckpt_path", type=str, required=True, help="Path to the Gaussian Splatting checkpoint (.pt) file.")
    parser.add_argument("--output_dir", type=str, default="results/pose_renders_colmap", help="Directory to save rendered images and point cloud.")
    parser.add_argument("--render_mode", type=str, default="RGB+D", help="Render mode (e.g., 'RGB', 'D', 'RGB+D'). Must include 'D' for point cloud generation.")

    #add image width and height arguments
    parser.add_argument("--render_width", type=int, default=640, help="Width of the rendered image.")
    parser.add_argument("--render_height", type=int, default=480, help="Height of the rendered image.")
    parser.add_argument("--use_render_width", action='store_true', help="If set, use render_width and render_height instead of COLMAP image dimensions.")

    # Arguments for point cloud generation
    parser.add_argument("--save_pointcloud", action='store_true', help="If set, generate and save an aggregated point cloud.")
    parser.add_argument("--pointcloud_filename", type=str, default="pointcloud_output", help="Base filename for the saved point cloud(s). Suffixes like '_rendered_views.ply' or '_splat_params_centers_colored_sh0.ply' will be appended.")
    parser.add_argument("--pointcloud_alpha_threshold", type=float, default=0.8, help="Minimum alpha value for a point to be included in the point cloud (rendered_views method).")
    parser.add_argument("--pointcloud_max_depth", type=float, default=10.0, help="Maximum depth value for a point to be included in the point cloud (rendered_views method, in scene units).")
    parser.add_argument("--pointcloud_voxel_size", type=float, default=0.01, help="Voxel size for downsampling the point cloud with Open3D (rendered_views method only). If 0.0 or not positive, no downsampling. E.g., 0.01.")
    parser.add_argument("--pointcloud_export_method", type=str, default="rendered_views", choices=["rendered_views", "splat_params"], help="Method to generate the point cloud. 'rendered_views' uses back-projection from rendered depth. 'splat_params' directly exports Gaussian splat parameters (both full properties and centers colored by SH0).")
    
    # Arguments for robot data inspection/saving
    parser.add_argument("--save_robot_data", action='store_true', help="If set, enables robot data saving/inspection mode.")
    parser.add_argument("--robot_output_dir", type=str, default="results/robot_data_exports", help="Directory to save exported robot-specific data (images, depth, poses) when --save_robot_data is active.")
    return parser.parse_args(), parser

# Helper function to convert rotation matrix to WXYZ quaternion
def rotation_matrix_to_quaternion_wxyz(matrix):
    """Converts a 3x3 rotation matrix to a WXYZ quaternion.
    Args:
        matrix (np.ndarray): A 3x3 rotation matrix.
    Returns:
        list[float]: A list representing the quaternion [w, x, y, z].
    """
    tr = np.trace(matrix)
    if tr > 0:
        S = math.sqrt(tr + 1.0) * 2  # S=4*qw
        qw = 0.25 * S
        qx = (matrix[2, 1] - matrix[1, 2]) / S
        qy = (matrix[0, 2] - matrix[2, 0]) / S
        qz = (matrix[1, 0] - matrix[0, 1]) / S
    elif (matrix[0, 0] > matrix[1, 1]) and (matrix[0, 0] > matrix[2, 2]):
        S = math.sqrt(1.0 + matrix[0, 0] - matrix[1, 1] - matrix[2, 2]) * 2  # S=4*qx
        qw = (matrix[2, 1] - matrix[1, 2]) / S
        qx = 0.25 * S
        qy = (matrix[0, 1] + matrix[1, 0]) / S
        qz = (matrix[0, 2] + matrix[2, 0]) / S
    elif matrix[1, 1] > matrix[2, 2]:
        S = math.sqrt(1.0 + matrix[1, 1] - matrix[0, 0] - matrix[2, 2]) * 2  # S=4*qy
        qw = (matrix[0, 2] - matrix[2, 0]) / S
        qx = (matrix[0, 1] + matrix[1, 0]) / S
        qy = 0.25 * S
        qz = (matrix[1, 2] + matrix[2, 1]) / S
    else:
        S = math.sqrt(1.0 + matrix[2, 2] - matrix[0, 0] - matrix[1, 1]) * 2  # S=4*qz
        qw = (matrix[1, 0] - matrix[0, 1]) / S
        qx = (matrix[0, 2] + matrix[2, 0]) / S
        qy = (matrix[1, 2] + matrix[2, 1]) / S
        qz = 0.25 * S
    return [qw, qx, qy, qz]

def load_colmap_data(args):
    """Loads camera poses, intrinsics, and image information from COLMAP data.
    Args:
        args: Parsed command-line arguments containing COLMAP data directory and parameters.
    Returns:
        datasets.colmap.Parser: An instance of the COLMAP parser with loaded data.
    Raises:
        Exception: If there is an error during COLMAP data loading.
    """
    print(f"Loading COLMAP data from: {args.colmap_data_dir} with factor {args.colmap_factor}, normalize: {args.colmap_normalize}")
    try:
        colmap_parser = Parser(
            data_dir=args.colmap_data_dir,
            factor=args.colmap_factor,
            normalize=args.colmap_normalize,
            test_every=1_000_000  # Effectively load all images
        )
        print(f"Successfully loaded COLMAP data. Found {len(colmap_parser.image_names)} images/poses.")
        return colmap_parser
    except Exception as e:
        print(f"Error loading COLMAP data: {e}")
        raise

def load_splat_checkpoint(ckpt_path, device):
    """Loads a Gaussian Splatting model checkpoint.
    Args:
        ckpt_path (str): Path to the .pt checkpoint file.
        device (torch.device): The device to load the tensors onto (e.g., 'cuda', 'cpu').
    Returns:
        tuple:
            - dict: Processed splat data for rendering (exponentiated scales, sigmoided opacities, etc.).
            - dict: Raw splat data dictionary from the checkpoint for export.
    Raises:
        FileNotFoundError: If the checkpoint file is not found.
        ValueError: If the checkpoint is missing required keys or has an unexpected structure.
        Exception: For other loading errors.
    """
    print(f"Loading checkpoint from: {ckpt_path}")
    try:
        ckpt_data = torch.load(ckpt_path, map_location=device)
        
        if "splats" not in ckpt_data:
            raise ValueError(f"'splats' key not found in checkpoint. Available keys: {list(ckpt_data.keys())}")

        raw_splats_dict = ckpt_data["splats"]
        
        required_keys = {"means", "quats", "scales", "opacities", "sh0", "shN"}
        if not required_keys.issubset(raw_splats_dict.keys()):
            missing_keys = required_keys - raw_splats_dict.keys()
            raise ValueError(f"Checkpoint 'splats' dictionary is missing required keys: {missing_keys}")

        # Process for rendering
        means_render = raw_splats_dict["means"].to(device)
        quats_render = F.normalize(raw_splats_dict["quats"], p=2, dim=-1).to(device)
        scales_render = torch.exp(raw_splats_dict["scales"]).to(device)
        opacities_render = torch.sigmoid(raw_splats_dict["opacities"]).to(device)
        sh0_render = raw_splats_dict["sh0"].to(device)
        shN_render = raw_splats_dict["shN"].to(device)
        colors_feat_render = torch.cat([sh0_render, shN_render], dim=-2)
        sh_degree_render = int(math.sqrt(colors_feat_render.shape[-2]) - 1)
        
        print(f"Loaded checkpoint with {means_render.shape[0]} Gaussians. SH degree: {sh_degree_render}")
        
        processed_splats_for_rendering = {
            "means": means_render, "quats": quats_render, "scales": scales_render,
            "opacities": opacities_render, "colors_feat": colors_feat_render, "sh_degree": sh_degree_render
        }
        
        # Ensure all raw tensors are on the specified device for consistency, though splat2ply_bytes moves to CPU
        raw_splats_dict_on_device = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in raw_splats_dict.items()}

        return processed_splats_for_rendering, raw_splats_dict_on_device
    except FileNotFoundError:
        print(f"Error: Checkpoint file not found at {ckpt_path}")
        raise
    except Exception as e:
        print(f"Error loading checkpoint: {e}")
        raise

def render_single_view(splat_data, c2w, K, width, height, render_mode, device):
    """Renders a single view using the Gaussian Splatting model.
    Args:
        splat_data (dict): Dictionary containing Gaussian splat model parameters.
        c2w (torch.Tensor): Camera-to-world transformation matrix (4x4).
        K (torch.Tensor): Camera intrinsics matrix (3x3 or 4x4).
        width (int): Width of the rendered image.
        height (int): Height of the rendered image.
        render_mode (str): Rendering mode (e.g., "RGB+D", "RGB", "D").
        device (torch.device): Device to perform rendering on.
    Returns:
        tuple:
            - render_colors (torch.Tensor): Rendered colors (B, H, W, C).
            - render_alphas (torch.Tensor): Rendered alpha values (B, H, W, 1).
            - meta (dict): Meta information from the rasterization process.
    """
    viewmat = torch.linalg.inv(c2w).to(device)
    K_tensor = K.to(device)

    render_colors, render_alphas, meta = rasterization(
        means=splat_data["means"],
        quats=splat_data["quats"],
        scales=splat_data["scales"],
        opacities=splat_data["opacities"],
        colors=splat_data["colors_feat"], # Use 'colors_feat' from loaded splat_data
        viewmats=viewmat.unsqueeze(0),
        Ks=K_tensor.unsqueeze(0),
        width=width,
        height=height,
        sh_degree=splat_data["sh_degree"],
        render_mode=render_mode,
        packed=False
    )
    return render_colors, render_alphas, meta

def process_rendered_output(render_colors_batched, render_alphas_batched, render_mode):
    """Processes the raw output from the rasterization step.
    Separates RGB, depth, and alpha channels based on the render_mode.
    Assumes a batch size of 1 for the input tensors.
    Args:
        render_colors_batched (torch.Tensor): Batched rendered colors/data (1, H, W, C).
                                             Channel content depends on render_mode.
        render_alphas_batched (torch.Tensor): Batched rendered alpha values (1, H, W, 1).
        render_mode (str): The rendering mode used (e.g., "RGB+D", "RGB", "D").
    Returns:
        tuple:
            - render_rgb (torch.Tensor | None): Processed RGB image (H, W, 3), or None.
            - render_depth (torch.Tensor | None): Processed depth map (H, W), or None.
            - render_alphas_single (torch.Tensor): Processed alpha map (H, W, 1).
    """
    # Remove batch dimension (B=1)
    render_colors_single = render_colors_batched[0]
    render_alphas_single = render_alphas_batched[0] # Shape (H, W, 1)

    render_rgb = None
    render_depth = None

    if "D" in render_mode.upper():
        if render_colors_single.shape[-1] == 4:  # RGB+D
            render_rgb = render_colors_single[..., :3].clamp(0, 1)
            render_depth = render_colors_single[..., 3]
        elif render_colors_single.shape[-1] == 1:  # D only
            render_depth = render_colors_single[..., 0]
    elif render_colors_single.shape[-1] >= 3 : # RGB or other modes (assuming at least 3 channels for RGB)
         render_rgb = render_colors_single[..., :3].clamp(0, 1)
    
    return render_rgb, render_depth, render_alphas_single

def save_image_outputs(output_dir, clean_base_name, rgb_image, depth_image):
    """Saves rendered RGB and depth images to disk for regular mode.
    Args:
        output_dir (str): Directory to save the images.
        clean_base_name (str): Clean base name for the output files (without prefix or extension).
        rgb_image (torch.Tensor | None): RGB image tensor (H, W, 3) to save.
        depth_image (torch.Tensor | None): Depth image tensor (H, W) to save.
    """
    if rgb_image is not None:
        output_image_path = os.path.join(output_dir, f"color_{clean_base_name}.jpg")
        img_np = (rgb_image.cpu().numpy() * 255).astype(np.uint8)
        imageio.imsave(output_image_path, img_np)
        print(f"Saved RGB image to {output_image_path}")

    if depth_image is not None:
        output_depth_path = os.path.join(output_dir, f"depth_{clean_base_name}.png")
        depth_vis = depth_image.cpu().numpy()
        # Normalize depth for visualization (0-255)
        if np.any(depth_vis > 0):
            depth_vis_norm = np.clip(depth_vis, 0, np.percentile(depth_vis[depth_vis > 0], 99))
            if np.any(depth_vis_norm > 0): # Check again after clipping
                 depth_vis_norm = (depth_vis_norm / depth_vis_norm.max() * 255).astype(np.uint8)
            else:
                depth_vis_norm = np.zeros_like(depth_vis_norm, dtype=np.uint8)
        else:
            depth_vis_norm = np.zeros_like(depth_vis, dtype=np.uint8)
        imageio.imsave(output_depth_path, depth_vis_norm)
        print(f"Saved depth image to {output_depth_path}")

def extract_points_for_cloud(render_depth, render_rgb, render_alphas, c2w, K, args):
    """Extracts 3D points and their colors from a single rendered view.

    Points are filtered based on depth range and alpha values.
    Args:
        render_depth (torch.Tensor | None): Depth map for the view (H, W).
        render_rgb (torch.Tensor | None): RGB image for the view (H, W, 3).
        render_alphas (torch.Tensor): Alpha map for the view (H, W, 1).
        c2w (torch.Tensor): Camera-to-world transformation matrix for the view.
        K (torch.Tensor): Camera intrinsics matrix for the view.
        args: Parsed command-line arguments containing point cloud parameters
              (pointcloud_max_depth, pointcloud_alpha_threshold).

    Returns:
        tuple:
            - points_xyz_view (np.ndarray | None): (N, 3) array of 3D point coordinates, or None.
            - points_rgb_view (np.ndarray | None): (N, 3) array of RGB colors (0-255), or None.
    """
    if render_depth is None:
        return None, None

    # Create masks for valid points
    # render_depth is (H,W), render_alphas is (H,W,1)
    valid_depth_mask = (render_depth > 1e-3) & (render_depth < args.pointcloud_max_depth)
    valid_alpha_mask = render_alphas[..., 0] > args.pointcloud_alpha_threshold
    final_mask = valid_depth_mask & valid_alpha_mask
    
    if not torch.any(final_mask):
        return None, None

    world_points = depth_to_points(
        depths=render_depth.unsqueeze(-1),  # Expects [H,W,1]
        camtoworlds=c2w,
        Ks=K,
        z_depth=True  # Assuming depth from rasterization is z-depth
    )  # Output is [H,W,3]

    points_xyz_view = world_points[final_mask].cpu().numpy()

    if render_rgb is not None:
        points_rgb_view = (render_rgb[final_mask].cpu().numpy() * 255).astype(np.uint8)
    else:
        # Default to gray if no RGB info
        points_rgb_view = np.full_like(points_xyz_view, 128, dtype=np.uint8)
    
    return points_xyz_view, points_rgb_view

def save_aggregated_pointcloud_o3d(output_path, all_xyz_points, all_rgb_points, voxel_size):
    """Saves an aggregated point cloud (from rendered views) to a PLY file using Open3D.

    Optionally performs voxel downsampling if voxel_size > 0.

    Args:
        output_path (str): Path to save the .ply file.
        all_xyz_points (list[np.ndarray]): List of (N_i, 3) arrays of point coordinates from all views.
        all_rgb_points (list[np.ndarray]): List of (N_i, 3) arrays of RGB colors (0-255) from all views.
        voxel_size (float): Voxel size for Open3D's voxel_down_sample. If <= 0, no downsampling.
    """
    if not all_xyz_points:
        print("Point cloud saving was enabled, but no valid points were collected.")
        return

    aggregated_points_xyz = np.concatenate(all_xyz_points, axis=0)
    aggregated_points_rgb_uint8 = np.concatenate(all_rgb_points, axis=0)
    
    # Convert RGB from 0-255 (uint8) to 0-1 (float64) for Open3D
    aggregated_points_rgb_float = aggregated_points_rgb_uint8.astype(np.float64) / 255.0

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(aggregated_points_xyz)
    pcd.colors = o3d.utility.Vector3dVector(aggregated_points_rgb_float)

    if voxel_size > 0.0:
        print(f"Original point count before downsampling: {len(pcd.points)}")
        pcd = pcd.voxel_down_sample(voxel_size=voxel_size)
        print(f"Point count after downsampling with voxel size {voxel_size}: {len(pcd.points)}")
    
    o3d.io.write_point_cloud(output_path, pcd, write_ascii=False)
    print(f"Saved point cloud with {len(pcd.points)} points to {output_path} using Open3D")

def save_splat_params_as_ply(raw_splats_dict, output_path, output_type):
    """Saves Gaussian splat parameters to a PLY file.

    Behavior depends on output_type:
    - "full_properties": Saves all splat parameters using gsplat.exporter.splat2ply_bytes.
    - "centers_colored_sh0": Saves splat centers as a point cloud with RGB colors derived from sh0.

    Args:
        raw_splats_dict (dict): Dictionary containing raw splat parameters.
        output_path (str): Path to save the .ply file.
        output_type (str): The type of output ("full_properties" or "centers_colored_sh0").
    """
    if output_type == "centers_colored_sh0":
        print(f"Exporting splat centers with SH-derived RGB to PLY: {output_path}")
        try:
            means_raw = raw_splats_dict["means"].cpu()
            sh0_raw = raw_splats_dict["sh0"].cpu()  # Expects (N, 1, 3) or (N, 3)

            if sh0_raw.ndim == 3 and sh0_raw.shape[1] == 1:
                sh0_raw = sh0_raw.squeeze(1)  # Ensure sh0 is (N, 3)
            
            if not (sh0_raw.ndim == 2 and sh0_raw.shape[1] == 3):
                raise ValueError(f"sh0 for color conversion must be (N,3), but got shape {sh0_raw.shape}")
            if not (means_raw.ndim == 2 and means_raw.shape[1] == 3):
                raise ValueError(f"Means for export must be (N,3), but got shape {means_raw.shape}")

            # Convert sh0 to RGB
            SH_C0 = 0.28209479177387814  # Constant for SH to RGB conversion
            colors_rgb_float_tensor = SH_C0 * sh0_raw + 0.5
            colors_rgb_float_tensor = torch.clamp(colors_rgb_float_tensor, 0.0, 1.0)
            
            points_xyz_np = means_raw.numpy()
            points_rgb_np_float = colors_rgb_float_tensor.numpy()

            pcd = o3d.geometry.PointCloud()
            pcd.points = o3d.utility.Vector3dVector(points_xyz_np)
            pcd.colors = o3d.utility.Vector3dVector(points_rgb_np_float)

            o3d.io.write_point_cloud(output_path, pcd, write_ascii=False)
            print(f"Successfully saved point cloud with {len(pcd.points)} splat centers (colored by sh0) to {output_path}")

        except Exception as e:
            print(f"Error saving splat centers as colored PLY: {e}")
            raise
    elif output_type == "full_properties":
        print(f"Exporting full splat parameters to PLY: {output_path}")
        try:
            means_raw = raw_splats_dict["means"]
            scales_raw = raw_splats_dict["scales"] 
            quats_raw = raw_splats_dict["quats"]
            opacities_raw = raw_splats_dict["opacities"]
            sh0_raw = raw_splats_dict["sh0"]
            shN_raw = raw_splats_dict["shN"]

            opacities_export = opacities_raw.squeeze(1) if opacities_raw.ndim == 2 and opacities_raw.shape[1] == 1 else opacities_raw
            if opacities_export.ndim != 1:
                raise ValueError(f"Opacities for export must be 1D (N,), but got shape {opacities_export.shape} from raw shape {opacities_raw.shape}")

            sh0_export = sh0_raw.squeeze(1) if sh0_raw.ndim == 3 and sh0_raw.shape[1] == 1 else sh0_raw
            if not (sh0_export.ndim == 2 and sh0_export.shape[1] == 3):
                 raise ValueError(f"sh0 for export must be 2D (N,3), but got shape {sh0_export.shape} from raw shape {sh0_raw.shape}")

            shN_export = shN_raw.reshape(shN_raw.shape[0], -1) if shN_raw.ndim == 3 else shN_raw
            if shN_export.ndim != 2:
                if not (shN_export.ndim == 2 and shN_export.shape[1] == 0 and sh0_export.shape[0] == shN_export.shape[0]):
                    raise ValueError(f"shN for export must be 2D (N,C), but got shape {shN_export.shape} from raw shape {shN_raw.shape}")
            
            if not (means_raw.ndim == 2 and means_raw.shape[1] == 3):
                raise ValueError(f"Means for export must be 2D (N,3), but got shape {means_raw.shape}")
            if not (scales_raw.ndim == 2 and scales_raw.shape[1] == 3):
                raise ValueError(f"Scales for export must be 2D (N,3), but got shape {scales_raw.shape}")
            if not (quats_raw.ndim == 2 and quats_raw.shape[1] == 4):
                raise ValueError(f"Quats for export must be 2D (N,4), but got shape {quats_raw.shape}")

            ply_bytes = splat2ply_bytes(
                means=means_raw,
                scales=scales_raw,
                quats=quats_raw,
                opacities=opacities_export,
                sh0=sh0_export,
                shN=shN_export
            )
            with open(output_path, "wb") as f:
                f.write(ply_bytes)
            print(f"Successfully saved full splat parameters PLY file with {means_raw.shape[0]} splats to {output_path}")
        except Exception as e:
            print(f"Error saving full splat parameters PLY: {e}")
            raise
    else:
        raise ValueError(f"Unknown splat_params_output_type: {output_type}")


def main():
    """Main function to orchestrate loading, rendering, and saving."""
    args, parser = parse_arguments() # Unpack both args and parser

    if args.save_robot_data:
        print(f"Robot data export mode enabled. Output directory: {args.robot_output_dir}")
        os.makedirs(args.robot_output_dir, exist_ok=True)
        
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {device}")
        try:
            colmap_parser = load_colmap_data(args)
            splat_data_for_rendering, _ = load_splat_checkpoint(args.ckpt_path, device) # Raw splats not needed for this export
        except Exception as e:
            print(f"Initialization failed for robot data export: {e}")
            sys.exit(1)

        robot_pose_all_data = {} # For all valid poses
        robot_pose_data_filtered = {} # For poses from base_names ending in "-0"
        skipped_renderings_base_names = [] # List to track base_names for skipped renderings

        print(f"Starting rendering for robot data export to {args.robot_output_dir}...")
        with torch.no_grad():
            for idx in range(len(colmap_parser.image_names)):
                image_filename = os.path.basename(colmap_parser.image_names[idx])
                base_name, _ = os.path.splitext(image_filename)
                base_name = base_name.replace("color_", "") # e.g., "waypointID-0", "waypointID-1"

                print(f"Processing for robot export {idx+1}/{len(colmap_parser.image_names)}: {image_filename} (base_name: {base_name})")

                c2w_np = colmap_parser.camtoworlds[idx]
                camera_id = colmap_parser.camera_ids[idx]
                K_np = colmap_parser.Ks_dict[camera_id]

                if args.use_render_width:
                    render_width = args.render_width
                    render_height = args.render_height
                else:
                    render_width, render_height = colmap_parser.imsize_dict[camera_id]

                c2w_tensor = torch.from_numpy(c2w_np).float().to(device)
                K_tensor = torch.from_numpy(K_np).float().to(device)

                current_render_mode = "RGB+D"
                render_colors_batched, render_alphas_batched, _ = render_single_view(
                    splat_data_for_rendering, c2w_tensor, K_tensor, render_width, render_height,
                    current_render_mode, device
                )
                rgb_output, depth_output, _ = process_rendered_output(
                    render_colors_batched, render_alphas_batched, current_render_mode
                )

                if rgb_output is None or depth_output is None:
                    print(f"Warning: RGB or Depth output is None for {image_filename} (base_name: {base_name}). Skipping image/depth saving and pose data for this view.")
                    skipped_renderings_base_names.append(base_name)
                    continue # Skip to the next iteration

                # Save color image
                color_filename_out = f"color_{base_name}.jpg"
                color_path = os.path.join(args.robot_output_dir, color_filename_out)
                img_np = (rgb_output.cpu().numpy() * 255).astype(np.uint8)
                imageio.imsave(color_path, img_np)

                # Save depth data (raw NumPy array)
                depth_filename_out = f"depth_{base_name}.pkl"
                depth_path = os.path.join(args.robot_output_dir, depth_filename_out)
                depth_np = depth_output.cpu().numpy() # Shape (H, W)

                if idx == 0: # Print for the first frame only to give a sample
                    print(f"  Sample of raw depth data for {image_filename} (robot data export):")
                    print(f"    Depth map shape: {depth_np.shape}")
                    print(f"    Min depth value: {np.min(depth_np):.4f}")
                    print(f"    Max depth value: {np.max(depth_np):.4f}")
                    h, w = depth_np.shape
                    if h > 0 and w > 0: 
                        print(f"    Depth at center ({h//2}, {w//2}): {depth_np[h//2, w//2]:.4f}")
                        print(f"    Depth at (0, 0): {depth_np[0, 0]:.4f}")
                        if h > 10 and w > 10:
                             print(f"    Depth at (10, 10): {depth_np[10, 10]:.4f}")
                        print(f"    Depth at ({h-1}, {w-1}): {depth_np[h-1, w-1]:.4f}")
                        print(f"    Depth patch (top-left 3x3):\n{depth_np[:3, :3]}")
                    else:
                        print("    Depth map is empty.")

                with open(depth_path, "wb") as f:
                    pickle.dump(depth_np, f)
                
                position = c2w_np[:3, 3].tolist() 
                rotation_matrix = c2w_np[:3, :3]
                quaternion_wxyz = rotation_matrix_to_quaternion_wxyz(rotation_matrix)
                
                current_pose_data = {
                    "position": position,
                    "quaternion(wxyz)": quaternion_wxyz,
                    "K": K_np.tolist(),  # Add intrinsic matrix
                    "camtoworld": c2w_np.tolist()  # Add extrinsic matrix
                }
                
                robot_pose_all_data[base_name] = current_pose_data
                
                if base_name.endswith("-0"):
                    filtered_base_name = base_name[:-2] 
                    robot_pose_data_filtered[filtered_base_name] = current_pose_data

            # After the loop, print skipped renderings and verify
            if skipped_renderings_base_names:
                print("\n--- Skipped Renderings Report ---")
                print(f"The following {len(skipped_renderings_base_names)} base_names had None for RGB or Depth and were skipped:")
                for bn in skipped_renderings_base_names:
                    print(f"  - {bn}")
                
                print("\n--- Verifying Pose Data Integrity ---")
                found_skipped_in_all_data = False
                for bn in skipped_renderings_base_names:
                    if bn in robot_pose_all_data:
                        print(f"  ERROR: Skipped base_name '{bn}' FOUND in robot_pose_all_data. This should not happen!")
                        found_skipped_in_all_data = True
                if not found_skipped_in_all_data:
                    print("  Verification PASSED: No skipped base_names found in robot_pose_all_data.")

                found_skipped_in_filtered_data = False
                for bn in skipped_renderings_base_names:
                    if bn.endswith("-0"):
                        filtered_bn = bn[:-2]
                        if filtered_bn in robot_pose_data_filtered:
                            print(f"  ERROR: Filtered skipped base_name '{filtered_bn}' (from '{bn}') FOUND in robot_pose_data_filtered. This should not happen!")
                            found_skipped_in_filtered_data = True
                if not found_skipped_in_filtered_data:
                    print("  Verification PASSED: No filtered skipped base_names found in robot_pose_data_filtered.")
                print("--- End of Verification ---")
            else:
                print("\nAll views processed successfully without any skipped renderings due to None RGB/Depth.")

            # Print a sample of the pose data before saving
            if robot_pose_all_data:
                sample_key_all = next(iter(robot_pose_all_data))
                print(f"Sample of robot_pose_all_data (key: {sample_key_all}):")
                print(f"  Position: {robot_pose_all_data[sample_key_all]['position']}")
                print(f"  Quaternion(wxyz): {robot_pose_all_data[sample_key_all]['quaternion(wxyz)']}")
                print(f"  K (intrinsics): {robot_pose_all_data[sample_key_all]['K']}")
                print(f"  camtoworld (extrinsics): {robot_pose_all_data[sample_key_all]['camtoworld']}")
            else:
                print("robot_pose_all_data is empty after processing all views.")
            
            if robot_pose_data_filtered:
                sample_key_filtered = next(iter(robot_pose_data_filtered))
                print(f"Sample of robot_pose_data_filtered (key: {sample_key_filtered}):")
                print(f"  Position: {robot_pose_data_filtered[sample_key_filtered]['position']}")
                print(f"  Quaternion(wxyz): {robot_pose_data_filtered[sample_key_filtered]['quaternion(wxyz)']}")
                print(f"  K (intrinsics): {robot_pose_data_filtered[sample_key_filtered]['K']}")
                print(f"  camtoworld (extrinsics): {robot_pose_data_filtered[sample_key_filtered]['camtoworld']}")
            else:
                print("robot_pose_data_filtered is empty (no valid poses ending in '-0' found or all such poses had issues).")

            # Save the aggregated pose data
            pose_all_pkl_path = os.path.join(args.robot_output_dir, "pose_all_data.pkl")
            with open(pose_all_pkl_path, "wb") as f:
                pickle.dump(robot_pose_all_data, f)
            print(f"Saved aggregated robot pose data (all valid views) to {pose_all_pkl_path}")

            pose_filtered_pkl_path = os.path.join(args.robot_output_dir, "pose_data.pkl") # Original name for the filtered data
            with open(pose_filtered_pkl_path, "wb") as f:
                pickle.dump(robot_pose_data_filtered, f)
            print(f"Saved aggregated robot pose data (filtered for '-0' waypoints) to {pose_filtered_pkl_path}")
            
            # --- Final Sanity Check ---
            print("\n--- Final Sanity Check: Verifying Poses Against Saved Files ---")
            try:
                with open(pose_all_pkl_path, "rb") as f:
                    loaded_pose_all_data = pickle.load(f)
                
                if not loaded_pose_all_data:
                    print("  Warning: pose_all_data.pkl is empty. Sanity check cannot be performed.")
                else:
                    print(f"  Loaded {len(loaded_pose_all_data)} poses from {pose_all_pkl_path} for verification.")
                    all_files_verified_successfully = True
                    for base_name_key in loaded_pose_all_data.keys():
                        color_file_to_check = f"color_{base_name_key}.jpg"
                        depth_file_to_check = f"depth_{base_name_key}.pkl"
                        
                        color_path_to_check = os.path.join(args.robot_output_dir, color_file_to_check)
                        depth_path_to_check = os.path.join(args.robot_output_dir, depth_file_to_check)
                        
                        color_loaded = False
                        depth_loaded = False
                        
                        try:
                            imageio.imread(color_path_to_check) # Try to load the image
                            color_loaded = True
                        except FileNotFoundError:
                            print(f"  ERROR: Color file NOT FOUND for base_name '{base_name_key}': {color_path_to_check}")
                            all_files_verified_successfully = False
                        except Exception as e:
                            print(f"  ERROR: Failed to load color file for base_name '{base_name_key}' ({color_path_to_check}): {e}")
                            all_files_verified_successfully = False

                        try:
                            with open(depth_path_to_check, "rb") as df:
                                pickle.load(df) # Try to load the depth pkl
                            depth_loaded = True
                        except FileNotFoundError:
                            print(f"  ERROR: Depth file NOT FOUND for base_name '{base_name_key}': {depth_path_to_check}")
                            all_files_verified_successfully = False
                        except Exception as e:
                            print(f"  ERROR: Failed to load depth file for base_name '{base_name_key}' ({depth_path_to_check}): {e}")
                            all_files_verified_successfully = False
                        
                        if color_loaded and depth_loaded:
                            print(f"  SUCCESS: Verified color and depth files for base_name '{base_name_key}'.")
                            
                    if all_files_verified_successfully:
                        print("  Sanity Check PASSED: All poses in pose_all_data.pkl have corresponding color and depth files.")
                    else:
                        print("  Sanity Check FAILED: Some poses in pose_all_data.pkl are missing corresponding color or depth files. See errors above.")

            except FileNotFoundError:
                print(f"  ERROR: Sanity check could not be performed. pose_all_data.pkl not found at {pose_all_pkl_path}")
            except Exception as e:
                print(f"  ERROR: An unexpected error occurred during sanity check: {e}")
            print("--- End of Final Sanity Check ---")

            print("Robot data export complete. Exiting.")
            sys.exit(0)

    # Mode 3: Original rendering logic (if not robot data saving)
    else:
        # Validate arguments related to point cloud saving
        if args.save_pointcloud and "D" not in args.render_mode.upper():
            print("Warning: --save_pointcloud is set, but render_mode does not include depth ('D'). Point cloud generation will be skipped.")
            args.save_pointcloud = False

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {device}")

        # Create output directory if it doesn't exist
        os.makedirs(args.output_dir, exist_ok=True)

        # Setup for saving robot data if the flag is set.
        robot_pose_data_dict = {}
        if args.save_robot_data:
            if not args.robot_output_dir:
                print("Error: --save_robot_data is set, but --robot_output_dir is not specified.")
                sys.exit(1)
            os.makedirs(args.robot_output_dir, exist_ok=True)
            print(f"Robot data saving is enabled. Output will be in: {args.robot_output_dir}")
            if "D" not in args.render_mode.upper():
                print("Warning: --save_robot_data is set, but render_mode does not include depth ('D'). Robot depth data will not be saved.")

        # Load data
        try:
            colmap_parser = load_colmap_data(args)
            splat_data_for_rendering, raw_splats_dict = load_splat_checkpoint(args.ckpt_path, device)
        except Exception as e:
            print(f"Initialization failed: {e}")
            return

        all_collected_points_xyz = []
        all_collected_points_rgb = []

        print(f"Starting rendering for {len(colmap_parser.image_names)} poses...")
        with torch.no_grad(): 
            for idx in range(len(colmap_parser.image_names)):
                image_filename = os.path.basename(colmap_parser.image_names[idx])
                base_name_from_colmap, _ = os.path.splitext(image_filename) # Renamed to avoid conflict
                base_name_from_colmap = base_name_from_colmap.replace("color_", "")
                print(f"base_name_from_colmap: {base_name_from_colmap}")
                print(f"Rendering pose {idx+1}/{len(colmap_parser.image_names)}: {image_filename}")
                
                c2w_np = colmap_parser.camtoworlds[idx]
                camera_id = colmap_parser.camera_ids[idx]
                K_np = colmap_parser.Ks_dict[camera_id]
                
                if args.use_render_width:
                    print(f"Using specified render dimensions: {args.render_width}x{args.render_height}")
                    render_width = args.render_width
                    render_height = args.render_height
                else:
                    print(f"Using COLMAP image dimensions for camera ID {camera_id}: {colmap_parser.imsize_dict[camera_id]}")
                    render_width, render_height = colmap_parser.imsize_dict[camera_id]


                c2w_tensor = torch.from_numpy(c2w_np).float().to(device)
                K_tensor = torch.from_numpy(K_np).float().to(device)
                
                render_colors_batched, render_alphas_batched, meta = render_single_view(
                    splat_data_for_rendering, c2w_tensor, K_tensor, render_width, render_height, 
                    args.render_mode, device
                )

                rgb_output, depth_output, alphas_output = process_rendered_output(
                    render_colors_batched, render_alphas_batched, args.render_mode
                )
                
                save_image_outputs(args.output_dir, base_name_from_colmap, rgb_output, depth_output)

                if args.save_robot_data:
                    unique_robot_id = f"{base_name_from_colmap}-{idx}"

                    # Save color image for robot data
                    if rgb_output is not None:
                        color_filename = f"color_{unique_robot_id}.jpg"
                        color_filepath = os.path.join(args.robot_output_dir, color_filename)
                        img_np = (rgb_output.cpu().numpy() * 255).astype(np.uint8)
                        imageio.imsave(color_filepath, img_np)
                        # print(f"Saved robot color image to {color_filepath}") # Optional: too verbose for many files
                    else:
                        print(f"Warning: RGB output is None for {unique_robot_id}, cannot save robot color image.")

                    # Save depth data for robot data
                    if depth_output is not None:
                        depth_filename = f"depth_{unique_robot_id}" # No extension, as per spec
                        depth_filepath = os.path.join(args.robot_output_dir, depth_filename)
                        depth_np = depth_output.cpu().numpy()
                        with open(depth_filepath, "wb") as f:
                            pickle.dump(depth_np, f)
                        # print(f"Saved robot depth data to {depth_filepath}") # Optional: too verbose
                    elif "D" in args.render_mode.upper(): # Only warn if depth was expected
                        print(f"Warning: Depth output is None for {unique_robot_id}, cannot save robot depth data.")

                    # Collect pose data
                    # c2w_np is the 4x4 camera-to-world matrix from COLMAP
                    position_list = c2w_np[:3, 3].tolist()
                    rotation_matrix_np = c2w_np[:3, :3]
                    quaternion_wxyz_list = rotation_matrix_to_quaternion_wxyz(rotation_matrix_np)
                    
                    robot_pose_data_dict[unique_robot_id] = {
                        'position': position_list,
                        'quaternion(wxyz)': quaternion_wxyz_list,
                        'K': K_np.tolist(),  # Add intrinsic matrix
                        'camtoworld': c2w_np.tolist()  # Add extrinsic matrix
                    }

                if args.save_pointcloud and args.pointcloud_export_method == "rendered_views" and depth_output is not None:
                    points_xyz_view, points_rgb_view = extract_points_for_cloud(
                        depth_output, rgb_output, alphas_output, c2w_tensor, K_tensor, args
                    )
                    if points_xyz_view is not None:
                        all_collected_points_xyz.append(points_xyz_view)
                        all_collected_points_rgb.append(points_rgb_view)
                        print(f"Collected {points_xyz_view.shape[0]} points from view {image_filename}")
                
                #stop at 10 images for testing purposes
                # if idx >= 1:
                #     print("Stopping after 10 images for testing purposes.")
                #     break
        
        print("All poses rendered.")

        if args.save_robot_data:
            pose_pkl_path = os.path.join(args.robot_output_dir, "pose_data.pkl")
            with open(pose_pkl_path, "wb") as f:
                pickle.dump(robot_pose_data_dict, f)
            print(f"Saved robot pose data for {len(robot_pose_data_dict)} frames to {pose_pkl_path}")

        if args.save_pointcloud:
            base_output_filename, _ = os.path.splitext(args.pointcloud_filename) # Remove .ply if user added it
            
            if args.pointcloud_export_method == "splat_params":
                print(f"Point cloud export method: splat_params. Saving both full properties and centers colored by SH0.")
                # Warn if rendered_views specific arguments are set to non-defaults
                if args.pointcloud_alpha_threshold != parser.get_default("pointcloud_alpha_threshold"):
                     print(f"Warning: --pointcloud_alpha_threshold ({args.pointcloud_alpha_threshold}) is ignored when using 'splat_params' export method.")
                if args.pointcloud_max_depth != parser.get_default("pointcloud_max_depth"):
                     print(f"Warning: --pointcloud_max_depth ({args.pointcloud_max_depth}) is ignored when using 'splat_params' export method.")
                if args.pointcloud_voxel_size != parser.get_default("pointcloud_voxel_size") and args.pointcloud_voxel_size > 0.0:
                     print(f"Warning: --pointcloud_voxel_size ({args.pointcloud_voxel_size}) is ignored when using 'splat_params' export method.")
                
                # Save full properties
                filename_full_props = os.path.join(args.output_dir, f"{base_output_filename}_splat_params_full_properties.ply")
                try:
                    save_splat_params_as_ply(raw_splats_dict, filename_full_props, "full_properties")
                except Exception as e:
                    print(f"Failed to save splat parameters (full_properties) PLY: {e}")

                # Save centers colored by sh0
                filename_centers_sh0 = os.path.join(args.output_dir, f"{base_output_filename}_splat_params_centers_colored_sh0.ply")
                try:
                    save_splat_params_as_ply(raw_splats_dict, filename_centers_sh0, "centers_colored_sh0")
                except Exception as e:
                    print(f"Failed to save splat parameters (centers_colored_sh0) PLY: {e}")
            
            elif args.pointcloud_export_method == "rendered_views":
                pointcloud_output_path = os.path.join(args.output_dir, f"{base_output_filename}_rendered_views.ply")
                print(f"Point cloud export method: rendered_views. Output path: {pointcloud_output_path}")
                if not all_collected_points_xyz:
                    print("No points collected from rendered views. Skipping point cloud saving.")
                else:
                    save_aggregated_pointcloud_o3d(pointcloud_output_path, all_collected_points_xyz, all_collected_points_rgb, args.pointcloud_voxel_size)
            else:
                print(f"Unknown pointcloud_export_method: {args.pointcloud_export_method}")

if __name__ == "__main__":
    main()
