"""

python render_trajectory_video.py \
    --ckpt_path results/spot_room/ckpts/ckpt_29999_rank0.pt \
    --colmap_data_dir data/spot_room \
    --trajectory_config_json_path traj.json \
    --output_video_path results/traj/output_video.mp4 \
    --render_width 640 \
    --render_height 480 \
    --fps 2

Script to render a video from a trajectory using a Gaussian Splatting model.

Prerequisites:
1. Install required Python packages:
   pip install torch torchvision torchaudio imageio imageio[ffmpeg] numpy opencv-python
   (Adjust torch installation according to your CUDA version if using GPU: https://pytorch.org/)
2. Ensure the gsplat project is structured correctly with __init__.py files in:
   - gsplat/
   - gsplat/splat_object_goal_nav/
   - gsplat/splat_object_goal_nav/datasets/
   This allows Python to recognize them as packages.
"""
import argparse
import torch # Requires torch to be installed
import os
import sys
import math
import numpy as np
import imageio # Requires imageio and imageio[ffmpeg] to be installed
import json # Added: For parsing JSON input
import torch.nn.functional as F # Added: For F.normalize
import cv2 # Added: For text overlay

from gsplat.rendering import rasterization
from datasets.colmap import Parser as ColmapParser
# If you have other utilities like depth_to_points in gsplat.utils:

def parse_arguments():
    parser = argparse.ArgumentParser(description="Render a video from a camera trajectory using a Gaussian Splatting checkpoint.")
    parser.add_argument("--ckpt_path", type=str, required=True, help="Path to the Gaussian Splatting checkpoint (.pt) file.")
    parser.add_argument("--colmap_data_dir", type=str, required=True, help="Path to the COLMAP data directory (for camera intrinsics).")
    # MODIFIED: Replaced individual trajectory arguments with a single JSON config path
    parser.add_argument('--trajectory_config_json_path', type=str, required=True, 
                        help='Path to a JSON file containing trajectory_points_2d, camera_z, look_at_point, and optionally up_vector.')
    
    parser.add_argument("--render_width", type=int, default=640, help="Width of the rendered video frames.")
    parser.add_argument("--render_height", type=int, default=480, help="Height of the rendered video frames.")
    parser.add_argument("--output_video_path", type=str, default="trajectory_render.mp4", help="Path to save the output MP4 video.")
    parser.add_argument("--fps", type=int, default=30, help="Frames per second for the output video.")

    parser.add_argument("--colmap_factor", type=int, default=1, help="Downsample factor for images when parsing COLMAP data (affects original image size for K scaling).")
    parser.add_argument("--colmap_normalize", type=bool, default=True, help="Normalize COLMAP poses and points when parsing (True/False).")
    
    return parser.parse_args()

def load_splat_checkpoint(ckpt_path, device):
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

        means_render = raw_splats_dict["means"].to(device)
        quats_render = F.normalize(raw_splats_dict["quats"], p=2, dim=-1).to(device)
        scales_render = torch.exp(raw_splats_dict["scales"]).to(device)
        opacities_render = torch.sigmoid(raw_splats_dict["opacities"]).to(device)
        
        sh0_render = raw_splats_dict["sh0"].to(device)
        shN_render = raw_splats_dict["shN"].to(device)

        if sh0_render.ndim == 2: # (N,3) -> (N,1,3)
            sh0_render = sh0_render.unsqueeze(1)
        
        if sh0_render.shape[1] != 1 or (sh0_render.ndim == 3 and sh0_render.shape[2] != 3):
             raise ValueError(f"sh0_render shape {sh0_render.shape} not compatible. Expected (N, 1, 3) or (N,3) that can be unsqueezed.")

        if shN_render.ndim == 2: # (N, K*3) -> (N, K, 3)
            if shN_render.shape[1] > 0 and shN_render.shape[1] % 3 != 0: # Allow (N,0) for degree 0
                raise ValueError(f"shN_render shape {shN_render.shape} not compatible for reshape to (N, K, 3)")
            if shN_render.shape[1] == 0: # (N,0) case for degree 0
                 shN_render = shN_render.reshape(shN_render.shape[0], 0, 3) # to (N,0,3)
            else:
                shN_render = shN_render.reshape(shN_render.shape[0], -1, 3)
        
        if shN_render.shape[0] != sh0_render.shape[0] or (shN_render.ndim ==3 and shN_render.shape[2] != 3):
            raise ValueError(f"shN_render shape {shN_render.shape} not compatible. Expected (N, K, 3) matching N and C of sh0.")

        colors_feat_render = torch.cat([sh0_render, shN_render], dim=1) 
        sh_degree_render = int(math.sqrt(colors_feat_render.shape[1]) - 1)
        
        if (sh_degree_render + 1)**2 != colors_feat_render.shape[1]:
             raise ValueError(f"Derived sh_degree {sh_degree_render} (coeffs: {(sh_degree_render + 1)**2}) does not match number of coefficients in colors_feat {colors_feat_render.shape[1]}")

        print(f"Loaded checkpoint with {means_render.shape[0]} Gaussians. SH degree: {sh_degree_render}")
        
        processed_splats = {
            "means": means_render, "quats": quats_render, "scales": scales_render,
            "opacities": opacities_render, "colors_feat": colors_feat_render, "sh_degree": sh_degree_render
        }
        return processed_splats
    except FileNotFoundError:
        print(f"Error: Checkpoint file not found at {ckpt_path}")
        raise
    except Exception as e:
        print(f"Error loading checkpoint: {e}")
        raise

def load_intrinsics_from_colmap(colmap_data_dir, colmap_factor, colmap_normalize, render_width, render_height, device):
    print(f"Loading COLMAP data from: {colmap_data_dir} to extract intrinsics.")
    try:
        colmap_parser = ColmapParser(
            data_dir=colmap_data_dir,
            factor=colmap_factor,
            normalize=colmap_normalize,
            test_every=1_000_000  # Load all camera models
        )
        if not colmap_parser.Ks_dict:
            raise ValueError("No intrinsics (Ks_dict) found in COLMAP parser.")
            
        first_camera_id_in_ks_dict = next(iter(colmap_parser.Ks_dict))
        K_colmap_np = colmap_parser.Ks_dict[first_camera_id_in_ks_dict]
        
        if first_camera_id_in_ks_dict not in colmap_parser.imsize_dict:
            raise ValueError(f"Original image dimensions for camera_id {first_camera_id_in_ks_dict} not found in imsize_dict.")
        orig_w, orig_h = colmap_parser.imsize_dict[first_camera_id_in_ks_dict]

        print(f"Using intrinsics from COLMAP camera_id {first_camera_id_in_ks_dict} with original dimensions {orig_w}x{orig_h}")

        K_render_np = K_colmap_np.copy()
        K_render_np[0, 0] *= (render_width / orig_w)   # fx
        K_render_np[1, 1] *= (render_height / orig_h)  # fy
        K_render_np[0, 2] *= (render_width / orig_w)   # cx
        K_render_np[1, 2] *= (render_height / orig_h)  # cy
        
        return torch.from_numpy(K_render_np).float().to(device)
    except Exception as e:
        print(f"Error loading intrinsics from COLMAP data: {e}")
        raise

def create_look_at_c2w(eye, target, up, device):
    eye_np = np.asarray(eye, dtype=np.float32)
    target_np = np.asarray(target, dtype=np.float32)
    up_np = np.asarray(up, dtype=np.float32)

    forward = target_np - eye_np
    forward_norm = np.linalg.norm(forward)
    if forward_norm < 1e-8: # Eye and target are too close
        print("Warning: Camera position and look_at_point are very close. Using default forward direction (0,0,-1).")
        forward = np.array([0.0,0.0,-1.0]) # Default: look along -Z world
    else:
        forward = forward / forward_norm

    right = np.cross(forward, up_np)
    right_norm = np.linalg.norm(right)
    if right_norm < 1e-8: # Forward and up are collinear
        print("Warning: Camera forward and up vectors are collinear. Adjusting up vector.")
        # Attempt to create a non-collinear up vector
        if abs(forward[1]) < 0.9: # If forward is not mostly vertical
            up_np = np.array([0.0, 1.0, 0.0])
        else: # If forward is mostly vertical
            up_np = np.array(1.0, 0.0, 0.0)
        right = np.cross(forward, up_np)
        right_norm = np.linalg.norm(right)


    right = right / (right_norm + 1e-8)
    
    new_up = np.cross(right, forward) # forward & right are unit & perp, so new_up is unit.

    c2w_np = np.eye(4, dtype=np.float32)
    c2w_np[:3, 0] = right
    c2w_np[:3, 1] = new_up 
    c2w_np[:3, 2] = -forward # Camera's Z axis points backward from view direction
    c2w_np[:3, 3] = eye_np
    
    return torch.from_numpy(c2w_np).to(device)

def render_rgb_frame(splat_data, c2w, K, width, height, device):
    viewmat = torch.linalg.inv(c2w).unsqueeze(0) 
    K_render = K.unsqueeze(0) 

    render_colors_batched, _, _ = rasterization(
        means=splat_data["means"],
        quats=splat_data["quats"],
        scales=splat_data["scales"],
        opacities=splat_data["opacities"],
        colors=splat_data["colors_feat"],
        viewmats=viewmat,
        Ks=K_render, # Expects 3x3 or 4x4, gsplat handles it
        width=width,
        height=height,
        sh_degree=splat_data["sh_degree"],
        render_mode="RGB",
        packed=False 
    )
    render_rgb_single = render_colors_batched[0].clamp(0, 1) # (H, W, 3)
    return render_rgb_single

def main():
    args = parse_arguments()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    try:
        with open(args.trajectory_config_json_path, 'r') as f:
            trajectory_config = json.load(f)
        
        trajectory_points_2d = trajectory_config["trajectory_points_2d"]
        camera_z = trajectory_config["camera_z"]
        look_at_point_1_config = trajectory_config["look_at_point_1"]
        up_vector = trajectory_config.get("up_vector", [0.0, 1.0, 0.0])

        calculate_look_at_point_2_opposite = trajectory_config.get("calculate_look_at_point_2_as_opposite", False)
        look_at_point_2_config = None
        if not calculate_look_at_point_2_opposite:
            look_at_point_2_config = trajectory_config["look_at_point_2"]

        if not isinstance(trajectory_points_2d, list) or \
           not all(isinstance(p, list) and len(p) == 2 for p in trajectory_points_2d):
            raise ValueError("'trajectory_points_2d' must be a list of [x,y] pairs.")
        if not isinstance(camera_z, (int, float)):
            raise ValueError("'camera_z' must be a number.")
        if not isinstance(look_at_point_1_config, list) or len(look_at_point_1_config) != 3:
            raise ValueError("'look_at_point_1' must be a list of 3 numbers [x,y,z].")
        if not calculate_look_at_point_2_opposite and (not isinstance(look_at_point_2_config, list) or len(look_at_point_2_config) != 3):
            raise ValueError("'look_at_point_2' must be a list of 3 numbers [x,y,z] when 'calculate_look_at_point_2_as_opposite' is false.")
        if not isinstance(up_vector, list) or len(up_vector) != 3:
            raise ValueError("'up_vector' must be a list of 3 numbers [x,y,z].")

    except FileNotFoundError:
        print(f"Error: Trajectory config JSON file not found at {args.trajectory_config_json_path}")
        sys.exit(1)
    except json.JSONDecodeError:
        print(f"Error: Invalid JSON format in {args.trajectory_config_json_path}.")
        sys.exit(1)
    except KeyError as e:
        print(f"Error: Missing required key {e} in trajectory_config_json. Ensure 'look_at_point_1' is present. If 'calculate_look_at_point_2_as_opposite' is false or absent, 'look_at_point_2' must also be present.")
        sys.exit(1)

    splat_data = load_splat_checkpoint(args.ckpt_path, device)
    
    # MODIFIED: Load intrinsics using the full specified render_width and render_height for each view.
    # The `args.render_width` now refers to the width of EACH individual side-by-side view.
    K_camera_full_res_per_view = load_intrinsics_from_colmap(
        args.colmap_data_dir, args.colmap_factor, args.colmap_normalize,
        args.render_width, args.render_height, device # Use full specified resolution
    )

    output_dir = os.path.dirname(args.output_video_path)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)
        print(f"Created output directory: {output_dir}")

    video_writer = imageio.get_writer(args.output_video_path, fps=args.fps)
    print(f"Rendering {len(trajectory_points_2d)} frames for video: {args.output_video_path}")

    # Determine text labels based on configuration
    text_view1 = "Simulated Front Camera"
    text_view2 = "Simulated Back Camera"
    if calculate_look_at_point_2_opposite:
        text_view1 = "Simulated Back Camera"  # View 1 uses look_at_point_1_config, which is the "back" reference
        text_view2 = "Simulated Front Camera" # View 2 uses the calculated opposite point

    with torch.no_grad():
        for i, (cam_x, cam_y) in enumerate(trajectory_points_2d):
            camera_position = [cam_x, cam_y, camera_z]
            
            current_target1 = look_at_point_1_config
            current_target2 = None

            if calculate_look_at_point_2_opposite:
                cam_pos_np = np.array(camera_position, dtype=np.float32)
                look_at1_np = np.array(look_at_point_1_config, dtype=np.float32)
                # Vector from camera to L1: L1 - P. Vector from P to L2 should be P - L1.
                # L2 = P + (P - L1) = 2*P - L1
                look_at2_np = 2 * cam_pos_np - look_at1_np
                current_target2 = look_at2_np.tolist()
            else:
                current_target2 = look_at_point_2_config

            print(f"  Frame {i+1}/{len(trajectory_points_2d)}: Cam: {camera_position}, Target1: {current_target1}, Target2: {current_target2}")

            c2w1 = create_look_at_c2w(camera_position, current_target1, up_vector, device)
            c2w2 = create_look_at_c2w(camera_position, current_target2, up_vector, device)
            
            rgb_frame_tensor1 = render_rgb_frame(
                splat_data, c2w1, K_camera_full_res_per_view, 
                args.render_width, args.render_height, device
            )
            rgb_frame_tensor2 = render_rgb_frame(
                splat_data, c2w2, K_camera_full_res_per_view, 
                args.render_width, args.render_height, device
            )

            # Convert tensors to NumPy arrays (HWC, RGB, 0-255)
            frame1_np_rgb = (rgb_frame_tensor1.cpu().numpy() * 255).astype(np.uint8)
            frame2_np_rgb = (rgb_frame_tensor2.cpu().numpy() * 255).astype(np.uint8)

            # --- Add text overlay using OpenCV ---
            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = 0.7
            font_color_bgr = (255, 255, 255) # White color in BGR for OpenCV
            thickness = 2
            line_type = cv2.LINE_AA

            org = (10, 30)

            annotated_frame1 = frame1_np_rgb.copy()
            cv2.putText(annotated_frame1, text_view1, org, font, font_scale, font_color_bgr, thickness, line_type)

            annotated_frame2 = frame2_np_rgb.copy()
            cv2.putText(annotated_frame2, text_view2, org, font, font_scale, font_color_bgr, thickness, line_type)
            # --- End text overlay ---
            
            # Concatenate annotated frames side-by-side
            # Ensure "Front Camera" is on the left, "Back Camera" is on the right
            if calculate_look_at_point_2_opposite:
                # In this case, text_view2 is "Front Camera" (on annotated_frame2)
                # and text_view1 is "Back Camera" (on annotated_frame1)
                # So, left should be annotated_frame2, right should be annotated_frame1
                combined_frame_annotated_np = np.concatenate((annotated_frame2, annotated_frame1), axis=1)
            else:
                # In this case, text_view1 is "Front Camera" (on annotated_frame1)
                # and text_view2 is "Back Camera" (on annotated_frame2)
                # So, left should be annotated_frame1, right should be annotated_frame2
                combined_frame_annotated_np = np.concatenate((annotated_frame1, annotated_frame2), axis=1)
            
            video_writer.append_data(combined_frame_annotated_np)

    video_writer.close()
    print(f"Successfully created video: {args.output_video_path}")

if __name__ == "__main__":
    main()
