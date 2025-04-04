import argparse
import os
from contextlib import nullcontext
from typing import Dict, List, Optional

import torch
from PIL import Image
from tqdm import tqdm
from transparent_background import Remover

from spar3d.models.mesh import QUAD_REMESH_AVAILABLE, TRIANGLE_REMESH_AVAILABLE
from spar3d.system import SPAR3D
from spar3d.utils import foreground_crop, get_device, remove_background


def check_positive(value):
    ivalue = int(value)
    if ivalue <= 0:
        raise argparse.ArgumentTypeError("%s is an invalid positive int value" % value)
    return ivalue


def process_multi_view_images(image_paths: Dict[str, str], bg_remover: Remover, foreground_ratio: float) -> Dict[str, Image.Image]:
    """Process multiple views of the same object.
    
    Args:
        image_paths: Dictionary mapping view names to image paths
        bg_remover: Background remover instance
        foreground_ratio: Ratio for foreground cropping
        
    Returns:
        Dictionary mapping view names to processed images
    """
    processed_images = {}
    for view_name, image_path in image_paths.items():
        image = remove_background(Image.open(image_path).convert("RGBA"), bg_remover)
        image = foreground_crop(image, foreground_ratio)
        processed_images[view_name] = image
    return processed_images


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    
    # Legacy mode arguments
    parser.add_argument(
        "--image", type=str, nargs="*", help="Path to input image(s) or folder (legacy mode)."
    )
    parser.add_argument(
        "--single-image", type=str, help="Path to single input image (legacy mode)."
    )
    
    # Multi-view arguments
    parser.add_argument(
        "--front", type=str, help="Path to front view image"
    )
    parser.add_argument(
        "--back", type=str, help="Path to back view image"
    )
    parser.add_argument(
        "--left", type=str, help="Path to left view image"
    )
    parser.add_argument(
        "--right", type=str, help="Path to right view image"
    )
    
    parser.add_argument(
        "--device",
        default=get_device(),
        type=str,
        help=f"Device to use. If no CUDA/MPS-compatible device is found, the baking will fail. Default: '{get_device()}'",
    )
    parser.add_argument(
        "--pretrained-model",
        default="stabilityai/stable-point-aware-3d",
        type=str,
        help="Path to the pretrained model. Could be either a huggingface model id is or a local path. Default: 'stabilityai/stable-point-aware-3d'",
    )
    parser.add_argument(
        "--foreground-ratio",
        default=1.3,
        type=float,
        help="Ratio of the foreground size to the image size. Only used when --no-remove-bg is not specified. Default: 0.85",
    )
    parser.add_argument(
        "--output-dir",
        default="output/",
        type=str,
        help="Output directory to save the results. Default: 'output/'",
    )
    parser.add_argument(
        "--texture-resolution",
        default=1024,
        type=int,
        help="Texture atlas resolution. Default: 1024",
    )
    parser.add_argument(
        "--low-vram-mode",
        action="store_true",
        help=(
            "Use low VRAM mode. SPAR3D consumes 10.5GB of VRAM by default. "
            "This mode will reduce the VRAM consumption to roughly 7GB but in exchange "
            "the model will be slower. Default: False"
        ),
    )

    remesh_choices = ["none"]
    if TRIANGLE_REMESH_AVAILABLE:
        remesh_choices.append("triangle")
    if QUAD_REMESH_AVAILABLE:
        remesh_choices.append("quad")
    parser.add_argument(
        "--remesh_option",
        choices=remesh_choices,
        default="none",
        help="Remeshing option",
    )
    if TRIANGLE_REMESH_AVAILABLE or QUAD_REMESH_AVAILABLE:
        parser.add_argument(
            "--reduction_count_type",
            choices=["keep", "vertex", "faces"],
            default="keep",
            help="Vertex count type",
        )
        parser.add_argument(
            "--target_count",
            type=check_positive,
            help="Selected target count.",
            default=2000,
        )
    parser.add_argument(
        "--batch_size", default=1, type=int, help="Batch size for inference"
    )
    args = parser.parse_args()

    # Ensure args.device contains cuda
    devices = ["cuda", "mps", "cpu"]
    if not any(args.device in device for device in devices):
        raise ValueError("Invalid device. Use cuda, mps or cpu")

    output_dir = args.output_dir
    os.makedirs(output_dir, exist_ok=True)

    device = args.device
    if not (torch.cuda.is_available() or torch.backends.mps.is_available()):
        device = "cpu"

    print("Device used: ", device)

    model = SPAR3D.from_pretrained(
        args.pretrained_model,
        config_name="config.yaml",
        weight_name="model.safetensors",
        low_vram_mode=args.low_vram_mode,
    )
    model.to(device)
    model.eval()

    bg_remover = Remover(device=device)
    
    # Calculate vertex count for remeshing
    vertex_count = (
        -1
        if args.reduction_count_type == "keep"
        else (
            args.target_count
            if args.reduction_count_type == "vertex"
            else args.target_count // 2
        )
    )
    
    # Check if we're using multi-view mode
    is_multi_view = any([args.front, args.back, args.left, args.right])
    
    if is_multi_view:
        # Multi-view mode
        image_paths = {}
        if args.front: image_paths["front"] = args.front
        if args.back: image_paths["back"] = args.back
        if args.left: image_paths["left"] = args.left
        if args.right: image_paths["right"] = args.right
        
        if not image_paths:
            raise ValueError("No input images provided. Use --single-image or specify views with --front, --back, etc.")
            
        processed_images = process_multi_view_images(image_paths, bg_remover, args.foreground_ratio)
        
        # Save processed images
        os.makedirs(output_dir, exist_ok=True)
        for view_name, image in processed_images.items():
            image.save(os.path.join(output_dir, f"{view_name}.png"))
            
        with torch.no_grad():
            with (
                torch.autocast(device_type=device, dtype=torch.bfloat16)
                if "cuda" in device
                else nullcontext()
            ):
                mesh, glob_dict = model.run_multi_view(
                    processed_images,
                    bake_resolution=args.texture_resolution,
                    remesh=args.remesh_option,
                    vertex_count=vertex_count,
                    return_points=True,
                )
        
        out_mesh_path = os.path.join(output_dir, "mesh.glb")
        mesh.export(out_mesh_path, include_normals=True)
        out_points_path = os.path.join(output_dir, "points.ply")
        glob_dict["point_clouds"][0].export(out_points_path)
        
    else:
        # Legacy single image mode
        images = []
        idx = 0
        
        # Handle single image or multiple images
        if args.single_image:
            image_paths = [args.single_image]
        elif args.image:
            image_paths = args.image
        else:
            raise ValueError("No input images provided. Use --single-image, --image, or specify views with --front, --back, etc.")
            
        for image_path in image_paths:
            def handle_image(image_path, idx):
                image = remove_background(Image.open(image_path).convert("RGBA"), bg_remover)
                image = foreground_crop(image, args.foreground_ratio)
                os.makedirs(os.path.join(output_dir, str(idx)), exist_ok=True)
                image.save(os.path.join(output_dir, str(idx), "input.png"))
                images.append(image)

            if os.path.isdir(image_path):
                image_paths_in_dir = [
                    os.path.join(image_path, f)
                    for f in os.listdir(image_path)
                    if f.endswith((".png", ".jpg", ".jpeg"))
                ]
                for image_path in image_paths_in_dir:
                    handle_image(image_path, idx)
                    idx += 1
            else:
                handle_image(image_path, idx)
                idx += 1

        for i in tqdm(range(0, len(images), args.batch_size)):
            image = images[i : i + args.batch_size]
            if torch.cuda.is_available():
                torch.cuda.reset_peak_memory_stats()
            with torch.no_grad():
                with (
                    torch.autocast(device_type=device, dtype=torch.bfloat16)
                    if "cuda" in device
                    else nullcontext()
                ):
                    mesh, glob_dict = model.run_image(
                        image,
                        bake_resolution=args.texture_resolution,
                        remesh=args.remesh_option,
                        vertex_count=vertex_count,
                        return_points=True,
                    )
            if torch.cuda.is_available():
                print("Peak Memory:", torch.cuda.max_memory_allocated() / 1024 / 1024, "MB")
            elif torch.backends.mps.is_available():
                print(
                    "Peak Memory:", torch.mps.driver_allocated_memory() / 1024 / 1024, "MB"
                )

            if len(image) == 1:
                out_mesh_path = os.path.join(output_dir, str(i), "mesh.glb")
                mesh.export(out_mesh_path, include_normals=True)
                out_points_path = os.path.join(output_dir, str(i), "points.ply")
                glob_dict["point_clouds"][0].export(out_points_path)
            else:
                for j in range(len(mesh)):
                    out_mesh_path = os.path.join(output_dir, str(i + j), "mesh.glb")
                    mesh[j].export(out_mesh_path, include_normals=True)
                    out_points_path = os.path.join(output_dir, str(i + j), "points.ply")
                    glob_dict["point_clouds"][j].export(out_points_path)
