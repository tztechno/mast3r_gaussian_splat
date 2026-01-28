import sys
import os
import torch
from random import randint
import uuid
from tqdm.auto import tqdm
import gradio as gr
import importlib.util
from dataclasses import dataclass, field
from demo_globals import DEVICE

import spaces
from simple_knn._C import distCUDA2


@dataclass
class PipelineParams:
    convert_SHs_python: bool = False
    compute_cov3D_python: bool = False
    debug: bool = False

@dataclass
class OptimizationParams:
    # DEFAULT PARAMETERS
    iterations: int = 30_000
    position_lr_init: float = 0.00016
    position_lr_final: float = 0.0000016
    position_lr_delay_mult: float = 0.01
    position_lr_max_steps: int = 30_000
    feature_lr: float = 0.0025
    opacity_lr: float = 0.05
    scaling_lr: float = 0.005
    rotation_lr: float = 0.001
    percent_dense: float = 0.01
    lambda_dssim: float = 0.2
    densification_interval: int = 100
    opacity_reset_interval: int = 3000
    densify_from_iter: int = 500
    densify_until_iter: int = 15_000
    densify_grad_threshold: float = 0.0002
    random_background: bool = False

@dataclass
class ModelParams:
    sh_degree: int = 3
    source_path: str = "../data/scenes/turtle/"  # Default path, adjust as needed
    model_path: str = ""
    images: str = "images"
    resolution: int = -1
    white_background: bool = True
    data_device: str = "cuda"
    eval: bool = False

@spaces.GPU(duration=160)
def train(
    data_source_path, iterations, position_lr_init, position_lr_final, position_lr_delay_mult,
    position_lr_max_steps, feature_lr, opacity_lr, scaling_lr, rotation_lr,
    percent_dense, lambda_dssim, densification_interval, opacity_reset_interval,
    densify_from_iter, densify_until_iter, densify_grad_threshold
):

    # Add the path to the gaussian-splatting repository
    if 'GaussianRasterizer' not in globals():
        gaussian_splatting_path = 'wild-gaussian-splatting/gaussian-splatting/'
        sys.path.append(gaussian_splatting_path)

        # Import necessary modules from the gaussian-splatting directory
        from utils.loss_utils import l1_loss, ssim
        from gaussian_renderer import render
        from scene import Scene, GaussianModel
        from utils.general_utils import safe_state
        from utils.image_utils import psnr
        from utils.graphics_utils import focal2fov, fov2focal, getProjectionMatrix

        # Dynamically import the train module from the gaussian-splatting directory
        train_spec = importlib.util.spec_from_file_location("gaussian_splatting_train", os.path.join(gaussian_splatting_path, "train.py"))
        gaussian_splatting_train = importlib.util.module_from_spec(train_spec)
        train_spec.loader.exec_module(gaussian_splatting_train)

        # Import the necessary functions from the dynamically loaded module
        prepare_output_and_logger = gaussian_splatting_train.prepare_output_and_logger
        training_report = gaussian_splatting_train.training_report

    print(data_source_path)
    # Create instances of the parameter dataclasses
    dataset = ModelParams(source_path=data_source_path,)
    
    pipe = PipelineParams()
    
    opt = OptimizationParams(
        iterations=iterations,
        position_lr_init=position_lr_init,
        position_lr_final=position_lr_final,
        position_lr_delay_mult=position_lr_delay_mult,
        position_lr_max_steps=position_lr_max_steps,
        feature_lr=feature_lr,
        opacity_lr=opacity_lr,
        scaling_lr=scaling_lr,
        rotation_lr=rotation_lr,
        percent_dense=percent_dense,
        lambda_dssim=lambda_dssim,
        densification_interval=densification_interval,
        opacity_reset_interval=opacity_reset_interval,
        densify_from_iter=densify_from_iter,
        densify_until_iter=densify_until_iter,
        densify_grad_threshold=densify_grad_threshold,
    )    
    gaussians = GaussianModel(dataset.sh_degree)
    scene = Scene(dataset, gaussians)
    gaussians.training_setup(opt)

    bg_color = [1, 1, 1] if dataset.white_background else [0, 0, 0]
    background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

    iter_start = torch.cuda.Event(enable_timing = True)
    iter_end = torch.cuda.Event(enable_timing = True)

    viewpoint_stack = None
    ema_loss_for_log = 0.0
    first_iter = 0
    progress_bar = tqdm(range(first_iter, opt.iterations), desc="Training progress")
    first_iter += 1

    point_cloud_path = ""
    progress = gr.Progress()  # Initialize the progress bar
    for iteration in range(first_iter, opt.iterations + 1):
        iter_start.record()
        gaussians.update_learning_rate(iteration)
        
        # Every 1000 its we increase the levels of SH up to a maximum degree
        if iteration % 1000 == 0:
            gaussians.oneupSHdegree()
            
        # Pick a random Camera
        if not viewpoint_stack:
            viewpoint_stack = scene.getTrainCameras().copy()
        viewpoint_cam = viewpoint_stack.pop(randint(0, len(viewpoint_stack)-1))
        
        bg = torch.rand((3), device=DEVICE) if opt.random_background else background
        
        render_pkg = render(viewpoint_cam, gaussians, pipe, bg)
        image, viewspace_point_tensor, visibility_filter, radii = render_pkg["render"], render_pkg["viewspace_points"], render_pkg["visibility_filter"], render_pkg["radii"]

        # Loss
        gt_image = viewpoint_cam.original_image.cuda()
        Ll1 = l1_loss(image, gt_image)
        loss = (1.0 - opt.lambda_dssim) * Ll1 + opt.lambda_dssim * (1.0 - ssim(image, gt_image))
        loss.backward()
        iter_end.record()
        
        with torch.no_grad():
            # Progress bar
            ema_loss_for_log = 0.4 * loss.item() + 0.6 * ema_loss_for_log
            if iteration % 10 == 0:
                progress_bar.set_postfix({"Loss": f"{ema_loss_for_log:.{7}f}"})
                progress_bar.update(10)
                progress(iteration / opt.iterations)  # Update Gradio progress bar
            if iteration == opt.iterations:
                progress_bar.close()

            # Log and save
            if (iteration == opt.iterations):
                point_cloud_path = os.path.join(os.path.join(data_source_path, "point_cloud/iteration_{}".format(iteration)), "point_cloud.ply")
                print("\n[ITER {}] Saving Gaussians to {}".format(iteration, point_cloud_path))
                scene.save(iteration)

            # Densification
            if iteration < opt.densify_until_iter:
                # Keep track of max radii in image-space for pruning
                gaussians.max_radii2D[visibility_filter] = torch.max(gaussians.max_radii2D[visibility_filter], radii[visibility_filter])
                gaussians.add_densification_stats(viewspace_point_tensor, visibility_filter)

                if iteration > opt.densify_from_iter and iteration % opt.densification_interval == 0:
                    size_threshold = 20 if iteration > opt.opacity_reset_interval else None
                    gaussians.densify_and_prune(opt.densify_grad_threshold, 0.005, scene.cameras_extent, size_threshold)

                if iteration % opt.opacity_reset_interval == 0 or (dataset.white_background and iteration == opt.densify_from_iter):
                    gaussians.reset_opacity()

                # Optimizer step
                if iteration < opt.iterations:
                    gaussians.optimizer.step()
                    gaussians.optimizer.zero_grad(set_to_none = True)

                # if (iteration == opt.iterations):
                #     print("\n[ITER {}] Saving Checkpoint".format(iteration))
                #     torch.save((gaussians.capture(), iteration), scene.model_path + "/chkpnt" + str(iteration) + ".pth")


    from os import makedirs
    import torchvision
    import subprocess

    @torch.no_grad()
    def render_path(dataset : ModelParams, iteration : int, pipeline : PipelineParams, render_resize_method='crop'):
        """
        render_resize_method: crop, pad
        """
        gaussians = GaussianModel(dataset.sh_degree)
        scene = Scene(dataset, gaussians, load_iteration=iteration, shuffle=False)

        iteration = scene.loaded_iter

        bg_color = [1,1,1] if dataset.white_background else [0, 0, 0]
        background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

        model_path = dataset.model_path
        name = "render"

        views = scene.getRenderCameras()

        # print(len(views))
        render_path = os.path.join(model_path, name, "ours_{}".format(iteration), "renders")

        makedirs(render_path, exist_ok=True)

        for idx, view in enumerate(tqdm(views, desc="Rendering progress")):
            if render_resize_method == 'crop':
                image_size = 256
            elif render_resize_method == 'pad':
                image_size = max(view.image_width, view.image_height)
            else:
                raise NotImplementedError
            view.original_image = torch.zeros((3, image_size, image_size), device=view.original_image.device)
            focal_length_x = fov2focal(view.FoVx, view.image_width)
            focal_length_y = fov2focal(view.FoVy, view.image_height)
            view.image_width = image_size
            view.image_height = image_size
            view.FoVx = focal2fov(focal_length_x, image_size)
            view.FoVy = focal2fov(focal_length_y, image_size)
            view.projection_matrix = getProjectionMatrix(znear=view.znear, zfar=view.zfar, fovX=view.FoVx, fovY=view.FoVy).transpose(0,1).cuda().float()
            view.full_proj_transform = (view.world_view_transform.unsqueeze(0).bmm(view.projection_matrix.unsqueeze(0))).squeeze(0)

            # print("background.device: ", background.device)
            # print("view.device: ", view.original_image.device)
            render_pkg = render(view, gaussians, pipeline, background)
            rendering = render_pkg["render"]
            torchvision.utils.save_image(rendering, os.path.join(render_path, '{0:05d}'.format(idx) + ".png"))

        # Use ffmpeg to output video
        renders_path = os.path.join(model_path, name, "ours_{}".format(iteration), "renders.mp4")
        # Use ffmpeg to output video
        subprocess.run(["ffmpeg", "-y", 
                    "-framerate", "24",
                    "-i", os.path.join(render_path, "%05d.png"), 
                    "-vf", "pad=ceil(iw/2)*2:ceil(ih/2)*2",
                    "-c:v", "libx264", 
                    "-pix_fmt", "yuv420p",
                    "-crf", "23", 
                    # "-pix_fmt", "yuv420p",  # Set pixel format for compatibility
                    renders_path], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL
                    )
        return renders_path
    
    renders_path = render_path(dataset, opt.iterations, pipe, render_resize_method='crop')
    torch.cuda.empty_cache()
    return renders_path, point_cloud_path