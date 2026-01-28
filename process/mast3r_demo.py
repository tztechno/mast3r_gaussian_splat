#!/usr/bin/env python3
# Copyright (C) 2024-present Naver Corporation. All rights reserved.
# Licensed under CC BY-NC-SA 4.0 (non-commercial use only).
#
# --------------------------------------------------------
# sparse gradio demo functions
# --------------------------------------------------------
import sys
import spaces

import math
import gradio
import os
import numpy as np
import functools
import trimesh
import copy
from scipy.spatial.transform import Rotation
import tempfile
import shutil
import typing

from mast3r.cloud_opt.sparse_ga import sparse_global_alignment
from mast3r.cloud_opt.tsdf_optimizer import TSDFPostProcess

from mast3r.model import AsymmetricMASt3R
from dust3r.image_pairs import make_pairs
from dust3r.utils.image import load_images
from dust3r.utils.device import to_numpy
from dust3r.viz import add_scene_cam, CAM_COLORS, OPENGL, pts3d_to_trimesh, cat_meshes
from dust3r.demo import get_args_parser as dust3r_get_args_parser
from copy import deepcopy

import matplotlib.pyplot as pl

import torch
import os.path as path
HERE_PATH = path.normpath(path.dirname(__file__))  # noqa

from demo_globals import CACHE_PATH, EXAMPLE_PATH, MODEL, DEVICE, SILENT, DATASET_DIR

class SparseGAState():
    def __init__(self, cache_dir=None, outfile_name=None):
        # self.sparse_ga = sparse_ga
        self.cache_dir = cache_dir
        self.outfile_name = outfile_name

    def __del__(self):
        if hasattr(self, 'cache_dir') and self.cache_dir is not None and os.path.isdir(self.cache_dir):
            shutil.rmtree(self.cache_dir)
        if hasattr(self, 'outfile_name') and self.outfile_name is not None and os.path.isfile(self.outfile_name):
            os.remove(self.outfile_name)


def get_args_parser():
    parser = dust3r_get_args_parser()
    parser.add_argument('--share', action='store_true')
    parser.add_argument('--gradio_delete_cache', default=None, type=int,
                        help='age/frequency at which gradio removes the file. If >0, matching cache is purged')

    actions = parser._actions
    for action in actions:
        if action.dest == 'model_name':
            action.choices = ["MASt3R_ViTLarge_BaseDecoder_512_catmlpdpt_metric"]
    # change defaults
    parser.prog = 'mast3r demo'
    return parser


def _convert_scene_output_to_glb(outfile, imgs, pts3d, mask, focals, cams2world, cam_size=0.05,
                                 cam_color=None, as_pointcloud=False,
                                 transparent_cams=False, silent=False):
    assert len(pts3d) == len(mask) <= len(imgs) <= len(cams2world) == len(focals)
    pts3d = to_numpy(pts3d)
    imgs = to_numpy(imgs)
    focals = to_numpy(focals)
    cams2world = to_numpy(cams2world)

    scene = trimesh.Scene()

    # full pointcloud
    if as_pointcloud:
        pts = np.concatenate([p[m.ravel()] for p, m in zip(pts3d, mask)]).reshape(-1, 3)
        col = np.concatenate([p[m] for p, m in zip(imgs, mask)]).reshape(-1, 3)
        valid_msk = np.isfinite(pts.sum(axis=1))
        pct = trimesh.PointCloud(pts[valid_msk], colors=col[valid_msk])
        scene.add_geometry(pct)
    else:
        meshes = []
        for i in range(len(imgs)):
            pts3d_i = pts3d[i].reshape(imgs[i].shape)
            msk_i = mask[i] & np.isfinite(pts3d_i.sum(axis=-1))
            meshes.append(pts3d_to_trimesh(imgs[i], pts3d_i, msk_i))
        mesh = trimesh.Trimesh(**cat_meshes(meshes))
        scene.add_geometry(mesh)

    # add each camera
    for i, pose_c2w in enumerate(cams2world):
        if isinstance(cam_color, list):
            camera_edge_color = cam_color[i]
        else:
            camera_edge_color = cam_color or CAM_COLORS[i % len(CAM_COLORS)]
        add_scene_cam(scene, pose_c2w, camera_edge_color,
                      None if transparent_cams else imgs[i], focals[i],
                      imsize=imgs[i].shape[1::-1], screen_width=cam_size)

    rot = np.eye(4)
    rot[:3, :3] = Rotation.from_euler('y', np.deg2rad(180)).as_matrix()
    scene.apply_transform(np.linalg.inv(cams2world[0] @ OPENGL @ rot))
    if not silent:
        print('(exporting 3D scene to', outfile, ')')
    scene.export(file_obj=outfile)
    return outfile


def get_3D_model_from_scene(scene, outfile, min_conf_thr=2, as_pointcloud=False, mask_sky=False,
                            clean_depth=False, transparent_cams=False, cam_size=0.05, TSDF_thresh=0):
    """
    extract 3D_model (glb file) from a reconstructed scene
    """

    # # get optimized values from scene
    # scene = scenescene_state.sparse_ga
    rgbimg = scene.imgs
    focals = scene.get_focals().cpu()
    cams2world = scene.get_im_poses().cpu()

    # 3D pointcloud from depthmap, poses and intrinsics
    if TSDF_thresh > 0:
        tsdf = TSDFPostProcess(scene, TSDF_thresh=TSDF_thresh)
        pts3d, _, confs = to_numpy(tsdf.get_dense_pts3d(clean_depth=clean_depth))
    else:
        pts3d, _, confs = to_numpy(scene.get_dense_pts3d(clean_depth=clean_depth))

    # torch.save(confs, '/app/data/confs.pt')
    msk = to_numpy([c > min_conf_thr for c in confs])
    return _convert_scene_output_to_glb(outfile, rgbimg, pts3d, msk, focals, cams2world, as_pointcloud=as_pointcloud,
                                        transparent_cams=transparent_cams, cam_size=cam_size, silent=SILENT)

def save_colmap_scene(scene, save_dir, min_conf_thr=2, clean_depth=False, mask_images=True):
    if 'save_pointcloud_with_normals' not in globals():
        sys.path.append(os.path.join(os.path.dirname(__file__), '../wild-gaussian-splatting/gaussian-splatting'))
        sys.path.append(os.path.join(os.path.dirname(__file__), '../wild-gaussian-splatting/src'))
        from colmap_dataset_utils import (
            inv,
            init_filestructure,
            save_images_masks,
            save_cameras,
            save_imagestxt,
            save_pointcloud,
            save_pointcloud_with_normals
        )

    cam2world = scene.get_im_poses().detach().cpu().numpy()
    world2cam = inv(cam2world) #
    principal_points = scene.get_principal_points().detach().cpu().numpy()
    focals = scene.get_focals().detach().cpu().numpy()[..., None]
    imgs = np.array(scene.imgs)

    pts3d, _, confs = scene.get_dense_pts3d(clean_depth=clean_depth)
    pts3d = [i.detach().reshape(imgs[0].shape) for i in pts3d] #

    masks = to_numpy([c > min_conf_thr for c in to_numpy(confs)])
    # move
    save_path, images_path, masks_path, sparse_path = init_filestructure(save_dir)
    save_images_masks(imgs, masks, images_path, masks_path, mask_images)
    save_cameras(focals, principal_points, sparse_path, imgs_shape=imgs.shape)
    save_imagestxt(world2cam, sparse_path)
    save_pointcloud_with_normals(imgs, pts3d, masks, sparse_path)
    return save_path

@spaces.GPU(duration=160)
def get_reconstructed_scene(snapshot,
                            min_conf_thr, matching_conf_thr,
                            as_pointcloud, cam_size, shared_intrinsics, clean_depth, filelist, example_name, req: gradio.Request, **kw):
    """
    from a list of images, run mast3r inference, sparse global aligner.
    then run get_3D_model_from_scene
    """

    if example_name != '':
        USER_DIR = os.path.join(CACHE_PATH, example_name)
    else:
        USER_DIR = os.path.join(CACHE_PATH, str(req.session_hash))
    os.makedirs(USER_DIR, exist_ok=True)
    
    image_size = 512
    imgs = load_images(filelist, size=image_size, verbose=not SILENT)
    if len(imgs) == 1:
        imgs = [imgs[0], copy.deepcopy(imgs[0])]
        imgs[1]['idx'] = 1
        filelist = [filelist[0], filelist[0] + '_2']

    lr1 = 0.07
    niter1 = 600
    lr2 = 0.014
    niter2 = 300
    optim_level = 'refine+depth'
    mask_sky, transparent_cams = False, False
    if len(filelist) < 13:
        scenegraph_type = 'complete'
        winsize = 1
    else:
        scenegraph_type = 'logwin'
        half_size = math.ceil((len(filelist) - 1) / 2)
        max_winsize = max(1, math.ceil(math.log(half_size, 2)))
        winsize = min(5, max_winsize)
    refid = 0
    win_cyclic = False
    TSDF_thresh = 0

    scene_graph_params = [scenegraph_type]
    if scenegraph_type in ["swin", "logwin"]:
        scene_graph_params.append(str(winsize))
    elif scenegraph_type == "oneref":
        scene_graph_params.append(str(refid))
    if scenegraph_type in ["swin", "logwin"] and not win_cyclic:
        scene_graph_params.append('noncyclic')
    scene_graph = '-'.join(scene_graph_params)
    pairs = make_pairs(imgs, scene_graph=scene_graph, prefilter=None, symmetrize=True)

    base_cache_dir = os.path.join(USER_DIR, 'cache')
    os.makedirs(base_cache_dir, exist_ok=True)
    def get_next_dir(base_dir):
        run_counter = 0
        while True:
            run_cache_dir = os.path.join(base_dir, f"run_{run_counter}")
            if not os.path.exists(run_cache_dir):
                os.makedirs(run_cache_dir)
                break
            run_counter += 1
        return run_cache_dir


    cache_dir = get_next_dir(base_cache_dir)
    scene = sparse_global_alignment(filelist, pairs, cache_dir,
                                    MODEL, lr1=lr1, niter1=niter1, lr2=lr2, niter2=niter2, device=DEVICE,
                                    opt_depth='depth' in optim_level, shared_intrinsics=shared_intrinsics,
                                    matching_conf_thr=matching_conf_thr, **kw)


    if example_name:
        colmap_data_dir = os.path.join(EXAMPLE_PATH, example_name)
    else:
        colmap_data_dir = get_next_dir(os.path.join(USER_DIR, DATASET_DIR))
    os.makedirs(colmap_data_dir, exist_ok=True)
    

    save_colmap_scene(scene, colmap_data_dir, min_conf_thr, clean_depth)

    outfile_name = os.path.join(USER_DIR, 'default_scene.glb')

    outfile = get_3D_model_from_scene(scene, outfile_name, min_conf_thr, as_pointcloud, mask_sky,
                                      clean_depth, transparent_cams, cam_size, TSDF_thresh)
    print(f"colmap_data_dir: {colmap_data_dir}")
    print(f"outfile_name: {outfile_name}")
    print(f"cache_dir: {cache_dir}")
    torch.cuda.empty_cache()
    return outfile


def mast3r_demo_tab():
    def get_context():
        css = """.gradio-container {margin: 0 !important; min-width: 100%};"""
        title = "MASt3R Demo"
        return gradio.Blocks(css=css, title=title, delete_cache=(True, True))

    with get_context() as demo:
        scene = gradio.State(None)
        
        # Title for the MASt3R demo
        gradio.HTML('<h2 style="text-align: center;">MASt3R Demo</h2>')
        
        gradio.HTML('''
        <div style="padding: 10px; border-radius: 5px; margin-bottom: 10px;">
            <h3>Instructions for MASt3R Demo</h3>
            <ul style="text-align: left; color: #333;">
                <li>Upload images. It is recommended to use no more than 7-10 images to avoid exceeding the 3-minute runtime limit for zeroGPU dynamic resources.</li>
                <li>Press the "Run" button to start the process.</li>
                <li>Once the stage is finished and the point cloud with cameras is visible below, switch to the 3DGS tab and follow the instructions there.</li>
            </ul>
        </div>
        ''')
        
        inputfiles = gradio.File(file_count="multiple")
        snapshot = gradio.Image(None, visible=False)
        run_btn = gradio.Button("Run")

        dummy_req = gradio.Request()
        dummy_text = gradio.Textbox(value="", visible=False)

        example_name = gradio.Textbox(value="", visible=False)


        with gradio.Row():
            matching_conf_thr = gradio.Slider(label="Matching Confidence Thr", value=2.,
                                minimum=0., maximum=30., step=0.1,
                                info="Before Fallback to Regr3D!")
            min_conf_thr = gradio.Slider(label="min_conf_thr", value=1.5, minimum=0.0, maximum=10, step=0.1)
            cam_size = gradio.Slider(label="cam_size", value=0.2, minimum=0.001, maximum=1.0, step=0.001)
        with gradio.Row():
            as_pointcloud = gradio.Checkbox(value=True, label="As pointcloud")
            shared_intrinsics = gradio.Checkbox(value=False, label="Shared intrinsics",
                                info="Only optimize one set of intrinsics for all views")
            clean_depth = gradio.Checkbox(value=False, label="Clean depth")
        
        outmodel = gradio.Model3D()
        run_btn.click(
            fn=get_reconstructed_scene,
            inputs=[snapshot, min_conf_thr, matching_conf_thr,
                    as_pointcloud, cam_size, shared_intrinsics, clean_depth, inputfiles, dummy_text],
            outputs=[outmodel]
        )

        tower_folder = os.path.join(HERE_PATH, '../wild-gaussian-splatting/mast3r/assets/NLE_tower/')
        turtle_folder = os.path.join(HERE_PATH, '../wild-gaussian-splatting/data/images/turtle_imgs/')
        puma_folder = os.path.join(HERE_PATH, '../wild-gaussian-splatting/data/images/puma_imgs/')
        tower_images = [os.path.join(tower_folder, file) for file in os.listdir(tower_folder) if file.endswith('.jpg') and not file.startswith('2679C386-1DC0-4443-81B5-93D7EDE4AB37-83120-000041DADB2EA917')] # my code not addpted to different size input
        turtle_images = [os.path.join(turtle_folder, file) for file in os.listdir(turtle_folder) if file.endswith('.jpg')]
        puma_images = [os.path.join(puma_folder, file) for file in os.listdir(puma_folder)[:12] if file.endswith('.jpg')]
        
        examples = gradio.Examples(
            examples=[
                [
                    puma_images[0],
                    1.5, 0.0, True, 0.2, True, False,
                    puma_images,
                    'puma',
                    None,
                ]
            ],
            inputs=[snapshot, min_conf_thr, matching_conf_thr, as_pointcloud, cam_size, shared_intrinsics, clean_depth, inputfiles, example_name],
            fn=get_reconstructed_scene,
            outputs=[outmodel],
            run_on_click=True,
            cache_examples='lazy',
        )
        examples = gradio.Examples(
            examples=[
                [
                    turtle_images[0],
                    1.5, 0.0, True, 0.2, True, False,
                    turtle_images,
                    'turtle',
                    None
                ]
            ],
            inputs=[snapshot, min_conf_thr, matching_conf_thr, as_pointcloud, cam_size, shared_intrinsics, clean_depth, inputfiles, example_name],
            fn=get_reconstructed_scene,
            outputs=[outmodel],
            run_on_click=True,
            cache_examples='lazy',
        )
        examples = gradio.Examples(
            examples=[
                [
                    tower_images[0],
                    1.5, 0.0, True, 0.2, True, False,
                    tower_images,
                    'tower',
                ]
            ],
            inputs=[snapshot, min_conf_thr, matching_conf_thr, as_pointcloud, cam_size, shared_intrinsics, clean_depth, inputfiles, example_name],
            fn=get_reconstructed_scene,
            outputs=[outmodel],
            run_on_click=True,
            cache_examples='lazy',
        )


    return demo
    
