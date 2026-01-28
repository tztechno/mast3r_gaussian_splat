# Copyright (C) 2024-present Naver Corporation. All rights reserved.
# Licensed under CC BY-NC-SA 4.0 (non-commercial use only).
#
# --------------------------------------------------------
# mast3r demo
# --------------------------------------------------------
import spaces
import os
import sys
import os.path as path
import torch
import tempfile
import gradio
import shutil
import math

HERE_PATH = path.normpath(path.dirname(__file__))  # noqa
MASt3R_REPO_PATH = path.normpath(path.join(HERE_PATH, './mast3r'))  # noqa
sys.path.insert(0, MASt3R_REPO_PATH)  # noqa

from mast3r.demo import get_reconstructed_scene
from mast3r.model import AsymmetricMASt3R
from mast3r.utils.misc import hash_md5

import mast3r.utils.path_to_dust3r  # noqa
from dust3r.demo import set_print_with_timestamp

import matplotlib.pyplot as pl
pl.ion()

# for gpu >= Ampere and pytorch >= 1.12
torch.backends.cuda.matmul.allow_tf32 = True
batch_size = 1
set_print_with_timestamp()

weights_path = "naver/MASt3R_ViTLarge_BaseDecoder_512_catmlpdpt_metric"
device = 'cuda' if torch.cuda.is_available() else 'cpu'
model = AsymmetricMASt3R.from_pretrained(weights_path).to(device)
chkpt_tag = hash_md5(weights_path)

tmpdirname = tempfile.mkdtemp(suffix='_mast3r_gradio_demo')
image_size = 512
silent = True
gradio_delete_cache = 7200


class FileState:
    def __init__(self, outfile_name=None):
        self.outfile_name = outfile_name

    def __del__(self):
        if self.outfile_name is not None and os.path.isfile(self.outfile_name):
            os.remove(self.outfile_name)
        self.outfile_name = None


@spaces.GPU(duration=180)
def local_get_reconstructed_scene(filelist, min_conf_thr, matching_conf_thr,
                                  as_pointcloud, cam_size,
                                  shared_intrinsics, **kw):
    lr1 = 0.07
    niter1 = 300
    lr2 = 0.01
    niter2 = 300
    optim_level = 'refine'
    mask_sky, clean_depth, transparent_cams = False, True, False
    if len(filelist) < 5:
        scenegraph_type = 'complete'
        winsize = 1
    else:
        scenegraph_type = 'logwin'
        half_size = math.ceil((len(filelist) - 1) / 2)
        max_winsize = max(1, math.ceil(math.log(half_size, 2)))
        winsize = min(5, max_winsize)
    refid = 0
    win_cyclic = False
    scene_state, outfile = get_reconstructed_scene(tmpdirname, gradio_delete_cache, model, None, device, silent, image_size, None,
                                                   filelist, optim_level, lr1, niter1, lr2, niter2, min_conf_thr, matching_conf_thr,
                                                   as_pointcloud, mask_sky, clean_depth, transparent_cams, cam_size, scenegraph_type, winsize,
                                                   win_cyclic, refid, TSDF_thresh=0, shared_intrinsics=shared_intrinsics, **kw)
    filestate = FileState(scene_state.outfile_name)
    scene_state.outfile_name = None
    del scene_state
    return filestate, outfile


def run_example(snapshot, matching_conf_thr, min_conf_thr, cam_size, as_pointcloud, shared_intrinsics, filelist, **kw):
    return local_get_reconstructed_scene(filelist, min_conf_thr, matching_conf_thr, as_pointcloud, cam_size, shared_intrinsics, **kw)


css = """.gradio-container {margin: 0 !important; min-width: 100%};"""
title = "MASt3R Demo"
with gradio.Blocks(css=css, title=title, delete_cache=(gradio_delete_cache, gradio_delete_cache)) as demo:
    filestate = gradio.State(None)
    gradio.HTML(
        '<h2 style="text-align: center;">3D Reconstruction with MASt3R</h2>')
    gradio.HTML('<p>Upload one or multiple images (wait for them to be fully uploaded before hitting the run button). '
                'We tested with up to 18 images before running into the allocation timeout - set at 3 minutes but your mileage may vary. '
                'At the very bottom of this page, you will find an example. If you click on it, it will pull the 3D reconstruction from 7 images of the small Naver Labs Europe tower from cache. '
                'If you want to try larger image collections, you can find the more complete version of this demo that you can run locally '
                'and more details about the method at <a href="https://github.com/naver/mast3r">github.com/naver/mast3r</a>. '
                'The checkpoint used in this demo is available at <a href="https://huggingface.co/naver/MASt3R_ViTLarge_BaseDecoder_512_catmlpdpt_metric">huggingface.co/naver/MASt3R_ViTLarge_BaseDecoder_512_catmlpdpt_metric</a>.</p>')
    with gradio.Column():
        inputfiles = gradio.File(file_count="multiple")
        snapshot = gradio.Image(None, visible=False)
        with gradio.Row():
            matching_conf_thr = gradio.Slider(label="Matching Confidence Thr", value=0.,
                                              minimum=0., maximum=30., step=0.1,
                                              info="Before Fallback to Regr3D!")
            # adjust the confidence threshold
            min_conf_thr = gradio.Slider(
                label="min_conf_thr", value=1.5, minimum=0.0, maximum=10, step=0.1)
            # adjust the camera size in the output pointcloud
            cam_size = gradio.Slider(
                label="cam_size", value=0.2, minimum=0.001, maximum=1.0, step=0.001)
        with gradio.Row():
            as_pointcloud = gradio.Checkbox(value=True, label="As pointcloud")
            shared_intrinsics = gradio.Checkbox(value=False, label="Shared intrinsics",
                                                info="Only optimize one set of intrinsics for all views")
        run_btn = gradio.Button("Run")
        outmodel = gradio.Model3D()

        examples = gradio.Examples(
            examples=[
                [
                    os.path.join(
                        HERE_PATH, 'mast3r/assets/NLE_tower/FF5599FD-768B-431A-AB83-BDA5FB44CB9D-83120-000041DADDE35483.jpg'),
                    0.0, 1.5, 0.2, True, False,
                     [os.path.join(HERE_PATH, 'mast3r/assets/NLE_tower/01D90321-69C8-439F-B0B0-E87E7634741C-83120-000041DAE419D7AE.jpg'),
                      os.path.join(
                          HERE_PATH, 'mast3r/assets/NLE_tower/1AD85EF5-B651-4291-A5C0-7BDB7D966384-83120-000041DADF639E09.jpg'),
                      os.path.join(
                          HERE_PATH, 'mast3r/assets/NLE_tower/28EDBB63-B9F9-42FB-AC86-4852A33ED71B-83120-000041DAF22407A1.jpg'),
                      os.path.join(
                          HERE_PATH, 'mast3r/assets/NLE_tower/91E9B685-7A7D-42D7-B933-23A800EE4129-83120-000041DAE12C8176.jpg'),
                      os.path.join(
                          HERE_PATH, 'mast3r/assets/NLE_tower/2679C386-1DC0-4443-81B5-93D7EDE4AB37-83120-000041DADB2EA917.jpg'),
                      os.path.join(
                          HERE_PATH, 'mast3r/assets/NLE_tower/CDBBD885-54C3-4EB4-9181-226059A60EE0-83120-000041DAE0C3D612.jpg'),
                      os.path.join(HERE_PATH, 'mast3r/assets/NLE_tower/FF5599FD-768B-431A-AB83-BDA5FB44CB9D-83120-000041DADDE35483.jpg')]
                ]
            ],
            inputs=[snapshot, matching_conf_thr, min_conf_thr,
                    cam_size, as_pointcloud, shared_intrinsics, inputfiles],
            outputs=[filestate, outmodel],
            fn=run_example,
            cache_examples="lazy",
        )

        # events
        run_btn.click(fn=local_get_reconstructed_scene,
                      inputs=[inputfiles, min_conf_thr, matching_conf_thr,
                              as_pointcloud,
                              cam_size, shared_intrinsics],
                      outputs=[filestate, outmodel])

demo.launch(show_error=True)
shutil.rmtree(tmpdirname)
