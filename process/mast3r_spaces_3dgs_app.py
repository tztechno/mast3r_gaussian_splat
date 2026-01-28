import sys
sys.path.append('wild-gaussian-splatting/mast3r/')
sys.path.append('demo/')

import os
import gradio as gr
import torch
from mast3r.demo import get_args_parser
from mast3r_demo import mast3r_demo_tab
from gs_demo import gs_demo_tab
from demo_globals import CACHE_PATH
import shutil

def start_session(req: gr.Request):
    user_dir = os.path.join(CACHE_PATH, str(req.session_hash))
    os.makedirs(user_dir, exist_ok=True)
    
def end_session(req: gr.Request):
    user_dir = os.path.join(CACHE_PATH, str(req.session_hash))
    shutil.rmtree(user_dir)

if __name__ == '__main__':
    with gr.Blocks() as demo:
        gr.HTML('''
        <div style="text-align: center; padding: 20px; color: #333;">
            <h2>MASt3R and 3DGS Pipeline Demo</h2>
            <p style="font-size: 16px;">This pipeline is designed for 3D reconstruction using MASt3R and 3DGS.</p>
            <p style="font-size: 16px;">The process is divided into two stages:</p>
            <ol style="text-align: left; display: inline-block; margin: 0 auto;">
                <li>MASt3R is used to obtain the initial point cloud and camera parameters.</li>
                <li>3DGS is then trained on the results from MASt3R to refine the 3D scene representation.</li>
            </ol>
            <p style="font-size: 16px; font-weight: bold;">Note: After a page reload, any generated MASt3R datasets in the 3DGS tab will be deleted.</p>
            <p style="font-size: 16px;">For a full version of this pipeline, please visit the repository at:</p>
            <a href="https://github.com/nerlfield/wild-gaussian-splatting" target="_blank" style="font-size: 16px; text-decoration: none;">nerlfield/wild-gaussian-splatting</a>
        </div>
        ''')

        with gr.Tabs():
            with gr.Tab("MASt3R"):
                mast3r_demo_tab()
            with gr.Tab("3DGS"):
                gs_demo_tab()

        demo.load(start_session)
        demo.unload(end_session)

        demo.launch(show_error=True, share=None, server_name=None, server_port=None)