import gradio as gr
from gs_train import train
import os

from demo_globals import CACHE_PATH, EXAMPLE_PATH, MODEL, DEVICE, SILENT, DATASET_DIR

def get_dataset_folders(datasets_path):

    folder = []
    if os.path.isdir(datasets_path):
        folder += [f for f in os.listdir(datasets_path) if os.path.isdir(os.path.join(datasets_path, f))]
    if os.path.isdir(EXAMPLE_PATH):
        folder += [f for f in os.listdir(EXAMPLE_PATH) if os.path.isdir(os.path.join(EXAMPLE_PATH, f))]
    return sorted(folder, key=lambda x: int(x.split('_')[-1]) if x.split('_')[-1].isdigit() else float('inf'))

def gs_demo_tab():
    # datasets_path = "/app/data/scenes/"

    def start_training(selected_folder, *args):
        selected_data_path = os.path.join(datasets_path, selected_folder)
        return train(selected_data_path, *args)
    
    def get_context():
        return gr.Blocks(delete_cache=(True, True))
    
    with get_context() as gs_demo:
        gr.Markdown("""
        <style>
        .fixed-size-video video {
            max-height: 400px !important;
            height: 400px !important;
            object-fit: contain;
        }
        </style>
        """)
        
        # Centered title
        gr.Markdown("""
        <h2 style="text-align: center;">3D Gaussian Splatting Reconstruction</h2>
        """)

        # Instructions
        gr.Markdown('''
        <div style="padding: 10px; border-radius: 5px; margin-bottom: 10px;">
            <h3>Instructions for 3DGS Demo</h3>
            <ul style="text-align: left; color: #333;">
                <li>Make sure to press "Refresh Datasets" to obtain an updated list of datasets from Stage 1. They are in the format run_0, run_1, run_...</li>
                <li>Adjust optimization parameters if needed, and press "Start Training".</li>
                <li>It is recommended to use 7k iterations to avoid exceeding the 3-minute limit. If you still exceed the limit, reduce the number of iterations.</li>
                <li>After reconstruction is finished, you can view it as a small video generated or download the full 3DGS reconstruction below the video.</li>
                <li>Press "Load 3D Model" to view the full 3DGS reconstruction.</li>
            </ul>
            <p><b>Note: 3DGS '.ply' models could be heavy, so it may take some time to download and view them in the 3D model section.</b></p>
        </div>
        ''')

        refresh_button = gr.Button("Refresh Datasets", elem_classes="refresh-button")
        dataset_dropdown = gr.Dropdown(label="Select Dataset", choices=[], value="")

        def update_dataset_dropdown(req: gr.Request):
            USER_DIR = os.path.join(CACHE_PATH, str(req.session_hash))
            print("update_dataset_dropdown, user_path", USER_DIR)
            dataset_path = os.path.join(USER_DIR, DATASET_DIR)
            # Update the dataset folders list
            dataset_folders = get_dataset_folders(dataset_path)
            print("dataset_folders", dataset_folders)
            # Set the default value to the last run if there are folders available
            default_value = dataset_folders[-1] if dataset_folders else None
            return gr.Dropdown(label="Select Dataset", choices=dataset_folders, value=default_value)
        
        # Set the update function to be called when the refresh button is clicked
        refresh_button.click(fn=update_dataset_dropdown, inputs=None, outputs=dataset_dropdown)
        
        with gr.Accordion("Optimization Parameters", open=False):
            with gr.Row():
                with gr.Column():
                    position_lr_init = gr.Number(label="Position LR Init", value=0.00032)
                    position_lr_final = gr.Number(label="Position LR Final", value=0.0000032)
                    position_lr_delay_mult = gr.Number(label="Position LR Delay Mult", value=0.02)
                    position_lr_max_steps = gr.Number(label="Position LR Max Steps", value=15000)
                    feature_lr = gr.Number(label="Feature LR", value=0.005)
                with gr.Column():
                    feature_lr = gr.Number(label="Feature LR", value=0.0025)
                    opacity_lr = gr.Number(label="Opacity LR", value=0.05)
                    scaling_lr = gr.Number(label="Scaling LR", value=0.005)
                    rotation_lr = gr.Number(label="Rotation LR", value=0.001)
                    percent_dense = gr.Number(label="Percent Dense", value=0.01)
                with gr.Column():
                    lambda_dssim = gr.Number(label="Lambda DSSIM", value=0.2)
                    densification_interval = gr.Number(label="Densification Interval", value=100)
                    opacity_reset_interval = gr.Number(label="Opacity Reset Interval", value=3000)
                    densify_from_iter = gr.Number(label="Densify From Iter", value=500)
                    densify_until_iter = gr.Number(label="Densify Until Iter", value=15000)
                    densify_grad_threshold = gr.Number(label="Densify Grad Threshold", value=0.0002)
        iterations = gr.Slider(label="Iterations", value=8000, minimum=1, maximum=15000, step=5)
        
        start_button = gr.Button("Start Training")
        
        # Add state variable to store model path
        model_path_state = gr.State()
        
        # Add video output and load model button with fixed scale
        video_output = gr.Video(
            label="Training Progress", 
            height=400,  # Fixed height
            width="100%",  # Full width of container
            autoplay=False,  # Prevent autoplay
            show_label=True,
            container=True,
            elem_classes="fixed-size-video"  # Add custom class for potential CSS
        )
        load_model_button = gr.Button("Load 3D Model", interactive=False)
        output = gr.Model3D(label="3D Model Output", visible=False)
        
        def handle_training_complete(selected_folder, req: gr.Request, *args):
            USER_DIR = os.path.join(CACHE_PATH, str(req.session_hash))
            if 'run' in selected_folder:
                dataset_path = os.path.join(USER_DIR, DATASET_DIR, selected_folder)
            else:
                dataset_path = os.path.join(EXAMPLE_PATH, selected_folder)
            # Call the training function with the full path
            video_path, model_path = train(dataset_path, *args)
            # Then return all required outputs
            return [
                video_path,           # video output
                gr.Button(value="Load 3D Model", interactive=True),  # Return new button with updated properties
                gr.Model3D(visible=False),  # keep 3D model hidden
                model_path            # store model path in state
            ]
        
        def load_model(model_path):
            if not model_path:
                return gr.Model3D(visible=False)
            return gr.Model3D(value=model_path, visible=True)
        
        # Connect the start training button
        start_button.click(
            fn=handle_training_complete,
            inputs=[
                dataset_dropdown, iterations, position_lr_init, position_lr_final, position_lr_delay_mult,
                position_lr_max_steps, feature_lr, opacity_lr, scaling_lr, rotation_lr,
                percent_dense, lambda_dssim, densification_interval, opacity_reset_interval,
                densify_from_iter, densify_until_iter, densify_grad_threshold
            ],
            outputs=[video_output, load_model_button, output, model_path_state]
        )
        
        # Connect the load model button
        load_model_button.click(
            fn=load_model,
            inputs=[model_path_state],
            outputs=output
        )
    return gs_demo