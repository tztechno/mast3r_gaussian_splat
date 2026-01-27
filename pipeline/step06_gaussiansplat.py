# ============================================================================
# Step 3: Gaussian Splatting Training
# ============================================================================

def setup_gaussian_splatting():
    """Setup Gaussian Splatting"""
    print("\n=== Setting up Gaussian Splatting ===")
    
    os.chdir('/kaggle/working')
    
    WORK_DIR = "gaussian-splatting"
    
    if not os.path.exists(WORK_DIR):
        print("Cloning Gaussian Splatting repository...")
        run_cmd([
            "git", "clone", "--recursive",
            "https://github.com/graphdeco-inria/gaussian-splatting.git",
            WORK_DIR
        ])
    else:
        print("✓ Repository already exists")
    
    os.chdir(WORK_DIR)
    
    # Install requirements
    print("Installing Gaussian Splatting requirements...")
    run_cmd([sys.executable, "-m", "pip", "install", "-r", "requirements.txt"])
    
    # Build submodules
    print("\n📦 Building Gaussian Splatting submodules...")
    
    submodules = {
        "diff-gaussian-rasterization":
            "https://github.com/graphdeco-inria/diff-gaussian-rasterization.git",
        "simple-knn":
            "https://github.com/camenduru/simple-knn.git"
    }
    
    for name, repo in submodules.items():
        print(f"\n📦 Installing {name}...")
        path = os.path.join("submodules", name)
        if not os.path.exists(path):
            run_cmd(["git", "clone", repo, path])
        run_cmd([sys.executable, "-m", "pip", "install", path])
    
    print("✓ Gaussian Splatting setup complete!")


def train_gaussian_splatting(colmap_dir, image_dir, output_dir, iterations=2000):
    """Train Gaussian Splatting model"""
    print("\n" + "="*70)
    print("Step 6: Training Gaussian Splatting")
    print("="*70)
    
    print("\n=== Training Gaussian Splatting ===")
    
    # Reduce memory usage with smaller resolution
    cmd = [
        'python', 'train.py',
        '-s', colmap_dir,
        '--images', image_dir,
        '-m', output_dir,
        '--iterations', str(iterations),
        '--test_iterations', '1000', str(iterations),
        '--save_iterations', '1000', str(iterations),
        '--resolution', '2',  # Reduce resolution to 1/2
        '--densify_grad_threshold', '0.001',  # Higher threshold = fewer Gaussians
        '--densification_interval', '200',  # Less frequent densification
        '--opacity_reset_interval', '5000',  # Less frequent reset
    ]
    
    print(f"Command: {' '.join(cmd)}\n")
    
    result = subprocess.run(
        cmd,
        cwd='/kaggle/working/gaussian-splatting',
        capture_output=True,
        text=True
    )
    
    print(result.stdout)
    if result.stderr:
        print("STDERR:", result.stderr)
    
    if result.returncode != 0:
        raise RuntimeError("Gaussian Splatting training failed")
    
    # Check output
    if not os.path.exists(os.path.join(output_dir, f'point_cloud/iteration_{iterations}/point_cloud.ply')):
        raise RuntimeError(f"Expected output not found at iteration {iterations}")
    
    print(f"\n✓ Gaussian Splatting training completed successfully")
    print(f"  Output: {output_dir}")
    
    return output_dir



def run(cfg):
    setup_gaussian_splatting()
    train_gaussian_splatting(colmap_dir, image_dir, output_dir, iterations=2000)    
    return cfg
    
