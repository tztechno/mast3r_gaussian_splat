import os
from .utils import clear_memory, get_memory_info
from .config import Config
from transformers import AutoImageProcessor, AutoModel


def load_torch_image(fname, device):
    """Load image as torch tensor"""
    import torchvision.transforms as T

    img = Image.open(fname).convert('RGB')
    transform = T.Compose([
        T.ToTensor(),
    ])
    return transform(img).unsqueeze(0).to(device)

def extract_dino_global(image_paths, model_path, device):
    """Extract DINO global descriptors with memory management"""
    print("\n=== Extracting DINO Global Features ===")
    print("Initial memory state:")
    get_memory_info()

    processor = AutoImageProcessor.from_pretrained(model_path)
    model = AutoModel.from_pretrained(model_path).eval().to(device)

    global_descs = []
    batch_size = 4  # Small batch to save memory
    
    for i in tqdm(range(0, len(image_paths), batch_size)):
        batch_paths = image_paths[i:i+batch_size]
        batch_imgs = []
        
        for img_path in batch_paths:
            img = load_torch_image(img_path, device)
            batch_imgs.append(img)
        
        batch_tensor = torch.cat(batch_imgs, dim=0)
        
        with torch.no_grad():
            inputs = processor(images=batch_tensor, return_tensors="pt", do_rescale=False).to(device)
            outputs = model(**inputs)
            desc = F.normalize(outputs.last_hidden_state[:, 1:].max(dim=1)[0], dim=1, p=2)
            global_descs.append(desc.cpu())
        
        # Clear batch memory
        del batch_tensor, inputs, outputs, desc
        clear_memory()

    global_descs = torch.cat(global_descs, dim=0)

    del model, processor
    clear_memory()
    
    print("After DINO extraction:")
    get_memory_info()

    return global_descs


def build_topk_pairs(global_feats, k, device):
    """Build top-k similar pairs from global features"""
    g = global_feats.to(device)
    sim = g @ g.T
    sim.fill_diagonal_(-1)

    N = sim.size(0)
    k = min(k, N - 1)

    topk_indices = torch.topk(sim, k, dim=1).indices.cpu()

    pairs = []
    for i in range(N):
        for j in topk_indices[i]:
            j = j.item()
            if i < j:
                pairs.append((i, j))

    # Remove duplicates
    pairs = list(set(pairs))
    
    return pairs


def select_diverse_pairs(pairs, max_pairs, num_images):
    """
    Select diverse pairs to ensure good image coverage
    Strategy: Select pairs that maximize image coverage
    """
    import random
    random.seed(42)
    
    if len(pairs) <= max_pairs:
        return pairs
    
    print(f"Selecting {max_pairs} diverse pairs from {len(pairs)} candidates...")
    
    # Count how many times each image appears in pairs
    image_counts = {i: 0 for i in range(num_images)}
    for i, j in pairs:
        image_counts[i] += 1
        image_counts[j] += 1
    
    # Sort pairs by: prefer pairs with less-connected images
    def pair_score(pair):
        i, j = pair
        # Lower score = images appear in fewer pairs = more diverse
        return image_counts[i] + image_counts[j]
    
    pairs_scored = [(pair, pair_score(pair)) for pair in pairs]
    pairs_scored.sort(key=lambda x: x[1])
    
    # Select pairs greedily to maximize coverage
    selected = []
    selected_images = set()
    
    # Phase 1: Select pairs that add new images (greedy coverage)
    for pair, score in pairs_scored:
        if len(selected) >= max_pairs:
            break
        i, j = pair
        # Prefer pairs that include new images
        if i not in selected_images or j not in selected_images:
            selected.append(pair)
            selected_images.add(i)
            selected_images.add(j)
    
    # Phase 2: Fill remaining slots with high-similarity pairs
    if len(selected) < max_pairs:
        remaining = [p for p, s in pairs_scored if p not in selected]
        random.shuffle(remaining)
        selected.extend(remaining[:max_pairs - len(selected)])
    
    print(f"Selected pairs cover {len(selected_images)} / {num_images} images ({100*len(selected_images)/num_images:.1f}%)")
    
    return selected


def get_image_pairs_dino(image_paths, max_pairs=None):
    """DINO-based pair selection with intelligent limiting"""
    device = Config.DEVICE

    # DINO global features
    global_feats = extract_dino_global(image_paths, Config.DINO_MODEL, device)
    pairs = build_topk_pairs(global_feats, Config.GLOBAL_TOPK, device)

    print(f"Initial pairs from DINO: {len(pairs)}")
    
    # Apply intelligent pair selection if limit specified
    if max_pairs and len(pairs) > max_pairs:
        pairs = select_diverse_pairs(pairs, max_pairs, len(image_paths))
    
    return pairs


def run(cfg):
    load_torch_image(fname, device)
    extract_dino_global(image_paths, model_path, device)
    build_topk_pairs(global_feats, k, device)
    select_diverse_pairs(pairs, max_pairs, num_images)   
    get_image_pairs_dino(image_paths, max_pairs=None)
    return cfg
    
