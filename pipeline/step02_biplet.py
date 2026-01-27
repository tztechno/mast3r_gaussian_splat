# ============================================================================
# Step 0: Biplet-Square Normalization (PRESERVED FROM ORIGINAL)
# ============================================================================
import os
from pipeline.config import Config

def normalize_image_sizes_biplet(input_dir, output_dir=None, size=1024):
    """
    Generates two square crops (Left & Right or Top & Bottom)
    from each image in a directory.
    """
    if output_dir is None:
        output_dir = 'output/images_biplet'

    os.makedirs(output_dir, exist_ok=True)

    print(f"Generating 2 cropped squares (Left/Right or Top/Bottom) for each image...")
    print()

    converted_count = 0
    size_stats = {}

    for img_file in sorted(os.listdir(input_dir)):
        if not img_file.lower().endswith(('.jpg', '.jpeg', '.png')):
            continue

        input_path = os.path.join(input_dir, img_file)

        try:
            img = Image.open(input_path)
            original_size = img.size

            size_key = f"{original_size[0]}x{original_size[1]}"
            size_stats[size_key] = size_stats.get(size_key, 0) + 1

            # Generate 2 crops
            crops = generate_two_crops(img, size)

            base_name, ext = os.path.splitext(img_file)
            for mode, cropped_img in crops.items():
                output_path = os.path.join(output_dir, f"{base_name}_{mode}{ext}")
                cropped_img.save(output_path, quality=95)

            converted_count += 1
            print(f"  ✓ {img_file}: {original_size} → 2 square images generated")

        except Exception as e:
            print(f"  ✗ Error processing {img_file}: {e}")

    print(f"\nProcessing complete: {converted_count} source images processed")
    print(f"Original size distribution: {size_stats}")
    return converted_count


def generate_two_crops(img, size):
    """
    Crops the image into a square and returns 2 variations
    (Left/Right for landscape, Top/Bottom for portrait).
    """
    width, height = img.size
    crop_size = min(width, height)
    crops = {}

    if width > height:
        # Landscape → Left & Right
        positions = {
            'left': 0,
            'right': width - crop_size
        }
        for mode, x_offset in positions.items():
            box = (x_offset, 0, x_offset + crop_size, crop_size)
            crops[mode] = img.crop(box).resize(
                (size, size),
                Image.Resampling.LANCZOS
            )

    else:
        # Portrait or Square → Top & Bottom
        positions = {
            'top': 0,
            'bottom': height - crop_size
        }
        for mode, y_offset in positions.items():
            box = (0, y_offset, crop_size, y_offset + crop_size)
            crops[mode] = img.crop(box).resize(
                (size, size),
                Image.Resampling.LANCZOS
            )

    return crops


def run(cfg):
    normalize_image_sizes_biplet(
        cfg.image_dir,
        cfg.processed_image_dir,
        cfg.square_size
    )
    generate_two_crops(img, size)
    return cfg
