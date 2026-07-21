"""Exploratory Data Analysis (EDA) for the Histopathological Image Dataset.

=============================================================================
WHAT IS EDA?
=============================================================================
Exploratory Data Analysis is the FIRST step in any Machine Learning project.
Before training a model, we need to deeply understand our data:
    - How many classes do we have?
    - How many images per class? (Is the dataset balanced?)
    - What are the image dimensions? (Do we need to resize?)
    - Are there corrupted files that could crash our training?
    - What format are the images in? (PNG, JPEG, etc.)

WHY IS THIS IMPORTANT FOR DEEP LEARNING?
    - If class counts are very unequal (imbalanced), the model will be biased
      towards the majority class. We'd need strategies like oversampling,
      undersampling, or class weights.
    - If images have different sizes, we MUST resize them because neural
      networks (like DenseNet121 or ResNet50V2) expect a FIXED input size.
    - Corrupted images will cause errors during training. We need to find
      and remove them BEFORE training starts.

=============================================================================
HOW TO RUN THIS SCRIPT
=============================================================================
From the project root directory, run:

    python -m src.data.explore_dataset

Or with a custom dataset path:

    python -m src.data.explore_dataset --data-dir data/raw/MyDataset

=============================================================================

Author: Yassine
Project: Histology AI Classification
"""

# =============================================================================
# IMPORTS
# =============================================================================
# 'pathlib' is a modern Python module for working with file paths.
# It's much cleaner than the older 'os.path' module.
# Example: Path("data") / "raw" creates "data/raw" (works on any OS).
from pathlib import Path

# 'argparse' lets us accept command-line arguments when running the script.
# This makes the script flexible — we can point it at any dataset directory.
import argparse

# 'sys' provides access to system-specific parameters.
# We use sys.exit() to stop the script if something goes wrong.
import sys

# ---------------------------------------------------------------------------
# WINDOWS ENCODING FIX
# ---------------------------------------------------------------------------
# On Windows, the default console encoding (cp1252) cannot display Unicode
# characters like ✓, ✗, ⚠, █. By reconfiguring stdout to UTF-8, we ensure
# these characters print correctly on all platforms.
# This is a common issue you'll encounter in Python on Windows.
if sys.stdout.encoding != "utf-8":
    sys.stdout.reconfigure(encoding="utf-8")

# 'collections.Counter' is a special dictionary that counts things.
# Example: Counter(["a", "b", "a"]) → {"a": 2, "b": 1}
from collections import Counter

# 'random' lets us pick random items — we use it to select sample images.
import random

# ---------------------------------------------------------------------------
# THIRD-PARTY LIBRARIES (installed via pip)
# ---------------------------------------------------------------------------

# 'PIL' (Pillow) is Python's main image processing library.
# PIL.Image lets us open, inspect, and manipulate images.
# We use it to check image dimensions, color mode, and detect corruption.
from PIL import Image

# 'matplotlib' is Python's core plotting library.
# 'pyplot' is its simple interface (similar to MATLAB).
# We use it to display sample images and plot histograms.
import matplotlib.pyplot as plt

# 'numpy' is the fundamental library for numerical computing in Python.
# Arrays in numpy are MUCH faster than Python lists for math operations.
# Deep learning frameworks (TensorFlow, PyTorch) use numpy arrays internally.
import numpy as np


# =============================================================================
# CONSTANTS
# =============================================================================
# These are the image file extensions we consider valid.
# Histopathological datasets are typically in PNG or JPEG format.
# We define them as a set (not a list) because checking membership
# in a set is O(1) — constant time — while in a list it's O(n).
VALID_IMAGE_EXTENSIONS: set[str] = {".png", ".jpg", ".jpeg", ".bmp", ".tiff", ".tif"}

# Default path to the dataset, relative to the project root.
# We use the actual subdirectory name from the Kaggle download.
DEFAULT_DATA_DIR: str = "data/raw/Human_Histopathological_H_E_Stained_Nuclei_Images"

# How many sample images to display per class in the grid visualization.
SAMPLES_PER_CLASS: int = 3

# We set a random seed for reproducibility.
# This means every time you run the script, you'll see the SAME random
# sample images. This is crucial in research — your results must be
# reproducible by others.
RANDOM_SEED: int = 42


# =============================================================================
# SECTION 1: DATASET DIRECTORY VERIFICATION
# =============================================================================
def verify_dataset_directory(data_dir: Path) -> bool:
    """Check that the dataset directory exists and is not empty.

    WHY THIS MATTERS:
        Before doing anything, we must confirm the data is actually there.
        A common mistake is pointing to the wrong path or forgetting to
        download the dataset. This function catches that early.

    Args:
        data_dir: Path to the root directory of the dataset.
                  Expected to contain one subdirectory per class.

    Returns:
        True if the directory exists and contains at least one subdirectory.
        False otherwise.
    """
    # Path.exists() returns True if the path points to an existing file or directory.
    if not data_dir.exists():
        print(f"[ERROR] Dataset directory not found: {data_dir}")
        print("        Make sure you've downloaded the dataset into data/raw/")
        return False

    # Path.is_dir() checks if the path is specifically a directory (not a file).
    if not data_dir.is_dir():
        print(f"[ERROR] Path exists but is not a directory: {data_dir}")
        return False

    # Check that there's at least one subdirectory (each class should be a folder).
    # list() converts the generator to a list so we can check its length.
    subdirs = [d for d in data_dir.iterdir() if d.is_dir()]
    if len(subdirs) == 0:
        print(f"[ERROR] No subdirectories found in: {data_dir}")
        print("        Expected one folder per tissue class.")
        return False

    print(f"[OK] Dataset directory verified: {data_dir}")
    print(f"     Found {len(subdirs)} subdirectories.")
    return True


# =============================================================================
# SECTION 2: CLASS DETECTION
# =============================================================================
def detect_classes(data_dir: Path) -> list[str]:
    """Detect all classes (tissue types) from the dataset directory structure.

    HOW IMAGE CLASSIFICATION DATASETS ARE ORGANIZED:
        Most image classification datasets follow this convention:

            dataset/
            ├── class_1/
            │   ├── image_001.png
            │   ├── image_002.png
            │   └── ...
            ├── class_2/
            │   ├── image_001.png
            │   └── ...
            └── class_N/
                └── ...

        Each subfolder name IS the class label. This is called
        "directory-based labeling" — the folder structure encodes the labels.
        TensorFlow's image_dataset_from_directory() and Keras's
        ImageDataGenerator.flow_from_directory() both expect this layout.

    Args:
        data_dir: Path to the dataset root.

    Returns:
        Sorted list of class names (subfolder names).
    """
    # sorted() ensures consistent ordering across runs and operating systems.
    # Without sorting, the order could vary depending on the filesystem.
    # .name gives us just the folder name, not the full path.
    classes = sorted([d.name for d in data_dir.iterdir() if d.is_dir()])

    print(f"\n{'='*60}")
    print("DETECTED CLASSES")
    print(f"{'='*60}")

    for i, class_name in enumerate(classes, start=1):
        # enumerate(list, start=1) gives us (index, item) pairs starting at 1.
        # This is cleaner than manually tracking a counter variable.
        print(f"  {i:>2}. {class_name}")

    return classes


# =============================================================================
# SECTION 3: IMAGE COUNTING PER CLASS
# =============================================================================
def count_images_per_class(
    data_dir: Path,
    classes: list[str],
) -> dict[str, int]:
    """Count the number of image files in each class directory.

    WHY THIS MATTERS FOR DEEP LEARNING:
        - If one class has 1000 images and another has only 50, the model
          will learn to favor the majority class (class imbalance problem).
        - Common fixes: oversampling the minority class, undersampling the
          majority, using class weights in the loss function, or applying
          data augmentation to balance the classes.

    WHAT'S A dict[str, int]?
        This is a Python type hint. It tells you (and your IDE) that this
        function returns a dictionary where:
            - Keys are strings (class names like "liver", "kidney")
            - Values are integers (image counts like 53, 46)

    Args:
        data_dir: Path to the dataset root.
        classes: List of class names (subfolder names).

    Returns:
        Dictionary mapping each class name to its image count.
    """
    class_counts: dict[str, int] = {}

    for class_name in classes:
        class_path = data_dir / class_name

        # We count only files with valid image extensions.
        # Path.suffix returns the file extension (e.g., ".png").
        # .lower() handles edge cases like ".PNG" or ".Jpg".
        image_files = [
            f for f in class_path.iterdir()
            if f.is_file() and f.suffix.lower() in VALID_IMAGE_EXTENSIONS
        ]
        class_counts[class_name] = len(image_files)

    return class_counts


def print_class_distribution(class_counts: dict[str, int]) -> None:
    """Display a formatted table of image counts per class.

    This function also prints a simple ASCII bar chart so you can
    visually spot imbalanced classes at a glance.

    Args:
        class_counts: Dictionary mapping class names to image counts.
    """
    print(f"\n{'='*60}")
    print("CLASS DISTRIBUTION")
    print(f"{'='*60}")

    # max() on the values gives us the largest count — used for scaling bars.
    max_count = max(class_counts.values())

    for class_name, count in class_counts.items():
        # We create a simple bar using '█' characters.
        # The bar length is proportional to the count.
        # int(30 * count / max_count) scales every bar to a max of 30 chars.
        bar_length = int(30 * count / max_count)
        bar = "█" * bar_length

        # f-string formatting:
        #   {class_name:<35} → left-align in 35 characters
        #   {count:>4}       → right-align in 4 characters
        print(f"  {class_name:<35} {count:>4} images  {bar}")


# =============================================================================
# SECTION 4: DATASET STATISTICS
# =============================================================================
def compute_dataset_statistics(class_counts: dict[str, int]) -> dict[str, float]:
    """Compute summary statistics about the dataset.

    WHY THESE STATISTICS MATTER:
        - total_classes: Determines the output layer size of our neural network.
          If we have 22 classes, the final Dense layer will have 22 neurons.
        - total_images: Tells us if we have enough data. Deep learning is
          data-hungry — too few images may require heavy augmentation or
          transfer learning (using a model pre-trained on ImageNet).
        - min/max/average: Reveals class imbalance. If min << max, we have
          an imbalance problem.

    Args:
        class_counts: Dictionary mapping class names to image counts.

    Returns:
        Dictionary of computed statistics.
    """
    counts = list(class_counts.values())

    stats = {
        "total_classes": len(counts),
        "total_images": sum(counts),
        "min_images_per_class": min(counts),
        "max_images_per_class": max(counts),
        "average_images_per_class": round(np.mean(counts), 2),
        "std_images_per_class": round(np.std(counts), 2),
    }

    # Find which class has the minimum and maximum.
    # min(dict, key=dict.get) returns the KEY with the smallest VALUE.
    stats["class_with_min"] = min(class_counts, key=lambda k: class_counts[k])
    stats["class_with_max"] = max(class_counts, key=lambda k: class_counts[k])

    print(f"\n{'='*60}")
    print("DATASET STATISTICS")
    print(f"{'='*60}")
    print(f"  Total classes:             {stats['total_classes']}")
    print(f"  Total images:              {stats['total_images']}")
    print(f"  Min images per class:      {stats['min_images_per_class']} ({stats['class_with_min']})")
    print(f"  Max images per class:      {stats['max_images_per_class']} ({stats['class_with_max']})")
    print(f"  Average images per class:  {stats['average_images_per_class']}")
    print(f"  Std deviation:             {stats['std_images_per_class']}")

    return stats


# =============================================================================
# SECTION 5: IMAGE FORMAT VERIFICATION
# =============================================================================
def verify_image_formats(
    data_dir: Path,
    classes: list[str],
) -> Counter:
    """Check the file extensions of all images in the dataset.

    WHY THIS MATTERS:
        - Different formats have different properties:
            PNG: Lossless compression (no quality loss). Best for medical images.
            JPEG: Lossy compression (some quality loss). Smaller file size.
            BMP/TIFF: Uncompressed or lossless. Rarely used in ML datasets.
        - Knowing the format helps us choose the right loading strategy.
        - Mixed formats can sometimes cause issues with data loaders.

    WHAT IS Counter?
        Counter is a specialized dictionary from Python's collections module.
        It counts how many times each element appears:
            Counter([".png", ".png", ".jpg"]) → {".png": 2, ".jpg": 1}

    Args:
        data_dir: Path to the dataset root.
        classes: List of class names.

    Returns:
        Counter object mapping each file extension to its count.
    """
    extensions: list[str] = []

    for class_name in classes:
        class_path = data_dir / class_name
        for f in class_path.iterdir():
            if f.is_file():
                extensions.append(f.suffix.lower())

    format_counts = Counter(extensions)

    print(f"\n{'='*60}")
    print("IMAGE FORMATS")
    print(f"{'='*60}")

    for ext, count in format_counts.most_common():
        # .most_common() returns items sorted by count (highest first).
        status = "✓ Valid" if ext in VALID_IMAGE_EXTENSIONS else "✗ Unexpected"
        print(f"  {ext:<10} {count:>5} files  [{status}]")

    return format_counts


# =============================================================================
# SECTION 6: COLOR MODE VERIFICATION
# =============================================================================
def verify_color_modes(
    data_dir: Path,
    classes: list[str],
) -> Counter:
    """Check the color mode of all images (RGB, Grayscale, RGBA, etc.).

    WHY THIS MATTERS FOR DEEP LEARNING:
        - DenseNet121 and ResNet50V2 expect RGB images (3 channels).
        - If some images are grayscale (1 channel) or RGBA (4 channels),
          we'll need to convert them during preprocessing.
        - The input shape of our model depends on this:
            RGB:       (224, 224, 3)
            Grayscale: (224, 224, 1)
            RGBA:      (224, 224, 4)
        - H&E stained images should be RGB because the staining produces
          distinct colors (blue for nuclei, pink for cytoplasm).

    WHAT ARE COLOR MODES?
        - "RGB":  3 channels — Red, Green, Blue. Standard for color images.
        - "L":    1 channel — Luminance. Grayscale images.
        - "RGBA": 4 channels — RGB + Alpha (transparency).
        - "CMYK": 4 channels — Cyan, Magenta, Yellow, Key (print format).
        - "P":    1 channel — Palette-based (indexed colors).

    Args:
        data_dir: Path to the dataset root.
        classes: List of class names.

    Returns:
        Counter mapping color modes to their frequency.
    """
    modes: list[str] = []

    for class_name in classes:
        class_path = data_dir / class_name
        for f in class_path.iterdir():
            if f.is_file() and f.suffix.lower() in VALID_IMAGE_EXTENSIONS:
                try:
                    # Image.open() doesn't load the full image into memory.
                    # It reads only the header (metadata) — this is fast!
                    # .mode tells us the color mode without decoding pixels.
                    with Image.open(f) as img:
                        modes.append(img.mode)
                except Exception:
                    # If an image can't be opened, skip it silently here.
                    # We'll catch corrupted images in a dedicated function.
                    pass

    mode_counts = Counter(modes)

    print(f"\n{'='*60}")
    print("COLOR MODES")
    print(f"{'='*60}")

    for mode, count in mode_counts.most_common():
        print(f"  {mode:<10} {count:>5} images")

    # Explain the practical implication.
    if len(mode_counts) == 1 and "RGB" in mode_counts:
        print("\n  ✓ All images are RGB — no color conversion needed.")
    elif len(mode_counts) == 1:
        only_mode = list(mode_counts.keys())[0]
        print(f"\n  ⚠ All images are {only_mode} — conversion to RGB will be needed.")
    else:
        print("\n  ⚠ Mixed color modes detected — conversion to a uniform mode needed.")

    return mode_counts


# =============================================================================
# SECTION 7: IMAGE DIMENSION ANALYSIS
# =============================================================================
def analyze_image_dimensions(
    data_dir: Path,
    classes: list[str],
) -> list[tuple[int, int]]:
    """Collect the width and height of every image in the dataset.

    WHY THIS MATTERS FOR DEEP LEARNING:
        Neural networks require a FIXED input size. For example:
            - DenseNet121 default input: 224 × 224
            - ResNet50V2 default input:  224 × 224

        If our images are 300×300, 256×256, or have mixed sizes, we MUST
        resize them all to the same dimensions during preprocessing.

        This function tells us:
            - Are all images the same size? (Best case — minimal preprocessing)
            - What's the size distribution? (Helps us choose the target size)

    WHAT IS tuple[int, int]?
        A tuple is an immutable (unchangeable) pair of values.
        tuple[int, int] means "a pair of two integers" — in our case,
        (width, height) of an image.

    Args:
        data_dir: Path to the dataset root.
        classes: List of class names.

    Returns:
        List of (width, height) tuples for all valid images.
    """
    dimensions: list[tuple[int, int]] = []

    for class_name in classes:
        class_path = data_dir / class_name
        for f in class_path.iterdir():
            if f.is_file() and f.suffix.lower() in VALID_IMAGE_EXTENSIONS:
                try:
                    with Image.open(f) as img:
                        # img.size returns (width, height) as a tuple.
                        dimensions.append(img.size)
                except Exception:
                    pass

    # Use a Counter to find unique dimensions and their frequencies.
    dimension_counts = Counter(dimensions)

    print(f"\n{'='*60}")
    print("IMAGE DIMENSIONS")
    print(f"{'='*60}")

    if len(dimension_counts) == 1:
        # All images have the same size — ideal scenario.
        size = list(dimension_counts.keys())[0]
        print(f"  ✓ All images have the same size: {size[0]} × {size[1]} pixels")
    else:
        # Multiple sizes detected — we'll need to resize.
        print(f"  Found {len(dimension_counts)} unique dimension(s):\n")
        for (w, h), count in dimension_counts.most_common(10):
            print(f"    {w:>5} × {h:<5}  →  {count:>5} images")

        if len(dimension_counts) > 10:
            print(f"    ... and {len(dimension_counts) - 10} more unique sizes.")

    # Compute min/max width and height across all images.
    widths = [w for w, h in dimensions]
    heights = [h for w, h in dimensions]

    print(f"\n  Width  range: {min(widths)} – {max(widths)} pixels")
    print(f"  Height range: {min(heights)} – {max(heights)} pixels")
    print(f"  Mean size:    {np.mean(widths):.0f} × {np.mean(heights):.0f} pixels")

    return dimensions


# =============================================================================
# SECTION 8: CORRUPTED IMAGE DETECTION
# =============================================================================
def detect_corrupted_images(
    data_dir: Path,
    classes: list[str],
) -> list[Path]:
    """Attempt to fully load every image to detect corrupted files.

    WHY THIS MATTERS:
        - Image.open() only reads the header. A file can have a valid header
          but corrupted pixel data. The only way to detect this is to force
          the library to read ALL the pixels.
        - img.verify() checks the file's internal integrity.
        - If corrupted images make it into training, they can cause:
            1. Crashes during training (data loader fails)
            2. Silent data corruption (garbage pixels treated as real data)

    HOW WE DETECT CORRUPTION:
        We use a two-step approach:
        1. img.verify() — checks the file structure (fast but incomplete).
        2. img.load()   — forces full pixel decoding (thorough but slower).
        We use verify() first because it's faster. We re-open and load()
        because verify() closes the file and doesn't catch all corruption.

    Args:
        data_dir: Path to the dataset root.
        classes: List of class names.

    Returns:
        List of Path objects pointing to corrupted image files.
    """
    corrupted: list[Path] = []
    total_checked = 0

    print(f"\n{'='*60}")
    print("CORRUPTED IMAGE DETECTION")
    print(f"{'='*60}")
    print("  Scanning all images (this may take a moment)...")

    for class_name in classes:
        class_path = data_dir / class_name
        for f in class_path.iterdir():
            if f.is_file() and f.suffix.lower() in VALID_IMAGE_EXTENSIONS:
                total_checked += 1
                try:
                    # STEP 1: Open and verify the file structure.
                    with Image.open(f) as img:
                        img.verify()

                    # STEP 2: Re-open (verify closes the file) and force-load pixels.
                    # This catches corruption that verify() misses.
                    with Image.open(f) as img:
                        img.load()

                except Exception as e:
                    # If either step fails, the image is corrupted.
                    corrupted.append(f)
                    print(f"    ✗ CORRUPTED: {f.name} in {class_name}")
                    print(f"      Error: {e}")

    if len(corrupted) == 0:
        print(f"\n  ✓ All {total_checked} images passed integrity checks.")
    else:
        print(f"\n  ✗ Found {len(corrupted)} corrupted image(s) out of {total_checked}.")
        print("    These files should be removed before training.")

    return corrupted


# =============================================================================
# SECTION 9: SAMPLE IMAGE VISUALIZATION
# =============================================================================
def display_sample_images(
    data_dir: Path,
    classes: list[str],
    samples_per_class: int = SAMPLES_PER_CLASS,
) -> None:
    """Display a grid of random sample images from each class.

    WHY THIS MATTERS:
        Looking at actual images is the most intuitive way to understand
        your dataset. You can visually check:
        - Are the classes visually distinct?
        - Is the image quality consistent?
        - Are there any obvious labeling errors?
        - How much variation exists within each class?

    HOW matplotlib SUBPLOTS WORK:
        fig, axes = plt.subplots(rows, cols)
        Creates a grid of "axes" (individual plot areas).
        Each axes[row][col] is one plot where we can show an image.

    Args:
        data_dir: Path to the dataset root.
        classes: List of class names.
        samples_per_class: Number of random images to show per class.
    """
    print(f"\n{'='*60}")
    print("SAMPLE IMAGES")
    print(f"{'='*60}")
    print(f"  Displaying {samples_per_class} random sample(s) per class...")

    # Set the random seed for reproducibility.
    random.seed(RANDOM_SEED)

    num_classes = len(classes)

    # Create a figure with a grid: one row per class, columns = samples_per_class.
    # figsize=(width, height) sets the figure size in inches.
    fig, axes = plt.subplots(
        nrows=num_classes,
        ncols=samples_per_class,
        figsize=(samples_per_class * 3, num_classes * 2.5),
    )

    # If there's only one sample per class, axes would be 1D. We ensure it's 2D.
    # np.atleast_2d doesn't work for axes arrays, so we handle it manually.
    if samples_per_class == 1:
        axes = axes.reshape(-1, 1)

    for row_idx, class_name in enumerate(classes):
        class_path = data_dir / class_name

        # Get all valid image files in this class folder.
        image_files = [
            f for f in class_path.iterdir()
            if f.is_file() and f.suffix.lower() in VALID_IMAGE_EXTENSIONS
        ]

        # random.sample() picks 'k' unique random items from a list.
        # min() ensures we don't try to sample more than available.
        k = min(samples_per_class, len(image_files))
        sampled_files = random.sample(image_files, k)

        for col_idx in range(samples_per_class):
            ax = axes[row_idx][col_idx]

            if col_idx < k:
                # Open and display the image.
                img = Image.open(sampled_files[col_idx])

                # ax.imshow() displays an image on the axes.
                # Converting to numpy array because imshow works with arrays.
                ax.imshow(np.array(img))

                # Show the class label only on the first column.
                if col_idx == 0:
                    # Extract a clean label: "human_tissue_image-liver" → "liver"
                    clean_label = class_name.replace("human_tissue_image-", "")
                    ax.set_ylabel(clean_label, fontsize=8, rotation=0, labelpad=60)

            # Turn off axis ticks and labels — they're meaningless for images.
            ax.set_xticks([])
            ax.set_yticks([])

    # plt.suptitle() adds a title above the entire figure (not just one subplot).
    plt.suptitle(
        "Random Sample Images per Class",
        fontsize=14,
        fontweight="bold",
        y=1.01,
    )

    # tight_layout() automatically adjusts spacing so subplots don't overlap.
    plt.tight_layout()

    # Save the figure to reports/figures/ for documentation.
    output_path = Path("reports/figures/sample_images_grid.png")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    print(f"  Saved to: {output_path}")

    # plt.show() opens the interactive window. Comment out if running headless.
    plt.show()


# =============================================================================
# SECTION 10: DISTRIBUTION PLOTS
# =============================================================================
def plot_class_distribution(class_counts: dict[str, int]) -> None:
    """Plot a horizontal bar chart of class distribution.

    WHY A BAR CHART?
        It immediately shows whether the dataset is balanced or imbalanced.
        In a balanced dataset, all bars would be roughly the same length.
        Large differences mean we might need class balancing strategies.

    WHY HORIZONTAL BARS?
        With 22 class names, vertical bars would have unreadable labels.
        Horizontal bars give each class name enough space.

    Args:
        class_counts: Dictionary mapping class names to image counts.
    """
    # Clean up class names for display: remove the "human_tissue_image-" prefix.
    clean_names = [
        name.replace("human_tissue_image-", "") for name in class_counts.keys()
    ]
    counts = list(class_counts.values())

    # Create the figure.
    fig, ax = plt.subplots(figsize=(10, 8))

    # Create a color gradient based on count values.
    # plt.cm.viridis is a perceptually uniform colormap — it's accessible
    # to people with color vision deficiency (unlike red-green colormaps).
    colors = plt.cm.viridis(np.linspace(0.3, 0.9, len(counts)))

    # Sort by count for a cleaner visual.
    # zip() pairs up names and counts, then sorted() sorts by count.
    sorted_pairs = sorted(zip(clean_names, counts, colors), key=lambda x: x[1])
    sorted_names, sorted_counts, sorted_colors = zip(*sorted_pairs)

    # barh() creates horizontal bars.
    bars = ax.barh(sorted_names, sorted_counts, color=sorted_colors, edgecolor="white")

    # Add count labels at the end of each bar.
    for bar, count in zip(bars, sorted_counts):
        ax.text(
            bar.get_width() + 0.5,     # X position (just past the bar end)
            bar.get_y() + bar.get_height() / 2,  # Y position (center of bar)
            str(count),                  # The text to display
            va="center",                 # Vertical alignment
            fontsize=9,
        )

    # Add a vertical line for the mean to visualize balance.
    mean_count = np.mean(counts)
    ax.axvline(
        x=mean_count,
        color="red",
        linestyle="--",
        linewidth=1.5,
        label=f"Mean: {mean_count:.1f}",
    )

    ax.set_xlabel("Number of Images", fontsize=12)
    ax.set_title("Class Distribution — Images per Tissue Type", fontsize=14, fontweight="bold")
    ax.legend(fontsize=10)

    plt.tight_layout()

    output_path = Path("reports/figures/class_distribution.png")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    print(f"\n  Class distribution plot saved to: {output_path}")

    plt.show()


def plot_image_size_distribution(dimensions: list[tuple[int, int]]) -> None:
    """Plot a scatter plot of image dimensions (width vs. height).

    WHY THIS PLOT?
        If all images are the same size, you'll see a single dot — no
        resizing needed. If dots are scattered, you need a resize step.

        This also helps you CHOOSE the target size for resizing.
        You want to pick a size close to the majority to minimize
        distortion from up/down-scaling.

    Args:
        dimensions: List of (width, height) tuples.
    """
    # Check if all dimensions are the same. If so, a scatter plot isn't useful.
    unique_dims = set(dimensions)
    if len(unique_dims) == 1:
        print("  All images have identical dimensions — size distribution plot skipped.")
        return

    widths = [w for w, h in dimensions]
    heights = [h for w, h in dimensions]

    fig, ax = plt.subplots(figsize=(8, 6))

    # alpha=0.5 makes dots semi-transparent so overlapping dots are visible.
    ax.scatter(widths, heights, alpha=0.5, s=10, c="steelblue")

    ax.set_xlabel("Width (pixels)", fontsize=12)
    ax.set_ylabel("Height (pixels)", fontsize=12)
    ax.set_title("Image Size Distribution", fontsize=14, fontweight="bold")

    # Add a reference point for common DL input sizes.
    ax.axhline(y=224, color="red", linestyle="--", alpha=0.7, label="224px (DenseNet/ResNet default)")
    ax.axvline(x=224, color="red", linestyle="--", alpha=0.7)
    ax.legend()

    plt.tight_layout()

    output_path = Path("reports/figures/image_size_distribution.png")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    print(f"  Image size distribution plot saved to: {output_path}")

    plt.show()


# =============================================================================
# SECTION 11: FINAL REPORT
# =============================================================================
def print_final_report(
    stats: dict[str, float],
    format_counts: Counter,
    mode_counts: Counter,
    dimensions: list[tuple[int, int]],
    corrupted: list[Path],
) -> None:
    """Print a comprehensive final summary with actionable recommendations.

    This is the most important output of the EDA. It tells you exactly
    what preprocessing steps will be needed before training.

    Args:
        stats: Dataset statistics from compute_dataset_statistics().
        format_counts: Image format distribution.
        mode_counts: Color mode distribution.
        dimensions: List of all image dimensions.
        corrupted: List of corrupted image paths.
    """
    print(f"\n{'='*60}")
    print("FINAL EDA REPORT")
    print(f"{'='*60}")

    # --- 1. Dataset Balance Assessment ---
    print("\n  1. DATASET BALANCE")
    ratio = stats["max_images_per_class"] / stats["min_images_per_class"]
    # A common rule of thumb: if the max/min ratio is below 1.5, the dataset
    # is considered reasonably balanced for classification.
    if ratio < 1.5:
        print(f"     ✓ Dataset is REASONABLY BALANCED (max/min ratio: {ratio:.2f})")
        print("       No class balancing strategy is strictly required.")
        print("       However, augmentation can still improve generalization.")
    elif ratio < 3.0:
        print(f"     ⚠ Dataset is SLIGHTLY IMBALANCED (max/min ratio: {ratio:.2f})")
        print("       Consider: class weights in loss function or mild oversampling.")
    else:
        print(f"     ✗ Dataset is HEAVILY IMBALANCED (max/min ratio: {ratio:.2f})")
        print("       Required: oversampling, SMOTE, or strong class weights.")

    # --- 2. Preprocessing Needs ---
    print("\n  2. PREPROCESSING REQUIREMENTS")

    # Color mode check.
    if len(mode_counts) == 1 and "RGB" in mode_counts:
        print("     ✓ Color: All images are RGB — no conversion needed.")
    else:
        print("     ✗ Color: Mixed modes detected — conversion to RGB needed.")

    # Format check.
    non_standard = [ext for ext in format_counts if ext not in VALID_IMAGE_EXTENSIONS]
    if non_standard:
        print(f"     ✗ Format: Non-standard formats found: {non_standard}")
    else:
        print(f"     ✓ Format: All images are in standard format(s): "
              f"{list(format_counts.keys())}")

    # --- 3. Resizing Assessment ---
    print("\n  3. RESIZING REQUIREMENTS")
    unique_dims = set(dimensions)
    if len(unique_dims) == 1:
        w, h = list(unique_dims)[0]
        print(f"     All images are {w}×{h} pixels.")
        if w == 224 and h == 224:
            print("     ✓ Already matches DenseNet121/ResNet50V2 default input (224×224).")
            print("       No resizing required.")
        else:
            print(f"     ⚠ Images need resizing from {w}×{h} to 224×224 (or chosen target).")
            print("       Use bilinear or bicubic interpolation for downscaling.")
    else:
        print(f"     ✗ {len(unique_dims)} different sizes detected — resizing is REQUIRED.")
        print("       All images must be resized to a uniform size (e.g., 224×224).")

    # --- 4. Corrupted Images ---
    print("\n  4. CORRUPTED IMAGES")
    if len(corrupted) == 0:
        print("     ✓ No corrupted images found. Dataset is clean.")
    else:
        print(f"     ✗ {len(corrupted)} corrupted image(s) found.")
        print("       Remove these files before training:")
        for path in corrupted:
            print(f"       - {path}")

    # --- 5. Overall Readiness ---
    print(f"\n{'='*60}")
    print("OVERALL READINESS")
    print(f"{'='*60}")

    issues = []
    if ratio >= 1.5:
        issues.append("class imbalance")
    if len(mode_counts) > 1 or "RGB" not in mode_counts:
        issues.append("color mode conversion")
    if len(unique_dims) > 1:
        issues.append("resizing")
    if len(corrupted) > 0:
        issues.append("corrupted image removal")

    if len(issues) == 0:
        print("  ✓ Dataset is in GOOD shape for training preparation.")
        print("    Recommended next steps:")
        print("      1. Normalize pixel values (ImageNet mean/std or [0, 1])")
        print("      2. Apply data augmentation to improve generalization")
        print("      3. Split into train/validation/test sets")
    else:
        print(f"  ⚠ {len(issues)} issue(s) to address before training:")
        for i, issue in enumerate(issues, start=1):
            print(f"    {i}. {issue}")

    print(f"\n{'='*60}")
    print("EDA COMPLETE")
    print(f"{'='*60}\n")


# =============================================================================
# MAIN FUNCTION
# =============================================================================
def main() -> None:
    """Entry point for the Exploratory Data Analysis script.

    This function orchestrates all EDA steps in logical order.

    WHAT IS if __name__ == '__main__'?
        When Python runs a .py file directly, it sets __name__ to '__main__'.
        When a .py file is imported by another module, __name__ is set to
        the module name instead. This check ensures main() only runs when
        you execute the script directly, NOT when you import it.

    WHAT IS argparse?
        argparse is Python's built-in library for parsing command-line
        arguments. It lets users customize the script's behavior without
        editing the code. Example:
            python script.py --data-dir my/custom/path
    """
    # -------------------------------------------------------------------------
    # Parse command-line arguments
    # -------------------------------------------------------------------------
    parser = argparse.ArgumentParser(
        description="Exploratory Data Analysis for Histopathological Image Dataset",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--data-dir",
        type=str,
        default=DEFAULT_DATA_DIR,
        help=f"Path to the dataset directory (default: {DEFAULT_DATA_DIR})",
    )
    args = parser.parse_args()

    # Convert the string path to a Path object.
    data_dir = Path(args.data_dir)

    # -------------------------------------------------------------------------
    # Print header
    # -------------------------------------------------------------------------
    print("\n" + "=" * 60)
    print("  EXPLORATORY DATA ANALYSIS")
    print("  Histopathological H&E Stained Nuclei Images")
    print("=" * 60)

    # -------------------------------------------------------------------------
    # Step 1: Verify dataset directory
    # -------------------------------------------------------------------------
    if not verify_dataset_directory(data_dir):
        sys.exit(1)  # Exit with error code 1 (convention for "something went wrong")

    # -------------------------------------------------------------------------
    # Step 2: Detect classes
    # -------------------------------------------------------------------------
    classes = detect_classes(data_dir)

    # -------------------------------------------------------------------------
    # Step 3: Count images per class
    # -------------------------------------------------------------------------
    class_counts = count_images_per_class(data_dir, classes)
    print_class_distribution(class_counts)

    # -------------------------------------------------------------------------
    # Step 4: Compute statistics
    # -------------------------------------------------------------------------
    stats = compute_dataset_statistics(class_counts)

    # -------------------------------------------------------------------------
    # Step 5: Verify image formats
    # -------------------------------------------------------------------------
    format_counts = verify_image_formats(data_dir, classes)

    # -------------------------------------------------------------------------
    # Step 6: Verify color modes
    # -------------------------------------------------------------------------
    mode_counts = verify_color_modes(data_dir, classes)

    # -------------------------------------------------------------------------
    # Step 7: Analyze image dimensions
    # -------------------------------------------------------------------------
    dimensions = analyze_image_dimensions(data_dir, classes)

    # -------------------------------------------------------------------------
    # Step 8: Detect corrupted images
    # -------------------------------------------------------------------------
    corrupted = detect_corrupted_images(data_dir, classes)

    # -------------------------------------------------------------------------
    # Step 9: Display sample images
    # -------------------------------------------------------------------------
    display_sample_images(data_dir, classes)

    # -------------------------------------------------------------------------
    # Step 10: Plot distributions
    # -------------------------------------------------------------------------
    plot_class_distribution(class_counts)
    plot_image_size_distribution(dimensions)

    # -------------------------------------------------------------------------
    # Step 11: Final report
    # -------------------------------------------------------------------------
    print_final_report(stats, format_counts, mode_counts, dimensions, corrupted)


# =============================================================================
# SCRIPT ENTRY POINT
# =============================================================================
# This is the standard Python idiom for making a file runnable as a script
# while also being importable as a module.
#
# When you run: python -m src.data.explore_dataset
#   → Python sets __name__ = "__main__" → main() gets called
#
# When you do: from src.data.explore_dataset import detect_classes
#   → Python sets __name__ = "src.data.explore_dataset" → main() is NOT called
#
if __name__ == "__main__":
    main()
