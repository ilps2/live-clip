"""
Analyze spatial motion distribution to find the "content region" in screen-recorded livestreams.
Computes per-row motion heatmap → identifies top/bottom boundaries for cropping.
"""
import sys, numpy as np
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from scene_classifier import SceneClassifier


def content_region(mv_path, margin=0.05):
    """
    Find the vertical region with most meaningful motion.
    Returns (y_top_pct, y_bottom_pct) as fractions of frame height.
    """
    d = np.load(mv_path, allow_pickle=True)
    mag = d["mag"]  # (T, grid_h, grid_w)
    
    # Average motion per grid row over entire video
    row_motion = mag.mean(axis=(0, 2))  # (grid_h,)
    
    # Smooth
    kernel = np.ones(3) / 3
    row_motion_smooth = np.convolve(row_motion, kernel, mode='same')
    
    # Find the band containing 80% of total motion
    total = row_motion_smooth.sum()
    cumsum = np.cumsum(row_motion_smooth)
    
    # Top boundary: first row where cumulative > 10% of total
    top_idx = np.searchsorted(cumsum, total * 0.10)
    # Bottom boundary: first row where cumulative > 90% of total
    bot_idx = np.searchsorted(cumsum, total * 0.90)
    
    # Convert to fractions
    h = len(row_motion_smooth)
    top_frac = max(0, top_idx / h - margin)
    bot_frac = min(1, bot_idx / h + margin)
    
    return top_frac, bot_frac, row_motion_smooth


if __name__ == "__main__":
    path = sys.argv[1] if len(sys.argv) > 1 else "直播带货样本 6_pipeline/mv.npz"
    top, bot, row_motion = content_region(path)
    
    print(f"Content region: {top*100:.0f}% – {bot*100:.0f}% of frame height")
    print(f"Crop: top {top*100:.0f}%, bottom {(1-bot)*100:.0f}%")
    
    # Visualize
    for i, v in enumerate(row_motion):
        bar = "█" * int(v / row_motion.max() * 40)
        marker = " ←" if i == int(top * len(row_motion)) or i == int(bot * len(row_motion)) else ""
        print(f"  row {i:2d}: {bar}{marker}")
