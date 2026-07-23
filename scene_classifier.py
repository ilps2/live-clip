"""
scene_classifier.py — reusable motion signature scene classifier.
API:
    sc = SceneClassifier(mv_npz_path)
    sc.classify()          → per-frame scene labels
    sc.per_second()        → per-second dominant scene
    sc.boundaries()        → list of timestamps where scene changes
    sc.weights()           → dict of per-second interest weights
    sc.best_broll(t0,t1)   → best product-demo window in [t0,t1]
"""
import numpy as np
import pandas as pd
from pathlib import Path
import sys

# Allow import from motion-signature/
sys.path.insert(0, str(Path(__file__).parent.parent))
from fast_features import batch_features, FEATURES


class SceneClassifier:
    def __init__(self, mv_path):
        d = np.load(mv_path, allow_pickle=True)
        self.pts = d["pts"]  # seconds
        self.dur = self.pts.max()
        feats = batch_features(d["mag"], d["dirx"], d["diry"])
        self.df = pd.DataFrame(feats, columns=FEATURES)
        self.df["t"] = self.pts
        self._classified = False
        self._sec_scene = None
        self._boundaries = None

    def classify(self, orientation="vertical"):
        """
        Classify each frame into scene type.
        orientation: 'vertical' (phone screen recording) or 'horizontal' (stage/presentation)
        """
        df = self.df
        med = df["global_mag"].median()

        if orientation == "vertical":
            # Vertical: product always near center. Use center/edge ratio + spread.
            cond_cam = (df["dir_consist"] > 0.6) & (df["move_ratio"] > 0.2)
            cond_face = (df["center_edge"] > 1.5) & (df["spread"] < 0.25) & \
                        (df["global_mag"] > med * 0.2) & ~cond_cam
            cond_static = (df["global_mag"] < med * 0.2) & (df["move_ratio"] < 0.04)
            # Product demo = center motion is much higher than edges, moderate spread
            cond_product = (df["center_edge"] > 3.0) & (df["spread"] < 0.2) & \
                           (df["global_mag"] > med) & ~cond_cam & ~cond_face
        else:
            # Horizontal: stage/presentation. Camera pans common.
            cond_cam = (df["dir_consist"] > 0.6) & (df["move_ratio"] > 0.25)
            cond_face = (df["center_edge"] > 1.3) & (df["spread"] < 0.3) & \
                        (df["global_mag"] > med * 0.2) & ~cond_cam
            cond_static = (df["global_mag"] < med * 0.25) & (df["move_ratio"] < 0.04)
            cond_product = (df["center_edge"] > 2.0) & (df["spread"] < 0.2) & \
                           (df["global_mag"] > med) & ~cond_cam & ~cond_face

        df["scene"] = "mixed"
        df.loc[cond_cam, "scene"] = "camera"
        df.loc[cond_face, "scene"] = "face"
        df.loc[cond_static, "scene"] = "static"
        df.loc[cond_product, "scene"] = "product"

        self._classified = True
        return df

    def per_second(self):
        """Aggregate to per-second dominant scene."""
        if not self._classified:
            self.classify()
        df = self.df.copy()
        df["sec"] = df["t"].astype(int)
        self._sec_scene = df.groupby("sec")["scene"].agg(
            lambda x: x.mode().iloc[0] if len(x.mode()) > 0 else "mixed"
        )
        return self._sec_scene

    def boundaries(self, min_gap=2):
        """Find timestamps where dominant scene changes."""
        ss = self.per_second()
        bounds = []
        prev = None
        prev_t = None
        for sec in sorted(ss.index):
            curr = ss[sec]
            if prev and curr != prev:
                if not bounds or sec - bounds[-1] >= min_gap:
                    bounds.append(sec)
            prev = curr
        self._boundaries = bounds
        return bounds

    def weights(self, format_type="livestream"):
        """Per-second interest weights based on scene type + format."""
        if format_type == "livestream":
            wmap = {"product": 2.0, "face": 1.2, "mixed": 0.8, "static": 0.3, "camera": 0.3}
        elif format_type == "launch":
            wmap = {"face": 1.5, "product": 1.5, "static": 0.8, "mixed": 0.6, "camera": 0.4}
        else:
            wmap = {"product": 1.5, "face": 1.0, "mixed": 0.7, "static": 0.4, "camera": 0.4}

        ss = self.per_second()
        max_sec = int(self.dur)
        weights = np.zeros(max_sec + 1)
        for sec in range(max_sec + 1):
            scene = ss.get(sec, "mixed")
            weights[sec] = wmap.get(scene, 0.6)
        return weights

    def best_broll(self, t0, t1, min_dur=2.0):
        """Find best product-demo window within [t0, t1] for B-roll overlay."""
        if not self._classified:
            self.classify()
        df = self.df
        mask = (df["t"] >= t0) & (df["t"] <= t1)
        sub = df[mask].copy()
        if len(sub) == 0:
            return (t0, t0 + min_dur)

        # Score = center_edge × center_mag → high = product in center with motion
        sub["broll_score"] = sub["center_edge"] * sub["center_mag"]
        # Find contiguous high-scoring windows
        threshold = sub["broll_score"].quantile(0.7)
        windows = []
        in_win = False
        ws = 0
        for _, row in sub.iterrows():
            if row["broll_score"] > threshold and not in_win:
                ws = row["t"]
                in_win = True
            elif row["broll_score"] <= threshold and in_win:
                if row["t"] - ws >= min_dur:
                    windows.append((ws, row["t"]))
                in_win = False
        if in_win and sub["t"].iloc[-1] - ws >= min_dur:
            windows.append((ws, sub["t"].iloc[-1]))

        if windows:
            # Return the longest window
            return max(windows, key=lambda w: w[1] - w[0])
        return (t0, t0 + min_dur)

    def nearest_visual_boundary(self, t, max_dist=5):
        """Find nearest visual boundary to timestamp t."""
        bounds = self.boundaries()
        best = t
        best_d = max_dist + 1
        for b in bounds:
            d = abs(b - t)
            if d < best_d and d <= max_dist:
                best = b
                best_d = d
        return best


# Quick CLI test
if __name__ == "__main__":
    import sys
    path = sys.argv[1] if len(sys.argv) > 1 else "real_fusion/sample5/mv.npz"
    sc = SceneClassifier(path)
    sc.classify()
    ss = sc.per_second()
    bounds = sc.boundaries()
    w = sc.weights()

    print(f"Duration: {sc.dur:.0f}s")
    print(f"Scenes: {dict(ss.value_counts())}")
    print(f"Boundaries: {len(bounds)}")
    print(f"Weight range: {w.min():.2f} - {w.max():.2f}")
