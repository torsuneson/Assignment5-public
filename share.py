'''
point cloud data is stored as a 2D matrix
each row has 3 values i.e. the x, y, z value for a point

Project has to be submitted to github in the private folder assigned to you
Readme file should have the numerical values as described in each task
Create a folder to store the images as described in the tasks.

Try to create commits and version for each task.

'''
#%%
import matplotlib
import numpy as np
from scipy.spatial import KDTree
from sklearn.cluster import DBSCAN
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

from pathlib import Path
from sklearn.neighbors import NearestNeighbors

import plotly.graph_objects as go

INTERACTIVE_SHOW = False  # True => show_cloud/show_scatter visas
MAKE_HTML_REPORT = True    # True => skapar images/report.html



#%% utility functions
def show_cloud(points_plt):
    ax = plt.axes(projection='3d')
    ax.scatter(points_plt[:,0], points_plt[:,1], points_plt[:,2], s=0.01)
    plt.show()

def show_scatter(x,y):
    plt.scatter(x, y)
    plt.show()

import base64
from pathlib import Path

def _img_to_base64(png_path: Path) -> str:
    data = png_path.read_bytes()
    return base64.b64encode(data).decode("utf-8")

def plotly_3d_div(points_xyz: np.ndarray, title: str, max_points: int = 20000) -> str:
    """
    Returnerar en HTML <div> med en interaktiv 3D scatter (Plotly).
    max_points: decimerar om dataset är stort så HTML inte blir enorm.
    """
    P = points_xyz
    if P.shape[0] > max_points:
        step = max(1, P.shape[0] // max_points)
        P = P[::step]

    fig = go.Figure(
        data=[go.Scatter3d(
            x=P[:,0], y=P[:,1], z=P[:,2],
            mode="markers",
            marker=dict(size=2, opacity=0.6)
        )]
    )
    fig.update_layout(
        title=title,
        margin=dict(l=0, r=0, b=0, t=40),
        scene=dict(
            xaxis_title="x", yaxis_title="y", zaxis_title="z",
            aspectmode="data"
        )
    )
    # include_plotlyjs='cdn' => liten HTML, kräver internet för Plotly JS.
    # Om du vill helt offline: include_plotlyjs=True (större fil).
    return fig.to_html(full_html=False, include_plotlyjs="cdn")

def write_html_report(out_path: Path, sections: list):
    """
    sections: list of dicts with keys:
      - title
      - html_3d (str) 
      - images (list of (caption, Path))
      - notes (str) 
    """
    parts = []
    parts.append("<html><head><meta charset='utf-8'><title>LiDAR Report</title>")
    parts.append("""
    <style>
      body { font-family: Arial, sans-serif; margin: 18px; }
      .dataset { margin-bottom: 40px; }
      .grid { display: grid; grid-template-columns: 1fr 1fr; gap: 16px; align-items: start; }
      .card { border: 1px solid #ddd; border-radius: 10px; padding: 12px; }
      img { max-width: 100%; height: auto; border-radius: 8px; }
      .caption { color: #444; font-size: 13px; margin-top: 6px; }
      .note { color: #333; background: #f6f6f6; padding: 10px; border-radius: 8px; }
    </style>
    </head><body>
    <h1>LiDAR Tasks 1–3 Report</h1>
    """)
    for sec in sections:
        parts.append(f"<div class='dataset'><h2>{sec['title']}</h2>")
        if sec.get("notes"):
            parts.append(f"<div class='note'>{sec['notes']}</div><br>")
        if sec.get("html_3d"):
            parts.append("<div class='card'>")
            parts.append(sec["html_3d"])
            parts.append("</div><br>")
        imgs = sec.get("images", [])
        if imgs:
            parts.append("<div class='grid'>")
            for caption, path in imgs:
                b64 = _img_to_base64(path)
                parts.append("<div class='card'>")
                parts.append(f"<img src='data:image/png;base64,{b64}'/>")
                parts.append(f"<div class='caption'>{caption}<br><code>{path.as_posix()}</code></div>")
                parts.append("</div>")
            parts.append("</div>")
        parts.append("</div>")
    parts.append("</body></html>")
    out_path.write_text("\n".join(parts), encoding="utf-8")


#%%  Task 1 
def get_ground_level(pcd, n_bins: int = 256, window_bins: int = 2) -> float:
    """
    Estimat the groundlevel from LiDAR z-values (pcd[:,2]) by np.histogram: 

    - Find histogram-peak at low height, 
    - Refine with median in a small window around the peak (to cancel noise)
    Return a float (z_ground).

    """
    z = np.asarray(pcd[:, 2]).reshape(-1)
    z = z[np.isfinite(z)]
    if z.size == 0:
        raise ValueError("get_ground_level: tomt z-array.")

    counts, edges = np.histogram(z, bins=n_bins)
    peak_bin = int(np.argmax(counts))
    centers = 0.5 * (edges[:-1] + edges[1:])
    z0 = centers[peak_bin]

    # Refine with median in a mall window at the peak-bin
    lo = max(0, peak_bin - window_bins)
    hi = min(len(centers) - 1, peak_bin + window_bins)
    z_lo = edges[lo]
    z_hi = edges[hi + 1]
    mask = (z >= z_lo) & (z <= z_hi)
    if np.any(mask):
        return float(np.median(z[mask]))
    return float(z0)


def save_ground_histogram(z_values, z_ground, out_png):
    counts, edges = np.histogram(z_values, bins=256)
    centers = 0.5*(edges[:-1] + edges[1:])
    plt.figure()
    plt.bar(centers, counts, width=(edges[1]-edges[0])*0.9, alpha=0.6)
    plt.axvline(z_ground, color='r', lw=2, label=f'Ground ≈ {z_ground:.2f} m')
    plt.xlabel("Höjd (z) [m]"); plt.ylabel("Antal punkter"); plt.legend(); plt.tight_layout()
    plt.savefig(out_png, dpi=150)
    plt.close()

#%% Task 2 

def k_distance_curve(points_xy: np.ndarray, k: int):
    """Return the sorted k-distans (distans to k:th nearest kneighbour) for each point."""
    nbrs = NearestNeighbors(n_neighbors=k, n_jobs=-1).fit(points_xy)
    dists, _ = nbrs.kneighbors(points_xy)
    k_dists = np.sort(dists[:, k-1])
    return k_dists

def find_elbow_eps(k_dists: np.ndarray):
    """
    Simple elbow-detection (Similar to Kneedle)
    Each Point with largest orhtogonal deaviation from line between start and end. 
    """  
    
    y = k_dists
    x = np.arange(len(y), dtype=np.float32)
    x0, y0 = x[0], y[0]
    x1, y1 = x[-1], y[-1]
    denom = np.hypot((y1 - y0), (x1 - x0))
    if denom == 0:
        return float(np.median(y)), 0
    distances = np.abs((y1 - y0)*x - (x1 - x0)*y + (x1*y0 - y1*x0)) / denom
    knee_idx = int(np.argmax(distances))
    return float(y[knee_idx]), knee_idx



def save_k_distance_plot(k_dists, eps, knee_idx, out_png):
    plt.figure()
    plt.plot(k_dists, lw=1.5)
    plt.axvline(knee_idx, color='r', ls='--', label=f'knee idx={knee_idx}')
    plt.axhline(eps, color='g', ls='--', label=f'eps≈{eps:.3f}')
    plt.xlabel("Punktindex (sorterad)"); plt.ylabel("k-distans"); plt.legend(); plt.tight_layout()
    plt.savefig(out_png, dpi=150)
    plt.close()


# support function for DBSCAN insted of inline 
def run_dbscan(points_xy: np.ndarray, eps: float, min_samples: int=5):
    db = DBSCAN(eps=eps, min_samples=min_samples, n_jobs=-1)
    labels = db.fit_predict(points_xy)
    return labels


def save_cluster_scatter_xy(points_xy, labels, title, out_png):
    labs = np.asarray(labels)
    uniq = np.unique(labs)
    # Skapa en färgkarta som rymmer alla kluster
    colors = [plt.cm.tab20(i % 20) for i in range(max(1, len(uniq)))]
    cmap = matplotlib.colors.ListedColormap(colors)
    plt.figure(figsize=(7,7))
    plt.scatter(points_xy[:,0], points_xy[:,1], c=labs, s=1.5, cmap=cmap)
    plt.title(title)
    plt.axis('equal'); plt.tight_layout()
    plt.savefig(out_png, dpi=150); 
    plt.close()

#%% Task 3 
def get_largest_cluster_by_xy_span(points_xy: np.ndarray, labels: np.ndarray):
    """
    Return (label, mask) for the largest non-noise-cluster according to XY-area ( (xmax-xmin)*(ymax-ymin) ).
    """

    lbls = np.asarray(labels)
    unique = np.unique(lbls)
    unique = unique[unique != -1]  # ignorera brus

    if unique.size == 0:
        raise ValueError("No clusters except noise was found.")

    best_label, best_area = None, -1.0
    for lab in unique:
        mask = (lbls == lab)
        xy = points_xy[mask]
        if xy.size == 0:
            continue
        xmin, ymin = xy.min(axis=0)
        xmax, ymax = xy.max(axis=0)
        area = float((xmax - xmin) * (ymax - ymin))
        if area > best_area:
            best_area = area
            best_label = int(lab)
    return best_label, (lbls == best_label)

def save_largest_cluster_plot(points_xy, target_mask, title, out_png):
    plt.figure(figsize=(7,7))
    plt.scatter(points_xy[~target_mask,0], points_xy[~target_mask,1], c='lightgray', s=1, alpha=0.25, label="others")
    plt.scatter(points_xy[target_mask,0], points_xy[target_mask,1], c='crimson', s=2, alpha=0.9, label="largest")
    plt.title(title); plt.axis('equal'); plt.legend(); plt.tight_layout()
    plt.savefig(out_png, dpi=150); plt.close()

#%% read & process datasets
# Create folder for images and HTML 
out_dir = Path("images")
out_dir.mkdir(exist_ok=True)


# This was messing up the code and I don't use it but for save external windowplot in Jupyter. 
# Works with clean .py execution. ( My way of developing )
try:
    get_ipython().run_line_magic('matplotlib', 'qt')
except Exception:
    pass

def process_dataset(npy_path: str, min_samples: int = 5):
    if not Path(npy_path).exists():
        print(f"[INFO] Could not find {npy_path} – skipping.")
        return None

    name = Path(npy_path).stem
    print(f"\n=== Processing {name} ===")
    pcd = np.load(npy_path)
    print("Shape:", pcd.shape)

    # --- Task 1 ---
    z_ground = get_ground_level(pcd)
    print(f"[{name}] Ground level ≈ {z_ground:.3f}")
    hist_png = out_dir / f"{name}_hist_ground.png"
    save_ground_histogram(pcd[:,2], z_ground, hist_png)

    pcd_above_ground = pcd[pcd[:,2] > z_ground]
    print(f"[{name}] Points above ground:", pcd_above_ground.shape[0])

    if INTERACTIVE_SHOW:
        # NOTE: can get heavy if there are a lot of points 
        show_cloud(pcd_above_ground[::10])  # show above ground to be faster. 

    # --- Task 2 ---
    XY = pcd_above_ground[:, :2].astype(np.float32)
    k_dists = k_distance_curve(XY, k=min_samples)
    eps, knee_idx = find_elbow_eps(k_dists)
    print(f"[{name}] Chosen eps ≈ {eps:.4f} (knee idx={knee_idx})")

    elbow_png = out_dir / f"{name}_kdist_elbow.png"
    save_k_distance_plot(k_dists, eps, knee_idx, elbow_png)

    labels = run_dbscan(XY, eps=eps, min_samples=min_samples)
    n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
    print(f"[{name}] DBSCAN clusters (excl. noise): {n_clusters}")

    cluster_png = out_dir / f"{name}_dbscan_clusters.png"
    save_cluster_scatter_xy(XY, labels,
                            f"DBSCAN ({name}) eps≈{eps:.3f}, clusters={n_clusters}",
                            cluster_png)

    if INTERACTIVE_SHOW:
        # Quick sanity check (from original code)
        show_scatter(XY[::10,0], XY[::10,1])

    # --- Task 3 ---
    notes = []
    largest_png = None
    html_3d = None

    try:
        largest_label, mask = get_largest_cluster_by_xy_span(XY, labels)
        xy_sel = XY[mask]
        xmin, ymin = xy_sel.min(axis=0)
        xmax, ymax = xy_sel.max(axis=0)

        notes.append(f"<b>Ground level</b>: {z_ground:.3f}")
        notes.append(f"<b>Optimal eps</b>: {eps:.4f} (knee idx={knee_idx})")
        notes.append(f"<b>Clusters (excl. noise)</b>: {n_clusters}")
        notes.append(f"<b>Catenary cluster</b> label={largest_label}")
        notes.append(f"<b>BBox</b>: min(x)={xmin:.3f}, min(y)={ymin:.3f}, max(x)={xmax:.3f}, max(y)={ymax:.3f}")

        largest_png = out_dir / f"{name}_largest_cluster.png"
        save_largest_cluster_plot(XY, mask, f"{name}: Largest cluster (catenary?)", largest_png)

        # Interactiv 3D: shows the larget cluster in 3D (use original z) 
        if MAKE_HTML_REPORT:
            # Fetch 3D-points that match the mask (mask from pcd_above_ground)
            pcd_cat = pcd_above_ground[mask]
            html_3d = plotly_3d_div(pcd_cat, title=f"{name}: Largest cluster (3D)")

    except ValueError as e:
        notes.append(str(e))

    section = {
        "title": f"{name}",
        "notes": "<br>".join(notes),
        "html_3d": html_3d,
        "images": [
            ("Task 1: Ground histogram", hist_png),
            ("Task 2: k-distance elbow", elbow_png),
            ("Task 2: DBSCAN clusters (XY)", cluster_png),
        ] + ([("Task 3: Largest cluster (XY)", largest_png)] if largest_png else [])
    }
    return section

# Ryn for both dataset (or more just add) that was required. 
report_sections = []
for npy in ["dataset1.npy", "dataset2.npy"]:
    sec = process_dataset(npy_path=npy, min_samples=5)
    if sec is not None:
        report_sections.append(sec)

if MAKE_HTML_REPORT and report_sections:
    report_path = out_dir / "report.html"
    write_html_report(report_path, report_sections)
    print(f"[OK] HTML report saved to: {report_path}")
