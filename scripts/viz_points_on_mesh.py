#!/usr/bin/env python
import argparse
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection


def _colors_from_values(values):
    if values is None:
        return None
    if values.size == 0:
        return None
    v = values.astype(np.float32)
    vmin = float(v.min())
    vmax = float(v.max())
    if vmax > vmin:
        v = (v - vmin) / (vmax - vmin)
    else:
        v = np.zeros_like(v)
    try:
        import matplotlib.cm as cm
        colors = cm.viridis(v)[:, :3]
    except Exception:
        colors = np.stack([v, 1.0 - v, np.zeros_like(v)], axis=1)
    return colors


def _save_matplotlib(verts, faces, points, colors, output_path, elev, azim, point_size):
    fig = plt.figure(figsize=(8, 8))
    ax = fig.add_subplot(111, projection="3d")
    ax.set_box_aspect([1, 1, 1])

    mesh = Poly3DCollection(verts[faces], linewidths=0.05, alpha=0.35)
    mesh.set_facecolor((0.7, 0.7, 0.7, 0.35))
    mesh.set_edgecolor((0.3, 0.3, 0.3, 0.1))
    ax.add_collection3d(mesh)

    if points is not None and points.size:
        if colors is None:
            colors = np.tile(np.array([0.1, 0.9, 0.2], dtype=np.float32), (points.shape[0], 1))
        ax.scatter(points[:, 0], points[:, 1], points[:, 2], s=point_size, c=colors)

    min_xyz = verts.min(axis=0)
    max_xyz = verts.max(axis=0)
    center = (min_xyz + max_xyz) / 2.0
    extent = (max_xyz - min_xyz).max() / 2.0
    ax.set_xlim(center[0] - extent, center[0] + extent)
    ax.set_ylim(center[1] - extent, center[1] + extent)
    ax.set_zlim(center[2] - extent, center[2] + extent)
    ax.view_init(elev=elev, azim=azim)
    ax.axis("off")

    fig.tight_layout(pad=0)
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mesh-obj", required=True)
    parser.add_argument("--points-npz", required=True)
    parser.add_argument("--points-key", default=None)
    parser.add_argument("--intensity-key", default=None)
    parser.add_argument("--output", required=True, help="Output PNG path")
    parser.add_argument("--elev", type=float, default=20.0)
    parser.add_argument("--azim", type=float, default=35.0)
    parser.add_argument("--point-size", type=float, default=6.0)
    args = parser.parse_args()

    npz = np.load(args.points_npz, allow_pickle=True)
    if args.points_key is None:
        if "points" in npz:
            points_key = "points"
        elif "target_points" in npz:
            points_key = "target_points"
        elif "source_contact_points" in npz:
            points_key = "source_contact_points"
        else:
            raise ValueError("No points key found in npz. Use --points-key.")
    else:
        points_key = args.points_key

    points = npz[points_key].astype(np.float32)
    intensity = None
    if args.intensity_key is not None and args.intensity_key in npz:
        intensity = npz[args.intensity_key].astype(np.float32)

    import trimesh
    mesh = trimesh.load(args.mesh_obj, process=False)
    verts = np.asarray(mesh.vertices, dtype=np.float32)
    faces = np.asarray(mesh.faces, dtype=np.int64)

    colors = _colors_from_values(intensity)

    output_path = os.path.expanduser(args.output)
    _save_matplotlib(verts, faces, points, colors, output_path, args.elev, args.azim, args.point_size)


if __name__ == "__main__":
    main()
