#!/usr/bin/env python
import argparse
import os
import sys
import numpy as np

import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

from densematcher.model import MeshFeaturizer
from densematcher.functional_map import compute_surface_map
from densematcher.utils import load_pytorch3d_mesh, recenter, get_groups_dmtx, get_uniform_SO3_RT, get_colors
from densematcher.pyFM.mesh.geometry import heat_geodmat_robust
from densematcher import diffusion_net
from densematcher.diffusion_net.utils import random_rotation_matrix


def get_mesh(instance, num_views=(1, 3), random_rotation=True, use_color_mesh=False):
    mesh_color = load_pytorch3d_mesh(f"{instance}/color_mesh.obj")
    mesh_simp = load_pytorch3d_mesh(f"{instance}/simple_mesh.obj")
    if use_color_mesh:
        mesh_simp = mesh_color

    def _remove_unreferenced_vertices(mesh):
        faces = mesh.faces_packed()
        verts = mesh.verts_packed()
        used = torch.unique(faces)
        if used.numel() == verts.shape[0]:
            return mesh, None
        used, _ = torch.sort(used)
        idx_map = torch.full((verts.shape[0],), -1, dtype=torch.long, device=verts.device)
        idx_map[used] = torch.arange(used.shape[0], device=verts.device)
        new_verts = verts[used]
        new_faces = idx_map[faces]

        new_textures = None
        textures = mesh.textures
        if textures is not None:
            try:
                from pytorch3d.renderer.mesh.textures import TexturesVertex
                if hasattr(textures, "verts_features_list"):
                    feats = textures.verts_features_list()[0]
                elif hasattr(textures, "verts_features_packed"):
                    feats = textures.verts_features_packed()
                else:
                    feats = None
                if feats is not None and feats.shape[0] == verts.shape[0]:
                    new_textures = TexturesVertex(verts_features=[feats[used]])
            except Exception:
                new_textures = None

        from pytorch3d.structures import Meshes
        return Meshes(verts=[new_verts], faces=[new_faces], textures=new_textures), idx_map

    groups = []
    groups_path = os.path.join(instance, "groups.txt")
    if os.path.exists(groups_path):
        with open(groups_path) as f:
            for line in f:
                line = line.strip()
                if line:
                    groups.append(list(map(int, line.split())))

    if not use_color_mesh:
        mesh_simp, remap = _remove_unreferenced_vertices(mesh_simp)
        if remap is not None and groups:
            remapped_groups = []
            for group in groups:
                new_group = []
                for g in group:
                    new_g = remap[g].item()
                    if new_g >= 0:
                        new_group.append(new_g)
                if new_group:
                    remapped_groups.append(new_group)
            groups = remapped_groups

    if mesh_simp.textures is None:
        from pytorch3d.renderer.mesh.textures import TexturesVertex
        verts = mesh_simp.verts_packed()
        fallback_rgb = torch.full((verts.shape[0], 3), 0.7, device=verts.device, dtype=verts.dtype)
        mesh_simp.textures = TexturesVertex(verts_features=[fallback_rgb])

    geodesic_distance = heat_geodmat_robust(mesh_simp.verts_packed().numpy(), mesh_simp.faces_packed().numpy())
    groups_dmtx = get_groups_dmtx(geodesic_distance, groups)

    bbox = mesh_simp.get_bounding_boxes()
    center = bbox.mean(2)
    recenter(mesh_color, mesh_simp)

    bb = mesh_color.get_bounding_boxes()
    cam_dist = bb.abs().max() * (np.random.rand() * 0.5 + 2.0)

    operators = diffusion_net.geometry.get_operators(
        mesh_simp.verts_list()[0].cpu(),
        mesh_simp.faces_list()[0].cpu(),
        k_eig=128,
        op_cache_dir=os.environ.get("OP_CACHE_DIR", None),
        normals=mesh_simp.verts_normals_list()[0],
    )
    frames, mass, L, evals, evecs, gradX, gradY = operators
    operators = (frames, mass, L.to_dense(), evals, evecs, gradX.to_dense(), gradY.to_dense())

    if random_rotation:
        R_inv = torch.from_numpy(random_rotation_matrix()).type_as(mesh_simp.verts_packed())
    else:
        R_inv = torch.eye(3).to(frames)
    new_verts_color = torch.matmul(mesh_color.verts_padded(), R_inv)
    new_verts_simp = torch.matmul(mesh_simp.verts_padded(), R_inv)
    mesh_color = mesh_color.update_padded(new_verts_color)
    mesh_simp = mesh_simp.update_padded(new_verts_simp)

    Rs, ts, _, _ = get_uniform_SO3_RT(num_azimuth=num_views[0], num_elevation=num_views[1], distance=cam_dist, center=bb.mean(2))
    cameras = [Rs, ts]
    return mesh_color, mesh_simp, groups, groups_dmtx, operators, cameras, geodesic_distance, center, R_inv


def build_model(device, imsize=384, width=512, num_blocks=8, aggre_net_weights_folder=None):
    if aggre_net_weights_folder is None:
        aggre_net_weights_folder = "checkpoints/SDDINO_weights"
    model = MeshFeaturizer(
        f"checkpoints/featup_imsize={imsize}_channelnorm=False_unitnorm=False_rotinv=True/final.ckpt",
        (3, 1),
        num_blocks,
        width,
        aggre_net_weights_folder=aggre_net_weights_folder,
    )
    ckpt_file = f"checkpoints/exp_mvmatcher_imsize={imsize}_width={width}_nviews=3x1_wrecon=10.0_cutprob=0.5_blocks={num_blocks}_release_jitter=0.0/final.ckpt"
    ckpt = torch.load(ckpt_file, map_location="cpu")
    state_dict = {}
    for key in ckpt["state_dict"].keys():
        if key.startswith("model.extractor_3d"):
            state_dict[key.removeprefix("model.extractor_3d.")] = ckpt["state_dict"][key]
    model.extractor_3d.load_state_dict(state_dict)

    model.to(device)
    if device.startswith("cuda"):
        model = model.half()
        model.extractor_2d.featurizer.mem_eff = True
    return model


def _nearest_vertices(points, verts):
    try:
        from scipy.spatial import cKDTree
        tree = cKDTree(verts)
        _, idx = tree.query(points, k=1)
        return idx
    except Exception:
        d2 = ((points[:, None, :] - verts[None, :, :]) ** 2).sum(axis=2)
        return d2.argmin(axis=1)

def _save_map_png(mesh1, mesh2, cmap1, cmap2, output_path, elev, azim):
    fig = plt.figure(figsize=(12, 6))
    ax1 = fig.add_subplot(1, 2, 1, projection="3d")
    ax2 = fig.add_subplot(1, 2, 2, projection="3d")

    def _face_colors(verts, faces, colors):
        # Average vertex colors per face.
        return colors[faces].mean(axis=1)

    def _draw(ax, verts, faces, colors):
        ax.set_box_aspect([1, 1, 1])
        face_colors = _face_colors(verts, faces, colors)
        mesh = Poly3DCollection(verts[faces], linewidths=0.05, alpha=0.9)
        mesh.set_facecolor(face_colors)
        mesh.set_edgecolor((0.2, 0.2, 0.2, 0.05))
        ax.add_collection3d(mesh)
        min_xyz = verts.min(axis=0)
        max_xyz = verts.max(axis=0)
        center = (min_xyz + max_xyz) / 2.0
        extent = (max_xyz - min_xyz).max() / 2.0
        ax.set_xlim(center[0] - extent, center[0] + extent)
        ax.set_ylim(center[1] - extent, center[1] + extent)
        ax.set_zlim(center[2] - extent, center[2] + extent)
        ax.view_init(elev=elev, azim=azim)
        ax.axis("off")

    verts1 = mesh1.verts_list()[0].detach().cpu().numpy()
    faces1 = mesh1.faces_list()[0].detach().cpu().numpy()
    verts2 = mesh2.verts_list()[0].detach().cpu().numpy()
    faces2 = mesh2.faces_list()[0].detach().cpu().numpy()

    _draw(ax1, verts1, faces1, cmap1)
    _draw(ax2, verts2, faces2, cmap2)

    fig.tight_layout(pad=0)
    fig.savefig(output_path, dpi=200)
    plt.close(fig)

def _save_contacts_png(mesh1, mesh2, src_points, tgt_points, intensity, output_path, elev, azim, point_size):
    fig = plt.figure(figsize=(12, 6))
    ax1 = fig.add_subplot(1, 2, 1, projection="3d")
    ax2 = fig.add_subplot(1, 2, 2, projection="3d")

    def _draw_mesh(ax, verts, faces):
        ax.set_box_aspect([1, 1, 1])
        mesh = Poly3DCollection(verts[faces], linewidths=0.05, alpha=0.2)
        mesh.set_facecolor((0.7, 0.7, 0.7, 0.2))
        mesh.set_edgecolor((0.2, 0.2, 0.2, 0.05))
        ax.add_collection3d(mesh)
        min_xyz = verts.min(axis=0)
        max_xyz = verts.max(axis=0)
        center = (min_xyz + max_xyz) / 2.0
        extent = (max_xyz - min_xyz).max() / 2.0
        ax.set_xlim(center[0] - extent, center[0] + extent)
        ax.set_ylim(center[1] - extent, center[1] + extent)
        ax.set_zlim(center[2] - extent, center[2] + extent)
        ax.view_init(elev=elev, azim=azim)
        ax.axis("off")

    verts1 = mesh1.verts_list()[0].detach().cpu().numpy()
    faces1 = mesh1.faces_list()[0].detach().cpu().numpy()
    verts2 = mesh2.verts_list()[0].detach().cpu().numpy()
    faces2 = mesh2.faces_list()[0].detach().cpu().numpy()

    _draw_mesh(ax1, verts1, faces1)
    _draw_mesh(ax2, verts2, faces2)

    if intensity is not None and intensity.size:
        norm = (intensity - intensity.min()) / (intensity.max() - intensity.min() + 1e-8)
        colors = plt.cm.viridis(norm)
    else:
        colors = None

    if src_points is not None and src_points.size:
        ax1.scatter(src_points[:, 0], src_points[:, 1], src_points[:, 2], s=point_size, c=colors)
    if tgt_points is not None and tgt_points.size:
        ax2.scatter(tgt_points[:, 0], tgt_points[:, 1], tgt_points[:, 2], s=point_size, c=colors)

    fig.tight_layout(pad=0)
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--contact-npz", required=True)
    parser.add_argument("--source-asset", required=True, help="Path to converted_new asset folder")
    parser.add_argument("--target-asset", required=True, help="Path to converted_new asset folder")
    parser.add_argument("--frame", type=int, default=0)
    parser.add_argument(
        "--frame-range",
        nargs=2,
        type=int,
        metavar=("START", "END"),
        default=None,
        help="Optional inclusive frame range to process (overrides --frame).",
    )
    parser.add_argument("--output", default=None)
    parser.add_argument(
        "--output-dir",
        default=None,
        help="When using --frame-range, write one NPZ per frame into this directory.",
    )
    parser.add_argument("--device", default=None)
    parser.add_argument("--seed", type=int, default=None, help="Optional seed for deterministic camera sampling.")
    parser.add_argument(
        "--map-method",
        choices=["inverse", "inverse_icp", "hungarian", "hungarian_icp"],
        default="hungarian",
        help="Mapping method to match example_new.ipynb (method 5 uses hungarian).",
    )
    parser.add_argument("--map-png", default=None, help="Optional output PNG path for vertex map.")
    parser.add_argument("--map-png-dir", default=None, help="Optional output directory for multi-angle vertex map PNGs.")
    parser.add_argument("--map-elev", type=float, default=20.0)
    parser.add_argument("--map-azim", type=float, default=35.0)
    parser.add_argument("--map-angles", type=int, default=0, help="If >0, render N angles around azimuth.")
    parser.add_argument("--contacts-png", default=None, help="Optional output PNG path for contact points.")
    parser.add_argument("--contacts-png-dir", default=None, help="Optional output directory for multi-angle contact PNGs.")
    parser.add_argument("--contacts-elev", type=float, default=20.0)
    parser.add_argument("--contacts-azim", type=float, default=35.0)
    parser.add_argument("--contacts-point-size", type=float, default=8.0)
    parser.add_argument("--contacts-angles", type=int, default=0, help="If >0, render N angles around azimuth.")
    parser.add_argument(
        "--frame-is-valid-index",
        action="store_true",
        help="Interpret --frame as index into object_frame_has_pose==1 frames.",
    )
    parser.add_argument(
        "--use-color-mesh",
        action="store_true",
        help="Use color_mesh.obj vertices instead of simple_mesh.obj for correspondence.",
    )
    parser.add_argument(
        "--no-contact-auto-scale",
        action="store_true",
        help="Disable auto-scaling contact points to match source mesh bbox extent.",
    )
    parser.add_argument(
        "--aggre-weights",
        default=None,
        help="Path to AggreNet weights folder (expects best_*.PTH inside).",
    )
    parser.add_argument(
        "--save-correspondence",
        default=None,
        help="Optional path to save a cached source->target vertex correspondence NPZ.",
    )
    parser.add_argument(
        "--load-correspondence",
        default=None,
        help="Optional path to a cached source->target vertex correspondence NPZ (skips model+FM solve).",
    )
    args = parser.parse_args()

    os.environ["INFERENCE"] = "1"

    if args.seed is not None:
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)

    if args.device is None:
        device = "cuda:0" if torch.cuda.is_available() else "cpu"
    else:
        device = args.device

    npz = np.load(args.contact_npz, allow_pickle=True)
    points = npz["object_points_local"].astype(np.float32)

    if "object_contact_binary" not in npz:
        raise ValueError("contact map missing object_contact_binary")
    contact_binary = npz["object_contact_binary"]
    if contact_binary.ndim != 2:
        raise ValueError(f"Expected object_contact_binary to have shape (F,P); got {contact_binary.shape}")
    if points.ndim != 2 or points.shape[1] != 3:
        raise ValueError(f"Expected object_points_local to have shape (P,3); got {points.shape}")
    if contact_binary.shape[1] != points.shape[0]:
        raise ValueError(
            f"object_contact_binary second dim ({contact_binary.shape[1]}) must match object_points_local ({points.shape[0]})."
        )

    contact_intensity_all = npz.get("object_contact_intensity", None)
    if contact_intensity_all is not None and contact_intensity_all.ndim != 2:
        raise ValueError(f"Expected object_contact_intensity to have shape (F,P); got {contact_intensity_all.shape}")
    if contact_intensity_all is not None and contact_intensity_all.shape != contact_binary.shape:
        raise ValueError(
            f"object_contact_intensity shape {contact_intensity_all.shape} must match object_contact_binary {contact_binary.shape}."
        )

    if args.frame_range is not None:
        start, end = args.frame_range
        if end < start:
            raise ValueError("--frame-range END must be >= START")
        requested_frames = list(range(start, end + 1))
        if args.output_dir is None:
            raise ValueError("--output-dir is required when using --frame-range")
        if args.map_png is not None or args.map_png_dir is not None or args.contacts_png is not None or args.contacts_png_dir is not None:
            raise ValueError("PNG outputs are only supported for single-frame runs (omit --frame-range).")
    else:
        requested_frames = [args.frame]

    def _to_frame_idx(requested_frame):
        frame_idx = int(requested_frame)
        if args.frame_is_valid_index and "object_frame_has_pose" in npz:
            valid = npz["object_frame_has_pose"].astype(bool)
            valid_idx = np.where(valid)[0]
            if frame_idx < 0 or frame_idx >= valid_idx.shape[0]:
                raise ValueError(f"--frame {requested_frame} is out of range for valid frames ({valid_idx.shape[0]}).")
            frame_idx = int(valid_idx[frame_idx])
        if frame_idx < 0 or frame_idx >= contact_binary.shape[0]:
            raise ValueError(f"frame_idx {frame_idx} out of range for object_contact_binary ({contact_binary.shape[0]}).")
        return frame_idx

    (
        source_dirty_mesh,
        source_clean_mesh,
        _,
        _,
        operators1,
        cameras1,
        _,
        source_center,
        source_R_inv,
    ) = get_mesh(args.source_asset, random_rotation=False, use_color_mesh=args.use_color_mesh)
    (
        target_dirty_mesh,
        target_clean_mesh,
        _,
        _,
        operators2,
        cameras2,
        _,
        target_center,
        target_R_inv,
    ) = get_mesh(args.target_asset, random_rotation=False, use_color_mesh=args.use_color_mesh)

    source_center_np = source_center.detach().cpu().numpy().reshape(1, 3)
    source_R_np = source_R_inv.detach().cpu().numpy()
    obj_verts = npz.get("object_mesh_vertices_local", None)
    if obj_verts is None:
        raise ValueError("contact map missing object_mesh_vertices_local")
    obj_verts = obj_verts.astype(np.float32)
    obj_center = (obj_verts.max(axis=0) + obj_verts.min(axis=0)) / 2.0
    if args.use_color_mesh:
        center_np = source_center_np
    else:
        center_np = obj_center.reshape(1, 3)

    points_src = points - center_np
    if not args.no_contact_auto_scale:
        src_verts_tmp = source_clean_mesh.verts_list()[0].detach().cpu().numpy()
        mesh_extent = (src_verts_tmp.max(axis=0) - src_verts_tmp.min(axis=0)).max()
        obj_extent = (obj_verts.max(axis=0) - obj_verts.min(axis=0)).max()
        if obj_extent > 0:
            scale = mesh_extent / obj_extent
            points_src = points_src * scale
    points_src = points_src @ source_R_np

    src_verts = source_clean_mesh.verts_list()[0].detach().cpu().numpy()
    tgt_verts = target_clean_mesh.verts_list()[0].detach().cpu().numpy()

    if args.load_correspondence is not None:
        def _npz_scalar_str(arr, default="unknown"):
            if arr is None:
                return default
            try:
                if isinstance(arr, np.ndarray):
                    if arr.shape == ():
                        return str(arr.item())
                    if arr.size == 1:
                        return str(arr.reshape(-1)[0])
                return str(arr)
            except Exception:
                return default

        cached = np.load(os.path.expanduser(args.load_correspondence), allow_pickle=True)
        source_to_target = cached["source_to_target"].astype(np.int64)
        cached_method = _npz_scalar_str(cached.get("map_method", None))
        if cached_method != "unknown" and cached_method != args.map_method:
            print(f"[warn] cached map_method={cached_method} but current --map-method={args.map_method}")
    else:
        model = build_model(device, aggre_net_weights_folder=args.aggre_weights)

        with torch.no_grad():
            if device.startswith("cuda"):
                with torch.autocast("cuda"):
                    f_source = model(source_dirty_mesh, source_clean_mesh, operators1, cameras1)
                    f_target = model(target_dirty_mesh, target_clean_mesh, operators2, cameras2)
            else:
                f_source = model(source_dirty_mesh, source_clean_mesh, operators1, cameras1)
                f_target = model(target_dirty_mesh, target_clean_mesh, operators2, cameras2)

        n_ev = 15
        maxiter = 5000
        fit_params = {
            "w_descr": 1e4,
            "w_lap": 1e3,
            "w_dcomm": 0.0,
            "w_orient": 0.0,
            "w_area": 0.0,
            "w_conformal": 0.0,
            "w_p2p": 0.0,
            "w_stochastic": 0.0,
            "w_ent": 1e-1,
            "w_range01": 0.0,
            "w_sumto1": 1e1,
            "optinit": "zeros",
            "maxiter": maxiter,
        }
        compute_extra = args.map_method == "hungarian"
        (
            surface_map,
            surface_map_inv,
            hungarian,
            _,
            surface_map_icp,
            surface_map_inv_icp,
            hungarian_icp,
            _,
            _,
            _,
            _,
            _,
            _,
            _,
        ) = compute_surface_map(
            source_clean_mesh,
            target_clean_mesh,
            f_source.clone().detach().cpu().numpy(),
            f_target.clone().detach().cpu().numpy(),
            n_ev=n_ev,
            descr_type="neural",
            compute_extra=compute_extra,
            optimizer="L-BFGS-B",
            maxiter=maxiter,
            optimize_p2p=False,
            fit_params=fit_params,
        )

        if args.map_method == "inverse":
            source_to_target = surface_map_inv
        elif args.map_method == "inverse_icp":
            source_to_target = surface_map_inv_icp
        elif args.map_method == "hungarian":
            if hungarian is None:
                raise RuntimeError("Hungarian mapping requested but compute_extra=False.")
            source_to_target = np.full_like(surface_map_inv, -1)
            for t_idx, s_idx in zip(hungarian[0], hungarian[1]):
                source_to_target[s_idx] = t_idx
            missing = source_to_target < 0
            source_to_target[missing] = surface_map_inv[missing]
        elif args.map_method == "hungarian_icp":
            source_to_target = np.full_like(surface_map_inv, -1)
            for t_idx, s_idx in zip(hungarian_icp[0], hungarian_icp[1]):
                source_to_target[s_idx] = t_idx
            missing = source_to_target < 0
            source_to_target[missing] = surface_map_inv_icp[missing]
        else:
            raise ValueError(f"Unknown map method: {args.map_method}")

        if args.save_correspondence is not None:
            out_path = os.path.expanduser(args.save_correspondence)
            os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
            np.savez(
                out_path,
                source_asset=np.array([args.source_asset]),
                target_asset=np.array([args.target_asset]),
                map_method=np.array([args.map_method]),
                source_vertex_count=np.array([src_verts.shape[0]], dtype=np.int64),
                target_vertex_count=np.array([tgt_verts.shape[0]], dtype=np.int64),
                source_to_target=source_to_target.astype(np.int64),
            )
            print("Wrote correspondence cache", out_path)

    if source_to_target.shape[0] != src_verts.shape[0]:
        raise ValueError(
            f"source_to_target length ({source_to_target.shape[0]}) does not match source verts ({src_verts.shape[0]})."
        )
    if source_to_target.min() < 0 or source_to_target.max() >= tgt_verts.shape[0]:
        raise ValueError(
            f"source_to_target has invalid target indices: min={int(source_to_target.min())} max={int(source_to_target.max())} "
            f"(target verts={tgt_verts.shape[0]})."
        )

    all_src_vertex_indices = _nearest_vertices(points_src, src_verts)

    def _process_one(requested_frame):
        frame_idx = _to_frame_idx(requested_frame)
        mask = contact_binary[frame_idx].astype(bool)
        intensity = None
        if contact_intensity_all is not None:
            intensity = contact_intensity_all[frame_idx].astype(np.float32)

        src_points = points_src[mask]
        src_intensity = intensity[mask] if intensity is not None else None

        if intensity is not None:
            count_binary = int(mask.sum())
            count_intensity = int((intensity > 0.0).sum())
            print(f"frame={requested_frame} frame_idx={frame_idx} binary_contacts={count_binary} intensity_gt0={count_intensity}")

        if src_points.size == 0:
            print("No contact points found for frame", requested_frame, "(frame_idx", frame_idx, ")")
            return None

        src_vertex_indices = all_src_vertex_indices[mask]
        tgt_vertex_indices = source_to_target[src_vertex_indices]
        tgt_points = tgt_verts[tgt_vertex_indices]

        return {
            "requested_frame": int(requested_frame),
            "frame_idx": int(frame_idx),
            "src_points": src_points,
            "src_intensity": src_intensity,
            "src_vertex_indices": src_vertex_indices,
            "tgt_vertex_indices": tgt_vertex_indices,
            "tgt_points": tgt_points,
        }

    if args.frame_range is not None:
        out_dir = os.path.expanduser(args.output_dir)
        os.makedirs(out_dir, exist_ok=True)
        wrote = 0
        for f in requested_frames:
            result = _process_one(f)
            if result is None:
                continue
            out_path = os.path.join(out_dir, f"transferred_contacts_frame{result['requested_frame']}.npz")
            np.savez(
                out_path,
                frame=np.array([result["requested_frame"]], dtype=np.int32),
                frame_idx=np.array([result["frame_idx"]], dtype=np.int32),
                source_contact_points=result["src_points"],
                source_contact_intensity=result["src_intensity"],
                source_vertex_indices=result["src_vertex_indices"],
                target_vertex_indices=result["tgt_vertex_indices"],
                target_points=result["tgt_points"],
                source_asset=np.array([args.source_asset]),
                target_asset=np.array([args.target_asset]),
                map_method=np.array([args.map_method]),
            )
            wrote += 1
            print("Wrote", out_path)
        print(f"Done. Wrote {wrote}/{len(requested_frames)} frames.")
        return

    single = _process_one(requested_frames[0])
    if single is None:
        return
    src_points = single["src_points"]
    src_intensity = single["src_intensity"]
    src_vertex_indices = single["src_vertex_indices"]
    tgt_vertex_indices = single["tgt_vertex_indices"]
    tgt_points = single["tgt_points"]

    if args.map_png is not None:
        cmap_source = get_colors(src_verts)
        cmap_target = np.zeros((tgt_verts.shape[0], 3), dtype=cmap_source.dtype)
        if args.map_method == "inverse":
            cmap_target[source_to_target] = cmap_source
        elif args.map_method == "inverse_icp":
            cmap_target[source_to_target] = cmap_source
        elif args.map_method == "hungarian":
            cmap_target[source_to_target] = cmap_source
        elif args.map_method == "hungarian_icp":
            cmap_target[source_to_target] = cmap_source

        # Fill unmatched target vertices with nearest matched color (matches notebook behavior).
        matched = (cmap_target != 0).any(axis=1)
        if not matched.all():
            try:
                from scipy.spatial import cKDTree
                tree = cKDTree(tgt_verts[matched])
                _, nn_idx = tree.query(tgt_verts[~matched], k=1)
                cmap_target[~matched] = cmap_target[matched][nn_idx]
            except Exception:
                # Fallback: brute-force nearest neighbor.
                for idx in np.where(~matched)[0]:
                    d2 = ((tgt_verts[idx] - tgt_verts[matched]) ** 2).sum(axis=1)
                    cmap_target[idx] = cmap_target[matched][d2.argmin()]

        output_path = os.path.expanduser(args.map_png)
        _save_map_png(source_clean_mesh, target_clean_mesh, cmap_source, cmap_target, output_path, args.map_elev, args.map_azim)

    if args.map_png_dir is not None and args.map_angles > 0:
        cmap_source = get_colors(src_verts)
        cmap_target = np.zeros((tgt_verts.shape[0], 3), dtype=cmap_source.dtype)
        if args.map_method == "inverse":
            cmap_target[source_to_target] = cmap_source
        elif args.map_method == "inverse_icp":
            cmap_target[source_to_target] = cmap_source
        elif args.map_method == "hungarian":
            cmap_target[source_to_target] = cmap_source
        elif args.map_method == "hungarian_icp":
            cmap_target[source_to_target] = cmap_source
        matched = (cmap_target != 0).any(axis=1)
        if not matched.all():
            try:
                from scipy.spatial import cKDTree
                tree = cKDTree(tgt_verts[matched])
                _, nn_idx = tree.query(tgt_verts[~matched], k=1)
                cmap_target[~matched] = cmap_target[matched][nn_idx]
            except Exception:
                for idx in np.where(~matched)[0]:
                    d2 = ((tgt_verts[idx] - tgt_verts[matched]) ** 2).sum(axis=1)
                    cmap_target[idx] = cmap_target[matched][d2.argmin()]
        out_dir = os.path.expanduser(args.map_png_dir)
        os.makedirs(out_dir, exist_ok=True)
        for i in range(args.map_angles):
            azim = (args.map_azim + (360.0 * i / args.map_angles)) % 360.0
            output_path = os.path.join(out_dir, f"vertex_map_{i:02d}.png")
            _save_map_png(source_clean_mesh, target_clean_mesh, cmap_source, cmap_target, output_path, args.map_elev, azim)

    if args.contacts_png is not None:
        output_path = os.path.expanduser(args.contacts_png)
        _save_contacts_png(
            source_clean_mesh,
            target_clean_mesh,
            src_points,
            tgt_points,
            src_intensity,
            output_path,
            args.contacts_elev,
            args.contacts_azim,
            args.contacts_point_size,
        )

    if args.contacts_png_dir is not None and args.contacts_angles > 0:
        out_dir = os.path.expanduser(args.contacts_png_dir)
        os.makedirs(out_dir, exist_ok=True)
        for i in range(args.contacts_angles):
            azim = (args.contacts_azim + (360.0 * i / args.contacts_angles)) % 360.0
            output_path = os.path.join(out_dir, f"contacts_{i:02d}.png")
            _save_contacts_png(
                source_clean_mesh,
                target_clean_mesh,
                src_points,
                tgt_points,
                src_intensity,
                output_path,
                args.contacts_elev,
                azim,
                args.contacts_point_size,
            )

    if args.output is None:
        args.output = f"transferred_contacts_frame{requested_frames[0]}.npz"

    np.savez(
        args.output,
        frame=np.array([requested_frames[0]], dtype=np.int32),
        frame_idx=np.array([_to_frame_idx(requested_frames[0])], dtype=np.int32),
        source_contact_points=src_points,
        source_contact_intensity=src_intensity,
        source_vertex_indices=src_vertex_indices,
        target_vertex_indices=tgt_vertex_indices,
        target_points=tgt_points,
        source_asset=np.array([args.source_asset]),
        target_asset=np.array([args.target_asset]),
        map_method=np.array([args.map_method]),
    )
    print("Wrote", args.output)


if __name__ == "__main__":
    main()
