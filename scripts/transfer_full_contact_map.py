#!/usr/bin/env python
"""
transfer_full_contact_map.py

Transfers a full contact map NPZ (60 keys) to a new object geometry using
DenseMatcher correspondence, preserving the exact same NPZ structure.

Unlike transfer_contact_points.py (which outputs a sparse ~9-key per-frame
result), this script:
  - Remaps all 6 contact arrays across ALL frames at once
  - Updates geometry keys (object_points_local, object_mesh_vertices_local, object_mesh_faces)
  - Updates object metadata (object_asset_path, object_uid)
  - Passes through all other 40+ keys unchanged (hand sampling, interaction
    segmentation, poses, timestamps, thresholds, etc.)

Output NPZ has the same 60-key structure as the input.
"""
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


# ---------------------------------------------------------------------------
# Shared helpers (identical to transfer_contact_points.py)
# ---------------------------------------------------------------------------

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


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Transfer a full 60-key contact map NPZ to a new object geometry, "
                    "preserving the exact same NPZ structure."
    )
    parser.add_argument("--contact-npz", required=True,
                        help="Path to the source full contact map NPZ.")
    parser.add_argument("--source-asset", required=True,
                        help="Path to source converted_new asset folder.")
    parser.add_argument("--target-asset", required=True,
                        help="Path to target converted_new asset folder.")
    parser.add_argument("--output", required=True,
                        help="Output NPZ path (will have same structure as input).")
    parser.add_argument("--device", default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument(
        "--map-method",
        choices=["inverse", "inverse_icp", "hungarian", "hungarian_icp"],
        default="hungarian",
    )
    parser.add_argument("--aggre-weights", default=None,
                        help="Path to AggreNet weights folder.")
    parser.add_argument("--save-correspondence", default=None,
                        help="Save computed src->tgt vertex correspondence to this path.")
    parser.add_argument("--load-correspondence", default=None,
                        help="Load cached src->tgt vertex correspondence (skips model).")
    parser.add_argument("--no-contact-auto-scale", action="store_true",
                        help="Disable bbox auto-scaling of contact points.")
    parser.add_argument("--use-color-mesh", action="store_true",
                        help="Use color_mesh.obj instead of simple_mesh.obj.")
    args = parser.parse_args()

    os.environ["INFERENCE"] = "1"

    if args.seed is not None:
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)

    if args.device is None:
        device = "cuda:0" if torch.cuda.is_available() else "cpu"
    else:
        device = args.device

    # ------------------------------------------------------------------
    # 1. Load input NPZ
    # ------------------------------------------------------------------
    print(f"Loading contact map: {args.contact_npz}")
    npz = np.load(args.contact_npz, allow_pickle=True)

    required = ["object_points_local", "object_contact_binary", "object_mesh_vertices_local"]
    for k in required:
        if k not in npz:
            raise ValueError(f"Input NPZ missing required key: {k}")

    points = npz["object_points_local"].astype(np.float32)           # (P, 3)
    obj_verts = npz["object_mesh_vertices_local"].astype(np.float32)  # (V, 3)
    contact_binary = npz["object_contact_binary"]                     # (F, P)

    num_frames, num_points = contact_binary.shape
    assert points.shape[0] == num_points, (
        f"object_points_local has {points.shape[0]} points but "
        f"object_contact_binary has {num_points} on axis 1"
    )
    print(f"  {num_frames} frames, {num_points} object surface points")

    # Optional contact arrays
    contact_binary_by_hand    = npz.get("object_contact_binary_by_hand", None)      # (F, 2, P)
    contact_intensity         = npz.get("object_contact_intensity", None)            # (F, P)
    contact_intensity_by_hand = npz.get("object_contact_intensity_by_hand", None)   # (F, 2, P)
    contact_sdf               = npz.get("object_contact_signed_distance_m", None)   # (F, P)
    contact_sdf_by_hand       = npz.get("object_contact_signed_distance_m_by_hand", None)  # (F, 2, P)

    # ------------------------------------------------------------------
    # 2. Load source and target meshes
    # ------------------------------------------------------------------
    print("Loading source mesh ...")
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

    print("Loading target mesh ...")
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

    src_verts = source_clean_mesh.verts_list()[0].detach().cpu().numpy()   # simplified
    tgt_verts = target_clean_mesh.verts_list()[0].detach().cpu().numpy()

    # Raw (color) mesh verts+faces for the output NPZ geometry fields.
    # get_mesh() calls recenter() which centers both meshes in-place using the
    # simple mesh bbox center. Add that center back so these are in the target
    # asset's natural coordinate space (same as color_mesh.obj on disk).
    target_center_np = target_center.detach().cpu().numpy().reshape(3)   # (3,)
    tgt_color_verts = target_dirty_mesh.verts_list()[0].detach().cpu().numpy()
    tgt_color_faces = target_dirty_mesh.faces_list()[0].detach().cpu().numpy()

    source_R_np = source_R_inv.detach().cpu().numpy()

    # ------------------------------------------------------------------
    # 3. Compute (or load) source -> target vertex correspondence
    # ------------------------------------------------------------------
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

    if args.load_correspondence is not None:
        print(f"Loading cached correspondence: {args.load_correspondence}")
        cached = np.load(os.path.expanduser(args.load_correspondence), allow_pickle=True)
        source_to_target = cached["source_to_target"].astype(np.int64)
        cached_method = _npz_scalar_str(cached.get("map_method", None))
        if cached_method != "unknown" and cached_method != args.map_method:
            print(f"[warn] cached map_method={cached_method} but current --map-method={args.map_method}")
    else:
        print("Running DenseMatcher to compute mesh correspondence ...")
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
            print(f"Saved correspondence cache: {args.save_correspondence}")

    # Validate correspondence
    if source_to_target.shape[0] != src_verts.shape[0]:
        raise ValueError(
            f"source_to_target length ({source_to_target.shape[0]}) != "
            f"source verts ({src_verts.shape[0]})"
        )
    if source_to_target.min() < 0 or source_to_target.max() >= tgt_verts.shape[0]:
        raise ValueError(
            f"source_to_target has invalid indices: "
            f"min={int(source_to_target.min())} max={int(source_to_target.max())} "
            f"(target verts={tgt_verts.shape[0]})"
        )

    # ------------------------------------------------------------------
    # 4. Build per-point remapping  (shape: num_points,)
    #    Each of the P object_points_local maps to a source mesh vertex,
    #    then to a target mesh vertex.
    # ------------------------------------------------------------------
    print("Computing point-level remapping ...")

    # Center + scale + rotate points into source mesh local space
    obj_center = (obj_verts.max(axis=0) + obj_verts.min(axis=0)) / 2.0
    if args.use_color_mesh:
        src_center_np = source_center.detach().cpu().numpy().reshape(1, 3)
        center_np = src_center_np
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

    points_src = points_src @ source_R_np  # (P, 3)

    # Nearest source vertex for each of the P points
    src_vtx_for_point = _nearest_vertices(points_src, src_verts)    # (P,)
    # Corresponding target vertex
    tgt_vtx_for_point = source_to_target[src_vtx_for_point]          # (P,)

    # New object_points_local = corresponding target mesh vertex positions,
    # un-centered back to the target asset's natural coordinate space.
    new_points_local = (tgt_verts[tgt_vtx_for_point] + target_center_np).astype(np.float32)  # (P, 3)

    print(f"  Point remapping complete: {num_points} points -> target mesh "
          f"({tgt_verts.shape[0]} verts)")

    # ------------------------------------------------------------------
    # 5. Contact arrays: pass through unchanged.
    #
    # contact_binary[f, i] records the contact at semantic point i.
    # After transfer, point i is placed at the *corresponding* location on
    # the target mesh (via DenseMatcher), so its contact value is preserved.
    # Re-indexing the array with vertex indices (as was done before) is wrong:
    # those indices have no relationship to the 5000-point indexing scheme.
    # ------------------------------------------------------------------
    print("Passing contact arrays through (preserving per-point semantics) ...")

    new_contact_binary           = contact_binary
    new_contact_binary_by_hand   = contact_binary_by_hand
    new_contact_intensity        = contact_intensity
    new_contact_intensity_by_hand = contact_intensity_by_hand
    new_contact_sdf              = contact_sdf
    new_contact_sdf_by_hand      = contact_sdf_by_hand

    # ------------------------------------------------------------------
    # 6. Assemble output dict: copy all keys, then overwrite what changed
    # ------------------------------------------------------------------
    print("Assembling output NPZ ...")
    out_dict = {}
    for k in npz.files:
        out_dict[k] = npz[k]

    # --- Geometry ---
    out_dict["object_points_local"]        = new_points_local
    out_dict["object_mesh_vertices_local"] = (tgt_color_verts + target_center_np).astype(np.float32)
    out_dict["object_mesh_faces"]          = tgt_color_faces.astype(np.int32)

    # --- Contact arrays ---
    out_dict["object_contact_binary"] = new_contact_binary
    if new_contact_binary_by_hand is not None:
        out_dict["object_contact_binary_by_hand"] = new_contact_binary_by_hand
    if new_contact_intensity is not None:
        out_dict["object_contact_intensity"] = new_contact_intensity
    if new_contact_intensity_by_hand is not None:
        out_dict["object_contact_intensity_by_hand"] = new_contact_intensity_by_hand
    if new_contact_sdf is not None:
        out_dict["object_contact_signed_distance_m"] = new_contact_sdf
    if new_contact_sdf_by_hand is not None:
        out_dict["object_contact_signed_distance_m_by_hand"] = new_contact_sdf_by_hand

    # --- Metadata ---
    out_dict["object_asset_path"] = np.array(args.target_asset)
    # Derive UID from the last path component (folder name = UID)
    target_uid = os.path.basename(os.path.normpath(args.target_asset))
    out_dict["object_uid"] = np.array(target_uid)

    # ------------------------------------------------------------------
    # 7. Save
    # ------------------------------------------------------------------
    out_path = os.path.expanduser(args.output)
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    np.savez(out_path, **out_dict)
    print(f"Wrote {out_path}  ({len(out_dict)} keys)")

    # Quick sanity checks
    result = np.load(out_path, allow_pickle=True)
    n_keys = len(result.files)
    expected_keys = len(npz.files)
    if n_keys != expected_keys:
        print(f"[warn] Output has {n_keys} keys but input had {expected_keys}")
    else:
        print(f"[ok] Key count matches: {n_keys}")

    total_src = int(contact_binary.sum())
    total_tgt = int(new_contact_binary.sum())
    print(f"[ok] Total contact activations: source={total_src}, target={total_tgt}")


if __name__ == "__main__":
    main()
