"""Run normalization on a native capture and write metrics, a comparison and Noa targets."""
import argparse
import json
from pathlib import Path

import torch

from . import Config, SignPoseNormalizer, invert
from .adapters import INDEX, NATIVE_LAYOUT, native_motion
from .avatar import export_motion, noa_skeleton
from .retargeting import retarget


def comparison(series, masks, times, contacts, path, *, layout=NATIVE_LAYOUT, max_frames=120):
    if len(times) > max_frames:
        sample = torch.linspace(0, len(times) - 1, max_frames).round().long()
        series = [points[sample] for points in series]
        masks = [mask[sample] for mask in masks]
        times, contacts = times[sample], contacts[sample]
    import matplotlib

    matplotlib.use("Agg")
    from matplotlib import pyplot as plt
    from PIL import Image

    front = ("Raw capture (rig units)", "Geometric (shoulder widths)",
             "Fixed bone lengths", "Optimized Noa targets")
    labels = ("head", "neck", "left_wrist", "right_wrist", "left_index_4", "right_index_4",
              "chin", "mouth", "forehead", "chest")
    names = {layout.names.index(name): name.replace("left_", "L ").replace("right_", "R ")
             .replace("index_4", "index tip") for name in labels if name in layout.names}
    fig = plt.figure(figsize=(14, 4.5), dpi=100)
    axes = [fig.add_subplot(1, 4, i + 1, projection="3d") for i in range(4)]
    ranges = []
    for points, mask in zip(series, masks):
        valid = points[mask]
        center = (valid.amin(dim=0) + valid.amax(dim=0)) / 2
        radius = max(float((valid - center).abs().max()) * 1.08, .1)
        ranges.append((center, radius))
    radius = max(size for _, size in ranges[1:])
    center = ranges[1][0]
    ranges[1:] = [(center, radius)] * 3
    images = []
    for t in range(len(times)):
        for idx, (axis, points, mask, title) in enumerate(zip(axes, series, masks, front)):
            axis.clear()
            center, radius = ranges[idx]
            x, y, z = center.tolist()
            axis.set(xlim=(x-radius, x+radius), ylim=(z-radius, z+radius),
                     zlim=(y-radius, y+radius), title=title)
            axis.set_box_aspect((1, 1, 1), zoom=1.25)
            axis.view_init(elev=12, azim=75)
            axis.set_axis_off()
            for j, parent in enumerate(layout.parents[1:], 1):
                if mask[t, j] and mask[t, parent]:
                    xyz = points[t, [parent, j]].T
                    side = layout.names[j].split("_")[0]
                    color = {"left": "#176a9a", "right": "#b64e22"}.get(side, "#444444")
                    axis.plot(xyz[0], xyz[2], xyz[1], color=color, linewidth=1.4)
            for j, label in names.items():
                if mask[t, j]:
                    x, y, z = points[t, j]
                    axis.scatter(x, z, y, color="#773f91", s=9)
                    axis.text(x, z, y, label, fontsize=6)
            for active, (_, a, b) in zip(contacts[t], layout.contact_pairs):
                if active and mask[t, a] and mask[t, b]:
                    xyz = points[t, [a, b]].T
                    axis.plot(xyz[0], xyz[2], xyz[1], color="#00866c", linewidth=2.0, linestyle=":")
        fig.suptitle(f"LSF motion comparison   |   source time {times[t]:.3f} s")
        fig.canvas.draw()
        images.append(Image.frombytes("RGBA", fig.canvas.get_width_height(), bytes(fig.canvas.buffer_rgba())).convert("RGB"))
    plt.close(fig)
    intervals = [max(1, round(float(b - a) * 1000)) for a, b in zip(times, times[1:])]
    intervals.append(intervals[-1] if intervals else 80)
    images[0].save(path / "comparison.gif", save_all=True, append_images=images[1:], duration=intervals, loop=0)
    images[len(images) // 2].save(path / "comparison.png")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("motion", type=Path)
    parser.add_argument("--avatar", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument("--steps", type=int, default=40)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    data = json.loads(args.motion.read_text())
    source = native_motion(data)
    inputs = {key: value.to(args.device) if isinstance(value, torch.Tensor) else value for key, value in source.items()}
    skeleton, shoulder_width = noa_skeleton(args.avatar)
    normalizer = SignPoseNormalizer(Config(NATIVE_LAYOUT, "rig_3d_xyz"))
    normalized = normalizer(**inputs, skeleton=skeleton)
    fitted = retarget(normalized, skeleton, steps=args.steps)
    result = export_motion(fitted, data, shoulder_width)
    (args.out / "noa-motion.json").write_text(json.dumps(result, ensure_ascii=False, allow_nan=False))
    valid = normalized["valid"]
    target = normalized["canonical_motion"]
    before = (normalized["anatomical_motion"] - target).norm(dim=-1)
    after = (fitted["positions"] - target).norm(dim=-1)
    wrist_indices = [INDEX["left_wrist"], INDEX["right_wrist"]]
    wrist_valid = valid[..., wrist_indices]
    recovered = invert(normalized["geometric_motion"], normalized["transforms"], torso_local=False)
    finite = torch.isfinite(inputs["pose"]).all(dim=-1)
    offsets = fitted["positions"][..., 1:, :] - fitted["positions"][..., list(NATIVE_LAYOUT.parents[1:]), :]
    summary = {
        "source": str(args.motion), "frames": target.shape[1],
        "duration_ms": data["duration_ms"], "timestamps_preserved": True,
        "observed_joints": int(normalized["observed"].sum()),
        "interpolated_joints": int(normalized["interpolated"].sum()),
        "contact_candidates": int(normalized["contacts"]["active"].sum()),
        "source_median_shoulder_width": float(normalized["transforms"]["scale"][0]),
        "noa_bind_shoulder_width": shoulder_width,
        "mean_direction_only_error_shoulder_widths": float(before[valid].mean()),
        "mean_ik_error_shoulder_widths": float(after[valid].mean()),
        "mean_direction_only_wrist_error_shoulder_widths": float(before[..., wrist_indices][wrist_valid].mean()),
        "mean_ik_wrist_error_shoulder_widths": float(after[..., wrist_indices][wrist_valid].mean()),
        "max_inverse_error_rig_units": float((recovered - inputs["pose"])[finite].abs().max()),
        "max_fk_bone_length_error_shoulder_widths": float((offsets.norm(dim=-1) - skeleton.offsets.to(target).norm(dim=-1)).abs().max()),
        "ik_losses": {name: float(value) for name, value in fitted["losses"].items()},
    }
    (args.out / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    mask = torch.isfinite(source["pose"][0]).all(dim=-1)
    series = [source["pose"][0], normalized["geometric_motion"][0].cpu(),
              normalized["anatomical_motion"][0].cpu(), fitted["positions"][0].cpu()]
    comparison(series, [mask, mask, valid[0].cpu(), valid[0].cpu()], source["times"][0], normalized["contacts"]["active"][0].cpu(), args.out)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
