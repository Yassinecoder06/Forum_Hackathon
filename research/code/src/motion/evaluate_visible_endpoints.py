"""Compare fixed and visibility-aware endpoints on existing France-LSF windows."""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import torch

from src.motion.inbetween import (
    interpolate, measure, visible_endpoint_interpolation, wrist_relative_interpolation,
)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    torch.set_num_threads(2)
    manifest = json.loads((args.data / "manifest.json").read_text())
    if manifest["sign_language"] != "fsl" or manifest["fps"] != 24:
        raise ValueError("Expected France-LSF pose windows at 24 FPS")
    report = {"task": "2D masked motion completion on existing validation and test windows", "splits": {}}
    for split in ("val", "test"):
        path = args.data / f"{split}.npz"
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        if digest != manifest["sha256"][split]:
            raise ValueError(f"Window checksum differs for {split}")
        with np.load(path, allow_pickle=False) as archive:
            x = torch.from_numpy(archive["xy"])
            mask = torch.from_numpy(archive["mask"])
        nominal_endpoints = mask[:, 3] & mask[:, 12]
        visible = visible_endpoint_interpolation(x, mask)
        # Altering the hidden interval must never affect the prediction.
        altered = x.clone()
        altered[:, 4:12] = 123
        if not torch.equal(visible, visible_endpoint_interpolation(altered, mask)):
            raise ValueError("Hidden frames affected the prediction")
        relative = wrist_relative_interpolation(x, mask)
        if not torch.equal(relative, wrist_relative_interpolation(altered, mask)):
            raise ValueError("Hidden frames affected wrist-relative prediction")
        linear = interpolate(x)
        if not torch.allclose(visible.masked_select(nominal_endpoints[:, None, :, None]),
                              linear.masked_select(nominal_endpoints[:, None, :, None]), atol=1e-6):
            raise ValueError("Visibility-aware method changed fully visible endpoints")
        report["splits"][split] = {
            "windows": len(x), "sha256": digest,
            "joint_windows_missing_nominal_endpoint": float((~nominal_endpoints).float().mean()),
            "linear": measure(linear, x, mask),
            "visible_endpoints": measure(visible, x, mask),
            "wrist_relative": measure(relative, x, mask),
        }
    with args.output.open("x") as output:
        json.dump(report, output, indent=2)
        output.write("\n")


if __name__ == "__main__":
    main()
