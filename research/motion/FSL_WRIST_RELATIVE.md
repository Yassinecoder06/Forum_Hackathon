# Wrist-relative 2D hand interpolation

Each finger is interpolated relative to the DWPose hand wrist using
context frames where both landmarks are valid. The finger offset is
added to the interpolated wrist. If there is no such context for a
finger, the method retains independent interpolation. The same 224
validation and 198 previously inspected test windows were used.

| Test metric | Independent visible endpoints | Wrist-relative fingers |
|---|---:|---:|
| Mean 2D joint error | 0.137817 | 0.145580 |
| Fingertip error | 0.333785 | 0.363656 |
| Hand joint error | 0.306602 | 0.328079 |
| Boundary velocity error | 1.779834 | 2.084470 |
| Projected bone length error | 0.050874 | 0.042963 |

Validation showed the same direction of change for all five metrics.
Projected bone lengths were closer to the references, while position and
velocity were worse. Independent visible endpoint interpolation remains
the comparison baseline. These figures cover masked 2D gap filling and
do not establish 3D hand shape or intelligibility.

The evaluator verifies the window checksums and checks that replacing
the hidden frames affects neither method.

```bash
.runtime/t5-local-venv/bin/python -m src.motion.evaluate_visible_endpoints \
  --data data/poses/private/windows-v1 --output /tmp/fsl-wrist-relative.json
```

[Full metrics](FSL_WRIST_RELATIVE.json)
