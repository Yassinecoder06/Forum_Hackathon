from dataclasses import replace

import pytest
import torch

from normalization import Config, Layout, SignPoseNormalizer, Skeleton, invert
from normalization.adapters import INDEX, mediapipe_world, native_motion
from normalization.retargeting import IKWeights, retarget
from normalization.skeleton import rotation_matrix

LAYOUT = Layout(
    ("hips", "left_shoulder", "right_shoulder", "elbow", "wrist", "index", "middle", "pinky",
     "chin", "forehead", "chest", "other_hand"),
    (-1, 0, 0, 1, 3, 4, 4, 4, 0, 0, 0, 0),
    (1, 2), (0,), palms=((4, 5, 6, 7),), fingers=((3, 4, 5),),
    spatial_pairs=(("wrist_to_shoulder", 4, 1),),
    contact_pairs=(("chin", 5, 8), ("forehead", 5, 9), ("chest", 5, 10), ("hand", 5, 11)),
)
REST = torch.tensor([
    [0, 0, 0], [.5, 1, 0], [-.5, 1, 0], [.55, .8, .20], [.30, .95, .30],
    [.2, 1, .35], [.21, .94, .38], [.22, .88, .36],
    [.2, 1, .35], [.15, 1.22, .36], [.20, .80, .38], [.05, .95, .4],
], dtype=torch.float64)
CONFIG = Config(LAYOUT, "shared_3d")
NORMALIZE = SignPoseNormalizer(CONFIG)


def fixture(frames=5):
    times = torch.arange(frames, dtype=REST.dtype)[None] * .04
    coords = REST[None, None].repeat(1, frames, 1, 1)
    coords[..., 2] += times[..., None]
    return coords, times


def test_geometric_invariance_inverse_and_motion():
    coords, times = fixture()
    spin = rotation_matrix(torch.tensor([.35, -.3, .2], dtype=coords.dtype))
    altered = coords @ spin.T * 2.4 + coords.new_tensor([8, -3, 2])
    out = NORMALIZE(coords, times=times)
    transformed = NORMALIZE(altered, times=times)
    for key in ("canonical_motion", "torso_local", "bone_direction", "velocity", "torso_local"):
        torch.testing.assert_close(out[key], transformed[key])
    torch.testing.assert_close(invert(transformed["torso_local"], transformed["transforms"]), altered)
    torch.testing.assert_close(invert(transformed["geometric_motion"], transformed["transforms"], torso_local=False), altered)
    torch.testing.assert_close(out["transforms"]["torso_translation"][0, -1, 2], times[0, -1] - .04)
    torch.testing.assert_close(out["canonical_motion"][0, :, 0, 2], times[0])
    assert torch.linalg.det(out["transforms"]["rotation"]).min() > .999


def test_turn_of_torso_is_retained_in_reference_and_orientation_track():
    coords, times = fixture()
    yaw = torch.zeros(5, 3, dtype=coords.dtype)
    yaw[:, 1] = torch.linspace(0, .6, 5)
    coords = coords @ rotation_matrix(yaw).transpose(-1, -2)
    output = SignPoseNormalizer(replace(CONFIG, frame_radius=0))(coords, times=times)
    assert output["transforms"]["torso_rotation"][0, -1, 0, 2] > .5
    assert output["canonical_motion"][0, -1, 1, 2] < -.1
    assert output["torso_local"][0, -1, 1, 2].abs() < 1e-8


def test_fixed_lengths_different_anatomies_and_batched_fk():
    coords, times = fixture(3)
    varied = coords.clone()
    varied[..., 1, 0] += .15
    varied[..., 2, 0] -= .15
    varied[..., 3:8, 1] += torch.linspace(0, .3, 5)
    lengths = REST[1:] - REST[list(LAYOUT.parents[1:])]
    avatar = Skeleton(LAYOUT, REST)
    result = NORMALIZE(torch.cat((coords, varied)), times=times.expand(2, -1), skeleton=avatar)
    p = result["anatomical_motion"]
    actual = (p[..., 1:, :] - p[..., list(LAYOUT.parents[1:]), :]).norm(dim=-1)
    torch.testing.assert_close(actual, lengths.norm(dim=-1).expand_as(actual))
    torch.testing.assert_close(avatar.forward_kinematics(p[..., 0, :], result["local_rotation"]), p)
    assert not torch.allclose(result["source_bone_length"][0], result["source_bone_length"][1])


@pytest.mark.parametrize("anchor", [8, 9, 10, 11])
def test_contact_objective_improves_chin_forehead_chest_and_hands(anchor):
    coords, times = fixture(1)
    coords[0, 0, 5] = REST[anchor]
    rest = coords[0, 0].clone()
    rest[3] += torch.tensor([.1, 0, 0])
    limits = torch.full((len(LAYOUT.names) - 1, 3), 2., dtype=coords.dtype)
    limits[7:] = 0  # fixed contact anchors in this fixture
    skeleton = Skeleton(LAYOUT, rest, limits)
    output = NORMALIZE(coords, times=times, skeleton=skeleton)
    active = output["contacts"]["active"][0, 0, anchor - 8]
    assert active
    before = (output["anatomical_motion"][0, 0, 5] - output["anatomical_motion"][0, 0, anchor]).norm()
    result = retarget(output, skeleton, steps=50, weights=IKWeights(contact=20))
    after = result["contact_distance"][0, 0, anchor - 8]
    assert after < min(.035, before * .65)
    assert (result["local_rotation"].abs() <= limits).all()
    actual = result["positions"][..., 1:, :] - result["positions"][..., list(LAYOUT.parents[1:]), :]
    torch.testing.assert_close(actual.norm(dim=-1), skeleton.offsets.norm(dim=-1).expand_as(actual[..., 0]))


def test_short_gap_outlier_confidence_velocity_and_contact_masks():
    coords, times = fixture(7)
    coords[..., 5, 0] += times
    confidence = torch.ones_like(coords[..., 0])
    coords[0, 1, 5] = torch.nan
    coords[0, 3, 5, 2] += 2
    confidence[0, 3, 5] = .7
    result = NORMALIZE(coords, times=times, confidence=confidence)
    assert result["interpolated"][0, 1, 5] and result["outlier"][0, 3, 5]
    torch.testing.assert_close(result["velocity"][0, :, 5, 0], torch.ones(6, dtype=coords.dtype))
    torch.testing.assert_close(result["acceleration"][0, :, 5, 0], torch.zeros(5, dtype=coords.dtype))
    assert not result["contacts"]["active"][0, 1].any()
    assert not result["contacts"]["active"][0, 3].any()
    confidence[..., 5] = 0
    assert not NORMALIZE(coords, times=times, confidence=confidence)["valid"][..., 5].any()


def test_degenerate_and_minimal_input():
    coords, times = fixture(1)
    coords[..., 1, :] = coords[..., 2, :]
    out = NORMALIZE(coords, times=times)
    assert not out["transforms"]["calibrated"].any()
    assert not out["valid"].any()
    assert torch.isfinite(out["canonical_motion"]).all()
    assert out["velocity"].shape[1] == 0
    with pytest.raises(ValueError, match="shared XYZ"):
        Config(LAYOUT, "image_2d")
    with pytest.raises(ValueError, match="strictly"):
        NORMALIZE(coords.expand(-1, 2, -1, -1), times=times.expand(-1, 2))


@pytest.mark.parametrize("device", ["cpu", pytest.param("cuda", marks=pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA unavailable"))])
def test_gradients_and_anatomical_handedness(device):
    coords, times = fixture(4)
    coords = coords.to(device).requires_grad_()
    confidence = torch.ones_like(coords[..., 0]).cpu()
    confidence[0, 2, 6] = 0
    out = NORMALIZE(coords, times=times, confidence=confidence, skeleton=Skeleton(LAYOUT, REST))
    assert out["canonical_motion"][0, 0, 1, 0] > out["canonical_motion"][0, 0, 2, 0]
    loss = sum(out[key].square().mean() for key in ("canonical_motion", "anatomical_motion", "finger_angle", "palm_rotation"))
    loss.backward()
    assert torch.isfinite(coords.grad).all() and coords.grad.abs().sum() > 0


def test_mediapipe_world_hands_use_body_wrist_and_explicit_scale():
    body = torch.zeros(1, 1, 33, 3)
    body[..., 15, :] = torch.tensor([3., 4., 5.])
    hand = torch.ones(1, 1, 21, 3) * 25
    hand[..., 5, 0] += .1
    out = mediapipe_world(body, hand, hand, hand_to_body_axes=(torch.eye(3), torch.eye(3)), hand_to_body_scale=(2, 1))
    torch.testing.assert_close(out[0, 0, INDEX["left_index_1"]], torch.tensor([3.2, 4., 5.]))
    torch.testing.assert_close(out[0, 0, INDEX["right_index_1"]], torch.tensor([.1, 0., 0.]))


def test_native_sparse_and_keeps_face():
    face = {"mouth_open": .2}
    record = {"rig_version": "signaculture-v1", "frames": [{"time_ms": 31, "joints": {"head": [0, 1.5, .1]}, "face": face}]}
    result = native_motion(record)
    assert result["face"] == [face]
    assert result["confidence"].sum() == 1
    assert result["times"].item() == .031


def test_variable_timing_head_rotation_and_face_payload():
    coords, _ = fixture(5)
    times = torch.tensor([[0, .027, .071, .103, .147]], dtype=REST.dtype)
    coords[..., 2] = REST[:, 2] + times[..., None] * 2
    turn = torch.zeros(1, 5, 3, dtype=REST.dtype)
    turn[..., 1] = times
    rotation = rotation_matrix(turn)
    output = NORMALIZE(coords, times=times, head_rotation=rotation, face={"jawOpen": times})
    assert output["head_valid"].all()
    torch.testing.assert_close(output["head_rotation"], rotation)
    torch.testing.assert_close(output["face"]["jawOpen"], times)
    torch.testing.assert_close(output["velocity"][..., 2], torch.full_like(output["velocity"][..., 2], 2))
    torch.testing.assert_close(output["acceleration"][..., 2], torch.zeros_like(output["acceleration"][..., 2]))


def test_transition_manifest_adapter_uses_selected_split(tmp_path):
    import json
    from normalization.adapters import NATIVE_LAYOUT, NativeMotionDataset

    motion = tmp_path / "motion.json"
    motion.write_text(json.dumps({
        "rig_version": "signaculture-v1",
        "frames": [{"time_ms": 0, "joints": {"hips": [0, 0, 0], "left_shoulder": [.5, 1, 0], "right_shoulder": [-.5, 1, 0]}}],
    }))
    manifest = tmp_path / "manifest.json"
    manifest.write_text(json.dumps({"clips": [{"split": "train", "path": str(motion)}, {"split": "test", "path": "other.json"}]}))
    normalize = SignPoseNormalizer(Config(NATIVE_LAYOUT, "rig_3d_xyz"))
    dataset = NativeMotionDataset.from_transition_manifest(manifest, "train", normalize)
    assert len(dataset) == 1
    assert dataset[0]["xyz"].shape == (1, 49, 3)
    assert dataset[0]["mask"].sum() == 3


def test_export_keeps_estimated_fingers_out_of_observed_hand_track():
    from normalization.adapters import NATIVE_LAYOUT
    from normalization.avatar import export_motion

    joints = len(NATIVE_LAYOUT.names)
    valid = torch.ones(1, 1, joints, dtype=torch.bool)
    observed = valid.clone()
    observed[..., INDEX["left_index_4"]] = False
    source = {"frames": [{"time_ms": 0, "face": {"mouth_open": .2},
                          "playback_quality": {"right_hand_rejected": False}}]}
    result = export_motion({"positions": torch.ones(1, 1, joints, 3), "valid": valid, "observed": observed}, source, .4)
    frame = result["frames"][0]
    assert "left_index_4" in frame["joints"]
    assert frame["hand_tracking"]["left"] == {"source": "normalization-short-gap", "observed": False}
    assert frame["hand_tracking"]["right"]["observed"] is True
    assert frame["playback_quality"] == source["frames"][0]["playback_quality"]
    assert frame["face"] == source["frames"][0]["face"]


def test_long_gap_stays_missing_and_confident_rapid_reversal_is_retained():
    coords, times = fixture(8)
    coords[0, 2:7, 5] = torch.nan
    coords[0, 3, 6, 1] += 1.2
    output = NORMALIZE(coords, times=times)
    assert not output["valid"][0, 2:7, 5].any()
    assert not output["outlier"][0, 3, 6]
    torch.testing.assert_close(output["canonical_motion"][0, 3, 6], output["geometric_motion"][0, 3, 6])
    assert not output["velocity_valid"][0, 1, 5]
