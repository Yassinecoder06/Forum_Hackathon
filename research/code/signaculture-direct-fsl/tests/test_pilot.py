import unittest

import numpy as np
import torch

from evaluate import dtw, mismatch_indices
from scripts.to_studio_motion import convert
from src.losses import objective
from src.model import Model
from src.retrieval import CaptionIndex, trajectory


class PilotTests(unittest.TestCase):
    def test_both_encoders_produce_pose_and_duration(self):
        torch.set_num_threads(2)
        for kind, data in (("learned", torch.ones(2,48,dtype=torch.long)),
                           ("frozen_french", torch.randn(2,48,768))):
            model=Model(4,32,8,kind)
            pose,duration=model(data)
            self.assertEqual(tuple(pose.shape),(2,8,128,2))
            self.assertEqual(tuple(duration.shape),(2,))
            mask=torch.ones(2,8,128,dtype=torch.bool)
            loss=objective(pose,duration,torch.zeros_like(pose),mask,torch.ones(2))
            loss.backward()
            self.assertTrue(torch.isfinite(loss).item())

    def test_ignored_landmarks_do_not_contribute_to_losses(self):
        pose=torch.zeros(1,4,128,2)
        mask=torch.ones(1,4,128,dtype=torch.bool)
        mask[:,2,22]=False
        target=pose.clone()
        target[:,2,22]=1000
        self.assertEqual(objective(pose,torch.zeros(1),target,mask,torch.ones(1)).item(),0)

    def test_mismatched_captions_use_different_recording_and_text(self):
        samples = [({"source_video_id": video, "french_caption": caption},)
                   for video, caption in [("a", "A"), ("a", "B"), ("b", "A"), ("b", "C"), ("c", "B")]]
        choices = [list(indices) for indices in mismatch_indices(samples)]
        self.assertEqual(choices, [list(indices) for indices in mismatch_indices(samples)])
        for (row,), indices in zip(samples, choices):
            self.assertTrue(indices)
            for j in indices:
                self.assertNotEqual(row["source_video_id"], samples[j][0]["source_video_id"])
                self.assertNotEqual(row["french_caption"], samples[j][0]["french_caption"])

    def test_caption_retrieval_falls_back_for_unsupported_landmarks_and_text(self):
        index = CaptionIndex(["le musée ouvre", "le train part"])
        self.assertEqual(index.neighbors("musée", 1).tolist(), [0])
        self.assertEqual(index.neighbors("zzz", 1).tolist(), [])
        template = np.zeros((1, 2, 2))
        train = [(None, np.full_like(template, 8), np.array([[True, False]]), 2)]
        motion, duration = trajectory(train, [0], template, 4, 0.5)
        np.testing.assert_array_equal(motion, [[[4, 4], [0, 0]]])
        self.assertEqual(duration, 3)

    def test_residual_pose_starts_from_template_and_stays_bounded(self):
        model = Model(4, 32, 8, "frozen_french", residual_scale=.15).eval()
        model.phase_template.fill_(.4)
        features = torch.randn(1, 48, 768)
        with torch.no_grad():
            initial, _ = model(features)
            torch.testing.assert_close(initial[0], model.phase_template)
            model.decoder.output.bias.fill_(5)
            moved, _ = model(features)
        self.assertGreater(float((moved - initial).mean()), .14)
        self.assertLessEqual(float((moved - initial).abs().max()), .15)

    def test_identical_dtw_and_studio_timing(self):
        xy=np.zeros((8,128,2),dtype=np.float32)
        mask=np.ones((8,128),dtype=bool)
        self.assertEqual(dtw(xy,xy,mask,mask),0)
        mask[2,1]=False
        mask[3,19]=False
        motion=convert(xy,mask,8)
        self.assertEqual(motion["frames"][0]["time_ms"],0)
        self.assertEqual(motion["frames"][-1]["time_ms"],1000)
        self.assertNotIn("left_thumb_1",motion["frames"][3]["joints"])


if __name__=="__main__":
    unittest.main()
