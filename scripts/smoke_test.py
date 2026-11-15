#!/usr/bin/env python3
"""
Integration and Smoke Test Suite for SignaCulture AI.

Strictly targets real backend endpoints per docs/INTEGRATION_CONTRACT.md.
No mock fallback: if the backend is offline or returns an error, the suite fails
immediately with a non-zero exit code.

Validated Endpoints:
1. GET /health -> {status: "ok"}
2. GET /glossary -> {entries: GlossEntry[]}
3. GET /motions/{motion_id} -> MotionAsset
4. GET /assets/{filename} -> serves only approved preview assets; unapproved files blocked
5. POST /analyze-text and POST /gloss-lookup -> known text vs unknown text (unknowns flagged, translation_validated: false)
6. POST /compose-animation -> AnimationPlan with render_allowed_for_publication: false
7. POST /analyze-audio -> multipart WAV (runtime generated 440Hz + silence + 220Hz); asserts F0 ~220Hz, pause detection, null emotion
8. POST /review -> reviewer='automated_test', notes='automated synthetic fixture test only', approval cannot bypass rights
9. POST /render -> purpose='preview' returns assets; purpose='publication' strictly blocked

Usage:
    python3 scripts/smoke_test.py [--base-url http://localhost:8000]
"""

import argparse
import json
import os
import sys
import urllib.error
import urllib.parse
import urllib.request
from typing import Any, Dict, List, Optional, Tuple

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SCRIPT_DIR)
from generate_test_audio import generate_sine_silence_wav_bytes


class Colors:
    GREEN = "\033[92m"
    RED = "\033[91m"
    YELLOW = "\033[93m"
    BLUE = "\033[94m"
    BOLD = "\033[1m"
    RESET = "\033[0m"


class SmokeTestRunner:
    def __init__(self, base_url: str):
        self.base_url = base_url.rstrip("/")
        self.passed = 0
        self.failed = 0
        self.errors: List[str] = []

    def log_success(self, test_name: str, detail: str = "") -> None:
        self.passed += 1
        msg = f"{Colors.GREEN}[PASS]{Colors.RESET} {test_name}"
        if detail:
            msg += f" - {Colors.BLUE}{detail}{Colors.RESET}"
        print(msg)

    def log_failure(self, test_name: str, reason: str) -> None:
        self.failed += 1
        self.errors.append(f"{test_name}: {reason}")
        print(f"{Colors.RED}[FAIL]{Colors.RESET} {test_name}: {reason}")

    def request_json(
        self,
        method: str,
        path: str,
        data: Optional[Dict[str, Any]] = None,
        headers: Optional[Dict[str, str]] = None,
        timeout: int = 10,
    ) -> Tuple[int, Any]:
        url = f"{self.base_url}{path}"
        req_headers = {"Accept": "application/json"}
        if headers:
            req_headers.update(headers)

        body = None
        if data is not None:
            body = json.dumps(data).encode("utf-8")
            req_headers["Content-Type"] = "application/json"

        req = urllib.request.Request(url, data=body, headers=req_headers, method=method)
        try:
            with urllib.request.urlopen(req, timeout=timeout) as response:
                status = response.getcode()
                raw = response.read().decode("utf-8")
                try:
                    return status, json.loads(raw)
                except json.JSONDecodeError:
                    return status, raw
        except urllib.error.HTTPError as e:
            raw = e.read().decode("utf-8")
            try:
                return e.code, json.loads(raw)
            except json.JSONDecodeError:
                return e.code, raw
        except Exception as e:
            raise ConnectionError(f"Connection failed to real backend at {url}: {e}")

    def verify_backend_reachable(self) -> None:
        """Strict check: if backend is not running, fail immediately without mock fallback."""
        url = f"{self.base_url}/health"
        try:
            req = urllib.request.Request(url, headers={"Accept": "application/json"})
            with urllib.request.urlopen(req, timeout=3) as resp:
                if resp.getcode() != 200:
                    raise RuntimeError(f"Backend /health returned HTTP {resp.getcode()}")
        except Exception as e:
            print(f"\n{Colors.RED}{Colors.BOLD}[CRITICAL] Real backend unreachable at {url}: {e}{Colors.RESET}")
            print("Ensure backend is running (e.g. `uvicorn app.main:app --port 8000`) before running smoke tests.")
            sys.exit(1)

    # -------------------------------------------------------------------------
    # Tests
    # -------------------------------------------------------------------------

    def test_health(self) -> None:
        name = "GET /health"
        try:
            status, data = self.request_json("GET", "/health")
            if status == 200 and isinstance(data, dict) and data.get("status") == "ok":
                self.log_success(name, f"status={data['status']}")
            else:
                self.log_failure(name, f"Expected HTTP 200 with {{status: 'ok'}}, got HTTP {status}, body={data}")
        except Exception as e:
            self.log_failure(name, str(e))

    def test_glossary(self) -> Optional[List[Dict[str, Any]]]:
        name = "GET /glossary"
        try:
            status, data = self.request_json("GET", "/glossary")
            if status != 200 or not isinstance(data, dict) or "entries" not in data:
                self.log_failure(name, f"Expected 200 with 'entries' list, got HTTP {status}, body={data}")
                return None

            entries = data["entries"]
            if not isinstance(entries, list) or len(entries) == 0:
                self.log_failure(name, "Glossary entries list is empty or not a list")
                return None

            required_keys = {"gloss_id", "concept_fr", "language", "variant", "source_id", "review_status"}
            first_entry = entries[0]
            missing = required_keys - set(first_entry.keys())
            if missing:
                self.log_failure(name, f"GlossEntry missing required keys: {missing}")
                return None

            self.log_success(name, f"{len(entries)} entries found (e.g. {first_entry.get('gloss_id')})")
            return entries
        except Exception as e:
            self.log_failure(name, str(e))
            return None

    def test_motions_and_assets(self, entries: Optional[List[Dict[str, Any]]]) -> None:
        name = "GET /motions/{motion_id} & GET /assets/{filename}"
        if not entries:
            self.log_failure(name, "Skipped because glossary was empty")
            return

        try:
            motion_id = entries[0].get("motion_id")
            if not motion_id:
                self.log_failure(name, "Gloss entry missing motion_id")
                return

            status, motion = self.request_json("GET", f"/motions/{motion_id}")
            if status != 200 or not isinstance(motion, dict):
                self.log_failure(name, f"GET /motions/{motion_id} returned HTTP {status}")
                return

            required_motion_keys = {"motion_id", "gloss_id", "path", "rig_version", "fps", "duration_ms", "license"}
            missing = required_motion_keys - set(motion.keys())
            if missing:
                self.log_failure(name, f"MotionAsset missing keys: {missing}")
                return

            asset_path = motion.get("path", "")
            asset_filename = os.path.basename(asset_path)
            status_asset, _ = self.request_json("GET", f"/assets/{asset_filename}")
            if status_asset != 200:
                self.log_failure(name, f"GET /assets/{asset_filename} returned HTTP {status_asset}")
                return

            # Gating check: unapproved assets must be blocked (HTTP 403 or 404)
            status_blocked, _ = self.request_json("GET", "/assets/unapproved_unregistered_fixture.txt")
            if status_blocked not in (403, 404):
                self.log_failure(name, f"Gating failure: unapproved asset returned HTTP {status_blocked} instead of 403/404")
                return

            self.log_success(name, f"Motion {motion_id} valid, asset {asset_filename} served, unapproved asset blocked ({status_blocked})")
        except Exception as e:
            self.log_failure(name, str(e))

    def test_text_analysis_known_and_unknowns(self) -> None:
        name = "POST /analyze-text & /gloss-lookup (known text vs unknowns)"
        try:
            # 1. Known text
            status_known, data_known = self.request_json("POST", "/gloss-lookup", {"text": "Bonjour"})
            if status_known != 200 or "gloss_sequence" not in data_known:
                self.log_failure(name, f"Known text failed: HTTP {status_known}, body={data_known}")
                return

            if len(data_known["gloss_sequence"]) == 0:
                self.log_failure(name, "Known concept 'Bonjour' produced empty gloss_sequence")
                return

            # 2. Unknown text
            unknown_text = "terme_completement_inconnu_xyz987"
            status_unk, data_unk = self.request_json("POST", "/analyze-text", {"text": unknown_text})
            if status_unk != 200:
                self.log_failure(name, f"Unknown text call failed: HTTP {status_unk}")
                return

            unknowns = data_unk.get("unknowns", [])
            trans_val = data_unk.get("translation_validated", True)
            ling_rev = data_unk.get("linguistic_review", "")

            if len(unknowns) == 0:
                self.log_failure(name, f"Expected unknowns list for '{unknown_text}', got empty list")
                return

            if trans_val is not False:
                self.log_failure(name, f"Contract violation: translation_validated must be False, got {trans_val}")
                return

            if ling_rev != "pending":
                self.log_failure(name, f"Contract violation: linguistic_review must be 'pending', got {ling_rev}")
                return

            self.log_success(name, f"Known matched {len(data_known['gloss_sequence'])} glosses; Unknowns detected: {unknowns}, validated=False")
        except Exception as e:
            self.log_failure(name, str(e))

    def test_compose_animation(self) -> Optional[Dict[str, Any]]:
        name = "POST /compose-animation"
        try:
            status, plan = self.request_json("POST", "/compose-animation", {"text": "Bonjour"})
            if status != 200 or not isinstance(plan, dict):
                self.log_failure(name, f"HTTP {status}, body={plan}")
                return None

            required_keys = {"plan_id", "source_text", "target_language", "segments", "unknowns", "duration_ms", "render_allowed_for_publication"}
            missing = required_keys - set(plan.keys())
            if missing:
                self.log_failure(name, f"AnimationPlan missing contract keys: {missing}")
                return None

            if plan.get("target_language") != "LSF":
                self.log_failure(name, f"target_language should be 'LSF', got {plan.get('target_language')}")
                return None

            if plan.get("render_allowed_for_publication") is not False:
                self.log_failure(name, "render_allowed_for_publication must be False for unapproved plan")
                return None

            self.log_success(name, f"Plan {plan.get('plan_id')}, duration={plan.get('duration_ms')}ms, segments={len(plan.get('segments', []))}")
            return plan
        except Exception as e:
            self.log_failure(name, str(e))
            return None

    def test_audio_analysis_sine_silence(self) -> None:
        name = "POST /analyze-audio (WAV sine+silence signal processing & F0 assert)"
        try:
            # Generate deterministic in-memory WAV bytes:
            # 0.0-1.0s: 440 Hz sine, 1.0-2.0s: silence, 2.0-3.0s: 220 Hz sine
            wav_bytes = generate_sine_silence_wav_bytes(
                sample_rate=16000,
                tone1_freq=440.0,
                tone1_duration=1.0,
                silence_duration=1.0,
                tone2_freq=220.0,
                tone2_duration=1.0,
            )

            boundary = "----SignaCultureSmokeBoundary84729104"
            body = (
                f"--{boundary}\r\n"
                f'Content-Disposition: form-data; name="file"; filename="synthetic_runtime_test.wav"\r\n'
                f"Content-Type: audio/wav\r\n\r\n"
            ).encode("utf-8") + wav_bytes + f"\r\n--{boundary}--\r\n".encode("utf-8")

            url = f"{self.base_url}/analyze-audio"
            headers = {"Content-Type": f"multipart/form-data; boundary={boundary}", "Accept": "application/json"}
            req = urllib.request.Request(url, data=body, headers=headers, method="POST")

            with urllib.request.urlopen(req, timeout=15) as resp:
                status = resp.getcode()
                raw_data = resp.read().decode("utf-8")
                features = json.loads(raw_data)

            if status != 200:
                self.log_failure(name, f"HTTP {status}, body={features}")
                return

            required_keys = {"duration_ms", "frames", "emotion_hypothesis", "model_version", "warnings"}
            missing = required_keys - set(features.keys())
            if missing:
                self.log_failure(name, f"AudioFeatures missing contract keys: {missing}")
                return

            dur = features.get("duration_ms", 0)
            if not (2700 <= dur <= 3300):
                self.log_failure(name, f"Expected duration ~3000ms, got {dur}ms")
                return

            # Contract verification: emotion_hypothesis MUST be null unless a local verified model is running
            if features.get("emotion_hypothesis") is not None:
                self.log_failure(name, f"Contract violation: raw signal analysis should have emotion_hypothesis=null, got {features['emotion_hypothesis']}")
                return

            frames = features.get("frames", [])
            if not frames:
                self.log_failure(name, "AudioFeatures has empty frames list")
                return

            # Frame subsets:
            # Voiced tone 2 (220 Hz): 2100ms - 2900ms
            # Silence: 1100ms - 1900ms
            tone2_frames = [f for f in frames if 2100 <= f.get("start_ms", 0) <= 2900]
            silence_frames = [f for f in frames if 1100 <= f.get("start_ms", 0) <= 1900]

            # 1. Assert F0 tone ~220 Hz in voiced interval (e.g. 180 Hz <= pitch_hz <= 260 Hz)
            tone2_pitches = [f.get("pitch_hz") for f in tone2_frames if f.get("pitch_hz") is not None]
            if not tone2_pitches:
                self.log_failure(name, "No pitch/F0 estimated in 220Hz voiced segment (2100-2900ms)")
                return

            avg_pitch_tone2 = sum(tone2_pitches) / len(tone2_pitches)
            if not (180.0 <= avg_pitch_tone2 <= 260.0):
                self.log_failure(name, f"Expected F0 ~220Hz in voiced interval, got average {avg_pitch_tone2:.1f} Hz (values: {tone2_pitches[:5]})")
                return

            # 2. Assert pauses detected in silence segment
            pauses_detected = sum(1 for f in silence_frames if f.get("is_pause") is True)
            if pauses_detected == 0 and len(silence_frames) > 0:
                self.log_failure(name, "Silence segment (1100-1900ms) had 0 frames flagged as is_pause=True")
                return

            self.log_success(
                name,
                f"Duration={dur}ms, F0 tone2 avg={avg_pitch_tone2:.1f}Hz (~220Hz), pauses={pauses_detected}/{len(silence_frames)}, emotion_hypothesis=null",
            )
        except Exception as e:
            self.log_failure(name, str(e))

    def test_review_and_publication_gating(self, plan: Optional[Dict[str, Any]]) -> None:
        name = "POST /review & POST /render (Publication Gating & Rights Checking)"
        if not plan:
            self.log_failure(name, "Skipped because compose-animation plan was missing")
            return

        plan_id = plan.get("plan_id")
        try:
            # 1. Preview render: MUST succeed
            status_prev, prev_data = self.request_json("POST", "/render", {"plan_id": plan_id, "purpose": "preview"})
            if status_prev != 200:
                self.log_failure(name, f"Preview render failed: HTTP {status_prev}, body={prev_data}")
                return

            # 2. Publication render BEFORE approval: MUST be blocked (HTTP 400 or 403)
            status_pub_blocked, pub_blocked = self.request_json("POST", "/render", {"plan_id": plan_id, "purpose": "publication"})
            if status_pub_blocked not in (400, 403):
                self.log_failure(name, f"GATING VIOLATION: Unapproved publication returned HTTP {status_pub_blocked} instead of 400/403")
                return

            # 3. Submit automated test review (Never record human validation!)
            status_rev, rev_record = self.request_json(
                "POST",
                "/review",
                {
                    "plan_id": plan_id,
                    "reviewer": "automated_test",
                    "status": "approved",
                    "notes": "automated synthetic fixture test only",
                },
            )
            if status_rev != 200:
                self.log_failure(name, f"POST /review failed: HTTP {status_rev}, body={rev_record}")
                return

            # 4. Publication render AFTER review: STILL blocked because rights_status is pending!
            # Contract: "approval cannot bypass rights/animation review"
            status_pub_still_blocked, _ = self.request_json("POST", "/render", {"plan_id": plan_id, "purpose": "publication"})
            if status_pub_still_blocked not in (400, 403):
                self.log_failure(name, f"GATING VIOLATION: Review approval bypassed rights check! Got HTTP {status_pub_still_blocked}")
                return

            self.log_success(
                name,
                f"Preview allowed (200), publication gated (HTTP {status_pub_blocked}), review recorded (reviewer='automated_test'), rights bypass blocked",
            )
        except Exception as e:
            self.log_failure(name, str(e))

    def run_all(self) -> bool:
        print(f"\n{Colors.BOLD}=== SignaCulture AI Integration & Smoke Test Suite ==={Colors.RESET}")
        print(f"Target Backend: {self.base_url}")
        print("Checking backend reachability...")
        self.verify_backend_reachable()

        print("\nExecuting contract verification tests:")
        self.test_health()
        entries = self.test_glossary()
        self.test_motions_and_assets(entries)
        self.test_text_analysis_known_and_unknowns()
        plan = self.test_compose_animation()
        self.test_audio_analysis_sine_silence()
        self.test_review_and_publication_gating(plan)

        total = self.passed + self.failed
        print(f"\n{Colors.BOLD}=== Test Summary ==={Colors.RESET}")
        print(f"Total:  {total}")
        print(f"Passed: {Colors.GREEN}{self.passed}{Colors.RESET}")
        print(f"Failed: {Colors.RED if self.failed > 0 else Colors.GREEN}{self.failed}{Colors.RESET}")

        if self.failed > 0:
            print("\nFailures:")
            for err in self.errors:
                print(f" - {err}")
            return False
        else:
            print(f"\n{Colors.GREEN}{Colors.BOLD}All integration and smoke tests passed against real backend!{Colors.RESET}\n")
            return True


def main() -> None:
    parser = argparse.ArgumentParser(description="SignaCulture AI Smoke Test Suite")
    parser.add_argument(
        "--base-url",
        default=os.environ.get("API_URL", "http://localhost:8000"),
        help="Base URL of running FastAPI backend (default: http://localhost:8000 or $API_URL)",
    )
    args = parser.parse_args()

    runner = SmokeTestRunner(args.base_url)
    success = runner.run_all()
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
