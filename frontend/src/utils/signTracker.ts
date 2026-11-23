/**
 * MediaPipe on the webcam, in the browser (camera frames never leave the page): pose + both hands + face, the same
 * three Google landmarkers as the webcam capture module (webcam/src/detector.js, @mediapipe/tasks-vision 0.10.32).
 * Each video frame -> the 157-point corpus layout (utils/signCapture.ts). The models and the WebAssembly
 * runtime are served by this app (public/mediapipe, `npm run setup:mediapipe`): nothing is fetched from the internet.
 */
import type { FaceLandmarker, HandLandmarker, PoseLandmarker } from "@mediapipe/tasks-vision";
import { assembleFrame, assignHands, type Landmark } from "./signCapture";

export const MEDIAPIPE_VERSION = "0.10.32";
// Everything MediaPipe needs is served by this app (frontend/public/mediapipe, filled by `npm run setup:mediapipe`).
const BASE = import.meta.env.BASE_URL;
const WASM = `${BASE}mediapipe/wasm`;
const MODELS = {
  pose: `${BASE}mediapipe/models/pose_landmarker_lite.task`,
  hands: `${BASE}mediapipe/models/hand_landmarker.task`,
  face: `${BASE}mediapipe/models/face_landmarker.task`,
};

export type TrackedFrame = {
  t_ms: number;
  points: Float32Array;
  pose: Landmark[] | null;
  left: Landmark[] | null;
  right: Landmark[] | null;
  ms: number;
};

export class SignTracker {
  private pose?: PoseLandmarker;
  private hands?: HandLandmarker;
  private face?: FaceLandmarker;
  private lastTs = -1;
  private frameNo = 0;
  private lastFace: Landmark[] | null = null;
  delegate: "GPU" | "CPU" = "GPU";

  async init() {
    const vision = await import("@mediapipe/tasks-vision");
    const files = await vision.FilesetResolver.forVisionTasks(WASM);
    const { pose, hands, face } = MODELS;
    const make = async (delegate: "GPU" | "CPU") => {
      this.pose = await vision.PoseLandmarker.createFromOptions(files, { baseOptions: { modelAssetPath: pose, delegate },
        runningMode: "VIDEO", numPoses: 1, minPoseDetectionConfidence: 0.5, minPosePresenceConfidence: 0.5, minTrackingConfidence: 0.5 });
      this.hands = await vision.HandLandmarker.createFromOptions(files, { baseOptions: { modelAssetPath: hands, delegate },
        runningMode: "VIDEO", numHands: 2, minHandDetectionConfidence: 0.4, minHandPresenceConfidence: 0.4, minTrackingConfidence: 0.4 });
      this.face = await vision.FaceLandmarker.createFromOptions(files, { baseOptions: { modelAssetPath: face, delegate },
        runningMode: "VIDEO", numFaces: 1, minFaceDetectionConfidence: 0.3, minFacePresenceConfidence: 0.4 });
    };
    try {
      await make("GPU");
    } catch {
      this.close();
      this.delegate = "CPU";
      await make("CPU");
    }
  }

  /** One video frame -> tracked landmarks (null if the frame is not new). */
  detect(video: HTMLVideoElement, now = performance.now()): TrackedFrame | null {
    if (!this.pose || !this.hands || !this.face || video.readyState < 2) return null;
    const ts = Math.max(now, this.lastTs + 1);
    this.lastTs = ts;
    const t0 = performance.now();
    const p = this.pose.detectForVideo(video, ts);
    const h = this.hands.detectForVideo(video, ts);
    // the face (lips, brows, eyes) changes slowly next to the hands: every other frame is enough
    if (this.frameNo++ % 2 === 0) this.lastFace = this.face.detectForVideo(video, ts).faceLandmarks[0] ?? null;
    const pose = (p.landmarks[0] as Landmark[] | undefined) ?? null;
    const { left, right } = assignHands(pose, h.landmarks as Landmark[][]);
    const points = assembleFrame(pose, left, right, this.lastFace, video.videoWidth, video.videoHeight);
    return { t_ms: now, points, pose, left, right, ms: performance.now() - t0 };
  }

  close() {
    this.pose?.close();
    this.hands?.close();
    this.face?.close();
    this.pose = this.hands = this.face = undefined;
  }
}
