/**
 * MediaPipe on the webcam, in the browser (camera frames never leave the page): pose + both hands + face, the same
 * three Google landmarkers as the webcam capture module (webcam/src/detector.js, @mediapipe/tasks-vision 0.10.32).
 * Each video frame -> the 157-point corpus layout (utils/signCapture.ts). The models load from the local webcam
 * module when it is set up (/webcam/models, `npm --prefix webcam run setup`), else from Google's pinned model URLs.
 */
import type { FaceLandmarker, HandLandmarker, PoseLandmarker } from "@mediapipe/tasks-vision";
import { assembleFrame, assignHands, type Landmark } from "./signCapture";

export const MEDIAPIPE_VERSION = "0.10.32";
const WASM = `https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@${MEDIAPIPE_VERSION}/wasm`;
const MODELS = {
  pose: "https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_lite/float16/1/pose_landmarker_lite.task",
  hands: "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task",
  face: "https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task",
};

export type TrackedFrame = {
  t_ms: number;
  points: Float32Array;
  pose: Landmark[] | null;
  left: Landmark[] | null;
  right: Landmark[] | null;
  ms: number;
};

async function modelUrl(name: keyof typeof MODELS) {
  const local = `/webcam/models/${name}.task`;
  try {
    const r = await fetch(local, { method: "HEAD", signal: AbortSignal.timeout(1500) });
    const type = r.headers.get("content-type") ?? "";
    if (r.ok && !type.includes("text/html")) return local;
  } catch {
    /* local webcam module not running: use the pinned model */
  }
  return MODELS[name];
}

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
    const [pose, hands, face] = await Promise.all([modelUrl("pose"), modelUrl("hands"), modelUrl("face")]);
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
