import * as THREE from "three";
import { AvatarLoadError, loadAvatarRig, type AvatarFrame, type AvatarRig } from "../AvatarPlayer/avatarRig";
import { SCALE_RANGE, easeAngle, facingYaw, fitSize, flipRows, type Placement } from "../../utils/companion";

/**
 * The AR companion's 3D view: the same MetaHuman and avatar runtime as the Studio (avatarRig.ts), drawn on a
 * transparent canvas, in one of two ways:
 *
 * - "xr" (Android Chrome, WebXR immersive-ar): the phone's camera is the background, a hit-test reticle follows real
 *   surfaces, a tap places the avatar there (life size), two fingers scale / turn it, and it keeps facing the person.
 *   With the "camera-access" feature, `capture()` reads the camera image of the current XR frame for the brain.
 * - "overlay" (iPhone Safari and every browser without WebXR AR): the page shows the rear camera video full screen;
 *   this canvas sits on top with a fixed camera, the avatar standing at a fixed spot (two fingers scale it, one
 *   finger slides it sideways).
 *
 * One render loop (renderer.setAnimationLoop) in both modes; nothing is drawn when the tab is hidden.
 */
export type StageMode = "overlay" | "xr";
export type StageStatus = { ready: boolean; placed: boolean; surface: boolean; error?: string };

type CameraBinding = { getCameraImage(camera: unknown): WebGLTexture | null };
type ViewWithCamera = XRView & { camera?: { width: number; height: number } };

export class CompanionStage {
  readonly renderer: THREE.WebGLRenderer;
  readonly scene = new THREE.Scene();
  /** Placed root: position on the surface, yaw toward the person, scale. The rig's model hangs under it. */
  readonly anchor = new THREE.Group();
  private camera: THREE.PerspectiveCamera;
  private rig: AvatarRig | null = null;
  private reticle: THREE.Mesh;
  private shadow: THREE.Mesh;
  mode: StageMode = "overlay";
  private session: XRSession | null = null;
  private hitSource: XRHitTestSource | null = null;
  private binding: CameraBinding | null = null;
  private pendingCapture: ((frame: { pixels: Uint8Array; width: number; height: number } | null) => void)[] = [];
  private disposed = false;
  private lastT = -1;
  private resizeObserver: ResizeObserver;
  /** Faces the person on its own; the two-finger twist adds this offset. */
  private yaw = 0;
  readonly placement: Placement = { scale: 1, yaw: 0 };
  /** Two-finger gestures also end as XR "select" events: the page sets this so they never move the avatar. */
  ignoreSelectUntil = 0;
  /** Overlay mode: sideways slide (metres, in the fixed camera's frame). */
  offsetX = 0;
  /** Asked once per drawn frame (also inside the XR loop): what the avatar signs now and where it looks. */
  frameSource: () => AvatarFrame = () => ({ motion: null, currentTimeMs: 0 });
  status: StageStatus = { ready: false, placed: false, surface: false };
  onStatus: (s: StageStatus) => void = () => {};

  constructor(private host: HTMLElement) {
    this.renderer = new THREE.WebGLRenderer({ antialias: true, alpha: true, preserveDrawingBuffer: false });
    this.renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
    this.renderer.setClearColor(0x000000, 0);
    this.renderer.outputColorSpace = THREE.SRGBColorSpace;
    this.renderer.toneMapping = THREE.ACESFilmicToneMapping;
    this.renderer.shadowMap.enabled = true;
    this.renderer.shadowMap.type = THREE.PCFSoftShadowMap;
    host.appendChild(this.renderer.domElement);
    this.camera = new THREE.PerspectiveCamera(40, 1, 0.01, 40);
    // Neutral daylight that sits well in most rooms (the Studio's café lighting would look orange in a real scene).
    this.scene.add(new THREE.HemisphereLight(0xfffaf2, 0x5a5048, 1.9));
    const key = new THREE.DirectionalLight(0xffffff, 2.2);
    key.position.set(-1.2, 3, 2);
    key.castShadow = true;
    key.shadow.mapSize.set(1024, 1024);
    key.shadow.camera.left = key.shadow.camera.bottom = -1.2;
    key.shadow.camera.right = key.shadow.camera.top = 1.2;
    key.shadow.radius = 6;
    key.shadow.bias = -0.0005;
    this.anchor.add(key, key.target);
    const fill = new THREE.DirectionalLight(0xfff1e4, 0.9);
    fill.position.set(1.8, 1.6, 1.5);
    this.anchor.add(fill);
    // Contact shadow on the real surface (XR only: the overlay view does not know where the floor is).
    this.shadow = new THREE.Mesh(new THREE.CircleGeometry(0.9, 48).rotateX(-Math.PI / 2), new THREE.ShadowMaterial({ opacity: 0.32 }));
    this.shadow.receiveShadow = true;
    this.shadow.visible = false;
    this.anchor.add(this.shadow);
    this.scene.add(this.anchor);
    // Placement reticle (a ring on the detected surface).
    this.reticle = new THREE.Mesh(
      new THREE.RingGeometry(0.11, 0.14, 40).rotateX(-Math.PI / 2),
      new THREE.MeshBasicMaterial({ color: 0xf0a37f, transparent: true, opacity: 0.9 }),
    );
    this.reticle.matrixAutoUpdate = false;
    this.reticle.visible = false;
    this.scene.add(this.reticle);
    this.resizeObserver = new ResizeObserver(() => this.resize());
    this.resizeObserver.observe(host);
    this.resize();
    loadAvatarRig("/api/avatar/model", () => this.disposed)
      .then((rig) => {
        if (!rig) return;
        this.rig = rig;
        this.anchor.add(rig.model);
        this.setStatus({ ready: true });
      })
      .catch((e) => this.setStatus({ error: e instanceof AvatarLoadError && e.kind === "skeleton" ? e.message : "Personnage 3D indisponible." }));
    this.useOverlay();
  }

  private setStatus(patch: Partial<StageStatus>) {
    this.status = { ...this.status, ...patch };
    this.onStatus(this.status);
  }

  private resize() {
    if (this.session) return;            // the XR session owns the size
    const w = this.host.clientWidth, h = this.host.clientHeight;
    if (!w || !h) return;
    this.renderer.setSize(w, h);
    this.camera.aspect = w / h;
    // Portrait phone: widen the vertical field so the signing space (hands to the sides) stays in view.
    this.camera.fov = THREE.MathUtils.radToDeg(2 * Math.atan(Math.tan(THREE.MathUtils.degToRad(20)) * Math.max(1, 0.6 / this.camera.aspect)));
    this.camera.updateProjectionMatrix();
  }

  /** Fixed view over the camera video: the avatar stands at the lower middle, from the thighs up, facing the person. */
  useOverlay() {
    this.mode = "overlay";
    this.camera.position.set(0, 1.3, 2.7);
    this.camera.lookAt(0, 1.05, 0);
    this.anchor.position.set(0, 0, 0);
    this.anchor.rotation.set(0, 0, 0);
    this.shadow.visible = false;
    this.reticle.visible = false;
    this.renderer.setAnimationLoop((t) => this.render(t, undefined));
    this.setStatus({ placed: true, surface: false });
  }

  /** Android Chrome: start the immersive AR session (call from a tap). `overlay` is the page's DOM overlay root. */
  async startXR(overlay: HTMLElement) {
    const xr = navigator.xr;
    if (!xr) throw new Error("WebXR indisponible");
    const session = await xr.requestSession("immersive-ar", {
      requiredFeatures: ["hit-test"],
      optionalFeatures: ["dom-overlay", "camera-access"],
      domOverlay: { root: overlay },
    } as XRSessionInit);
    this.session = session;
    this.mode = "xr";
    this.renderer.xr.enabled = true;
    // "local" is granted to every immersive session; the hit-test poses place the avatar on the real surface in it.
    this.renderer.xr.setReferenceSpaceType("local");
    await this.renderer.xr.setSession(session);
    const viewer = await session.requestReferenceSpace("viewer");
    this.hitSource = (await session.requestHitTestSource?.({ space: viewer })) ?? null;
    const Binding = (window as unknown as { XRWebGLBinding?: new (s: XRSession, gl: WebGL2RenderingContext) => CameraBinding }).XRWebGLBinding;
    try {
      this.binding = Binding && (session as unknown as { enabledFeatures?: string[] }).enabledFeatures?.includes("camera-access")
        ? new Binding(session, this.renderer.getContext() as WebGL2RenderingContext) : null;
    } catch {
      this.binding = null;
    }
    this.anchor.visible = false;
    this.setStatus({ placed: false, surface: false });
    session.addEventListener("select", () => this.placeAtReticle());
    session.addEventListener("end", () => {
      this.session = null;
      this.hitSource = null;
      this.binding = null;
      this.renderer.xr.enabled = false;
      this.anchor.visible = true;
      this.anchor.scale.setScalar(1);
      this.placement.scale = 1;
      this.placement.yaw = 0;
      this.flushCaptures(null);
      if (!this.disposed) {
        this.resize();
        this.useOverlay();
      }
      this.onSessionEnd();
    });
    this.renderer.setAnimationLoop((t, frame) => this.render(t, frame));
    return session;
  }

  onSessionEnd: () => void = () => {};

  get xrActive() {
    return Boolean(this.session);
  }

  /** True when `capture()` can read the camera image (overlay mode reads the page's video instead). */
  get cameraAccess() {
    return Boolean(this.binding);
  }

  endXR() {
    void this.session?.end();
  }

  /** A tap in AR: put the avatar on the surface under the reticle (it can be moved by tapping elsewhere). */
  placeAtReticle() {
    if (!this.reticle.visible || performance.now() < this.ignoreSelectUntil) return;
    this.anchor.position.setFromMatrixPosition(this.reticle.matrix);
    this.anchor.visible = true;
    this.shadow.visible = true;
    this.setStatus({ placed: true });
  }

  setPlacement(p: Placement) {
    this.placement.scale = THREE.MathUtils.clamp(p.scale, SCALE_RANGE[0], SCALE_RANGE[1]);
    this.placement.yaw = p.yaw;
  }

  /**
   * The current camera image, longest side <= maxSide (RGBA, top-down), or null. XR: from the camera-access image of
   * the next XR frame; overlay: the page grabs its own <video> (see grabVideo).
   */
  capture(maxSide = 512): Promise<{ pixels: Uint8Array; width: number; height: number } | null> {
    if (!this.session || !this.binding) return Promise.resolve(null);
    return new Promise((resolve) => {
      this.pendingCapture.push(resolve);
      this.captureSide = maxSide;
    });
  }

  private captureSide = 512;

  private flushCaptures(frame: { pixels: Uint8Array; width: number; height: number } | null) {
    const list = this.pendingCapture;
    this.pendingCapture = [];
    list.forEach((r) => r(frame));
  }

  /** Read the XR camera texture (valid only inside this XR frame), downscaled on the GPU by a framebuffer blit. */
  private readCamera(frame: XRFrame) {
    const ref = this.renderer.xr.getReferenceSpace();
    const pose = ref ? frame.getViewerPose(ref) : null;
    const view = pose?.views.find((v) => (v as ViewWithCamera).camera) as ViewWithCamera | undefined;
    if (!view?.camera || !this.binding) return this.flushCaptures(null);
    const gl = this.renderer.getContext() as WebGL2RenderingContext;
    const texture = this.binding.getCameraImage(view.camera);
    if (!texture) return this.flushCaptures(null);
    const { width, height } = view.camera;
    const size = fitSize(width, height, this.captureSide);
    const src = gl.createFramebuffer(), dst = gl.createFramebuffer(), rb = gl.createRenderbuffer();
    let result: { pixels: Uint8Array; width: number; height: number } | null = null;
    try {
      gl.bindFramebuffer(gl.READ_FRAMEBUFFER, src);
      gl.framebufferTexture2D(gl.READ_FRAMEBUFFER, gl.COLOR_ATTACHMENT0, gl.TEXTURE_2D, texture, 0);
      gl.bindRenderbuffer(gl.RENDERBUFFER, rb);
      gl.renderbufferStorage(gl.RENDERBUFFER, gl.RGBA8, size.width, size.height);
      gl.bindFramebuffer(gl.DRAW_FRAMEBUFFER, dst);
      gl.framebufferRenderbuffer(gl.DRAW_FRAMEBUFFER, gl.COLOR_ATTACHMENT0, gl.RENDERBUFFER, rb);
      gl.blitFramebuffer(0, 0, width, height, 0, 0, size.width, size.height, gl.COLOR_BUFFER_BIT, gl.LINEAR);
      gl.bindFramebuffer(gl.READ_FRAMEBUFFER, dst);
      const pixels = new Uint8Array(size.width * size.height * 4);
      gl.readPixels(0, 0, size.width, size.height, gl.RGBA, gl.UNSIGNED_BYTE, pixels);
      result = { pixels: flipRows(pixels, size.width, size.height), width: size.width, height: size.height };
    } catch {
      result = null;
    } finally {
      gl.bindFramebuffer(gl.FRAMEBUFFER, null);
      gl.deleteFramebuffer(src);
      gl.deleteFramebuffer(dst);
      gl.deleteRenderbuffer(rb);
      this.renderer.resetState();
    }
    this.flushCaptures(result);
  }

  private render(t: number, frame: XRFrame | undefined) {
    if (this.disposed) return;
    const dt = this.lastT < 0 ? 0 : Math.min(0.1, (t - this.lastT) / 1000);
    this.lastT = t;
    if (frame && this.session) {
      if (this.pendingCapture.length) this.readCamera(frame);
      const ref = this.renderer.xr.getReferenceSpace();
      const hit = this.hitSource && ref ? frame.getHitTestResults(this.hitSource)[0] : undefined;
      const hitPose = hit && ref ? hit.getPose(ref) : undefined;
      this.reticle.visible = Boolean(hitPose);
      if (hitPose) this.reticle.matrix.fromArray(hitPose.transform.matrix);
      if (Boolean(hitPose) !== this.status.surface) this.setStatus({ surface: Boolean(hitPose) });
      // Keep facing the person: turn toward the phone (the XR camera), plus the two-finger twist offset.
      const cam = this.renderer.xr.getCamera();
      const eye = new THREE.Vector3().setFromMatrixPosition(cam.matrixWorld);
      this.yaw = easeAngle(this.yaw, facingYaw(this.anchor.position, eye) + this.placement.yaw, dt);
      this.anchor.rotation.set(0, this.yaw, 0);
    } else {
      this.anchor.position.set(this.offsetX, 0, 0);
      this.anchor.rotation.set(0, this.placement.yaw, 0);
    }
    this.anchor.scale.setScalar(this.placement.scale);
    if (this.rig) this.rig.pose(this.frameSource(), true);
    this.renderer.render(this.scene, this.camera);
    this.rig?.afterDraw();
  }

  dispose() {
    this.disposed = true;
    this.flushCaptures(null);
    void this.session?.end().catch(() => {});
    this.renderer.setAnimationLoop(null);
    this.resizeObserver.disconnect();
    this.rig?.dispose();
    this.reticle.geometry.dispose();
    (this.reticle.material as THREE.Material).dispose();
    this.shadow.geometry.dispose();
    (this.shadow.material as THREE.Material).dispose();
    this.renderer.dispose();
    this.renderer.domElement.remove();
  }
}

/** Grab the current picture of a <video> as a JPEG data URL, longest side <= maxSide (null before the first frame). */
export function grabVideo(video: HTMLVideoElement, maxSide = 512, quality = 0.7): string | null {
  const size = fitSize(video.videoWidth, video.videoHeight, maxSide);
  if (!size.width) return null;
  const canvas = document.createElement("canvas");
  canvas.width = size.width;
  canvas.height = size.height;
  const g = canvas.getContext("2d");
  if (!g) return null;
  g.drawImage(video, 0, 0, size.width, size.height);
  return canvas.toDataURL("image/jpeg", quality);
}

/** RGBA pixels -> JPEG data URL. */
export function pixelsToJpeg(frame: { pixels: Uint8Array; width: number; height: number }, quality = 0.7): string | null {
  const canvas = document.createElement("canvas");
  canvas.width = frame.width;
  canvas.height = frame.height;
  const g = canvas.getContext("2d");
  if (!g) return null;
  g.putImageData(new ImageData(new Uint8ClampedArray(frame.pixels.buffer as ArrayBuffer), frame.width, frame.height), 0, 0);
  return canvas.toDataURL("image/jpeg", quality);
}

/** A tiny grayscale thumbnail of an image source, for the scene-change watcher. */
export function thumbnail(source: CanvasImageSource, width: number, height: number): Uint8ClampedArray | null {
  const canvas = document.createElement("canvas");
  canvas.width = width;
  canvas.height = height;
  const g = canvas.getContext("2d", { willReadFrequently: true });
  if (!g) return null;
  g.drawImage(source, 0, 0, width, height);
  return g.getImageData(0, 0, width, height).data;
}

/** Same thumbnail from RGBA pixels (XR camera image). */
export function thumbnailFromPixels(frame: { pixels: Uint8Array; width: number; height: number }, width: number, height: number) {
  const canvas = document.createElement("canvas");
  canvas.width = frame.width;
  canvas.height = frame.height;
  const g = canvas.getContext("2d");
  if (!g) return null;
  g.putImageData(new ImageData(new Uint8ClampedArray(frame.pixels.buffer as ArrayBuffer), frame.width, frame.height), 0, 0);
  return thumbnail(canvas, width, height);
}
