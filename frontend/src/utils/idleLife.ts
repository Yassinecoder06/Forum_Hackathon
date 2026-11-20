import * as THREE from "three";
import { findRigBone, RPM_PROFILE, type RigProfile } from "./rigProfiles";

/**
 * Display-only "aliveness" layer: breathing, blinking, gaze saccades and a slow
 * head sway. It is applied on top of the retargeted pose every frame (the
 * retargeter resets every bone first, so nothing accumulates) and never touches
 * the captured motion data. Head motion is damped while signing because head
 * movement can carry grammatical meaning in LSF.
 */
export type IdleSample = {
  blink: number;
  gaze: { x: number; y: number };
};

const random = (min: number, max: number) => min + Math.random() * (max - min);

export class IdleLife {
  private spine1?: THREE.Object3D;
  private spine2?: THREE.Object3D;
  private neck?: THREE.Object3D;
  private head?: THREE.Object3D;
  private nextBlink = random(1.2, 3);
  private blinkStart = -1;
  private doubleBlink = false;
  private nextSaccade = random(0.8, 2);
  private gazeTarget = new THREE.Vector2();
  private gaze = new THREE.Vector2();
  private last = -1;
  private signingWeight = 0;
  private euler = new THREE.Euler();
  private delta = new THREE.Quaternion();

  constructor(root: THREE.Object3D, profile: RigProfile = RPM_PROFILE) {
    this.spine1 = findRigBone(root, profile, "Spine1");
    this.spine2 = findRigBone(root, profile, "Spine2");
    this.neck = findRigBone(root, profile, "Neck");
    this.head = findRigBone(root, profile, "Head");
  }

  private rotate(bone: THREE.Object3D | undefined, x: number, y: number, z: number) {
    if (!bone) return;
    this.euler.set(x, y, z);
    bone.quaternion.multiply(this.delta.setFromEuler(this.euler));
  }

  /** Phrase-aware blink (interpreters blink at clause ends and when the hands return to rest): starts a blink
   * now unless one is running or happened in the last 0.8 s. */
  requestBlink(time: number) {
    if (this.blinkStart >= 0 || time - this.lastBlinkAt < 0.8) return;
    this.blinkStart = time;
    this.doubleBlink = false;
  }
  private lastBlinkAt = -10;

  update(time: number, signing: boolean): IdleSample {
    const dt = this.last < 0 ? 0 : Math.min(0.1, time - this.last);
    this.last = time;
    // Ease between idle and signing amplitudes instead of switching abruptly.
    this.signingWeight += ((signing ? 1 : 0) - this.signingWeight) * Math.min(1, dt * 3);
    const calm = 1 - 0.7 * this.signingWeight;

    // Breathing: ~14 breaths/min, chest leads the upper spine slightly.
    const breath = Math.sin((time * Math.PI * 2) / 4.3);
    this.rotate(this.spine1, -0.006 * breath, 0, 0);
    this.rotate(this.spine2, -0.01 * Math.sin((time * Math.PI * 2) / 4.3 - 0.5), 0, 0);

    // Slow, non-repeating head sway from incommensurate sines.
    const yaw = (0.03 * Math.sin(time * 0.31) + 0.012 * Math.sin(time * 0.77 + 1.3)) * calm;
    const pitch = (0.018 * Math.sin(time * 0.23 + 0.7) + 0.008 * Math.sin(time * 0.61)) * calm;
    const roll = 0.014 * Math.sin(time * 0.19 + 2.1) * calm;
    this.rotate(this.neck, pitch * 0.4, yaw * 0.4, roll * 0.4);
    this.rotate(this.head, pitch * 0.6, yaw * 0.6, roll * 0.6);

    // Gaze saccades: quick jumps between small fixation offsets around the viewer.
    if (time >= this.nextSaccade) {
      this.gazeTarget.set(random(-0.22, 0.22), random(-0.1, 0.14));
      if (Math.random() < 0.35) this.gazeTarget.set(0, 0);
      this.nextSaccade = time + random(0.9, 3.2);
    }
    this.gaze.lerp(this.gazeTarget, Math.min(1, dt * 22));

    // Blinks: 150 ms lid closure every 2.5-5.5 s, occasionally doubled.
    if (this.blinkStart < 0 && time >= this.nextBlink) {
      this.blinkStart = time;
      this.doubleBlink = Math.random() < 0.15;
    }
    let blink = 0;
    if (this.blinkStart >= 0) {
      const u = (time - this.blinkStart) / 0.15;
      if (u >= 1) {
        if (this.doubleBlink) {
          this.doubleBlink = false;
          this.blinkStart = time + 0.08;
        } else {
          this.blinkStart = -1;
          this.lastBlinkAt = time;
          this.nextBlink = time + random(2.5, 5.5);
        }
      } else if (u >= 0) blink = Math.sin(Math.PI * u);
    }
    return { blink, gaze: { x: this.gaze.x, y: this.gaze.y } };
  }
}
