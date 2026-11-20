import * as THREE from "three";

/**
 * Display-only "aliveness" layer: blinks and gaze saccades (morph targets only).
 * Breathing, weight shift and head sway are part of the idle animation clip
 * (utils/idleAnimation.ts), which is blended out of driven bones while signing
 * because head movement can carry grammatical meaning in LSF. Never touches the
 * captured motion data.
 */
export type IdleSample = {
  blink: number;
  gaze: { x: number; y: number };
};

const random = (min: number, max: number) => min + Math.random() * (max - min);

export class IdleLife {
  private nextBlink = random(1.2, 3);
  private blinkStart = -1;
  private doubleBlink = false;
  private nextSaccade = random(0.8, 2);
  private gazeTarget = new THREE.Vector2();
  private gaze = new THREE.Vector2();
  private last = -1;
  constructor(_root: THREE.Object3D) {}

  update(time: number, _signing: boolean): IdleSample {
    const dt = this.last < 0 ? 0 : Math.min(0.1, time - this.last);
    this.last = time;
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
          this.nextBlink = time + random(2.5, 5.5);
        }
      } else if (u >= 0) blink = Math.sin(Math.PI * u);
    }
    return { blink, gaze: { x: this.gaze.x, y: this.gaze.y } };
  }
}
