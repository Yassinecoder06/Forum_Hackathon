import * as THREE from "three";

/**
 * Procedural café / atelier backdrop: warm terracotta wall, defocused string
 * lights, a dark wood wainscot and a soft halo behind the signer. It is kept
 * mid-toned and low-contrast so hands stay readable against it, and the bokeh
 * avoids the central signing space.
 */
export function createCafeBackdrop(width = 1320, height = 1000) {
  const canvas = document.createElement("canvas");
  canvas.width = width;
  canvas.height = height;
  const ctx = canvas.getContext("2d")!;
  let seed = 7;
  const rand = () => ((seed = (seed * 16807) % 2147483647) - 1) / 2147483646;

  const wall = ctx.createLinearGradient(0, 0, 0, height);
  wall.addColorStop(0, "#2d4e5c");
  wall.addColorStop(0.45, "#467f97");
  wall.addColorStop(0.7, "#3c7087");
  wall.addColorStop(1, "#223d48");
  ctx.fillStyle = wall;
  ctx.fillRect(0, 0, width, height);

  // Plaster texture: faint blotches for a hand-finished wall.
  for (let i = 0; i < 90; i++) {
    const x = rand() * width, y = rand() * height * 0.72, r = 40 + rand() * 140;
    const g = ctx.createRadialGradient(x, y, 0, x, y, r);
    const light = rand() > 0.5;
    g.addColorStop(0, light ? "rgba(173, 228, 252,.05)" : "rgba(20, 47, 58,.06)");
    g.addColorStop(1, "rgba(0,0,0,0)");
    ctx.fillStyle = g;
    ctx.fillRect(x - r, y - r, r * 2, r * 2);
  }

  // Warm halo behind the head and shoulders separates the silhouette.
  const halo = ctx.createRadialGradient(width / 2, height * 0.36, 20, width / 2, height * 0.4, height * 0.55);
  halo.addColorStop(0, "rgba(133, 207, 239,.55)");
  halo.addColorStop(0.5, "rgba(100, 176, 209,.22)");
  halo.addColorStop(1, "rgba(0,0,0,0)");
  ctx.fillStyle = halo;
  ctx.fillRect(0, 0, width, height);

  // Wood wainscot with blurred vertical planks and a brass rail.
  const railY = height * 0.74;
  const wood = ctx.createLinearGradient(0, railY, 0, height);
  wood.addColorStop(0, "#264958");
  wood.addColorStop(1, "#14262e");
  ctx.fillStyle = wood;
  ctx.fillRect(0, railY, width, height - railY);
  ctx.save();
  ctx.filter = "blur(3px)";
  for (let x = 0; x < width; x += 66) {
    ctx.fillStyle = `rgba(7, 16, 19,${0.18 + rand() * 0.1})`;
    ctx.fillRect(x, railY + 8, 3, height - railY);
    ctx.fillStyle = "rgba(154, 222, 251,.05)";
    ctx.fillRect(x + 4, railY + 8, 2, height - railY);
  }
  ctx.fillStyle = "rgba(97, 175, 209,.55)";
  ctx.fillRect(0, railY - 4, width, 7);
  ctx.restore();

  // Defocused string lights along a catenary, plus scattered bokeh in the corners.
  const bokeh = (x: number, y: number, r: number, alpha: number, hue: string) => {
    const g = ctx.createRadialGradient(x, y, 0, x, y, r);
    g.addColorStop(0, `rgba(${hue},${alpha})`);
    g.addColorStop(0.65, `rgba(${hue},${alpha * 0.75})`);
    g.addColorStop(1, `rgba(${hue},0)`);
    ctx.fillStyle = g;
    ctx.beginPath();
    ctx.arc(x, y, r, 0, Math.PI * 2);
    ctx.fill();
  };
  const hues = ["255,214,150", "255,190,120", "255,236,200", "255,170,105"];
  ctx.globalCompositeOperation = "lighter";
  for (let i = 0; i <= 18; i++) {
    const u = i / 18, x = u * width, sag = Math.cosh((u - 0.5) * 2.4) - 1;
    const y = height * (0.04 + 0.16 - sag * 0.09);
    const central = Math.abs(u - 0.5) < 0.16;
    bokeh(x + (rand() - 0.5) * 18, y, 16 + rand() * 18, central ? 0.12 : 0.3, hues[i % hues.length]);
  }
  for (let i = 0; i < 26; i++) {
    const side = rand() < 0.5 ? rand() * 0.3 : 0.7 + rand() * 0.3;
    bokeh(side * width, height * (0.08 + rand() * 0.55), 10 + rand() * 34, 0.06 + rand() * 0.14, hues[i % hues.length]);
  }
  ctx.globalCompositeOperation = "source-over";

  // Vignette.
  const vignette = ctx.createRadialGradient(width / 2, height * 0.45, height * 0.35, width / 2, height * 0.45, height * 0.95);
  vignette.addColorStop(0, "rgba(0,0,0,0)");
  vignette.addColorStop(1, "rgba(7, 19, 24,.55)");
  ctx.fillStyle = vignette;
  ctx.fillRect(0, 0, width, height);

  const texture = new THREE.CanvasTexture(canvas);
  texture.colorSpace = THREE.SRGBColorSpace;
  return texture;
}
