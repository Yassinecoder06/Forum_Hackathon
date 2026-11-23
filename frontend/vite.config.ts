import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";
import { env } from "node:process";
import { readFileSync } from "node:fs";

const apiTarget = env.SIGNACULTURE_API_TARGET || "http://127.0.0.1:8000";
// Phone access (AR companion): camera, microphone and WebXR need https. Either `tailscale serve` in front of this dev
// server (its *.ts.net name must be an allowed host), or a local certificate given by these two variables.
const devCert = env.SIGNACULTURE_DEV_CERT, devKey = env.SIGNACULTURE_DEV_KEY;

// https://vite.dev/config/
export default defineConfig(({ command }) => ({
  // The built Studio is served under /studio/; the showcase site owns /.
  base: command === "build" ? "/studio/" : "/",
  plugins: [react()],
  server: {
    port: 5173,
    host: "127.0.0.1",
    allowedHosts: [".ts.net"],
    https: devCert && devKey ? { cert: readFileSync(devCert), key: readFileSync(devKey) } : undefined,
    proxy: {
      "/webcam": {
        target: env.SIGNACULTURE_WEBCAM_TARGET || "http://127.0.0.1:5180",
        changeOrigin: true,
      },
      "/api": {
        target: apiTarget,
        changeOrigin: true,
        rewrite: (path) => path.replace(/^\/api/, ""),
      },
      // Direct proxy for /assets in case any asset is requested directly
      "/assets": {
        target: apiTarget,
        changeOrigin: true,
      },
    },
  },
}));
