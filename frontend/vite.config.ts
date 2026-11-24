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
  plugins: [react(), {
    // presentation.* / interprete.* open the live interpreter alone, not the Studio.
    name: "presentation-host",
    configureServer(server) {
      server.middlewares.use((req, res, next) => {
        // SIGNACULTURE_HOME_PAGE: a server that shows one page only (e.g. the transcription page on its own port)
        const home = env.SIGNACULTURE_HOME_PAGE || (/^(presentation|interprete)\./.test(req.headers.host ?? "") ? "/interprete/" : "");
        if (home && (req.url === "/" || req.url === "/index.html")) {
          res.statusCode = 302;
          res.setHeader("Location", home);
          res.end();
        } else next();
      });
    },
  }],
  server: {
    port: 5173,
    host: "127.0.0.1",
    allowedHosts: [".ts.net", ".trycloudflare.com", ".yassinejemmali.me", ".sc", ".local"],
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
