import { useEffect, useRef, useState } from "react";
import { Download, Loader2 } from "lucide-react";
import { request } from "../services/api";
type Result = {
  glosses: string[];
  frames: number;
  fps: number;
  pose_url: string;
  video_url: string;
  report_url: string;
  translation_validated: false;
};
export function LexicalPipelinePanel({ text }: { text: string }) {
  const [result, setResult] = useState<Result | null>(null),
    [busy, setBusy] = useState(false),
    [error, setError] = useState("");
  const version = useRef(0);
  useEffect(() => {
    version.current++;
    setResult(null);
    setError("");
  }, [text]);
  async function run() {
    const current = version.current;
    setBusy(true);
    setResult(null);
    setError("");
    try {
      const response = await request<Result>("/pipeline/lexical", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ text }),
      });
      if (current === version.current) setResult(response);
    } catch (e) {
      if (current === version.current)
        setError(e instanceof Error ? e.message : "Séquence indisponible.");
    } finally {
      setBusy(false);
    }
  }
  return (
    <details className="concept-composer">
      <summary>Exporter la chaîne glosses → poses</summary>
      <p>
        Générez un fichier .pose compatible avec spoken-to-signed et une vidéo
        de diagnostic du squelette. Les mots manquants ou ambigus bloquent la
        génération ; aucune langue de secours n’est utilisée.
      </p>
      <button
        className="primary-button"
        onClick={() => void run()}
        disabled={busy || !text.trim() || text.length > 500}
      >
        {busy ? <Loader2 className="spin" size={16} /> : <Download size={16} />}
        {busy
          ? "Construction du lexique et des poses…"
          : "Générer les poses et le diagnostic"}
      </button>
      {error && (
        <p role="alert" className="research-error">
          {error}
        </p>
      )}
      {result && (
        <div className="pipeline-result">
          <strong>{result.glosses.join(" → ")}</strong>
          <p>
            {result.frames} images · {result.fps} fps · prototype lexical à
            relire
          </p>
          <video
            src={"/api" + result.video_url}
            controls
            playsInline
            preload="metadata"
          />
          <div>
            <a href={"/api" + result.pose_url} download>
              Télécharger .pose
            </a>
            <a href={"/api" + result.video_url} download>
              Vidéo squelette
            </a>
            <a href={"/api" + result.report_url} download>
              Rapport et sources
            </a>
          </div>
        </div>
      )}
    </details>
  );
}
