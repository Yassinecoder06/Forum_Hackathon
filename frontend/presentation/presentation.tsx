/* eslint-disable react-refresh/only-export-components */
// Presentation of SignaCulture: full-screen slides + the live MetaHuman interpreter in the bottom-right corner.
import { createRoot } from "react-dom/client";
import { useCallback, useEffect, useState } from "react";
import { Interpreter } from "./interpreter";
import { SLIDES } from "./slides";
import "../src/fonts.css";
import "./presentation.css";

const slideFromHash = () => {
  const n = parseInt(location.hash.slice(1), 10);
  return Number.isFinite(n) ? Math.max(0, Math.min(SLIDES.length - 1, n - 1)) : 0;
};

function Deck() {
  const [index, setIndex] = useState(slideFromHash);
  const [signRequest, setSignRequest] = useState<{ text: string; n: number } | null>(null);
  const go = useCallback((d: number) => setIndex((i) => Math.max(0, Math.min(SLIDES.length - 1, i + d))), []);
  const signSlide = useCallback(() => setSignRequest((r) => ({ text: SLIDES[index].sign, n: (r?.n ?? 0) + 1 })), [index]);

  useEffect(() => { history.replaceState(null, "", `#${index + 1}`); }, [index]);
  useEffect(() => {
    const onHash = () => setIndex(slideFromHash());
    window.addEventListener("hashchange", onHash);
    return () => window.removeEventListener("hashchange", onHash);
  }, []);
  useEffect(() => {
    const key = (e: KeyboardEvent) => {
      if (e.target instanceof HTMLInputElement || e.ctrlKey || e.metaKey || e.altKey) return;
      if (["ArrowRight", "PageDown", " ", "Enter"].includes(e.key)) { e.preventDefault(); go(1); }
      else if (["ArrowLeft", "PageUp", "Backspace"].includes(e.key)) { e.preventDefault(); go(-1); }
      else if (e.key === "Home") setIndex(0);
      else if (e.key === "End") setIndex(SLIDES.length - 1);
      else if (e.key.toLowerCase() === "s") signSlide();
      else if (e.key.toLowerCase() === "f") void (document.fullscreenElement ? document.exitFullscreen() : document.documentElement.requestFullscreen());
    };
    window.addEventListener("keydown", key);
    return () => window.removeEventListener("keydown", key);
  }, [go, signSlide]);

  const slide = SLIDES[index];
  return (
    <div className={`deck tone-${slide.tone ?? "light"}`}>
      <div className="brand">Signe<em>PourMoi</em></div>
      {SLIDES.map((s, i) => (
        <section key={s.id} className={`slide${i === index ? " active" : i < index ? " past" : ""}`} aria-hidden={i !== index}>
          <div className="slide-inner">
            {s.kicker && <p className="kicker">{s.kicker}</p>}
            {s.body}
          </div>
        </section>
      ))}
      <nav className="deck-nav">
        <button onClick={() => go(-1)} disabled={index === 0} aria-label="Diapositive précédente">‹</button>
        <span>{index + 1} / {SLIDES.length}</span>
        <button onClick={() => go(1)} disabled={index === SLIDES.length - 1} aria-label="Diapositive suivante">›</button>
        <button className="sign-slide" onClick={signSlide} title="Faire signer la phrase de la diapo (S)">Signer la diapo</button>
      </nav>
      <div className="progress"><i style={{ width: `${((index + 1) / SLIDES.length) * 100}%` }} /></div>
      <Interpreter signRequest={signRequest} />
    </div>
  );
}

createRoot(document.getElementById("root")!).render(<Deck />);
