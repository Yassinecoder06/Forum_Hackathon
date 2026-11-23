import type { CSSProperties } from "react";
import { Users } from "lucide-react";
import { LOCUS_LABEL, type CastEntry, type Locus } from "../utils/speakers";
import "./filmSpeakers.css";

/** Name chip of a speaker (cue list, first appearance on screen, avatar badge). */
export function SpeakerChip({ entry, side, big }: { entry?: CastEntry; side?: boolean; big?: boolean }) {
  if (!entry) return <span className="speaker-chip narrator">Narration</span>;
  return (
    <span className={`speaker-chip${big ? " big" : ""}`} style={{ "--chip": entry.color } as CSSProperties}>
      {entry.label}
      {side && entry.locus !== "N" && <small> · {LOCUS_LABEL[entry.locus].toLowerCase()}</small>}
    </span>
  );
}

/** Small stage: the three loci seen from the viewer, the active speaker lit. */
export function StageDots({ cast, active }: { cast: CastEntry[]; active?: CastEntry }) {
  return (
    <span className="stage-dots" aria-label={active ? `L’avatar prend le rôle de ${active.label}` : "Narration face au public"}>
      {(["L", "C", "R"] as Locus[]).map((l) => {
        const who = cast.filter((c) => c.locus === l);
        const lit = active?.locus === l;
        return (
          <i key={l} className={lit ? "lit" : who.length ? "taken" : ""} style={lit ? ({ "--chip": active!.color } as CSSProperties) : undefined}
            title={who.map((c) => c.label).join(", ") || LOCUS_LABEL[l]} />
        );
      })}
    </span>
  );
}

/** Cast sheet: each speaker found in the subtitles, with an editable name and place in signing space. */
export function CastSheet({ cast, onChange, counts }: { cast: CastEntry[]; onChange: (cast: CastEntry[]) => void; counts: Record<string, number> }) {
  if (!cast.length) return null;
  const set = (id: string, patch: Partial<CastEntry>) => onChange(cast.map((c) => (c.id === id ? { ...c, ...patch } : c)));
  return (
    <div className="cast-sheet card">
      <div className="cast-head">
        <Users size={16} />
        <strong>Personnages</strong>
        <small>Chaque personnage a sa place dans l’espace de signation : l’avatar se tourne vers elle quand il prend sa parole.</small>
      </div>
      <ul>
        {cast.map((c) => (
          <li key={c.id}>
            <i className="cast-dot" style={{ background: c.color }} />
            <input aria-label="Nom du personnage" value={c.label} maxLength={40} onChange={(e) => set(c.id, { label: e.target.value })} />
            <select aria-label={`Place de ${c.label}`} value={c.locus} onChange={(e) => set(c.id, { locus: e.target.value as Locus })}>
              {(["L", "C", "R", "N"] as Locus[]).map((l) => <option key={l} value={l}>{LOCUS_LABEL[l]}</option>)}
            </select>
            <small>{counts[c.id] ?? 0} réplique{(counts[c.id] ?? 0) > 1 ? "s" : ""}</small>
          </li>
        ))}
      </ul>
    </div>
  );
}
