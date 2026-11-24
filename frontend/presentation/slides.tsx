/* eslint-disable react-refresh/only-export-components */
import type { ReactNode } from "react";

/** One slide: its content, and the French sentence the avatar signs with "Signer la diapo". */
export type Slide = { id: string; kicker?: string; sign: string; body: ReactNode; tone?: "dark" | "light" };

const Flow = ({ steps }: { steps: { t: string; s?: string }[] }) => (
  <ol className="flow">
    {steps.map((x, i) => (
      <li key={i}><b>{x.t}</b>{x.s && <small>{x.s}</small>}</li>
    ))}
  </ol>
);

export const SLIDES: Slide[] = [
  {
    id: "titre",
    tone: "dark",
    sign: "Bonjour, bienvenue dans la présentation de SignePourMoi.",
    body: (
      <div className="hero">
        <p className="kicker">Forum · Hackathon 2026</p>
        <h1>Signe<em>PourMoi</em></h1>
        <p className="lead">Le français, signé en langue des signes française par un avatar 3D, en direct.</p>
        <p className="hint">Parlez : l’interprète en bas à droite écoute, transcrit et signe.</p>
      </div>
    ),
  },
  {
    id: "constat",
    kicker: "Le constat",
    sign: "L'information est en français, mais beaucoup de personnes sourdes vivent en langue des signes.",
    body: (
      <>
        <h2>L’information parle français. Beaucoup de personnes sourdes vivent en LSF.</h2>
        <div className="cards three">
          <div><h3>Interprètes rares</h3><p>Conférences, films, rendez-vous du quotidien : un interprète n’est presque jamais disponible au bon moment.</p></div>
          <div><h3>Le texte ne suffit pas</h3><p>La LSF est une langue à part entière, avec sa grammaire, son espace et son visage. Les sous-titres ne la remplacent pas.</p></div>
          <div><h3>Tout, tout de suite</h3><p>Il faut une traduction immédiate, partout où l’on parle : à l’écran, en face-à-face, dans la rue.</p></div>
        </div>
      </>
    ),
  },
  {
    id: "reponse",
    kicker: "Notre réponse",
    sign: "SignePourMoi traduit le français en langue des signes, sur un seul ordinateur.",
    body: (
      <>
        <h2>Du français en entrée. De la LSF signée en sortie.</h2>
        <p className="lead">Texte, voix, film ou personne qui signe : SignePourMoi comprend le français et le fait signer par un avatar MetaHuman, avec le texte à côté.</p>
        <ul className="ticks">
          <li>Une seule chaîne de traduction pour tous les usages</li>
          <li>Un générateur de signes appris sur de vrais interprètes</li>
          <li>Tout tourne en local, sur un PC portable : aucune IA dans le cloud</li>
        </ul>
      </>
    ),
  },
  {
    id: "chaine",
    kicker: "Une chaîne, cinq entrées",
    sign: "Studio, direct, film, conversation et compagnon utilisent la même chaîne.",
    body: (
      <>
        <h2>Cinq façons d’entrer, un seul moteur.</h2>
        <div className="cards five">
          <div><h3>Studio</h3><p>On écrit une phrase, l’avatar la signe.</p></div>
          <div><h3>Direct</h3><p>On parle au micro, chaque phrase est signée dès la pause.</p></div>
          <div><h3>Film</h3><p>Vidéo + sous-titres : l’avatar signe et se tourne vers chaque personnage.</p></div>
          <div><h3>Conversation</h3><p>On signe à la webcam, l’assistant répond en LSF.</p></div>
          <div><h3>Compagnon</h3><p>Sur le téléphone, en réalité augmentée, l’avatar regarde ce dont il parle.</p></div>
        </div>
        <p className="note">Toutes les entrées appellent les mêmes services : Whisper, le générateur LSF, l’assistant local et l’avatar.</p>
      </>
    ),
  },
  {
    id: "direct",
    kicker: "Ce que vous voyez en bas à droite",
    sign: "Je vous écoute, je transcris votre voix, puis je signe chaque phrase.",
    body: (
      <>
        <h2>De la voix aux signes en quelques secondes.</h2>
        <Flow steps={[
          { t: "Micro", s: "le navigateur coupe la parole aux pauses" },
          { t: "Whisper", s: "large-v3-turbo, sur la carte graphique, chargé une seule fois" },
          { t: "Grammaire LSF", s: "ordre des signes, visage, référents" },
          { t: "Générateur C0", s: "le mouvement complet du corps, des mains et du visage" },
          { t: "Avatar", s: "MetaHuman en 3D, dans la page" },
        ]} />
        <p className="note">De la fin d’une phrase aux premiers signes : environ 3 à 5 secondes.</p>
      </>
    ),
  },
  {
    id: "donnees",
    kicker: "Les données",
    sign: "Nous avons appris sur des milliers de phrases signées par de vrais interprètes.",
    body: (
      <>
        <h2>Appris sur de vrais interprètes.</h2>
        <div className="stats">
          <div><b>37 600</b><span>phrases françaises alignées avec leur interprétation en LSF</span></div>
          <div><b>156</b><span>points de presse avec interprète à l’écran</span></div>
          <div><b>157</b><span>points suivis par image : corps, mains, lèvres, sourcils, yeux</span></div>
        </div>
        <p className="note">Chaque phrase est recalée sur le décalage réel de l’interprète (environ 3 secondes derrière l’orateur). Les vidéos et les mouvements restent privés.</p>
      </>
    ),
  },
  {
    id: "c0",
    kicker: "Le générateur C0",
    sign: "Le générateur transforme une phrase en mouvement de signes.",
    body: (
      <>
        <h2>Une phrase entre, un mouvement signé sort.</h2>
        <div className="cards three">
          <div><h3>Un vocabulaire de mouvements</h3><p>Un tokeniseur découpe le geste en codes : corps et visage d’un côté, mains de l’autre.</p></div>
          <div><h3>Un transformeur</h3><p>Il lit le sens de la phrase et écrit la suite de codes, avec des temps de tenue naturels.</p></div>
          <div><h3>Deux affineurs</h3><p>L’un corrige la profondeur des doigts, l’autre rend le visage plus expressif.</p></div>
        </div>
      </>
    ),
  },
  {
    id: "grammaire",
    kicker: "La grammaire de la LSF",
    sign: "Demain, toi, aller où ?",
    body: (
      <>
        <h2>Signer, ce n’est pas traduire mot à mot.</h2>
        <ul className="ticks two">
          <li><b>Ordre LSF</b> le temps et le lieu d’abord, le verbe tard, la question à la fin</li>
          <li><b>Visage grammatical</b> sourcils levés ou froncés pour les questions, tête qui dit non pour la négation</li>
          <li><b>Référents</b> chaque personne reçoit une place dans l’espace, l’avatar la pointe</li>
          <li><b>Prise de rôle</b> le corps se tourne vers celui qui parle</li>
          <li><b>Prosodie</b> un mot accentué à la voix devient un geste plus ample, avec un hochement</li>
          <li><b>Émotion</b> la voix colore la vitesse des mains et le visage</li>
        </ul>
      </>
    ),
  },
  {
    id: "avatar",
    kicker: "L’avatar",
    sign: "L'avatar respire, regarde et signe avec des mains réalistes.",
    body: (
      <>
        <h2>Un MetaHuman qui signe dans le navigateur.</h2>
        <div className="cards three">
          <div><h3>Du jeu vidéo au web</h3><p>MetaHuman d’Unreal Engine 5, converti en un seul fichier 3D léger, avec 51 expressions du visage.</p></div>
          <div><h3>Des mains crédibles</h3><p>Limites articulaires doigt par doigt, coudes en vraie charnière, ressorts et inertie.</p></div>
          <div><h3>Vivant</h3><p>Respiration, regard, clignements, tenue habillée sombre pour que les mains se lisent bien.</p></div>
        </div>
      </>
    ),
  },
  {
    id: "mesures",
    kicker: "Les mesures",
    sign: "Les doigts sont plus justes et le visage est plus vivant.",
    body: (
      <>
        <h2>Ce que nous mesurons, phrase après phrase.</h2>
        <table className="metrics">
          <thead><tr><th /><th>Interprète réel</th><th>C0</th><th>C0 + affineurs</th></tr></thead>
          <tbody>
            <tr><td>Accord texte ↔ mouvement (z)</td><td>1,12</td><td>1,26</td><td><b>1,26</b></td></tr>
            <tr><td>Visage vivant (× interprète)</td><td>1</td><td>0,67</td><td><b>0,86</b></td></tr>
            <tr><td>Doigts hors limites</td><td>13 %</td><td>15 %</td><td><b>6,6 %</b></td></tr>
          </tbody>
        </table>
        <p className="note">Indicateurs automatiques, sur 600 phrases jamais vues à l’entraînement.</p>
      </>
    ),
  },
  {
    id: "local",
    kicker: "Local et privé",
    sign: "Tout fonctionne sur cet ordinateur, sans envoyer votre voix ailleurs.",
    body: (
      <>
        <h2>Votre voix ne quitte pas l’ordinateur.</h2>
        <ul className="ticks">
          <li>Reconnaissance vocale, générateur, assistant et avatar : tout tourne sur une carte graphique de portable (8 Go)</li>
          <li>Aucun service d’IA externe, aucune donnée envoyée</li>
          <li>Les données d’entraînement et les modèles restent chez nous</li>
        </ul>
      </>
    ),
  },
  {
    id: "suite",
    kicker: "La suite",
    sign: "La suite, c'est travailler avec la communauté sourde.",
    body: (
      <>
        <h2>La suite.</h2>
        <ul className="ticks">
          <li>Construire avec des interprètes et des personnes sourdes signantes, phrase après phrase</li>
          <li>69 nouveaux points de presse (38 heures) pour un jeu de données encore plus riche</li>
          <li>Le compagnon en réalité augmentée sur téléphone, dans la vie de tous les jours</li>
        </ul>
      </>
    ),
  },
  {
    id: "merci",
    tone: "dark",
    sign: "Merci. Avez-vous des questions ?",
    body: (
      <div className="hero">
        <p className="kicker">Merci</p>
        <h1>À vous la <em>parole</em>.</h1>
        <p className="lead">Posez votre question à voix haute : l’avatar la signe.</p>
      </div>
    ),
  },
];
