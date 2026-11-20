import { createRoot } from "react-dom/client";
import { AvatarCanvas } from "./components/AvatarPlayer/AvatarCanvas";
import type { MotionFile } from "./types/contract";

export function mountAvatar(host: HTMLElement) {
  const root = createRoot(host);
  return (motion: MotionFile | null, timeMs: number, faceBlendshapes?: Record<string, number>, headRotation?: number[]) => {
    root.render(
      <AvatarCanvas
        avatarCredit="Avatar Noa · UPF-GTI / Ready Player Me"
        fitNarrowViewport
        renderOnChange
        motion={motion}
        currentTimeMs={timeMs}
        faceBlendshapes={faceBlendshapes}
        headRotation={headRotation}
      />,
    );
  };
}
