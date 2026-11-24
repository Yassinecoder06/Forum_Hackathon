// The live LSF interpreter alone: the whole window is the box (mic -> Whisper -> C0 -> avatar). Keys: M micro, F full screen.
import { createRoot } from "react-dom/client";
import { Interpreter } from "../presentation/interpreter";
import "../src/fonts.css";
import "./interprete.css";

createRoot(document.getElementById("root")!).render(<Interpreter standalone />);
