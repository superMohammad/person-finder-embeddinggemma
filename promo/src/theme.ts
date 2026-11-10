import { loadFont as loadSerif } from "@remotion/google-fonts/Newsreader";
import { loadFont as loadMono } from "@remotion/google-fonts/IBMPlexMono";
import { loadFont as loadSans } from "@remotion/google-fonts/IBMPlexSans";

/** Same palette and type as the web demo and the figures: off-white paper, black ink, green only for the match. */
export const PAPER: string = "#f3f0e8";
export const INK: string = "#141413";
export const INK_2: string = "#4f4c45";
export const MUTED: string = "#8a857a";
export const LINE: string = "#cfc9bb";
export const MATCH: string = "#1f9d6b";
export const ZS: string = "#2a78d6";
export const FT: string = "#eb6834";

export const SERIF: string = loadSerif("normal", { weights: ["400", "500"], subsets: ["latin"] }).fontFamily;
export const SERIF_ITALIC: string = loadSerif("italic", { weights: ["400"], subsets: ["latin"] }).fontFamily;
export const MONO: string = loadMono("normal", { weights: ["400", "500", "600"], subsets: ["latin"] }).fontFamily;
export const SANS: string = loadSans("normal", { weights: ["400", "500"], subsets: ["latin"] }).fontFamily;
