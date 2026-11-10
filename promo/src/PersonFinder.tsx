import React from "react";
import { AbsoluteFill, Easing, Img, OffthreadVideo, Sequence, interpolate, spring, staticFile, useCurrentFrame, useVideoConfig } from "remotion";
import { FT, INK, INK_2, LINE, MATCH, MONO, MUTED, PAPER, SANS, SERIF, ZS } from "./theme";

/** Scene boundaries in frames at 30 fps. */
const INTRO: number = 110;
const SEARCH: number = 190;
const VIDEO: number = 230;
const NUMBERS: number = 130;
const OUTRO: number = 100;
export const TOTAL_FRAMES: number = INTRO + SEARCH + VIDEO + NUMBERS + OUTRO;

type Hit = { img: string; score: number; same: boolean };

const SEARCH_QUERY: string = "a man wearing a red jacket, black trousers and black shoes";
const VIDEO_QUERY: string = "a man in a red and white striped shirt";
const HITS: Hit[] = [
  { img: "people/3920_c1_0001.jpg", score: 0.579, same: true },
  { img: "people/3920_c14_0015.jpg", score: 0.559, same: true },
  { img: "people/3920_c7_0005.jpg", score: 0.537, same: true },
  { img: "people/3920_c15_0022.jpg", score: 0.528, same: true },
  { img: "people/3946_c7_0005.jpg", score: 0.476, same: false },
];

const clamp = { extrapolateLeft: "clamp", extrapolateRight: "clamp" } as const;

const fadeInOut = (frame: number, length: number): number =>
  interpolate(frame, [0, 10, length - 10, length], [0, 1, 1, 0], clamp);

/** Text revealed one character at a time, with a blinking block caret. */
const Typed: React.FC<{ text: string; start: number; cps: number; style: React.CSSProperties }> = ({ text, start, cps, style }) => {
  const frame = useCurrentFrame();
  const { fps } = useVideoConfig();
  const n = Math.max(0, Math.min(text.length, Math.floor(((frame - start) / fps) * cps)));
  const caretOn = Math.floor(frame / 15) % 2 === 0 || n < text.length;
  return (
    <span style={style}>
      {text.slice(0, n)}
      <span style={{ display: "inline-block", width: "0.5em", height: "0.95em", marginLeft: 4, verticalAlign: "-0.1em", background: caretOn ? "currentColor" : "transparent" }} />
    </span>
  );
};

const Eyebrow: React.FC<{ children: React.ReactNode; color: string }> = ({ children, color }) => (
  <div style={{ fontFamily: MONO, fontSize: 24, letterSpacing: "0.14em", textTransform: "uppercase", color }}>{children}</div>
);

const Masthead: React.FC<{ dark: boolean; label: string }> = ({ dark, label }) => (
  <div style={{ position: "absolute", top: 0, left: 0, right: 0, height: 92, display: "flex", alignItems: "center", padding: "0 96px",
    borderBottom: `2px solid ${dark ? PAPER : INK}`, color: dark ? PAPER : INK }}>
    <span style={{ fontFamily: SERIF, fontSize: 38, fontWeight: 500 }}>Person Finder</span>
    <span style={{ marginLeft: "auto", fontFamily: MONO, fontSize: 20, letterSpacing: "0.12em", textTransform: "uppercase", opacity: 0.7 }}>{label}</span>
  </div>
);

const Intro: React.FC = () => {
  const frame = useCurrentFrame();
  const lift = interpolate(frame, [0, 24], [30, 0], { ...clamp, easing: Easing.out(Easing.cubic) });
  return (
    <AbsoluteFill style={{ background: PAPER, opacity: fadeInOut(frame, INTRO) }}>
      <Masthead dark={false} label="EmbeddingGemma-2 × RSTPReid" />
      <div style={{ position: "absolute", left: 96, right: 96, top: 250, transform: `translateY(${lift}px)` }}>
        <Eyebrow color={MUTED}>Find a person from one sentence</Eyebrow>
        <div style={{ marginTop: 36, borderTop: `2px solid ${INK}`, borderBottom: `2px solid ${INK}`, padding: "48px 0 56px", display: "flex", gap: 34 }}>
          <span style={{ fontFamily: SERIF, fontStyle: "italic", fontSize: 120, lineHeight: 0.8, color: MUTED }}>“</span>
          <Typed text={SEARCH_QUERY} start={12} cps={26} style={{ fontFamily: SERIF, fontSize: 76, lineHeight: 1.18, color: INK, letterSpacing: "-0.01em" }} />
        </div>
      </div>
    </AbsoluteFill>
  );
};

const Search: React.FC = () => {
  const frame = useCurrentFrame();
  const { fps } = useVideoConfig();
  return (
    <AbsoluteFill style={{ background: PAPER, opacity: fadeInOut(frame, SEARCH) }}>
      <Masthead dark={false} label="Image search · 1,000 unseen photos" />
      <div style={{ position: "absolute", left: 96, top: 150, right: 96 }}>
        <div style={{ fontFamily: SERIF, fontStyle: "italic", fontSize: 40, color: INK_2 }}>“{SEARCH_QUERY}”</div>
        <div style={{ display: "flex", gap: 34, marginTop: 44 }}>
          {HITS.map((h, i) => {
            const s = spring({ frame: frame - 8 - i * 7, fps, config: { damping: 200 } });
            const ring = interpolate(frame, [60 + i * 6, 76 + i * 6], [0, 1], clamp);
            const counted = h.score * interpolate(frame, [8 + i * 7, 40 + i * 7], [0, 1], clamp);
            return (
              <div key={h.img} style={{ width: 300, opacity: s, transform: `translateY(${(1 - s) * 60}px)` }}>
                <div style={{ position: "relative", width: 300, height: 560, outline: h.same ? `${6 * ring}px solid ${MATCH}` : "none", outlineOffset: 4 }}>
                  <Img src={staticFile(h.img)} style={{ width: 300, height: 560, objectFit: "cover", display: "block", border: `2px solid ${LINE}` }} />
                  <span style={{ position: "absolute", top: 0, left: 0, background: INK, color: PAPER, fontFamily: MONO, fontSize: 24, padding: "6px 12px" }}>{String(i + 1).padStart(2, "0")}</span>
                </div>
                <div style={{ display: "flex", justifyContent: "space-between", marginTop: 20, fontFamily: MONO, fontSize: 30, color: INK }}>
                  <span>{counted.toFixed(3)}</span>
                  <span style={{ color: h.same ? MATCH : MUTED, opacity: ring, fontWeight: 600 }}>{h.same ? "✓ same" : "✗ other"}</span>
                </div>
              </div>
            );
          })}
        </div>
        <div style={{ marginTop: 40, fontFamily: SANS, fontSize: 30, color: INK_2, opacity: interpolate(frame, [96, 116], [0, 1], clamp) }}>
          Four of the top five are the same man, seen by four different cameras.
        </div>
      </div>
    </AbsoluteFill>
  );
};

const Video: React.FC = () => {
  const frame = useCurrentFrame();
  const zoom = interpolate(frame, [0, VIDEO], [1, 1.06], clamp);
  return (
    <AbsoluteFill style={{ background: INK, opacity: fadeInOut(frame, VIDEO) }}>
      <Masthead dark label="Live video · Mall dataset" />
      <div style={{ position: "absolute", left: 96, top: 150, width: 1056, height: 792, overflow: "hidden", border: `2px solid ${PAPER}` }}>
        <OffthreadVideo src={staticFile("mall_annot.mp4")} trimBefore={105} muted
          style={{ width: 1056, height: 833, marginTop: -41, objectFit: "cover", display: "block", transform: `scale(${zoom})`, transformOrigin: "45% 80%" }} />
      </div>
      <div style={{ position: "absolute", left: 1220, right: 96, top: 170, color: PAPER }}>
        <Eyebrow color={MUTED}>Describe</Eyebrow>
        <div style={{ marginTop: 18, minHeight: 190 }}>
          <Typed text={VIDEO_QUERY} start={6} cps={30} style={{ fontFamily: SERIF, fontSize: 52, lineHeight: 1.2, color: PAPER }} />
        </div>
        <div style={{ borderTop: `2px solid ${PAPER}`, marginTop: 30, paddingTop: 26, display: "grid", gap: 26, opacity: interpolate(frame, [50, 70], [0, 1], clamp) }}>
          {[["Detect", "YOLO26s finds every person"], ["Track", "ByteTrack keeps their ids"], ["Read", "the model scores each crop"], ["Choose", "best ≥ 0.45 gets the green box"]].map(([k, v], i) => (
            <div key={k} style={{ display: "grid", gridTemplateColumns: "150px 1fr", alignItems: "baseline", opacity: interpolate(frame, [56 + i * 10, 70 + i * 10], [0, 1], clamp) }}>
              <span style={{ fontFamily: MONO, fontSize: 22, letterSpacing: "0.12em", textTransform: "uppercase", color: i === 3 ? "#3fbf86" : MUTED }}>{k}</span>
              <span style={{ fontFamily: SANS, fontSize: 30, color: PAPER }}>{v}</span>
            </div>
          ))}
        </div>
      </div>
    </AbsoluteFill>
  );
};

const CountUp: React.FC<{ from: number; to: number; label: string; start: number; color: string }> = ({ from, to, label, start, color }) => {
  const frame = useCurrentFrame();
  const t = interpolate(frame, [start, start + 40], [0, 1], { ...clamp, easing: Easing.out(Easing.cubic) });
  return (
    <div style={{ borderTop: `2px solid ${INK}`, paddingTop: 24 }}>
      <div style={{ fontFamily: SERIF, fontSize: 150, lineHeight: 1, letterSpacing: "-0.03em", color: INK }}>{(from + (to - from) * t).toFixed(1)}%</div>
      <div style={{ display: "flex", alignItems: "center", gap: 14, marginTop: 18 }}>
        <span style={{ width: 22, height: 22, background: color }} />
        <span style={{ fontFamily: MONO, fontSize: 26, color: INK_2 }}>{label}</span>
      </div>
      <div style={{ fontFamily: MONO, fontSize: 24, color: MUTED, marginTop: 8 }}>from {from.toFixed(1)}% zero-shot</div>
    </div>
  );
};

const Numbers: React.FC = () => {
  const frame = useCurrentFrame();
  return (
    <AbsoluteFill style={{ background: PAPER, opacity: fadeInOut(frame, NUMBERS) }}>
      <Masthead dark={false} label="Fine-tuned · 200 people it never saw" />
      <div style={{ position: "absolute", left: 96, right: 96, top: 230, display: "grid", gridTemplateColumns: "1fr 1fr 1fr", gap: 60 }}>
        <CountUp from={40.1} to={71.4} label="Rank-1 · first guess right" start={6} color={FT} />
        <CountUp from={75.4} to={93.2} label="Rank-10 · on the first screen" start={16} color={FT} />
        <CountUp from={28.8} to={54.8} label="mAP · every photo found" start={26} color={ZS} />
      </div>
      <div style={{ position: "absolute", left: 96, right: 96, bottom: 120, display: "flex", alignItems: "baseline", gap: 40, borderTop: `2px solid ${LINE}`, paddingTop: 28,
        opacity: interpolate(frame, [60, 80], [0, 1], clamp) }}>
        <span style={{ fontFamily: SERIF, fontStyle: "italic", fontSize: 46, color: INK }}>Every grade nearly doubles.</span>
        <span style={{ fontFamily: MONO, fontSize: 24, color: MUTED }}>37,010 training pairs · 3 epochs · one RTX 3090</span>
      </div>
    </AbsoluteFill>
  );
};

const Outro: React.FC = () => {
  const frame = useCurrentFrame();
  const o = interpolate(frame, [0, 16], [0, 1], clamp);
  return (
    <AbsoluteFill style={{ background: INK, color: PAPER, opacity: o }}>
      <div style={{ position: "absolute", left: 96, right: 96, top: 300 }}>
        <Eyebrow color={MUTED}>Person Finder</Eyebrow>
        <div style={{ fontFamily: SERIF, fontSize: 128, lineHeight: 1.0, letterSpacing: "-0.025em", marginTop: 26 }}>
          One sentence. <span style={{ fontStyle: "italic", color: "#b9b4a8" }}>Any crowd.</span>
        </div>
        <div style={{ display: "flex", gap: 18, marginTop: 64 }}>
          {["EmbeddingGemma-2", "fine-tuned on RSTPReid", "YOLO26 + ByteTrack", "≈ 26 fps live"].map((t, i) => (
            <span key={t} style={{ fontFamily: MONO, fontSize: 26, border: `2px solid #4a4843`, padding: "10px 18px", opacity: interpolate(frame, [14 + i * 6, 26 + i * 6], [0, 1], clamp) }}>{t}</span>
          ))}
        </div>
      </div>
    </AbsoluteFill>
  );
};

export const PersonFinder: React.FC = () => (
  <AbsoluteFill style={{ background: PAPER }}>
    <Sequence durationInFrames={INTRO}><Intro /></Sequence>
    <Sequence from={INTRO} durationInFrames={SEARCH}><Search /></Sequence>
    <Sequence from={INTRO + SEARCH} durationInFrames={VIDEO}><Video /></Sequence>
    <Sequence from={INTRO + SEARCH + VIDEO} durationInFrames={NUMBERS}><Numbers /></Sequence>
    <Sequence from={INTRO + SEARCH + VIDEO + NUMBERS} durationInFrames={OUTRO}><Outro /></Sequence>
  </AbsoluteFill>
);
