import React from "react";
import { Composition } from "remotion";
import { PersonFinder, TOTAL_FRAMES } from "./PersonFinder";

export const FPS: number = 30;

export const RemotionRoot: React.FC = () => (
  <Composition id="PersonFinder" component={PersonFinder} durationInFrames={TOTAL_FRAMES} fps={FPS} width={1920} height={1080} />
);
