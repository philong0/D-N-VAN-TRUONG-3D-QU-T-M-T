import type { MorphGroup } from "./services-catalog";

export interface AnchorPoint {
  x: number;
  y: number;
}

/**
 * Illustrative anchor points (fraction of image width/height) used to place
 * the measurement overlay on the patient's real photo, indexed by
 * [morph group][slot index 0-3]. These are NOT derived from real facial
 * landmark detection — there is no computer-vision model finding the actual
 * nose bridge/canthus/etc. in the uploaded photo. They are fixed, reasonable
 * default positions assuming a roughly centered subject, tuned per slot to
 * roughly track how that angle is typically framed. Treat as a visual guide
 * for the doctor, not a precise clinical measurement tool.
 */
export const REGION_ANCHORS: Record<MorphGroup, [AnchorPoint, AnchorPoint, AnchorPoint, AnchorPoint]> = {
  nose: [
    { x: 0.5, y: 0.48 },
    { x: 0.56, y: 0.47 },
    { x: 0.64, y: 0.46 },
    { x: 0.5, y: 0.4 },
  ],
  eye: [
    { x: 0.5, y: 0.4 },
    { x: 0.5, y: 0.4 },
    { x: 0.5, y: 0.38 },
    { x: 0.58, y: 0.4 },
  ],
  chin: [
    { x: 0.5, y: 0.76 },
    { x: 0.52, y: 0.76 },
    { x: 0.58, y: 0.74 },
    { x: 0.5, y: 0.7 },
  ],
  breast: [
    { x: 0.5, y: 0.55 },
    { x: 0.56, y: 0.55 },
    { x: 0.62, y: 0.55 },
    { x: 0.5, y: 0.5 },
  ],
};
