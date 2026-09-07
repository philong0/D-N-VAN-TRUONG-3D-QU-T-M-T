/**
 * iOS TrueDepth / ARKit Capture Package Schema & Contracts.
 * Defines the canonical binary/JSON structure exported by iPhone/iPad Pro
 * and ingested by the ScanSession API for 3D reconstruction.
 */

import type { ScanCaptureView } from "@/lib/types";

export interface CameraIntrinsics {
  fx: number;
  fy: number;
  cx: number;
  cy: number;
  imageWidth: number;
  imageHeight: number;
  lensDistortionCoefficients?: number[];
}

export interface ARKitTransformMatrix {
  columns: [
    [number, number, number, number],
    [number, number, number, number],
    [number, number, number, number],
    [number, number, number, number]
  ];
  translationMeters: { x: number; y: number; z: number };
  eulerRotationDeg: { pitch: number; yaw: number; roll: number };
}

export interface ARKitFaceGeometryData {
  vertexCount: number; // exactly 1220 in Apple ARKit
  triangleCount: number; // exactly 2304 in Apple ARKit
  verticesMeters: number[]; // Flat array [x0, y0, z0, x1, y1, z1, ...]
  triangleIndices: number[]; // Flat array [t0_a, t0_b, t0_c, ...]
  textureCoordinates: number[]; // Flat array [u0, v0, u1, v1, ...]
  blendShapes?: Record<string, number>; // 52 ARKit blendshapes
}

export interface TrueDepthFrameCapture {
  view: ScanCaptureView;
  timestamp: number;
  rgbFileName: string; // e.g. "front_rgb.jpg"
  depthFileName?: string; // e.g. "front_depth.raw" (16-bit float millimeters)
  depthWidth?: number;
  depthHeight?: number;
  intrinsics: CameraIntrinsics;
  pose: ARKitTransformMatrix;
  geometry: ARKitFaceGeometryData;
  lightingEstimate?: {
    ambientIntensity: number;
    ambientColorTemperature: number;
  };
}

export interface TrueDepthCapturePackageManifest {
  schemaVersion: "1.0.0";
  deviceModel: string; // e.g. "iPhone15,2" (iPhone 14 Pro), "iPad13,8" (iPad Pro 12.9)
  systemVersion: string; // e.g. "iOS 17.5"
  hasTrueDepth: boolean;
  hasLiDAR: boolean;
  patientId: string;
  sessionId: string;
  capturedAt: string;
  frames: TrueDepthFrameCapture[];
}

