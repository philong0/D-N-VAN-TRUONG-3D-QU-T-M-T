import { readFile } from "fs/promises";
import path from "path";
import { NextRequest, NextResponse } from "next/server";
import { getPatient } from "@/lib/db";
import { PHOTO_ANGLES } from "@/lib/photo-angles";
import type { PhotoAngle } from "@/lib/types";

const REPO_ROOT = process.cwd();
const AI_ENGINE_URL = process.env.AI_ENGINE_URL ?? "http://localhost:8001";

export interface DetectedCameraPose {
  rotationMatrix: [[number, number, number], [number, number, number], [number, number, number]];
  translation: [number, number, number];
  focalLength: number;
  principalPoint: [number, number];
  imageSize: [number, number];
  meanReprojectionErrorPx: number;
  meanReprojectionErrorPctWidth: number;
}

export interface DetectedPose {
  hasFace: boolean;
  /** Real yaw in degrees, from ai-engine's YuNet+PIPNet(98pt) detection — see ai-engine/detect_pose.py. Null when hasFace is false. */
  yaw: number | null;
  /** Real SolvePnP camera pose (Giai đoạn B, validated ~0.5% reprojection error) — null if solvePnP didn't converge even though a face was detected. Never a fabricated fallback. */
  cameraPose: DetectedCameraPose | null;
}

interface AiEngineDetectPoseResponse {
  has_face: boolean;
  yaw: number | null;
  landmarks_98: [number, number][] | null;
  camera_pose: {
    rotation_matrix: [[number, number, number], [number, number, number], [number, number, number]];
    translation: [number, number, number];
    focal_length: number;
    principal_point: [number, number];
    image_size: [number, number];
    mean_reprojection_error_px: number;
    mean_reprojection_error_pct_width: number;
  } | null;
  error?: string;
}

/**
 * Real per-photo pose (yaw + SolvePnP camera pose) for every angle a patient
 * has uploaded in Bước 2 — proxies each photo to ai-engine's
 * `POST /detect-pose` (YuNet face box + PIPNet 98-point landmarks + a real
 * SolvePnP camera pose from the validated 8-point jaw+nose correspondence,
 * see ai-engine/detect_pose.py), which unlike the browser's MediaPipe
 * pipeline (lib/face-geometry.ts) can detect a genuine ~90° profile photo.
 * Never trusts the frontend's angle1..4 slot label as the real yaw/pose —
 * src/lib/gnm/camera-pose.ts is the caller that actually needs this real
 * value for multi-view texture blending.
 */
async function detectPoseForPhoto(imgPath: string): Promise<DetectedPose> {
  const buffer = await readFile(imgPath);
  const formData = new FormData();
  formData.set("file", new Blob([new Uint8Array(buffer)]), path.basename(imgPath));

  const res = await fetch(`${AI_ENGINE_URL}/detect-pose`, { method: "POST", body: formData });
  if (!res.ok) {
    throw new Error(`ai-engine /detect-pose returned ${res.status}`);
  }
  const body = (await res.json()) as AiEngineDetectPoseResponse;
  const cp = body.camera_pose;
  return {
    hasFace: body.has_face,
    yaw: body.yaw,
    cameraPose: cp
      ? {
          rotationMatrix: cp.rotation_matrix,
          translation: cp.translation,
          focalLength: cp.focal_length,
          principalPoint: cp.principal_point,
          imageSize: cp.image_size,
          meanReprojectionErrorPx: cp.mean_reprojection_error_px,
          meanReprojectionErrorPctWidth: cp.mean_reprojection_error_pct_width,
        }
      : null,
  };
}

export async function POST(request: NextRequest) {
  const body = await request.json().catch(() => null);
  const patientId: string | undefined = body?.patientId;
  if (!patientId) {
    return NextResponse.json({ error: "patientId is required" }, { status: 400 });
  }

  const patient = await getPatient(patientId);
  if (!patient) {
    return NextResponse.json({ error: "Patient not found" }, { status: 404 });
  }

  const poses: Partial<Record<PhotoAngle, DetectedPose>> = {};
  const errors: Partial<Record<PhotoAngle, string>> = {};

  for (const angle of PHOTO_ANGLES) {
    const asset = patient.photos[angle];
    if (!asset) continue;
    const imgPath = path.join(REPO_ROOT, ".data", "patients", patientId, "photos", asset.fileName);
    try {
      poses[angle] = await detectPoseForPhoto(imgPath);
    } catch (err) {
      errors[angle] = err instanceof Error ? err.message : String(err);
    }
  }

  if (Object.keys(poses).length === 0) {
    return NextResponse.json({ patientId, poses, errors, error: "No photo could be processed" }, { status: 502 });
  }

  return NextResponse.json({ patientId, poses, errors });
}
