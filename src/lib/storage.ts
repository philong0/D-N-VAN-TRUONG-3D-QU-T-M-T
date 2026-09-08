import { mkdir } from "fs/promises";
import path from "path";
import os from "os";

// Trên Vercel / AWS Lambda, chỉ có thư mục os.tmpdir() (/tmp) là được phép ghi
const isVercel = Boolean(process.env.VERCEL || process.env.AWS_LAMBDA_FUNCTION_NAME);
export const DATA_DIR = isVercel
  ? path.join(os.tmpdir(), "dr_vantruong_data")
  : path.join(process.cwd(), ".data");

export const PATIENTS_FILE = path.join(DATA_DIR, "patients.json");
export const PATIENTS_DIR = path.join(DATA_DIR, "patients");

export function patientDir(patientId: string): string {
  return path.join(PATIENTS_DIR, patientId);
}

export function patientPhotosDir(patientId: string): string {
  return path.join(patientDir(patientId), "photos");
}

export function patientScansDir(patientId: string): string {
  return path.join(patientDir(patientId), "scans");
}

export function scanFramesDir(patientId: string, sessionId: string): string {
  return path.join(patientScansDir(patientId), sessionId, "frames");
}

export function scanSessionDir(patientId: string, sessionId: string): string {
  return path.join(patientScansDir(patientId), sessionId);
}

export async function ensureDir(dir: string): Promise<void> {
  try {
    await mkdir(dir, { recursive: true });
  } catch (e) {
    // Ignore error if already exists
  }
}

export function resolvePatientFile(patientId: string, segments: string[]): string {
  const base = patientDir(patientId);
  const resolved = path.join(base, ...segments);
  if (resolved !== base && !resolved.startsWith(base + path.sep)) {
    throw new Error("Invalid file path");
  }
  return resolved;
}
