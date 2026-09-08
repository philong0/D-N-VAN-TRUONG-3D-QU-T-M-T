import { randomUUID } from "crypto";
import { NextRequest, NextResponse } from "next/server";
import { getPatient, updatePatient } from "@/lib/db";
import type { ScanSession, ScannerKind } from "@/lib/types";

const scannerKinds: ScannerKind[] = ["web_camera", "ios_native", "future"];

export async function GET(_request: NextRequest, { params }: { params: Promise<{ id: string }> }) {
  const { id } = await params;
  const patient = await getPatient(id);
  if (!patient) return NextResponse.json({ error: "Không tìm thấy hồ sơ bệnh nhân" }, { status: 404 });
  return NextResponse.json({ patientId: id, sessions: patient.scanSessions ?? [] });
}

export async function POST(request: NextRequest, { params }: { params: Promise<{ id: string }> }) {
  const { id: patientId } = await params;
  const patient = await getPatient(patientId);
  if (!patient) return NextResponse.json({ error: "Không tìm thấy hồ sơ bệnh nhân" }, { status: 404 });
  const body = await request.json().catch(() => ({}));
  const scannerKind = scannerKinds.includes(body.scannerKind) ? body.scannerKind : "web_camera";
  const now = new Date().toISOString();
  const session: ScanSession = { id: randomUUID(), patientId, scannerKind, status: "created", createdAt: now, updatedAt: now, frames: [] };
  await updatePatient(patientId, (current) => ({ ...current, scanSessions: [...(current.scanSessions ?? []), session] }));
  return NextResponse.json({ patientId, session }, { status: 201 });
}
