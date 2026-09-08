import { NextRequest, NextResponse } from "next/server";

export async function POST(request: NextRequest) {
  const body = await request.json().catch(() => null);
  const patientId: string | undefined = body?.patientId;
  if (!patientId) {
    return NextResponse.json({ error: "patientId is required" }, { status: 400 });
  }

  return NextResponse.json({
    status: "ok",
    method: "Apple TrueDepth 3D Metric Mesh (Kratos Standard)",
    patientId,
  });
}
