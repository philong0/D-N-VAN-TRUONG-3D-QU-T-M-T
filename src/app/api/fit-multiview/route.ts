import { NextRequest, NextResponse } from "next/server";

export async function POST(request: NextRequest) {
  const body = await request.json().catch(() => null);
  const patientId: string | undefined = body?.patientId;
  if (!patientId) {
    return NextResponse.json({ error: "patientId is required" }, { status: 400 });
  }

  return NextResponse.json({
    viewsUsed: ["angle1", "angle2", "angle3", "angle4"],
    warnings: [],
    coverageFraction: 1.0,
    status: "ok",
  });
}
