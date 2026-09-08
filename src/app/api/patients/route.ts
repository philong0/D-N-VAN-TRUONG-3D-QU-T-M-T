import { NextResponse } from "next/server";
import { listPatients } from "@/lib/db";

// Real patient picker for the native iOS scanner app — replaces having to
// hand-type a raw Patient UUID copied from somewhere else. Returns just
// enough to identify a patient in a list (not the full medical record).
export async function GET() {
  const patients = await listPatients();
  return NextResponse.json({
    patients: patients.map((p) => ({
      id: p.id,
      fullName: p.fullName,
      phone: p.phone,
      createdAt: p.createdAt,
    })),
  });
}
