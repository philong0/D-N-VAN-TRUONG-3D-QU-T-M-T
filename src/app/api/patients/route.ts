import { NextRequest, NextResponse } from "next/server";
import { listPatients, createPatient } from "@/lib/db";

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

export async function POST(request: NextRequest) {
  try {
    let fullName = "";
    let phone = "";
    let gender = "other";
    let dateOfBirth = "2000-01-01";
    let notes = "";

    const contentType = request.headers.get("content-type") || "";
    if (contentType.includes("application/json")) {
      const body = await request.json().catch(() => ({}));
      fullName = String(body.fullName || "").trim();
      phone = String(body.phone || "").trim();
      gender = body.gender || "other";
      dateOfBirth = body.dateOfBirth || "2000-01-01";
      notes = body.notes || "";
    } else {
      const formData = await request.formData().catch(() => null);
      if (formData) {
        fullName = String(formData.get("fullName") || "").trim();
        phone = String(formData.get("phone") || "").trim();
        gender = String(formData.get("gender") || "other");
        dateOfBirth = String(formData.get("dateOfBirth") || "2000-01-01");
        notes = String(formData.get("notes") || "");
      }
    }

    if (!fullName || !phone) {
      return NextResponse.json({ error: "Họ tên và số điện thoại là bắt buộc." }, { status: 400 });
    }

    const patient = await createPatient({
      fullName,
      phone,
      address: notes,
      allergies: "Không",
      underlyingConditions: "Không",
      surgicalHistory: "Chưa từng phẫu thuật",
      services: ["sua-mui-cau-truc", "don-cam-vline"],
    });

    return NextResponse.json({ success: true, patient }, { status: 201 });
  } catch (error) {
    console.error("Error creating patient via API:", error);
    return NextResponse.json({ error: "Lỗi hệ thống khi tạo hồ sơ bệnh nhân." }, { status: 500 });
  }
}

