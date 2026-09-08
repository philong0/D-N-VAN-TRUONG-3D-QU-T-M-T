import { NextRequest, NextResponse } from "next/server";
import { writeFile } from "fs/promises";
import path from "path";
import { getPatient, updatePatient } from "@/lib/db";
import { ensureDir } from "@/lib/storage";

export async function POST(
  request: NextRequest,
  context: { params: Promise<{ id: string }> }
) {
  try {
    const { id: patientId } = await context.params;
    const patient = await getPatient(patientId);

    if (!patient) {
      return NextResponse.json({ error: "Không tìm thấy hồ sơ bệnh nhân" }, { status: 404 });
    }

    const formData = await request.formData();
    const modelFile = (formData.get("model3d") || formData.get("file")) as File | null;
    if (!modelFile) {
      return NextResponse.json({ error: "Ảnh camera không phải baseline 3D. Hãy tạo Scan Session hoặc nạp file .glb/.obj do scanner tạo." }, { status: 400 });
    }

    // Thư mục lưu trữ: public/data/patients/[id]/ và public/models/patients/[id]/
    const publicDataDir = path.join(process.cwd(), "public", "data", "patients", patientId);
    const publicModelsDir = path.join(process.cwd(), "public", "models", "patients", patientId);

    await ensureDir(publicDataDir);
    await ensureDir(publicModelsDir);

    let modelFileName = "model.glb";

    const extension = path.extname(modelFile.name).toLowerCase();
    if (extension !== ".glb" && extension !== ".obj") {
      return NextResponse.json({ error: "Chỉ nhận file baseline .glb hoặc .obj từ scanner/reconstruction provider." }, { status: 400 });
    }
    const modelBuffer = Buffer.from(await modelFile.arrayBuffer());
    modelFileName = extension === ".obj" ? "model.obj" : "model.glb";
    await writeFile(path.join(publicDataDir, modelFileName), modelBuffer);
    // Keep the original extension. Never write OBJ bytes under a .glb name.
    await writeFile(path.join(publicModelsDir, extension === ".obj" ? "head.obj" : "head.glb"), modelBuffer);

    // 3. Cập nhật database hồ sơ bệnh nhân
    await updatePatient(patientId, (p) => {
      return {
        ...p,
        status: "da-tao-mo-hinh",
        model3d: {
          before: {
            generatedAt: new Date().toISOString(),
            sourcePhotos: (Object.keys(p.photos) as (keyof typeof p.photos)[]),
            method: `Imported baseline (${extension.slice(1).toUpperCase()})`,
          },
        },
      };
    });

    return NextResponse.json({
      success: true,
      message: "Đã nạp dữ liệu quét 3D của bệnh nhân thành công!",
      patientId,
      modelUrl: `/data/patients/${patientId}/${modelFileName}`,
    });
  } catch (error) {
    console.error("Lỗi upload 3D scan:", error);
    return NextResponse.json(
      { error: "Lỗi hệ thống khi lưu trữ dữ liệu 3D", details: String(error) },
      { status: 500 }
    );
  }
}
