"use client";

import Link from "next/link";
import { useState, useTransition } from "react";
import { useRouter } from "next/navigation";

interface Dropzone3DProps { patientId: string; has3DModel: boolean; }

export default function Dropzone3D({ patientId, has3DModel }: Dropzone3DProps) {
  const router = useRouter();
  const [isUploading, startUploading] = useTransition();
  const [message, setMessage] = useState<string | null>(null);
  const [isError, setIsError] = useState(false);
  function upload(file: File) {
    if (!file.name.toLowerCase().endsWith(".glb") && !file.name.toLowerCase().endsWith(".obj")) { setIsError(true); setMessage("Chỉ nhận baseline .glb hoặc .obj từ thiết bị scan."); return; }
    startUploading(async () => { try { setMessage("Đang lưu baseline 3D vào hồ sơ..."); setIsError(false); const form = new FormData(); form.set("model3d", file); const response = await fetch(`/api/patients/${patientId}/upload-3d`, { method: "POST", body: form }); const data = await response.json(); if (!response.ok) throw new Error(data.error || "Không thể nạp file."); setMessage("Baseline 3D đã được nạp. Bạn có thể mở Studio."); router.refresh(); } catch (error) { setIsError(true); setMessage(error instanceof Error ? error.message : "Tải file thất bại."); } });
  }
  return <div className="space-y-4"><Link href={`/patients/${patientId}/scan`} className="block rounded-3xl bg-[linear-gradient(125deg,#a74048,#7f3038)] p-6 text-white shadow-[0_14px_38px_rgba(127,48,56,0.2)]"><span className="inline-flex rounded-full bg-white/15 px-3 py-1 text-[11px] font-bold">SCAN SESSION</span><h2 className="mt-4 text-xl font-bold">Bắt đầu guided face capture</h2><p className="mt-2 max-w-lg text-sm leading-6 text-white/80">Thu 5 góc theo hướng dẫn, lưu frame, quality check và gắn session vào bệnh án. Web fallback không tạo 3D hoặc giả lập TrueDepth.</p><span className="mt-5 inline-flex text-sm font-bold">Mở màn hình scan →</span></Link><div className="rounded-3xl border border-border bg-card p-5"><p className="text-xs font-bold tracking-[0.14em] text-muted">NHẬP TỪ THIẾT BỊ SCAN</p><h2 className="mt-2 text-lg font-bold text-card-foreground">Đã có baseline 3D?</h2><p className="mt-1 text-sm leading-6 text-muted">Nạp file GLB hoặc OBJ do máy scan/reconstruction provider tạo. File này được lưu theo đúng patient ID.</p><label className="mt-5 flex cursor-pointer items-center justify-center rounded-2xl border border-dashed border-border bg-muted-bg px-4 py-6 text-center text-sm font-bold text-foreground transition hover:border-accent hover:bg-accent/5"><input type="file" accept=".glb,.obj" className="sr-only" disabled={isUploading} onChange={(event) => { const file = event.target.files?.[0]; if (file) upload(file); }} />{isUploading ? "Đang nạp baseline..." : "Chọn file baseline (.GLB / .OBJ)"}</label>{message && <p className={`mt-4 rounded-xl p-3 text-sm ${isError ? "bg-danger/10 text-danger" : "bg-success/10 text-success"}`}>{message}</p>}</div>{has3DModel && <div className="flex items-center justify-between rounded-2xl border border-success/20 bg-success/5 p-4"><div><p className="text-sm font-bold text-success">Baseline 3D đã sẵn sàng</p><p className="mt-1 text-xs text-muted">Được nạp từ dữ liệu model hiện có.</p></div><button onClick={() => router.push(`/patients/${patientId}/studio`)} className="rounded-xl bg-success px-3 py-2 text-xs font-bold text-white">Mở Studio</button></div>}</div>;
}
