import Link from "next/link";
import { notFound } from "next/navigation";
import { getPatient } from "@/lib/db";
import { PHOTO_ANGLES, getSlotLabels } from "@/lib/photo-angles";
import { serviceLabel } from "@/lib/services-catalog";
import { STATUS_LABELS, STATUS_STEP } from "@/lib/status";

function recordCode(id: string) { return `VT-${id.replace(/-/g, "").slice(0, 6).toUpperCase()}`; }
function date(value: string) { return new Intl.DateTimeFormat("vi-VN", { day: "2-digit", month: "2-digit", year: "numeric" }).format(new Date(value)); }

export default async function PatientDossierPage({ params }: { params: Promise<{ id: string }> }) {
  const { id } = await params;
  const patient = await getPatient(id);
  if (!patient) notFound();
  const uploadedCount = PHOTO_ANGLES.filter((angle) => patient.photos[angle]).length;
  const labels = getSlotLabels(patient.services);
  const latestScan = patient.scanSessions?.slice().sort((a, b) => b.updatedAt.localeCompare(a.updatedAt))[0];
  const has3DModel = Boolean(patient.model3d?.before);

  let scanStatusText = "Chưa quét";
  let scanActionText = "Bắt đầu quét";
  let scanActionHref = `/patients/${patient.id}/scan`;
  let scanBadgeColor = "bg-muted-bg text-muted";

  if (has3DModel) {
    scanStatusText = "Sẵn sàng (3D Ready)";
    scanActionText = "Mở 3D Studio";
    scanActionHref = `/patients/${patient.id}/studio`;
    scanBadgeColor = "bg-success/15 text-success";
  } else if (latestScan) {
    switch (latestScan.status) {
      case "capturing":
      case "uploading":
        scanStatusText = "Đang quét (Capturing)";
        scanActionText = "Tiếp tục quét";
        scanBadgeColor = "bg-accent/15 text-accent";
        break;
      case "processing":
      case "reconstructing":
        scanStatusText = "Đang xử lý (Processing)";
        scanActionText = "Xem tiến trình";
        scanBadgeColor = "bg-amber-500/15 text-amber-300";
        break;
      case "needs_rescan":
      case "failed":
        scanStatusText = "Cần quét lại (Needs Rescan)";
        scanActionText = "Quét lại ngay";
        scanBadgeColor = "bg-danger/15 text-danger";
        break;
      case "quality_check":
        scanStatusText = "Đã thu thập (Chờ kiểm tra chất lượng)";
        scanActionText = "Xem kết quả scan";
        scanBadgeColor = "bg-warning/15 text-warning";
        break;
      case "ready":
        scanStatusText = "Baseline sẵn sàng";
        scanActionText = "Mở 3D Studio";
        scanActionHref = `/patients/${patient.id}/studio`;
        scanBadgeColor = "bg-success/15 text-success";
        break;
      default:
        scanStatusText = "Chưa quét";
        scanActionText = "Bắt đầu quét";
        scanBadgeColor = "bg-muted-bg text-muted";
    }
  }

  const workflow = [
    { title: "1. Hồ sơ", detail: "Thông tin & dịch vụ", href: `/patients/${patient.id}`, done: true },
    { title: "2. Ảnh bệnh án", detail: `${uploadedCount}/${PHOTO_ANGLES.length} góc ảnh`, href: `/patients/${patient.id}/photos`, done: uploadedCount > 0 },
    { title: "3. Scan 3D", detail: scanStatusText, href: `/patients/${patient.id}/scan`, done: has3DModel },
    { title: "4. 3D Studio", detail: patient.simulation ? "Đã có phương án" : "Chưa có phương án", href: `/patients/${patient.id}/studio`, done: Boolean(patient.simulation) },
  ];

  return <div className="mx-auto w-full max-w-5xl px-4 py-5 sm:px-6 sm:py-8 lg:px-10 lg:py-10">
    <Link href="/patients" className="text-sm font-bold text-muted transition hover:text-accent">← Hồ sơ bệnh án</Link>
    <section className="mt-5 rounded-3xl border border-border bg-card p-5 sm:p-7"><div className="flex flex-col gap-5 sm:flex-row sm:items-start sm:justify-between"><div className="flex gap-4"><span className="flex h-14 w-14 shrink-0 items-center justify-center rounded-2xl bg-accent/10 text-xl font-bold text-accent">{patient.fullName.trim().slice(0, 1).toUpperCase()}</span><div><p className="text-xs font-bold tracking-[0.14em] text-muted">{recordCode(patient.id)}</p><h1 className="mt-1 text-2xl font-bold tracking-tight text-card-foreground sm:text-3xl">{patient.fullName}</h1><p className="mt-1 text-sm text-muted">{patient.phone}{patient.address ? ` · ${patient.address}` : ""}</p></div></div><div className="sm:text-right"><span className={`inline-flex rounded-full px-3 py-1.5 text-xs font-bold ${scanBadgeColor}`}>{patient.status === "da-tao-mo-hinh" ? "3D Ready" : STATUS_LABELS[patient.status]}</span><p className="mt-2 text-xs text-muted">Cập nhật {date(patient.updatedAt)}</p></div></div><div className="mt-5 flex flex-wrap gap-2">{patient.services.length ? patient.services.map((service) => <span key={service} className="rounded-full bg-muted-bg px-3 py-1.5 text-xs font-bold text-foreground">{serviceLabel(service)}</span>) : <span className="text-sm text-muted">Chưa chọn dịch vụ tư vấn.</span>}</div></section>
    <section className="mt-5"><div className="flex items-center justify-between"><div><h2 className="text-lg font-bold text-card-foreground">Tiến trình hồ sơ</h2><p className="mt-1 text-sm text-muted">Mọi dữ liệu được gắn với mã hồ sơ này.</p></div><span className="text-xs font-bold text-accent">Bước {STATUS_STEP[patient.status]}/4</span></div><div className="mt-4 grid gap-3 sm:grid-cols-2 lg:grid-cols-4">{workflow.map((item, index) => <Link key={item.title} href={item.href} className="rounded-2xl border border-border bg-card p-4 transition hover:border-accent/30"><div className="flex items-center justify-between"><span className={`flex h-7 w-7 items-center justify-center rounded-full text-xs font-bold ${item.done ? "bg-success/10 text-success" : "bg-muted-bg text-muted"}`}>{item.done ? "✓" : index + 1}</span><span className="text-xs font-bold text-accent">Mở →</span></div><h3 className="mt-4 font-bold text-card-foreground">{item.title}</h3><p className="mt-1 text-xs leading-5 text-muted">{item.detail}</p></Link>)}</div></section>
    <section className="mt-7 grid gap-5 lg:grid-cols-[1.2fr_0.8fr]"><div className="rounded-3xl border border-border bg-card p-5"><div className="flex items-center justify-between"><div><h2 className="font-bold text-card-foreground">Ảnh bệnh án</h2><p className="mt-1 text-sm text-muted">Ảnh được lưu theo đúng hồ sơ khách hàng.</p></div><Link href={`/patients/${patient.id}/photos`} className="text-sm font-bold text-accent">Quản lý ảnh</Link></div><div className="mt-4 grid grid-cols-4 gap-2">{PHOTO_ANGLES.map((angle, index) => { const asset = patient.photos[angle]; return <div key={angle} className="min-w-0"><div className="aspect-[3/4] overflow-hidden rounded-xl border border-border bg-muted-bg">{asset ? <img src={`/api/files/${patient.id}/photos/${asset.fileName}`} alt={labels[index].label} className="h-full w-full object-cover" /> : <div className="flex h-full items-center justify-center text-center text-[10px] font-medium text-muted">Chưa có ảnh</div>}</div><p className="mt-1.5 truncate text-center text-[10px] text-muted">{labels[index].label}</p></div>; })}</div></div><div className="rounded-3xl border border-border bg-card p-5"><h2 className="font-bold text-card-foreground">Thông tin tư vấn</h2><dl className="mt-4 space-y-4 text-sm"><div><dt className="text-xs font-bold uppercase tracking-wide text-muted">Dị ứng thuốc</dt><dd className="mt-1 text-foreground">{patient.allergies || "Chưa ghi nhận"}</dd></div><div><dt className="text-xs font-bold uppercase tracking-wide text-muted">Bệnh lý nền</dt><dd className="mt-1 text-foreground">{patient.underlyingConditions || "Chưa ghi nhận"}</dd></div><div><dt className="text-xs font-bold uppercase tracking-wide text-muted">Tiền sử phẫu thuật</dt><dd className="mt-1 text-foreground">{patient.surgicalHistory || "Chưa ghi nhận"}</dd></div></dl></div></section>
    {patient.profilePreview && Object.keys(patient.profilePreview).length > 0 && (
      <section className="mt-5 rounded-3xl border border-border bg-card p-5">
        <div>
          <h2 className="font-bold text-card-foreground">Ảnh đại diện 3D Scan</h2>
          <p className="mt-1 text-sm text-muted">4 ảnh đại diện được tự động chọn từ dữ liệu quét (chỉ để bác sĩ xem nhanh — dữ liệu tái tạo 3D vẫn dùng toàn bộ khung hình đã quét).</p>
        </div>
        <div className="mt-4 grid grid-cols-2 gap-2 sm:max-w-md">
          {(["front", "left", "right", "three_quarter"] as const).map((role) => {
            const img = patient.profilePreview?.[role];
            const roleLabel = { front: "Chính diện", left: "Trái", right: "Phải", three_quarter: "Góc 3/4" }[role];
            return (
              <div key={role} className="min-w-0">
                <div className="aspect-[3/4] overflow-hidden rounded-xl border border-border bg-muted-bg">
                  {img ? (
                    <img src={`/api/files/${patient.id}/profile-preview/${img.fileName}`} alt={roleLabel} className="h-full w-full object-cover" />
                  ) : (
                    <div className="flex h-full items-center justify-center text-center text-[10px] font-medium text-muted">Chưa đủ dữ liệu</div>
                  )}
                </div>
                <p className="mt-1.5 truncate text-center text-[10px] text-muted">{roleLabel}{img ? ` · yaw ${Math.round(img.yaw)}°` : ""}</p>
              </div>
            );
          })}
        </div>
      </section>
    )}
    <section className="mt-5 rounded-3xl border border-accent/20 bg-accent/5 p-5"><div className="flex flex-col gap-4 sm:flex-row sm:items-center sm:justify-between"><div><p className="text-xs font-bold tracking-[0.14em] text-accent">3D SCAN &amp; CONSULTATION</p><h2 className="mt-1 text-lg font-bold text-card-foreground">{has3DModel ? "Baseline 3D sẵn sàng để bác sĩ tư vấn" : scanStatusText}</h2><p className="mt-1 max-w-2xl text-sm leading-6 text-muted">{has3DModel ? "Studio dùng baseline làm khung trước; khung sau là bản clone được biến dạng theo phương án bác sĩ." : "Quy trình scan 5 góc chuẩn y khoa. Web camera hoạt động ở chế độ Guided Photographic Capture. Để có mesh 3D TrueDepth cần kết nối thiết bị iPad/iPhone Pro chuyên dụng."}</p></div><Link href={scanActionHref} className="inline-flex shrink-0 items-center justify-center rounded-xl bg-accent px-5 py-3 text-sm font-bold text-white shadow-[var(--accent-glow)] hover:scale-[1.01] transition">{scanActionText}</Link></div></section>
  </div>;
}
