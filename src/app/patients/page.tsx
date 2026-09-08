import Link from "next/link";
import { listPatients } from "@/lib/db";
import { serviceLabel } from "@/lib/services-catalog";
import { STATUS_LABELS } from "@/lib/status";
import type { PatientStatus } from "@/lib/types";

const statuses: PatientStatus[] = ["moi-tao", "da-tai-anh", "da-tao-mo-hinh", "da-mo-phong"];

function recordCode(id: string) {
  return `VT-${id.replace(/-/g, "").slice(0, 6).toUpperCase()}`;
}

function date(value: string) {
  return new Intl.DateTimeFormat("vi-VN", { day: "2-digit", month: "2-digit", year: "numeric" }).format(new Date(value));
}

function statusTone(status: PatientStatus) {
  return status === "da-tao-mo-hinh" || status === "da-mo-phong"
    ? "bg-success/15 text-emerald-400 border border-emerald-500/30"
    : status === "da-tai-anh"
    ? "bg-warning/15 text-amber-400 border border-amber-500/30"
    : "bg-muted-bg text-muted border border-border";
}

export default async function PatientsPage({
  searchParams,
}: {
  searchParams: Promise<{ q?: string; status?: string; sort?: string }>;
}) {
  const query = await searchParams;
  const q = query.q?.trim().toLowerCase() ?? "";
  const status = statuses.includes(query.status as PatientStatus) ? (query.status as PatientStatus) : undefined;
  const sort = query.sort === "name" ? "name" : "recent";
  const all = await listPatients();

  const patients = all
    .filter(
      (patient) =>
        (!q || `${patient.fullName} ${patient.phone} ${patient.id}`.toLowerCase().includes(q)) &&
        (!status || patient.status === status)
    )
    .sort((a, b) =>
      sort === "name" ? a.fullName.localeCompare(b.fullName, "vi") : b.updatedAt.localeCompare(a.updatedAt)
    );

  return (
    <div className="mx-auto w-full max-w-5xl px-3 py-4 sm:px-6 sm:py-8 lg:px-10 lg:py-10">
      {/* Header */}
      <div className="flex flex-col sm:flex-row sm:items-start sm:justify-between gap-3">
        <div>
          <p className="text-[11px] font-bold tracking-[0.16em] text-accent uppercase">KHÁCH HÀNG & BỆNH ÁN</p>
          <h1 className="mt-1 text-2xl font-black tracking-tight text-card-foreground sm:text-3xl">Hồ sơ bệnh án</h1>
          <p className="mt-1 text-xs sm:text-sm text-muted">{all.length} hồ sơ đang được quản lý trên hệ thống.</p>
        </div>
        <Link
          href="/patients/new"
          className="inline-flex items-center justify-center rounded-2xl bg-accent px-5 py-3 text-sm font-bold text-white shadow-lg hover:scale-105 active:scale-95 transition"
        >
          <span>+ Tạo hồ sơ mới</span>
        </Link>
      </div>

      {/* Search & Filter Form */}
      <form className="mt-5 rounded-2xl border border-border bg-card p-3 flex flex-col sm:flex-row sm:items-center gap-2.5 shadow-sm">
        <input
          name="q"
          defaultValue={q}
          placeholder="Tìm tên, SĐT hoặc mã hồ sơ..."
          className="w-full bg-transparent px-3 py-2 text-sm outline-none placeholder:text-muted"
        />
        <div className="flex items-center gap-2">
          <select
            name="status"
            defaultValue={status ?? ""}
            className="flex-1 sm:flex-none rounded-xl bg-muted-bg px-3 py-2.5 text-xs font-bold text-foreground outline-none border border-border"
          >
            <option value="">Tất cả trạng thái</option>
            {statuses.map((item) => (
              <option key={item} value={item}>
                {STATUS_LABELS[item]}
              </option>
            ))}
          </select>
          <select
            name="sort"
            defaultValue={sort}
            className="rounded-xl bg-muted-bg px-3 py-2.5 text-xs font-bold text-foreground outline-none border border-border"
          >
            <option value="recent">Mới cập nhật</option>
            <option value="name">Tên A–Z</option>
          </select>
          <button type="submit" className="rounded-xl bg-accent px-4 py-2.5 text-xs font-bold text-white">
            Lọc
          </button>
        </div>
      </form>

      {/* Horizontal Status Filter Chips */}
      <div className="mt-4 flex gap-2 overflow-x-auto pb-1 no-scrollbar">
        {[
          { label: "Tất cả", value: undefined, count: all.length },
          ...statuses.map((item) => ({
            label: STATUS_LABELS[item],
            value: item,
            count: all.filter((p) => p.status === item).length,
          })),
        ].map((item) => (
          <Link
            key={item.label}
            href={`/patients${item.value ? `?status=${item.value}` : ""}`}
            className={`whitespace-nowrap rounded-full px-3.5 py-1.5 text-xs font-bold transition ${
              status === item.value
                ? "bg-accent text-white shadow-sm"
                : "bg-card text-muted hover:text-foreground border border-border"
            }`}
          >
            {item.label} · {item.count}
          </Link>
        ))}
      </div>

      {/* Patient List Cards */}
      <div className="mt-5 space-y-3">
        {patients.length === 0 ? (
          <div className="rounded-3xl border border-dashed border-border bg-card px-5 py-12 text-center">
            <p className="font-bold text-card-foreground">Không tìm thấy hồ sơ phù hợp</p>
            <p className="mt-1 text-xs text-muted">Thử thay đổi từ khoá hoặc bộ lọc.</p>
          </div>
        ) : (
          patients.map((patient) => (
            <Link
              key={patient.id}
              href={`/patients/${patient.id}`}
              className="block rounded-2xl border border-border bg-card p-4 transition hover:border-accent/40 hover:shadow-md active:scale-[0.99]"
            >
              <div className="flex gap-3.5 items-start">
                <span className="flex h-11 w-11 shrink-0 items-center justify-center rounded-2xl bg-accent/15 text-sm font-black text-accent">
                  {patient.fullName.trim().slice(0, 1).toUpperCase()}
                </span>
                <div className="min-w-0 flex-1">
                  <div className="flex flex-wrap items-start justify-between gap-2">
                    <div>
                      <h2 className="font-bold text-card-foreground text-sm sm:text-base">{patient.fullName}</h2>
                      <p className="mt-0.5 text-xs text-muted">
                        {recordCode(patient.id)} · {patient.phone}
                      </p>
                    </div>
                    <span className={`rounded-full px-2.5 py-1 text-[10px] font-black uppercase ${statusTone(patient.status)}`}>
                      {patient.status === "da-tao-mo-hinh" ? "3D Ready" : STATUS_LABELS[patient.status]}
                    </span>
                  </div>
                  <p className="mt-2.5 text-xs sm:text-sm text-foreground/90 font-medium">
                    {patient.services.length ? patient.services.map(serviceLabel).join(" · ") : "Chưa chọn dịch vụ"}
                  </p>
                  <div className="mt-3 flex items-center justify-between text-xs text-muted border-t border-border/50 pt-2.5">
                    <span>Cập nhật: {date(patient.updatedAt)}</span>
                    <span className="font-bold text-accent">Mở hồ sơ →</span>
                  </div>
                </div>
              </div>
            </Link>
          ))
        )}
      </div>
    </div>
  );
}
