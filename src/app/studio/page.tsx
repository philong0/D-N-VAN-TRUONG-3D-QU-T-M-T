import Link from "next/link";
import { listPatients } from "@/lib/db";
import { serviceLabel } from "@/lib/services-catalog";
import { BUTTON_PRIMARY_CLASS, CARD_CLASS } from "@/lib/ui";

export default async function StudioLandingPage() {
  const patients = await listPatients();
  const ready = patients.filter((p) => p.model3d?.before);
  const notReady = patients.filter((p) => !p.model3d?.before);

  return (
    <div className="mx-auto w-full max-w-4xl px-6 py-12">
      <h1 className="text-2xl font-semibold tracking-tight text-card-foreground">
        🧊 3D Consultation Studio
      </h1>
      <p className="mt-1 text-sm text-muted">
        Chọn một hồ sơ đã có mô hình 3D để bắt đầu nắn chỉnh mô phỏng và xem đánh giá AI Clinical Advisor.
      </p>

      {ready.length === 0 ? (
        <div className={CARD_CLASS + " mt-8 border-dashed p-12 text-center text-sm text-muted"}>
          Chưa có hồ sơ nào sẵn sàng — hoàn tất Bước 1–3 (hồ sơ, ảnh, mô hình 3D) trước.
          <div className="mt-4">
            <Link href="/patients/new" className={BUTTON_PRIMARY_CLASS}>
              + Tạo hồ sơ mới
            </Link>
          </div>
        </div>
      ) : (
        <ul className={CARD_CLASS + " mt-8 divide-y divide-border"}>
          {ready.map((p) => (
            <li key={p.id}>
              <Link
                href={`/patients/${p.id}/studio`}
                className="flex flex-col gap-1 p-4 hover:bg-accent/5 sm:flex-row sm:items-center sm:justify-between"
              >
                <div>
                  <p className="font-medium text-card-foreground">{p.fullName}</p>
                  <p className="text-sm text-muted">
                    {p.services.length > 0 ? p.services.map(serviceLabel).join(", ") : "Chưa đăng ký dịch vụ"}
                  </p>
                </div>
                <span className="shrink-0 text-sm font-medium text-accent">Vào Studio →</span>
              </Link>
            </li>
          ))}
        </ul>
      )}

      {notReady.length > 0 && (
        <div className="mt-8">
          <h2 className="mb-2 text-sm font-semibold text-muted">Hồ sơ chưa sẵn sàng</h2>
          <ul className={CARD_CLASS + " divide-y divide-border"}>
            {notReady.map((p) => (
              <li key={p.id}>
                <Link
                  href={`/patients/${p.id}`}
                  className="flex items-center justify-between p-4 text-sm hover:bg-accent/5"
                >
                  <span className="text-foreground">{p.fullName}</span>
                  <span className="text-muted">Tiếp tục hồ sơ →</span>
                </Link>
              </li>
            ))}
          </ul>
        </div>
      )}
    </div>
  );
}
