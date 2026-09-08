import Link from "next/link";
import { listPatients } from "@/lib/db";
import { PHOTO_ANGLES, getSlotLabels } from "@/lib/photo-angles";
import { CARD_CLASS } from "@/lib/ui";

export default async function GalleryPage() {
  const patients = await listPatients();
  const withPhotos = patients.filter((p) => Object.keys(p.photos).length > 0);

  return (
    <div className="mx-auto w-full max-w-5xl px-6 py-12">
      <h1 className="text-2xl font-semibold tracking-tight text-card-foreground">
        🖼️ Thư viện 4 góc ảnh Y tế
      </h1>
      <p className="mt-1 text-sm text-muted">
        Tổng hợp bộ 4 góc ảnh chuẩn Y khoa của toàn bộ hồ sơ — nhãn góc thay đổi theo dịch vụ đăng ký của
        từng bệnh nhân.
      </p>

      {withPhotos.length === 0 ? (
        <div className={CARD_CLASS + " mt-8 border-dashed p-12 text-center text-sm text-muted"}>
          Chưa có ảnh nào được tải lên.
        </div>
      ) : (
        <div className="mt-8 flex flex-col gap-8">
          {withPhotos.map((patient) => {
            const slotLabels = getSlotLabels(patient.services);
            return (
              <div key={patient.id} className={CARD_CLASS + " p-5"}>
                <div className="mb-3 flex items-center justify-between">
                  <Link
                    href={`/patients/${patient.id}`}
                    className="text-sm font-semibold text-card-foreground hover:text-accent"
                  >
                    {patient.fullName}
                  </Link>
                  <span className="text-xs text-muted">
                    {Object.keys(patient.photos).length}/{PHOTO_ANGLES.length} góc
                  </span>
                </div>
                <div className="grid grid-cols-4 gap-2">
                  {PHOTO_ANGLES.map((angle, i) => {
                    const asset = patient.photos[angle];
                    const label = slotLabels[i].label;
                    return (
                      <div key={angle} className="flex flex-col gap-1">
                        <div className="aspect-[3/4] overflow-hidden rounded-md border border-border bg-background">
                          {asset ? (
                            // eslint-disable-next-line @next/next/no-img-element
                            <img
                              src={`/api/files/${patient.id}/photos/${asset.fileName}`}
                              alt={label}
                              className="h-full w-full object-cover"
                            />
                          ) : (
                            <div className="flex h-full items-center justify-center text-[9px] text-muted">
                              Trống
                            </div>
                          )}
                        </div>
                        <p className="text-center text-[9px] leading-tight text-muted">{label}</p>
                      </div>
                    );
                  })}
                </div>
              </div>
            );
          })}
        </div>
      )}
    </div>
  );
}
