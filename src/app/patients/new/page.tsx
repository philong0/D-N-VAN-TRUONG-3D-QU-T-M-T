import WorkflowSteps from "@/components/WorkflowSteps";
import { SERVICES_CATALOG } from "@/lib/services-catalog";
import { BUTTON_PRIMARY_CLASS, INPUT_CLASS } from "@/lib/ui";
import { createPatientAction } from "../actions";

export default function NewPatientPage() {
  return (
    <div className="mx-auto w-full max-w-2xl px-4 sm:px-6 py-6 sm:py-12">
      <WorkflowSteps current={1} />

      <h1 className="mt-6 text-xl sm:text-2xl font-semibold tracking-tight text-card-foreground">
        Bước 1 · Tạo hồ sơ bệnh án
      </h1>
      <p className="mt-1 text-xs sm:text-sm text-muted">
        Nhập thông tin bệnh nhân, tiền sử y khoa và chọn các dịch vụ đăng ký thực hiện.
      </p>

      <form action={createPatientAction} className="mt-6 sm:mt-8 flex flex-col gap-5 sm:gap-6">
        <div>
          <label htmlFor="fullName" className="mb-1.5 block text-sm font-medium text-card-foreground">
            Họ tên <span className="text-danger">*</span>
          </label>
          <input id="fullName" name="fullName" required className={INPUT_CLASS} placeholder="Nguyễn Thị A" />
        </div>

        <div>
          <label htmlFor="phone" className="mb-1.5 block text-sm font-medium text-card-foreground">
            Số điện thoại <span className="text-danger">*</span>
          </label>
          <input id="phone" name="phone" required type="tel" className={INPUT_CLASS} placeholder="09xx xxx xxx" />
        </div>

        <div>
          <label htmlFor="address" className="mb-1.5 block text-sm font-medium text-card-foreground">
            Địa chỉ
          </label>
          <input id="address" name="address" className={INPUT_CLASS} placeholder="Số nhà, đường, quận/huyện..." />
        </div>

        <div>
          <label htmlFor="allergies" className="mb-1.5 block text-sm font-medium text-card-foreground">
            Tiền sử dị ứng thuốc
          </label>
          <textarea
            id="allergies"
            name="allergies"
            rows={2}
            className={INPUT_CLASS}
            placeholder="Kháng sinh, thuốc gây tê/mê, mỹ phẩm..."
          />
        </div>

        <div>
          <label
            htmlFor="underlyingConditions"
            className="mb-1.5 block text-sm font-medium text-card-foreground"
          >
            Bệnh lý nền
          </label>
          <textarea
            id="underlyingConditions"
            name="underlyingConditions"
            rows={2}
            className={INPUT_CLASS}
            placeholder="Tim mạch, tiểu đường, rối loạn đông máu..."
          />
        </div>

        <div>
          <label htmlFor="surgicalHistory" className="mb-1.5 block text-sm font-medium text-card-foreground">
            Tiền sử phẫu thuật
          </label>
          <textarea
            id="surgicalHistory"
            name="surgicalHistory"
            rows={2}
            className={INPUT_CLASS}
            placeholder="Các phẫu thuật thẩm mỹ / y khoa đã thực hiện trước đây..."
          />
        </div>

        <fieldset>
          <legend className="mb-2 text-sm font-medium text-card-foreground">Dịch vụ đăng ký</legend>
          <div className="grid grid-cols-1 gap-2 sm:grid-cols-2">
            {SERVICES_CATALOG.map((service) => (
              <label
                key={service.key}
                className="flex items-center gap-2.5 rounded-lg border border-border px-3 py-2.5 text-sm text-foreground hover:bg-accent/5"
              >
                <input type="checkbox" name="services" value={service.key} className="h-4 w-4 accent-accent" />
                {service.label}
              </label>
            ))}
          </div>
        </fieldset>

        <button type="submit" className={BUTTON_PRIMARY_CLASS + " mt-2 w-full sm:w-fit py-3 justify-center"}>
          Tạo hồ sơ &amp; tiếp tục tải ảnh →
        </button>
      </form>
    </div>
  );
}
