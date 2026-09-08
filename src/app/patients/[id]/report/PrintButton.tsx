"use client";

import { BUTTON_PRIMARY_CLASS } from "@/lib/ui";

export default function PrintButton() {
  return (
    <button
      type="button"
      onClick={() => window.print()}
      className={BUTTON_PRIMARY_CLASS + " shrink-0 print:hidden"}
    >
      In báo cáo (PDF)
    </button>
  );
}
