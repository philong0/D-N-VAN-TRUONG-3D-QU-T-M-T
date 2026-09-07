import type { Metadata, Viewport } from "next";
import Sidebar from "@/components/Sidebar";
import "./globals.css";

export const metadata: Metadata = {
  title: "Viện Thẩm Mỹ Văn Trường — CRM & 3D Consultation Studio",
  description:
    "Hệ thống quản lý hồ sơ bệnh án, mô phỏng 3D và tư vấn thẩm mỹ hỗ trợ AI.",
};

export const viewport: Viewport = {
  width: "device-width",
  initialScale: 1,
  maximumScale: 1,
  userScalable: false,
  viewportFit: "cover",
  themeColor: "#8B263E",
};

export default function RootLayout({
  children,
}: {
  children: React.ReactNode;
}) {
  return (
    <html lang="vi" data-theme="light" className="h-full antialiased">
      <body className="flex min-h-full flex-col bg-background text-foreground overflow-x-hidden">
        <Sidebar />
        <main className="min-w-0 w-full flex-1 pb-24 pt-[124px] lg:pl-[272px] lg:pb-0 lg:pt-0 print:p-0">{children}</main>
      </body>
    </html>
  );
}
