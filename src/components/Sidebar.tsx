"use client";

import Link from "next/link";
import { usePathname } from "next/navigation";
import type { SVGProps } from "react";

type IconProps = SVGProps<SVGSVGElement>;

function IconHome(props: IconProps) { return <svg width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.8" {...props}><path d="m3 10 9-7 9 7v10a1 1 0 0 1-1 1h-5v-6H9v6H4a1 1 0 0 1-1-1V10Z" strokeLinecap="round" strokeLinejoin="round" /></svg>; }
function IconPatients(props: IconProps) { return <svg width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.8" {...props}><circle cx="9" cy="8" r="3" /><path d="M3.5 20c.5-3.3 2.3-5 5.5-5s5 1.7 5.5 5M16 5.5a3 3 0 0 1 0 5M17.5 15.2c2 .5 3 2.1 3.3 4.8" strokeLinecap="round" /></svg>; }
function IconPlus(props: IconProps) { return <svg width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" {...props}><path d="M12 5v14M5 12h14" strokeLinecap="round" /></svg>; }
function IconCube(props: IconProps) { return <svg width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.8" {...props}><path d="m12 3 8 4.5v9L12 21l-8-4.5v-9L12 3Z" strokeLinejoin="round" /><path d="m4.5 7.8 7.5 4.3 7.5-4.3M12 12v8.4" /></svg>; }
function IconPhoto(props: IconProps) { return <svg width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.8" {...props}><rect x="3" y="4" width="18" height="16" rx="2" /><circle cx="8.2" cy="9" r="1.4" /><path d="m4 17 5.1-5.1 3.5 3.4 2.4-2.4L20 18" strokeLinecap="round" strokeLinejoin="round" /></svg>; }

const desktopItems = [
  { href: "/", label: "Tổng quan", Icon: IconHome },
  { href: "/patients", label: "Hồ sơ bệnh án", Icon: IconPatients },
  { href: "/studio", label: "3D Consultation", Icon: IconCube },
  { href: "/gallery", label: "Thư viện ảnh", Icon: IconPhoto },
];
const mobileItems = [
  { href: "/", label: "Trang chủ", Icon: IconHome },
  { href: "/patients", label: "Hồ sơ", Icon: IconPatients },
  { href: "/patients/new", label: "Tạo hồ sơ", Icon: IconPlus, primary: true },
  { href: "/studio", label: "3D Studio", Icon: IconCube },
  { href: "/gallery", label: "Ảnh", Icon: IconPhoto },
];

function active(pathname: string, href: string) {
  if (href === "/") return pathname === "/";
  if (href === "/patients") return pathname === "/patients" || /^\/patients\/[^/]+(\/|$)/.test(pathname);
  return pathname === href || pathname.startsWith(`${href}/`);
}

export default function Sidebar() {
  const pathname = usePathname() ?? "/";
  if (pathname.endsWith("/scan")) return null;
  return <>
    <header className="fixed inset-x-0 top-0 z-30 flex flex-col justify-end border-b border-border/80 bg-sidebar-bg/95 backdrop-blur-xl lg:hidden print:hidden" style={{ height: "114px", paddingTop: "58px" }}>
      <div className="flex h-14 w-full items-center justify-between px-4">
        <Link href="/" className="flex items-center gap-2.5" aria-label="Về tổng quan">
          <span className="flex h-9 w-9 items-center justify-center rounded-xl bg-accent text-xs font-bold tracking-tight text-white shadow-[var(--accent-glow)]">VT</span>
          <span className="leading-tight">
            <span className="block text-[10px] font-semibold tracking-wider text-muted">VIỆN THẨM MỸ</span>
            <span className="block text-sm font-bold text-card-foreground">Văn Trường</span>
          </span>
        </Link>
        <Link href="/patients/new" className="flex h-9 items-center gap-1.5 rounded-xl bg-accent px-3 text-xs font-bold text-white shadow-[var(--accent-glow)]">
          <IconPlus className="h-4 w-4" />Hồ sơ mới
        </Link>
      </div>
    </header>
    <aside className="fixed inset-y-0 left-0 z-40 hidden w-[272px] flex-col border-r border-border bg-sidebar-bg px-4 py-5 lg:flex print:hidden">
      <Link href="/" className="flex items-center gap-3 px-2"><span className="flex h-11 w-11 items-center justify-center rounded-2xl bg-accent text-sm font-bold text-white shadow-[var(--accent-glow)]">VT</span><span><span className="block text-xs font-medium tracking-[0.12em] text-muted">VIỆN THẨM MỸ</span><span className="block text-base font-bold text-card-foreground">Văn Trường</span></span></Link>
      <Link href="/patients/new" className="mt-8 flex items-center justify-center gap-2 rounded-2xl bg-accent px-4 py-3.5 text-sm font-bold text-white shadow-[var(--accent-glow)] transition hover:bg-accent-hover"><IconPlus className="h-5 w-5" />Tạo hồ sơ mới</Link>
      <nav className="mt-7 space-y-1.5">{desktopItems.map(({ href, label, Icon }) => { const isActive = active(pathname, href); return <Link key={href} href={href} className={`flex items-center gap-3 rounded-xl px-3.5 py-3 text-sm font-semibold transition ${isActive ? "bg-accent/10 text-accent" : "text-muted hover:bg-muted-bg hover:text-foreground"}`}><Icon className="h-5 w-5" />{label}</Link>; })}</nav>
      <div className="mt-auto rounded-2xl border border-border bg-card p-4"><div className="flex items-center gap-2 text-xs font-semibold text-success"><span className="h-2 w-2 rounded-full bg-success" />Hệ thống sẵn sàng</div><p className="mt-2 text-xs leading-5 text-muted">Nền tảng tư vấn 3D. Kết quả mô phỏng cần được bác sĩ thẩm định.</p></div>
    </aside>
    <nav className="fixed inset-x-0 bottom-0 z-40 grid h-[76px] grid-cols-5 border-t border-border/80 bg-sidebar-bg/95 px-1 pb-[max(0.45rem,env(safe-area-inset-bottom))] pt-2 backdrop-blur-xl lg:hidden print:hidden" aria-label="Điều hướng chính">{mobileItems.map(({ href, label, Icon, primary }) => { const isActive = active(pathname, href); return <Link key={href} href={href} className={`flex min-w-0 flex-col items-center justify-start gap-1 text-[10px] font-semibold ${primary ? "-mt-6 text-accent" : isActive ? "text-accent" : "text-muted"}`}><span className={`flex items-center justify-center ${primary ? "h-12 w-12 rounded-2xl bg-accent text-white shadow-[var(--accent-glow)]" : "h-7 w-7"}`}><Icon className={primary ? "h-6 w-6" : "h-5 w-5"} /></span><span className="truncate">{label}</span></Link>; })}</nav>
  </>;
}
