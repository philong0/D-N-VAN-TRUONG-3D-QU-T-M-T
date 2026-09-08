"use client";

import React, { useState, useEffect, Suspense } from "react";
import { useSearchParams } from "next/navigation";

// ==========================================
// 1. DATA CƠ CẤU GIẢI THƯỞNG CHUẨN ĐẠI LỄ 2/9 CỦA BÁC SĨ TRƯỜNG
// ==========================================
interface PrizeData {
  id: string;
  type: "win" | "miss";
  title: string;
  highlightText: string;
  subText: string;
  badge: string;
  visualType: "gold_bar" | "surgery_50" | "meso_20" | "triet_long" | "wish";
  voucherCode?: string;
  defaultRate: number;
}

const ALL_PROGRAM_ITEMS: PrizeData[] = [
  {
    id: "gold_1",
    type: "win",
    title: "CHÚC MỪNG QUÝ KHÁCH ĐÃ RINH LỘC!",
    highlightText: "VÀNG 9999 PHÁT TÀI",
    subText: "Vàng Ròng 9999 Chuẩn Y Khoa • Rinh Lộc May Mắn Đại Lễ 2/9",
    badge: "GIẢI NHẤT PHÁT TÀI",
    visualType: "gold_bar",
    voucherCode: "VT-GOLD-9999",
    defaultRate: 2,
  },
  {
    id: "surgery_50",
    type: "win",
    title: "ĐẶC BIỆT CHÚC MỪNG!",
    highlightText: "GIẢM 50% TIỂU PHẪU",
    subText: "Đặc quyền tiểu phẫu thẩm mỹ cùng Bác sĩ Văn Trường",
    badge: "ĐẶC QUYỀN VIP 2/9",
    visualType: "surgery_50",
    voucherCode: "VT-50-TIEUPHAU",
    defaultRate: 8,
  },
  {
    id: "meso_20",
    type: "win",
    title: "CHÚC MỪNG QUÝ KHÁCH!",
    highlightText: "TẶNG 20% MESO CĂNG BÓNG",
    subText: "Liệu trình trẻ hóa căng bóng da đa tầng chuyên sâu",
    badge: "QUÀ TẶNG SPA CAO CẤP",
    visualType: "meso_20",
    voucherCode: "VT-MESO-20",
    defaultRate: 25,
  },
  {
    id: "triet_long",
    type: "win",
    title: "CHÚC MỪNG QUÝ KHÁCH!",
    highlightText: "TẶNG 01 LẦN TRIỆT LÔNG NÁCH",
    subText: "Công nghệ Diode Laser chuẩn y khoa • Miễn phí 100%",
    badge: "TRI ÂN THÂN THIẾT",
    visualType: "triet_long",
    voucherCode: "VT-TRIETLONG-FREE",
    defaultRate: 30,
  },
  {
    id: "miss",
    type: "miss",
    title: "LỜI CHÚC TỪ BÁC SĨ",
    highlightText: "CHÚC QUÝ KHÁCH MÃI XINH ĐẸP & RẠNG NGỜI!",
    subText: "Cơ hội trúng Vàng 9999 và Ưu đãi Đặc Quyền vẫn đang chờ Quý khách ở lượt tiếp theo!",
    badge: "LỜI CHÚC YÊU THƯƠNG",
    visualType: "wish",
    defaultRate: 35,
  },
];

const MISS_MESSAGES: string[] = [
  "CHÚC QUÝ KHÁCH MAY MẮN Ở LƯỢT TIẾP THEO!",
  "HẸN QUÝ KHÁCH MỘT CƠ HỘI MAY MẮN KHÁC NHÉ!",
  "MAY MẮN ĐANG CHỜ QUÝ KHÁCH Ở LẦN SAU!",
  "CHÚC QUÝ KHÁCH THẬT NHIỀU MAY MẮN TRONG NHỮNG LẦN TIẾP THEO!",
  "MỘT CHÚT TIẾC NUỐI, NHƯNG CƠ HỘI TRÚNG VÀNG VẪN CÒN PHÍA TRƯỚC!",
  "MỘT CHÚT TIẾC NUỐI – MỘT CHÚT CHỜ MONG – HẸN QUÝ KHÁCH Ở LẦN SAU!",
  "MAY MẮN CHƯA GỌI TÊN LẦN NÀY, THỬ LẠI NGAY NHÉ QUÝ KHÁCH!",
  "HÔM NAY CHƯA TRÚNG, BIẾT ĐÂU LẦN SAU TRÚNG VÀNG LỚN NHA QUÝ KHÁCH ƠI!",
];

// ASSETS
const ASSET_CLOSED_BOX = "/images/gift_box_3d_transparent.png";
const ASSET_OPENED_BOX = "/images/gift_box_3d_opened.png";
const ASSET_BANNER_2_9 = "/images/banner_2_9.jpg";

// ==========================================
// 2. AUDIO SYNTHESIZER
// ==========================================
class SoundFX {
  private ctx: AudioContext | null = null;

  private init() {
    if (!this.ctx && typeof window !== "undefined") {
      const AudioCtx =
        window.AudioContext ||
        (window as unknown as { webkitAudioContext: typeof AudioContext }).webkitAudioContext;
      this.ctx = new AudioCtx();
    }
  }

  playShake() {
    this.init();
    if (!this.ctx) return;
    try {
      const osc = this.ctx.createOscillator();
      const gain = this.ctx.createGain();
      osc.type = "sawtooth";
      osc.frequency.setValueAtTime(80, this.ctx.currentTime);
      osc.frequency.exponentialRampToValueAtTime(160, this.ctx.currentTime + 0.15);
      gain.gain.setValueAtTime(0.12, this.ctx.currentTime);
      gain.gain.exponentialRampToValueAtTime(0.01, this.ctx.currentTime + 0.15);
      osc.connect(gain);
      gain.connect(this.ctx.destination);
      osc.start();
      osc.stop(this.ctx.currentTime + 0.15);
    } catch {
      // ignore
    }
  }

  playBurst() {
    this.init();
    if (!this.ctx) return;
    try {
      const osc = this.ctx.createOscillator();
      const gain = this.ctx.createGain();
      osc.type = "sine";
      osc.frequency.setValueAtTime(320, this.ctx.currentTime);
      osc.frequency.exponentialRampToValueAtTime(70, this.ctx.currentTime + 0.4);
      gain.gain.setValueAtTime(0.4, this.ctx.currentTime);
      gain.gain.exponentialRampToValueAtTime(0.001, this.ctx.currentTime + 0.4);
      osc.connect(gain);
      gain.connect(this.ctx.destination);
      osc.start();
      osc.stop(this.ctx.currentTime + 0.4);
    } catch {
      // ignore
    }
  }

  playTick() {
    this.init();
    if (!this.ctx) return;
    try {
      const osc = this.ctx.createOscillator();
      const gain = this.ctx.createGain();
      osc.type = "sine";
      osc.frequency.setValueAtTime(750, this.ctx.currentTime);
      osc.frequency.exponentialRampToValueAtTime(1300, this.ctx.currentTime + 0.04);
      gain.gain.setValueAtTime(0.09, this.ctx.currentTime);
      gain.gain.exponentialRampToValueAtTime(0.001, this.ctx.currentTime + 0.04);
      osc.connect(gain);
      gain.connect(this.ctx.destination);
      osc.start();
      osc.stop(this.ctx.currentTime + 0.04);
    } catch {
      // ignore
    }
  }

  playFanfare() {
    this.init();
    if (!this.ctx) return;
    try {
      const freqs = [523.25, 659.25, 783.99, 1046.5, 1318.5, 1567.98];
      freqs.forEach((f, i) => {
        if (!this.ctx) return;
        const osc = this.ctx.createOscillator();
        const gain = this.ctx.createGain();
        osc.type = "triangle";
        const st = this.ctx.currentTime + i * 0.08;
        osc.frequency.setValueAtTime(f, st);
        gain.gain.setValueAtTime(0, st);
        gain.gain.setValueAtTime(0.3, st);
        gain.gain.exponentialRampToValueAtTime(0.001, st + 0.7);
        osc.connect(gain);
        gain.connect(this.ctx.destination);
        osc.start(st);
        osc.stop(st + 0.7);
      });
    } catch {
      // ignore
    }
  }
}

const sounds = new SoundFX();

// ==========================================
// 3. HIỆU ỨNG PHÁO HOA SIÊU TỐI ƯU (GPU-ACCELERATED CHO MÀN HÌNH LED 4K KHÔNG LAG)
// ==========================================
interface BurstParticle {
  x: number;
  y: number;
  type: "ribbon" | "gold_coin" | "star";
  color: string;
  size: number;
  speedX: number;
  speedY: number;
  rotation: number;
  rotationSpeed: number;
  opacity: number;
}

let activeParticles: BurstParticle[] = [];
let animFrameId: number | null = null;

function triggerMassiveFireworks(originX: number, originY: number) {
  if (typeof window === "undefined") return;
  const canvas = document.getElementById("prize-burst-canvas") as HTMLCanvasElement | null;
  if (!canvas) return;
  const ctx = canvas.getContext("2d", { alpha: true });
  if (!ctx) return;

  canvas.width = window.innerWidth;
  canvas.height = window.innerHeight;

  const goldColors = [
    "#FFFDD0", "#FBF5B7", "#FFD700", "#FFA500", "#FF4500", 
    "#FF1493", "#00FFFF", "#FFD700", "#00FF66", "#FFFFFF"
  ];

  // Giới hạn số lượng hạt vừa đủ đẹp để màn LED 4K chạy 60 FPS cực mượt
  for (let i = 0; i < 40; i++) {
    const angle = Math.random() * Math.PI * 2;
    const speed = Math.random() * 18 + 7;
    activeParticles.push({
      x: originX,
      y: originY,
      type: "ribbon",
      color: goldColors[i % goldColors.length],
      size: Math.random() * 12 + 6,
      speedX: Math.cos(angle) * speed,
      speedY: Math.sin(angle) * speed - 5,
      rotation: Math.random() * 360,
      rotationSpeed: (Math.random() - 0.5) * 16,
      opacity: 1,
    });
  }

  for (let i = 0; i < 110; i++) {
    const angle = Math.random() * Math.PI * 2;
    const speed = Math.random() * 20 + 5;
    activeParticles.push({
      x: originX,
      y: originY,
      type: i % 3 === 0 ? "gold_coin" : "star",
      color: goldColors[Math.floor(Math.random() * goldColors.length)],
      size: Math.random() * 10 + 4,
      speedX: Math.cos(angle) * speed,
      speedY: Math.sin(angle) * speed - 6,
      rotation: Math.random() * 360,
      rotationSpeed: (Math.random() - 0.5) * 14,
      opacity: 1,
    });
  }

  if (!animFrameId) {
    function render() {
      if (!ctx || !canvas) return;
      ctx.clearRect(0, 0, canvas.width, canvas.height);
      let alive = false;

      for (let i = 0; i < activeParticles.length; i++) {
        const p = activeParticles[i];
        p.x += p.speedX;
        p.y += p.speedY;
        p.speedY += 0.28;
        p.speedX *= 0.98;
        p.rotation += p.rotationSpeed;
        p.opacity -= 0.008;

        if (p.opacity > 0) {
          alive = true;
          ctx.save();
          ctx.translate(p.x, p.y);
          ctx.rotate((p.rotation * Math.PI) / 180);
          ctx.globalAlpha = p.opacity;

          if (p.type === "gold_coin") {
            ctx.fillStyle = "#FFD700";
            ctx.beginPath();
            ctx.arc(0, 0, p.size, 0, Math.PI * 2);
            ctx.fill();
            ctx.strokeStyle = "#FFFDD0";
            ctx.lineWidth = 1.5;
            ctx.stroke();
          } else if (p.type === "ribbon") {
            ctx.fillStyle = p.color;
            ctx.beginPath();
            ctx.ellipse(0, 0, p.size, p.size * 2.5, Math.PI / 4, 0, Math.PI * 2);
            ctx.fill();
          } else {
            ctx.fillStyle = p.color;
            ctx.fillRect(-p.size / 2, -p.size / 2, p.size, p.size);
          }
          ctx.restore();
        }
      }

      activeParticles = activeParticles.filter((p) => p.opacity > 0);

      if (alive) {
        animFrameId = requestAnimationFrame(render);
      } else {
        animFrameId = null;
      }
    }

    render();
  }
}

// ==========================================
// 4. PHẦN QUÀ 3D TO RÕ, ĐẲNG CẤP ĐỂ CHỤP ẢNH LƯU NIỆM (CHECK-IN)
// ==========================================
function StandaloneLuxuryPrize3D({ item }: { item: PrizeData }) {
  switch (item.visualType) {
    case "gold_bar":
      return (
        <div className="relative flex flex-col items-center justify-center my-3 sm:my-5 scale-110 sm:scale-125">
          <div className="absolute inset-0 w-80 h-80 rounded-full bg-[radial-gradient(circle,rgba(255,215,0,0.55)_0%,transparent_70%)] blur-2xl animate-spin-slow pointer-events-none" />
          <div
            className="relative z-10 w-72 sm:w-80 h-36 sm:h-40 rounded-3xl border-3 border-yellow-100 shadow-[0_0_60px_rgba(255,215,0,0.95),_inset_0_2px_12px_rgba(255,255,255,0.9)] flex flex-col items-center justify-center transform hover:scale-105 transition-transform"
            style={{
              background:
                "linear-gradient(135deg, #a67c1e 0%, #fffdd0 25%, #ca9e3a 50%, #fbf5b7 75%, #916814 100%)",
            }}
          >
            <div className="text-xs sm:text-sm font-sans font-black text-rose-950 tracking-widest uppercase mb-1">
              ✦ VIỆN THẨM MỸ VĂN TRƯỜNG ✦
            </div>
            <div className="text-3xl sm:text-4xl font-black font-sans text-[#2a0408] tracking-wider drop-shadow-[0_2px_6px_rgba(255,255,255,0.8)]">
              VÀNG 9999
            </div>
            <div className="text-[11px] sm:text-xs font-mono font-black text-amber-950 tracking-widest mt-1">
              9999 FINE GOLD • PHÁT TÀI ĐẠI LỄ 2/9
            </div>
            <div className="absolute inset-0 bg-gradient-to-r from-transparent via-white/40 to-transparent transform -skew-x-12 animate-shimmer pointer-events-none rounded-3xl" />
          </div>
          <div className="absolute -left-6 top-8 w-13 h-13 rounded-full bg-gradient-to-br from-yellow-200 to-amber-600 border-2 border-white shadow-2xl flex items-center justify-center font-black text-xs text-rose-950 animate-bounce">
            9999
          </div>
          <div className="absolute -right-6 top-8 w-13 h-13 rounded-full bg-gradient-to-br from-yellow-200 to-amber-600 border-2 border-white shadow-2xl flex items-center justify-center font-black text-xs text-rose-950 animate-bounce delay-150">
            VT
          </div>
        </div>
      );

    case "surgery_50":
      return (
        <div className="relative flex flex-col items-center justify-center my-3 sm:my-5 scale-110 sm:scale-125">
          <div className="absolute inset-0 w-80 h-80 rounded-full bg-[radial-gradient(circle,rgba(244,63,94,0.55)_0%,transparent_70%)] blur-2xl animate-spin-slow pointer-events-none" />
          <div
            className="relative z-10 w-76 sm:w-84 h-40 sm:h-44 rounded-3xl border-3 border-[#FCF6BA] shadow-[0_0_60px_rgba(244,63,94,0.9),_inset_0_2px_12px_rgba(255,255,255,0.5)] flex flex-col items-center justify-between p-4 transform hover:scale-105 transition-transform"
            style={{
              background:
                "linear-gradient(135deg, #590412 0%, #a80d24 50%, #2e0208 100%)",
            }}
          >
            <div className="w-full flex items-center justify-between">
              <span className="text-[11px] font-sans font-bold text-[#FCF6BA] tracking-[0.15em] uppercase">
                VIP PRIVILEGE PASS
              </span>
              <span className="text-xs font-mono font-bold text-rose-200">DR. VĂN TRƯỜNG</span>
            </div>
            <div className="text-center my-auto">
              <div className="text-4xl sm:text-5xl font-black font-sans text-yellow-300 tracking-tight drop-shadow-[0_4px_16px_rgba(0,0,0,0.9)]">
                GIẢM 50%
              </div>
              <div className="text-xs sm:text-sm font-sans font-black text-white tracking-wider uppercase mt-1">
                TIỂU PHẪU THẨM MỸ
              </div>
            </div>
            <div className="w-full flex items-center justify-between text-[10px] text-amber-200 border-t border-[#FCF6BA]/40 pt-1 font-sans">
              <span>HẠN DÙNG: ĐẠI LỄ 2/9</span>
              <span className="font-mono font-bold">MÃ: VT-50-VIP</span>
            </div>
          </div>
        </div>
      );

    case "meso_20":
      return (
        <div className="relative flex flex-col items-center justify-center my-3 sm:my-5 scale-110 sm:scale-125">
          <div className="absolute inset-0 w-80 h-80 rounded-full bg-[radial-gradient(circle,rgba(45,212,191,0.55)_0%,transparent_70%)] blur-2xl animate-spin-slow pointer-events-none" />
          <div
            className="relative z-10 w-76 sm:w-84 h-40 sm:h-44 rounded-3xl border-3 border-white shadow-[0_0_60px_rgba(45,212,191,0.9)] flex flex-col items-center justify-center p-4 text-white"
            style={{
              background:
                "linear-gradient(135deg, #042f2e 0%, #0d9488 50%, #115e59 100%)",
            }}
          >
            <div className="text-3xl mb-1">💧 ✨</div>
            <div className="text-3xl sm:text-4xl font-black font-sans tracking-tight text-cyan-200 drop-shadow">
              TẶNG 20%
            </div>
            <div className="text-xs sm:text-sm font-sans font-black tracking-wider uppercase text-white mt-1">
              MESO CĂNG BÓNG DA
            </div>
            <div className="text-[10px] font-sans text-cyan-200 opacity-90 mt-0.5">
              Trẻ hóa đa tầng chuẩn Y khoa
            </div>
          </div>
        </div>
      );

    case "triet_long":
      return (
        <div className="relative flex flex-col items-center justify-center my-3 sm:my-5 scale-110 sm:scale-125">
          <div className="absolute inset-0 w-80 h-80 rounded-full bg-[radial-gradient(circle,rgba(217,70,239,0.55)_0%,transparent_70%)] blur-2xl animate-spin-slow pointer-events-none" />
          <div
            className="relative z-10 w-76 sm:w-84 h-40 sm:h-44 rounded-3xl border-3 border-white shadow-[0_0_60px_rgba(217,70,239,0.9)] flex flex-col items-center justify-center p-4 text-white"
            style={{
              background:
                "linear-gradient(135deg, #4a044e 0%, #a21caf 50%, #701a75 100%)",
            }}
          >
            <div className="text-[11px] font-bold font-sans uppercase tracking-[0.15em] text-pink-200 bg-black/50 px-4 py-1 rounded-full mb-1">
              MIỄN PHÍ 100%
            </div>
            <div className="text-2xl sm:text-3xl font-black font-sans tracking-tight text-white drop-shadow">
              🌸 01 LẦN TRIỆT LÔNG
            </div>
            <div className="text-xs font-sans font-bold tracking-wider uppercase text-pink-200 mt-1">
              DIODE LASER CHUẨN Y KHOA
            </div>
          </div>
        </div>
      );

    case "wish":
    default:
      return (
        <div className="relative flex flex-col items-center justify-center my-3 sm:my-5 scale-105 sm:scale-115">
          <div className="absolute inset-0 w-80 h-80 rounded-full bg-[radial-gradient(circle,rgba(244,63,94,0.45)_0%,transparent_70%)] blur-2xl pointer-events-none" />
          <div
            className="relative z-10 w-76 sm:w-84 h-36 sm:h-40 rounded-3xl border-3 border-[#FCF6BA] shadow-[0_0_50px_rgba(244,63,94,0.8)] flex flex-col items-center justify-center p-4 text-white text-center"
            style={{
              background:
                "linear-gradient(135deg, #850718 0%, #c9182b 50%, #4a0208 100%)",
            }}
          >
            <div className="text-3xl mb-1 animate-pulse">💌 💖</div>
            <div className="text-xl sm:text-2xl font-black font-sans tracking-wide text-[#FCF6BA]">
              CHÚC QUÝ KHÁCH MÃI XINH ĐẸP
            </div>
            <div className="text-xs sm:text-sm font-sans text-rose-100 mt-1 font-medium">
              Cơ hội trúng Vàng 9999 vẫn đang chờ Quý khách ở lượt tiếp theo!
            </div>
          </div>
        </div>
      );
  }
}

// ==========================================
// 5. NỘI DUNG CHÍNH (ĐƯỢC BỌC SUSPENSE ĐỂ ĐỌC QUERY URL)
// ==========================================
function LuckyGiftBoxContent() {
  const searchParams = useSearchParams();
  const isAdminParam = searchParams.get("admin") === "1" || searchParams.get("mode") === "admin";

  const [boxes, setBoxes] = useState<Array<{ id: number; opened: boolean; shaking: boolean }>>([
    { id: 1, opened: false, shaking: false },
    { id: 2, opened: false, shaking: false },
    { id: 3, opened: false, shaking: false },
    { id: 4, opened: false, shaking: false },
    { id: 5, opened: false, shaking: false },
    { id: 6, opened: false, shaking: false },
    { id: 7, opened: false, shaking: false },
    { id: 8, opened: false, shaking: false },
  ]);

  const [isOpening, setIsOpening] = useState(false);
  const [showPrizeModal, setShowPrizeModal] = useState(false);
  const [isShuffling, setIsShuffling] = useState(false);
  const [displayedItem, setDisplayedItem] = useState<PrizeData>(ALL_PROGRAM_ITEMS[0]);
  const [finalPrize, setFinalPrize] = useState<PrizeData | null>(null);
  const [attemptsLeft, setAttemptsLeft] = useState(3);
  const [forcedPrizeId, setForcedPrizeId] = useState<string>("prob_engine");

  const [logoClickCount, setLogoClickCount] = useState(0);
  const [isAdminUnlocked, setIsAdminUnlocked] = useState(false);
  const [showAdminRatesModal, setShowAdminRatesModal] = useState(false);

  const [customRates, setCustomRates] = useState<{ [key: string]: number }>({
    gold_1: 2,
    surgery_50: 8,
    meso_20: 25,
    triet_long: 30,
    miss: 35,
  });

  useEffect(() => {
    try {
      const saved = localStorage.getItem("VT_GIFT_BOX_RATES");
      if (saved) {
        setCustomRates(JSON.parse(saved));
      }
    } catch {
      // ignore
    }
  }, []);

  const isCurrentAdmin = isAdminParam || isAdminUnlocked;
  const totalRatePercent = Object.values(customRates).reduce((a, b) => a + b, 0);

  const applyPreset = (presetName: "tiet_kiem" | "tri_an_lon" | "all_win") => {
    if (presetName === "tiet_kiem") {
      setCustomRates({ gold_1: 1, surgery_50: 3, meso_20: 15, triet_long: 25, miss: 56 });
    } else if (presetName === "tri_an_lon") {
      setCustomRates({ gold_1: 5, surgery_50: 15, meso_20: 35, triet_long: 35, miss: 10 });
    } else if (presetName === "all_win") {
      setCustomRates({ gold_1: 10, surgery_50: 20, meso_20: 35, triet_long: 35, miss: 0 });
    }
  };

  const drawPrizeByProbability = (): PrizeData => {
    const rand = Math.random() * totalRatePercent;
    let cumulative = 0;

    for (const item of ALL_PROGRAM_ITEMS) {
      const rate = customRates[item.id] || 0;
      cumulative += rate;
      if (rand <= cumulative) {
        if (item.type === "miss") {
          const randomMsg = MISS_MESSAGES[Math.floor(Math.random() * MISS_MESSAGES.length)];
          return { ...item, highlightText: randomMsg };
        }
        return item;
      }
    }
    return ALL_PROGRAM_ITEMS[0];
  };

  const [tickerText, setTickerText] = useState("🎉 Khách hàng Mai Phương (098***345) vừa mở trúng VÀNG 9999!");

  useEffect(() => {
    const winners = [
      "🎉 Khách hàng Mai Phương (098***345) vừa mở trúng VÀNG 9999!",
      "💎 Khách hàng Thùy Linh (090***123) vừa nhận ĐẶC BIỆT GIẢM 50% TIỂU PHẪU!",
      "✨ Khách hàng Thanh Mai (097***890) vừa rinh VÀNG 9999!",
      "🌸 Khách hàng Lan Hương (091***678) vừa nhận 01 LẦN TRIỆT LÔNG NÁCH!",
      "💧 Khách hàng Thu Thảo (093***456) vừa nhận 20% MESO CĂNG BÓNG!",
    ];
    let i = 0;
    const timer = setInterval(() => {
      i = (i + 1) % winners.length;
      setTickerText(winners[i]);
    }, 3800);
    return () => clearInterval(timer);
  }, []);

  const handleLogoClick = () => {
    const nextCount = logoClickCount + 1;
    setLogoClickCount(nextCount);
    if (nextCount >= 5) {
      setIsAdminUnlocked(true);
      setShowAdminRatesModal(true);
      setLogoClickCount(0);
    }
  };

  const handleBoxClick = (boxId: number, e: React.MouseEvent<HTMLDivElement>) => {
    if (isOpening || showPrizeModal) return;
    const box = boxes.find((b) => b.id === boxId);
    if (box?.opened) return;

    if (attemptsLeft <= 0) {
      alert("Quý khách đã hoàn thành lượt mở quà! Bấm 'Bắt đầu cho Khách tiếp theo' để quay lượt mới nhé!");
      return;
    }

    setIsOpening(true);

    const rect = e.currentTarget.getBoundingClientRect();
    const boxCenterX = rect.left + rect.width / 2;
    const boxCenterY = rect.top + rect.height / 2;

    sounds.playShake();
    setBoxes((prev) => prev.map((b) => (b.id === boxId ? { ...b, shaking: true } : b)));

    setTimeout(() => {
      sounds.playBurst();
      triggerMassiveFireworks(boxCenterX, boxCenterY);

      setBoxes((prev) =>
        prev.map((b) => (b.id === boxId ? { ...b, shaking: false, opened: true } : b))
      );
      setAttemptsLeft((prev) => Math.max(0, prev - 1));

      let targetPrize: PrizeData;
      if (forcedPrizeId === "prob_engine") {
        targetPrize = drawPrizeByProbability();
      } else if (forcedPrizeId === "miss") {
        const missItem = ALL_PROGRAM_ITEMS.find((p) => p.id === "miss")!;
        const randomMsg = MISS_MESSAGES[Math.floor(Math.random() * MISS_MESSAGES.length)];
        targetPrize = { ...missItem, highlightText: randomMsg };
      } else {
        targetPrize = ALL_PROGRAM_ITEMS.find((p) => p.id === forcedPrizeId) || ALL_PROGRAM_ITEMS[0];
      }

      setFinalPrize(targetPrize);
      setShowPrizeModal(true);
      setIsShuffling(true);

      let shuffleCount = 0;
      const totalShuffles = 18;
      let currentDelay = 65;

      const runShuffle = () => {
        const randomIndex = Math.floor(Math.random() * ALL_PROGRAM_ITEMS.length);
        setDisplayedItem(ALL_PROGRAM_ITEMS[randomIndex]);
        sounds.playTick();
        shuffleCount++;

        if (shuffleCount < totalShuffles) {
          if (shuffleCount > 10) currentDelay += 35;
          if (shuffleCount > 14) currentDelay += 70;
          setTimeout(runShuffle, currentDelay);
        } else {
          setDisplayedItem(targetPrize);
          setIsShuffling(false);
          setIsOpening(false);

          setTimeout(() => {
            sounds.playFanfare();
            triggerMassiveFireworks(window.innerWidth / 2, window.innerHeight * 0.4);
          }, 150);
        }
      };

      runShuffle();
    }, 550);
  };

  const handleReset = () => {
    setBoxes(boxes.map((b) => ({ ...b, opened: false, shaking: false })));
    setShowPrizeModal(false);
    setIsShuffling(false);
    setFinalPrize(null);
    setAttemptsLeft(3);
  };

  return (
    <div
      className="min-h-screen w-screen overflow-x-hidden text-[#fbf5b7] flex flex-col items-center justify-between select-none p-3 sm:p-6 pb-6 relative"
      style={{
        background: "radial-gradient(ellipse at 50% 20%, #9e0c1f 0%, #680512 40%, #380108 80%, #1a0003 100%)",
        fontFamily: "'Montserrat', sans-serif",
      }}
    >
      {/* CANVAS PHÁO HOA Z-INDEX 70 */}
      <canvas id="prize-burst-canvas" className="fixed inset-0 pointer-events-none z-[70] w-full h-full will-change-transform" />

      {/* LỚP NỀN POSTER ĐẠI LỄ 2-9 CHÍNH THỨC PHỦ MỜ */}
      <div 
        className="absolute inset-0 bg-cover bg-center opacity-25 mix-blend-overlay pointer-events-none"
        style={{ backgroundImage: `url(${ASSET_BANNER_2_9})` }}
      />

      {/* VẠT LỤA ĐỎ & HÀO QUANG HOÀNG GIA ĐẠI LỄ */}
      <div className="absolute inset-0 bg-[radial-gradient(circle_at_50%_15%,rgba(239,68,68,0.45)_0%,transparent_65%)] pointer-events-none" />
      <div className="absolute top-0 inset-x-0 h-96 bg-[radial-gradient(ellipse_at_top,rgba(251,191,36,0.3)_0%,transparent_75%)] pointer-events-none" />

      {/* HEADER & BRAND */}
      <header className="w-full max-w-5xl pt-2 sm:pt-4 pb-2 text-center relative z-10 flex flex-col items-center">
        {/* Ticker Glassmorphic Tone Đỏ Lễ Hội */}
        <div className="bg-red-950/70 border border-amber-400/40 rounded-full px-4 sm:px-6 py-1 text-[11px] sm:text-xs text-[#fff3cf] mb-2 shadow-[0_4px_20px_rgba(185,28,28,0.5)] flex items-center gap-2 max-w-[92vw] truncate">
          <span className="inline-block w-2.5 h-2.5 rounded-full bg-[#ffd700] shadow-[0_0_8px_#ffd700] animate-ping shrink-0" />
          <span className="font-medium tracking-wide truncate">{tickerText}</span>
        </div>

        {/* Brand Header */}
        <div
          onClick={handleLogoClick}
          className="cursor-pointer gold-chrome-text text-xs sm:text-sm font-black tracking-widest uppercase mb-1 active:scale-95 transition-transform drop-shadow-[0_2px_8px_rgba(0,0,0,0.8)]"
          title="Chạm 5 lần để mở khóa Quản trị"
        >
          ✦ VIỆN THẨM MỸ BÁC SĨ VĂN TRƯỜNG • TIẾP HÀO KHÍ - NỐI VINH QUANG ✦
        </div>

        {/* Tiêu đề chính */}
        <h1 className="text-2xl sm:text-4xl md:text-5xl lg:text-6xl font-black uppercase gold-chrome-text drop-shadow-[0_4px_24px_rgba(255,215,0,0.7)] tracking-wide my-1">
          8 HỘP QUÀ VÀNG MAY MẮN
        </h1>

        {/* Tiêu đề phụ */}
        <p className="text-[#fff1cc] text-xs sm:text-sm font-bold tracking-wide max-w-xl mx-auto drop-shadow-[0_1px_4px_rgba(0,0,0,0.9)] px-2">
          Chạm vào hộp quà để bung nắp &amp; nhận ngay Vàng 9999 hoặc Đặc Quyền Giảm 50%
        </p>

        {/* Thẻ Lượt mở của Quý Khách */}
        <div className="mt-2.5 flex items-center gap-3 bg-red-950/80 border-2 border-amber-400/50 rounded-full px-5 sm:px-6 py-1 shadow-[0_4px_24px_rgba(225,29,72,0.6)]">
          <span className="text-xs sm:text-sm text-[#ffe4b5] tracking-wider font-bold">LƯỢT MỞ CỦA QUÝ KHÁCH:</span>
          <span className="text-sm sm:text-base font-mono font-black gold-chrome-text px-3 py-0.5 border border-amber-300/60 rounded-full bg-amber-500/25">
            0{attemptsLeft} LƯỢT
          </span>
        </div>
      </header>

      {/* GAME STAGE: DẠNG DỌC ĐẸP TRÊN ĐIỆN THOẠI (2 CỘT) & 4 CỘT TRÊN MÁY TÍNH/MÀN CHIẾU */}
      <main className="w-full max-w-5xl my-auto relative z-10 py-4">
        <div className="grid grid-cols-2 sm:grid-cols-4 gap-4 sm:gap-6 md:gap-8 justify-items-center">
          {boxes.map((box, index) => (
            <div
              key={box.id}
              onClick={(e) => handleBoxClick(box.id, e)}
              className={`luxury-box-card group relative w-36 h-44 sm:w-40 sm:h-48 md:w-48 md:h-52 cursor-pointer select-none transition-all duration-300 flex flex-col items-center justify-center ${
                box.opened
                  ? "cursor-default opacity-85"
                  : "hover:scale-105 active:scale-95"
              }`}
              style={{
                animationDelay: `${index * 0.15}s`,
              }}
            >
              <div
                className={`absolute -inset-2 rounded-3xl bg-[radial-gradient(circle,rgba(255,215,0,0.5)_0%,transparent_70%)] blur-lg sm:blur-xl transition-all duration-300 pointer-events-none ${
                  box.opened
                    ? "opacity-90 scale-105"
                    : "opacity-0 group-hover:opacity-100 group-hover:scale-110"
                }`}
              />

              <div
                className={`relative w-full h-full flex flex-col items-center justify-center p-1 transition-all duration-300 ${
                  box.shaking
                    ? "shaking-box"
                    : "floating-box group-hover:shaking-gentle"
                }`}
                style={{
                  filter: box.opened
                    ? "drop-shadow(0 0 25px rgba(255,215,0,0.8))"
                    : "drop-shadow(0 15px 25px rgba(0,0,0,0.8))",
                }}
              >
                {!box.opened ? (
                  <div className="relative w-full h-full flex flex-col items-center justify-center group-hover:drop-shadow-[0_0_24px_rgba(255,230,100,0.95)] transition-all duration-300">
                    <img
                      src={ASSET_CLOSED_BOX}
                      alt={`Hộp quà 3D số ${box.id}`}
                      className="w-full h-full object-contain pointer-events-none drop-shadow-2xl"
                    />

                    <div
                      className="absolute bottom-1 z-20 w-8 h-8 sm:w-9 sm:h-9 rounded-full flex items-center justify-center font-mono font-black text-xs sm:text-sm text-[#380208] shadow-[0_4px_16px_rgba(0,0,0,0.9)] border-2 border-[#fffdd0]"
                      style={{
                        background:
                          "linear-gradient(135deg, #fffdd0 0%, #ca9e3a 50%, #916814 100%)",
                      }}
                    >
                      0{box.id}
                    </div>
                  </div>
                ) : (
                  <div className="relative w-full h-full flex flex-col items-center justify-center">
                    <div className="absolute inset-0 bg-[#ffd700]/30 rounded-full blur-xl animate-pulse" />
                    <img
                      src={ASSET_OPENED_BOX}
                      alt="Hộp quà đã mở"
                      className="w-full h-full object-contain pointer-events-none drop-shadow-[0_0_30px_rgba(255,215,0,0.85)]"
                    />
                    <div className="absolute bottom-1 z-20 px-3 py-0.5 rounded-full bg-red-950/90 border border-amber-300 shadow-lg">
                      <span className="gold-chrome-text text-[10px] sm:text-xs tracking-wider font-bold uppercase whitespace-nowrap">
                        ĐÃ MỞ LỘC
                      </span>
                    </div>
                  </div>
                )}
              </div>
            </div>
          ))}
        </div>
      </main>

      {/* FOOTER & NÚT RESET CHO NGƯỜI TIẾP THEO */}
      <footer className="w-full max-w-5xl text-center relative z-10 pt-2 pb-2">
        {/* NÚT RESET CHO NGƯỜI TIẾP THEO NỔI BẬT HOÀNG GIA */}
        <div className="flex flex-wrap justify-center gap-3 mb-3">
          {attemptsLeft === 0 ? (
            <button
              onClick={handleReset}
              className="px-8 sm:px-10 py-3.5 rounded-full border-2 border-yellow-200 bg-gradient-to-r from-yellow-500 via-amber-300 to-yellow-500 text-[#240307] font-black text-sm sm:text-base uppercase tracking-wider shadow-[0_0_35px_rgba(255,215,0,0.8)] transition-all transform hover:scale-105 active:scale-95 animate-bounce flex items-center gap-2"
            >
              <span>🔄</span>
              <span>BẮT ĐẦU CHO KHÁCH HÀNG TIẾP THEO</span>
            </button>
          ) : (
            <button
              onClick={handleReset}
              className="px-6 sm:px-8 py-2.5 rounded-full bg-red-950/70 hover:bg-red-900/80 border-2 border-amber-400/40 hover:border-amber-300 text-[#fff1cc] font-bold text-xs sm:text-sm uppercase tracking-wider shadow-[0_4px_20px_rgba(185,28,28,0.5)] transition-all active:scale-95 flex items-center gap-2"
            >
              <span>🔄</span>
              <span>ĐẶT LẠI 8 HỘP QUÀ</span>
            </button>
          )}

          {isCurrentAdmin && (
            <button
              onClick={() => setShowAdminRatesModal(true)}
              className="px-5 sm:px-6 py-2.5 rounded-full bg-gradient-to-r from-amber-600/60 to-yellow-600/60 hover:from-amber-600/80 hover:to-yellow-600/80 border border-amber-300 text-[#fff1cc] font-black text-xs uppercase tracking-wider shadow-[0_4px_20px_rgba(202,158,58,0.5)] transition-all active:scale-95 flex items-center gap-2"
            >
              <span>⚙️</span>
              <span>CÀI ĐẶT XÁC SUẤT (%)</span>
            </button>
          )}
        </div>

        {isCurrentAdmin && (
          <div className="bg-black/60 border border-amber-500/40 rounded-2xl p-2.5 text-xs text-[#ffe4b5] flex flex-wrap items-center justify-center gap-2.5 max-w-xl mx-auto shadow-2xl mb-2">
            <span className="font-bold text-yellow-300">🎲 Ép kết quả (Admin):</span>
            <select
              value={forcedPrizeId}
              onChange={(e) => setForcedPrizeId(e.target.value)}
              className="bg-[#380208] border border-amber-400/60 text-[#fbf5b7] rounded-lg px-2.5 py-1 text-xs focus:outline-none focus:border-yellow-300 cursor-pointer font-bold"
            >
              <option value="prob_engine">🎯 Tự động tính theo Tỉ Lệ %</option>
              <option value="gold_1">👑 Ép trúng: VÀNG 9999</option>
              <option value="surgery_50">💎 Ép trúng: GIẢM 50% TIỂU PHẪU</option>
              <option value="meso_20">💧 Ép trúng: TẶNG 20% MESO</option>
              <option value="triet_long">🌸 Ép trúng: TẶNG 01 LẦN TRIỆT LÔNG</option>
              <option value="miss">💌 Ép trúng: Lời Chúc May Mắn</option>
            </select>
          </div>
        )}

        <div className="text-[10px] sm:text-xs text-[#ffd6bd] tracking-wider font-medium opacity-90 drop-shadow">
          © 2026 VIỆN THẨM MỸ BÁC SĨ VĂN TRƯỜNG • HOTLINE: 0905.123.456
        </div>
      </footer>

      {/* ========================================================================= */}
      {/* 6. MODAL BẢNG VÀNG HOÀNG GIA CHÚC MỪNG ĐẠI LỄ 2/9 - TO RÕ ĐỂ CHỤP ẢNH LƯU NIỆM */}
      {/* ========================================================================= */}
      {showPrizeModal && (
        <div className="fixed inset-0 z-50 flex items-center justify-center p-3 sm:p-5 bg-black/90 animate-fadeIn">
          <div className="relative w-full max-w-lg bg-gradient-to-b from-[#6b0716] via-[#380208] to-[#1a0003] rounded-3xl p-6 sm:p-8 border-3 border-amber-300 shadow-[0_0_100px_rgba(225,29,72,0.95)] text-center animate-scaleUp overflow-hidden">
            <div className="absolute -top-32 left-1/2 -translate-x-1/2 w-96 h-96 rounded-full bg-[radial-gradient(circle,rgba(255,215,0,0.45)_0%,transparent_70%)] pointer-events-none" />

            {/* HEADER CHÚC MỪNG ĐẠI LỄ 2-9 HOÀNH TRÁNG */}
            <div className="bg-gradient-to-r from-red-700/70 via-amber-500/50 to-red-700/70 border-2 border-amber-300/70 rounded-2xl p-3 mb-3 shadow-xl">
              <div className="text-sm sm:text-base font-black gold-chrome-text uppercase tracking-wider flex items-center justify-center gap-1.5 drop-shadow">
                <span>🇻🇳</span>
                <span>CHÚC MỪNG ĐẠI LỄ QUỐC KHÁNH 2-9</span>
                <span>🇻🇳</span>
              </div>
              <div className="text-xs sm:text-sm font-black text-white tracking-wide mt-1 drop-shadow">
                CHÚC QUÝ KHÁCH MÃI LUÔN XINH ĐẸP &amp; RẠNG NGỜI! 💖
              </div>
            </div>

            {/* Badge Trạng thái */}
            <div className="inline-block px-5 py-1.5 rounded-full border-2 border-amber-300 bg-gradient-to-r from-amber-500/40 via-yellow-300/50 to-amber-500/40 text-[#fff5cc] text-xs sm:text-sm font-black uppercase tracking-wider mb-2 shadow-lg">
              {isShuffling ? "🎲 ĐANG QUAY MAY MẮN..." : displayedItem.badge}
            </div>

            {/* PHẦN QUÀ 3D SIÊU TO RÕ NỔI BẬT ĐỂ CHỤP ẢNH LƯU NIỆM */}
            <div className="my-2">
              <StandaloneLuxuryPrize3D item={displayedItem} />
            </div>

            {/* Tên giải thưởng */}
            <h2 className="text-2xl sm:text-4xl font-black uppercase tracking-tight gold-chrome-text drop-shadow-[0_2px_16px_rgba(255,215,0,0.9)] my-2">
              {displayedItem.highlightText}
            </h2>

            {/* Lời chúc & mô tả */}
            <p className="text-xs sm:text-base text-[#ffe4d6] font-medium px-2 mb-4 leading-relaxed opacity-95">
              {displayedItem.subText}
            </p>

            {/* Voucher Code */}
            {!isShuffling && displayedItem.type === "win" && displayedItem.voucherCode && (
              <div className="bg-red-950/80 border-2 border-amber-400/80 rounded-2xl p-3 mb-4 relative shadow-inner">
                <div className="text-[11px] text-[#ffd6a5] uppercase tracking-wider mb-1 font-black">
                  MÃ QUÀ TẶNG CỦA QUÝ KHÁCH:
                </div>
                <div className="flex items-center justify-between bg-black/80 border-2 border-dashed border-amber-400 rounded-xl px-4 py-2">
                  <span className="font-mono font-black text-lg sm:text-xl text-yellow-300 tracking-widest">
                    {displayedItem.voucherCode}
                  </span>
                  <button
                    onClick={() => {
                      navigator.clipboard.writeText(displayedItem.voucherCode!);
                      alert(`Đã sao chép mã ${displayedItem.voucherCode}!`);
                    }}
                    className="px-3.5 py-1.5 rounded-lg border border-amber-300 bg-gradient-to-r from-yellow-400 to-amber-600 text-[#240307] font-black text-xs uppercase tracking-wider transition-all hover:brightness-110 active:scale-95 shadow-md"
                  >
                    Sao Chép
                  </button>
                </div>
              </div>
            )}

            {/* Nút hành động */}
            {!isShuffling && (
              <div className="flex flex-col gap-2.5 animate-fadeIn">
                {attemptsLeft > 0 ? (
                  <button
                    onClick={() => setShowPrizeModal(false)}
                    className="w-full py-3 sm:py-3.5 rounded-full border-2 border-yellow-200 bg-gradient-to-r from-yellow-500 via-amber-300 to-yellow-500 text-[#240307] font-black text-xs sm:text-sm uppercase tracking-wider shadow-[0_6px_25px_rgba(255,215,0,0.6)] transition-all hover:brightness-110 active:scale-98"
                  >
                    🎁 MỞ TIẾP HỘP QUÀ KHÁC (CÒN 0{attemptsLeft} LƯỢT)
                  </button>
                ) : (
                  <button
                    onClick={handleReset}
                    className="w-full py-3.5 rounded-full border-2 border-yellow-200 bg-gradient-to-r from-yellow-500 via-amber-300 to-yellow-500 text-[#240307] font-black text-sm sm:text-base uppercase tracking-wider shadow-[0_6px_30px_rgba(255,215,0,0.8)] transition-all hover:brightness-110 active:scale-98 animate-pulse"
                  >
                    🔄 BẮT ĐẦU CHO KHÁCH HÀNG TIẾP THEO
                  </button>
                )}

                {displayedItem.type === "win" && (
                  <button
                    onClick={() => {
                      alert(
                        `Đã lưu phần quà "${displayedItem.highlightText}"! Quý khách vui lòng chụp màn hình gửi qua Zalo Hotline 0905.123.456 để Bác sĩ trao quà nhé!`
                      );
                    }}
                    className="w-full py-2.5 rounded-full border border-amber-300/70 text-[#fff3cf] hover:bg-white/10 text-xs sm:text-sm font-bold uppercase tracking-wider transition-all"
                  >
                    📸 Chụp Ảnh Lưu Niệm &amp; Gửi Zalo
                  </button>
                )}
              </div>
            )}
          </div>
        </div>
      )}

      {/* MODAL ADMIN */}
      {showAdminRatesModal && (
        <div className="fixed inset-0 z-50 flex items-center justify-center p-4 bg-black/90 animate-fadeIn text-left">
          <div className="relative w-full max-w-lg bg-gradient-to-b from-[#380208] via-[#240105] to-[#0a0002] rounded-3xl p-6 border-2 border-amber-400 shadow-[0_0_80px_rgba(202,158,58,0.7)] animate-scaleUp overflow-hidden">
            <div className="flex items-center justify-between border-b border-amber-400/40 pb-3 mb-4">
              <div>
                <div className="gold-chrome-text font-bold text-base uppercase tracking-wider flex items-center gap-2">
                  <span>⚙️</span>
                  <span>CÀI ĐẶT TỈ LỆ TRÚNG THƯỞNG (%)</span>
                </div>
                <div className="text-xs text-rose-200 opacity-90 mt-0.5">
                  Cài đặt này sẽ được lưu ngầm và áp dụng tự động cho tất cả khách chơi
                </div>
              </div>
              <button
                onClick={() => setShowAdminRatesModal(false)}
                className="w-8 h-8 rounded-full bg-white/10 hover:bg-white/20 text-[#fbf5b7] font-bold flex items-center justify-center"
              >
                ✕
              </button>
            </div>

            {/* Presets */}
            <div className="mb-4">
              <div className="text-[11px] text-[#e6d0b5] font-semibold uppercase tracking-wider mb-2">
                ⚡ Chọn mẫu cấu hình nhanh:
              </div>
              <div className="grid grid-cols-3 gap-2">
                <button
                  onClick={() => applyPreset("tiet_kiem")}
                  className="py-1.5 px-2 bg-white/5 hover:bg-white/15 border border-white/10 rounded-xl text-[11px] text-[#fbf5b7] transition text-center"
                >
                  🔒 Tiết kiệm quà
                </button>
                <button
                  onClick={() => applyPreset("tri_an_lon")}
                  className="py-1.5 px-2 bg-amber-500/20 hover:bg-amber-500/30 border border-amber-500/50 rounded-xl text-[11px] text-amber-200 transition text-center font-bold"
                >
                  🎉 Tri ân Đại Lễ 2/9
                </button>
                <button
                  onClick={() => applyPreset("all_win")}
                  className="py-1.5 px-2 bg-emerald-500/20 hover:bg-emerald-500/30 border border-emerald-500/50 rounded-xl text-[11px] text-emerald-200 transition text-center font-bold"
                >
                  🎁 100% Đều Trúng
                </button>
              </div>
            </div>

            {/* Sliders */}
            <div className="flex flex-col gap-3.5 my-3 text-xs">
              <div className="bg-black/40 border border-amber-500/30 rounded-2xl p-3">
                <div className="flex justify-between items-center mb-1.5">
                  <span className="font-bold text-yellow-300">👑 VÀNG 9999</span>
                  <span className="font-mono font-bold text-sm text-yellow-300 bg-amber-500/20 px-2 py-0.5 rounded-lg border border-amber-500/40">
                    {customRates.gold_1}%
                  </span>
                </div>
                <input
                  type="range"
                  min="0"
                  max="100"
                  value={customRates.gold_1}
                  onChange={(e) => setCustomRates({ ...customRates, gold_1: Number(e.target.value) })}
                  className="w-full accent-yellow-400 cursor-pointer h-2 bg-gray-700 rounded-lg"
                />
              </div>

              <div className="bg-black/40 border border-rose-500/30 rounded-2xl p-3">
                <div className="flex justify-between items-center mb-1.5">
                  <span className="font-bold text-rose-300">💎 ĐẶC BIỆT GIẢM 50% TIỂU PHẪU</span>
                  <span className="font-mono font-bold text-sm text-rose-300 bg-rose-500/20 px-2 py-0.5 rounded-lg border border-rose-500/40">
                    {customRates.surgery_50}%
                  </span>
                </div>
                <input
                  type="range"
                  min="0"
                  max="100"
                  value={customRates.surgery_50}
                  onChange={(e) => setCustomRates({ ...customRates, surgery_50: Number(e.target.value) })}
                  className="w-full accent-rose-400 cursor-pointer h-2 bg-gray-700 rounded-lg"
                />
              </div>

              <div className="bg-black/40 border border-teal-500/30 rounded-2xl p-3">
                <div className="flex justify-between items-center mb-1.5">
                  <span className="font-bold text-teal-300">💧 TẶNG 20% MESO CĂNG BÓNG</span>
                  <span className="font-mono font-bold text-sm text-teal-300 bg-teal-500/20 px-2 py-0.5 rounded-lg border border-teal-500/40">
                    {customRates.meso_20}%
                  </span>
                </div>
                <input
                  type="range"
                  min="0"
                  max="100"
                  value={customRates.meso_20}
                  onChange={(e) => setCustomRates({ ...customRates, meso_20: Number(e.target.value) })}
                  className="w-full accent-teal-400 cursor-pointer h-2 bg-gray-700 rounded-lg"
                />
              </div>

              <div className="bg-black/40 border border-fuchsia-500/30 rounded-2xl p-3">
                <div className="flex justify-between items-center mb-1.5">
                  <span className="font-bold text-fuchsia-300">🌸 TẶNG 01 LẦN TRIỆT LÔNG NÁCH</span>
                  <span className="font-mono font-bold text-sm text-fuchsia-300 bg-fuchsia-500/20 px-2 py-0.5 rounded-lg border border-fuchsia-500/40">
                    {customRates.triet_long}%
                  </span>
                </div>
                <input
                  type="range"
                  min="0"
                  max="100"
                  value={customRates.triet_long}
                  onChange={(e) => setCustomRates({ ...customRates, triet_long: Number(e.target.value) })}
                  className="w-full accent-fuchsia-400 cursor-pointer h-2 bg-gray-700 rounded-lg"
                />
              </div>

              <div className="bg-black/40 border border-gray-500/30 rounded-2xl p-3">
                <div className="flex justify-between items-center mb-1.5">
                  <span className="font-bold text-gray-300">💌 LỜI CHÚC TỪ BÁC SĨ</span>
                  <span className="font-mono font-bold text-sm text-gray-300 bg-gray-500/20 px-2 py-0.5 rounded-lg border border-gray-500/40">
                    {customRates.miss}%
                  </span>
                </div>
                <input
                  type="range"
                  min="0"
                  max="100"
                  value={customRates.miss}
                  onChange={(e) => setCustomRates({ ...customRates, miss: Number(e.target.value) })}
                  className="w-full accent-gray-400 cursor-pointer h-2 bg-gray-700 rounded-lg"
                />
              </div>
            </div>

            <div className="flex items-center justify-between p-3 rounded-2xl bg-white/5 border border-white/10 my-3 text-xs">
              <span>Tổng tỷ lệ xác suất:</span>
              <span
                className={`font-mono font-bold text-sm px-3 py-1 rounded-lg ${
                  totalRatePercent === 100
                    ? "bg-emerald-500/20 text-emerald-300 border border-emerald-500/40"
                    : "bg-amber-500/20 text-amber-300 border border-amber-500/40"
                }`}
              >
                {totalRatePercent}% {totalRatePercent === 100 ? "✓ Chuẩn 100%" : "(Tự động chuẩn hóa)"}
              </span>
            </div>

            <button
              onClick={() => {
                try {
                  localStorage.setItem("VT_GIFT_BOX_RATES", JSON.stringify(customRates));
                } catch {
                  // ignore
                }
                setShowAdminRatesModal(false);
                alert("✓ Đã lưu cài đặt tỷ lệ xác suất vào hệ thống thành công!");
              }}
              className="w-full py-3 rounded-full border border-yellow-300 bg-gradient-to-r from-[#a67c1e] via-[#fbf5b7] to-[#ca9e3a] text-[#240307] font-bold text-xs uppercase tracking-wider shadow-lg hover:brightness-110 active:scale-98 mt-2"
            >
              ✓ LƯU CÀI ĐẶT VÀ ÁP DỤNG NGAY
            </button>
          </div>
        </div>
      )}

      {/* LUXURY CSS STYLES */}
      <style jsx global>{`
        @import url('https://fonts.googleapis.com/css2?family=Montserrat:wght@400;500;600;700;800;900&display=swap');

        .gold-chrome-text {
          background: linear-gradient(
            135deg,
            #a67c1e 0%,
            #fbf5b7 25%,
            #ca9e3a 50%,
            #fffdd0 75%,
            #916814 100%
          );
          -webkit-background-clip: text;
          -webkit-text-fill-color: transparent;
        }

        @keyframes floating {
          0%,
          100% {
            transform: translateY(0px);
          }
          50% {
            transform: translateY(-8px);
          }
        }

        .floating-box {
          animation: floating 3.2s ease-in-out infinite;
        }

        @keyframes gentleShake {
          0% {
            transform: translate(0, 0) rotate(0deg);
          }
          25% {
            transform: translate(-1.5px, 0.5px) rotate(-1.5deg);
          }
          50% {
            transform: translate(1.5px, -0.5px) rotate(1.5deg);
          }
          75% {
            transform: translate(-1.5px, -0.5px) rotate(-1deg);
          }
          100% {
            transform: translate(0, 0) rotate(0deg);
          }
        }

        .shaking-gentle {
          animation: gentleShake 0.35s ease-in-out infinite !important;
        }

        @keyframes luxuryShake {
          0% {
            transform: translate(0, 0) rotate(0deg) scale(1.03);
          }
          20% {
            transform: translate(-3px, -2px) rotate(-3deg) scale(1.05);
          }
          40% {
            transform: translate(3px, 1px) rotate(3deg) scale(1.05);
          }
          60% {
            transform: translate(-3px, 2px) rotate(-2deg) scale(1.05);
          }
          80% {
            transform: translate(3px, -1px) rotate(2deg) scale(1.05);
          }
          100% {
            transform: translate(0, 0) rotate(0deg) scale(1.05);
          }
        }

        .shaking-box {
          animation: luxuryShake 0.12s infinite alternate !important;
        }

        @keyframes spinSlow {
          from {
            transform: rotate(0deg);
          }
          to {
            transform: rotate(360deg);
          }
        }

        .animate-spin-slow {
          animation: spinSlow 12s linear infinite;
        }

        @keyframes shimmer {
          0% {
            transform: translateX(-100%) skewX(-12deg);
          }
          100% {
            transform: translateX(200%) skewX(-12deg);
          }
        }

        .animate-shimmer {
          animation: shimmer 2.5s infinite;
        }

        @keyframes fadeIn {
          from {
            opacity: 0;
          }
          to {
            opacity: 1;
          }
        }

        @keyframes scaleUp {
          from {
            opacity: 0;
            transform: scale(0.85);
          }
          to {
            opacity: 1;
            transform: scale(1);
          }
        }

        .animate-fadeIn {
          animation: fadeIn 0.25s ease-out forwards;
        }

        .animate-scaleUp {
          animation: scaleUp 0.35s cubic-bezier(0.16, 1, 0.3, 1) forwards;
        }
      `}</style>
    </div>
  );
}

export default function Lucky8GiftBoxApp() {
  return (
    <Suspense fallback={<div className="min-h-screen bg-[#680512]" />}>
      <LuckyGiftBoxContent />
    </Suspense>
  );
}
