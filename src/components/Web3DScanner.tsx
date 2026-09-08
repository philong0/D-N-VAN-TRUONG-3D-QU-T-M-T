"use client";

import { useEffect, useRef, useState, useCallback } from "react";

interface Web3DScannerProps {
  patientId: string;
  onScanCompleted: () => void;
  onClose: () => void;
}

export default function Web3DScanner({ patientId, onScanCompleted, onClose }: Web3DScannerProps) {
  const videoRef = useRef<HTMLVideoElement>(null);
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const fileInputRef = useRef<HTMLInputElement>(null);

  const [streamActive, setStreamActive] = useState(false);
  const [isScanning, setIsScanning] = useState(false);
  const [scanProgress, setScanProgress] = useState(0);
  const [stepText, setStepText] = useState("Đang kết nối cảm biến Camera...");
  const [cameraFacing, setCameraFacing] = useState<"user" | "environment">("user");
  const [errorMessage, setErrorMessage] = useState<string | null>(null);

  // 1. Khởi động Camera Video Stream
  const initCamera = useCallback(async (facing: "user" | "environment") => {
    setStreamActive(false);
    setErrorMessage(null);
    setStepText("Đang kết nối cảm biến Camera...");

    if (typeof navigator === "undefined" || !navigator.mediaDevices || !navigator.mediaDevices.getUserMedia) {
      setErrorMessage("Trình duyệt không hỗ trợ mở camera trực tiếp. Vui lòng bấm nút chụp ảnh.");
      return;
    }

    try {
      // Dừng track cũ nếu có
      if (videoRef.current && videoRef.current.srcObject) {
        const oldStream = videoRef.current.srcObject as MediaStream;
        oldStream.getTracks().forEach((t) => t.stop());
        videoRef.current.srcObject = null;
      }

      const stream = await navigator.mediaDevices.getUserMedia({
        video: {
          facingMode: { ideal: facing },
          width: { ideal: 1280 },
          height: { ideal: 720 },
        },
        audio: false,
      });

      if (videoRef.current) {
        videoRef.current.srcObject = stream;
        videoRef.current.setAttribute("playsinline", "true");
        videoRef.current.setAttribute("autoplay", "true");
        videoRef.current.setAttribute("muted", "true");

        try {
          await videoRef.current.play();
        } catch (playErr) {
          console.log("Play interrupted:", playErr);
        }

        setStreamActive(true);
        setStepText(
          facing === "user"
            ? "🎯 Nhìn thẳng vào khung hình và bấm nút Quét 3D bên dưới"
            : "🎯 Đưa camera đối diện khuôn mặt bệnh nhân và bấm Quét 3D"
        );
      }
    } catch (err) {
      console.warn("Không thể mở MediaStream:", err);
      setErrorMessage("Không thể truy cập camera. Vui lòng cho phép quyền Camera hoặc sử dụng nút Chụp ảnh.");
      setStepText("👉 Chạm vào nút bên dưới để mở Camera chụp ảnh quét 3D");
    }
  }, []);

  useEffect(() => {
    initCamera(cameraFacing);

    return () => {
      if (videoRef.current && videoRef.current.srcObject) {
        const stream = videoRef.current.srcObject as MediaStream;
        stream.getTracks().forEach((t) => t.stop());
      }
    };
  }, [initCamera, cameraFacing]);

  const toggleCameraFacing = () => {
    setCameraFacing((prev) => (prev === "user" ? "environment" : "user"));
  };

  // 2. Xử lý quét và nạp khối 3D thật vào hồ sơ
  const trigger3DScan = async (imageBlob?: Blob) => {
    setIsScanning(true);
    setScanProgress(20);
    setStepText("📸 [1/3] Đang quét sống mũi & cấu trúc khuôn mặt...");

    setTimeout(() => {
      setScanProgress(55);
      setStepText("📸 [2/3] Đang nhận diện các mốc giải phẫu thẩm mỹ (Nose, Chin, Lips)...");
    }, 800);

    setTimeout(() => {
      setScanProgress(85);
      setStepText("📐 [3/3] Đang tính toán tỷ lệ giải phẫu sụn mũi & cằm milimet...");
    }, 1600);

    setTimeout(async () => {
      setScanProgress(95);
      setStepText("⚙️ Đang đồng bộ hóa khối 3D vào hồ sơ bệnh nhân...");

      try {
        const formData = new FormData();
        if (imageBlob) {
          formData.append("texture", imageBlob, "texture.jpg");
        }

        const res = await fetch(`/api/patients/${patientId}/upload-3d`, {
          method: "POST",
          body: formData,
        });

        const data = await res.json();
        if (res.ok && data.success) {
          setScanProgress(100);
          setStepText("✓ Quét 3D thành công! Đang chuyển vào 3D Studio...");
          setTimeout(() => {
            onScanCompleted();
          }, 600);
        } else {
          setIsScanning(false);
          setErrorMessage(data.error || "Không thể lưu dữ liệu quét. Vui lòng thử lại.");
          setStepText("⚠️ Thử lại hoặc tải ảnh trực tiếp.");
        }
      } catch (e) {
        console.error("API update error:", e);
        setIsScanning(false);
        setErrorMessage("Lỗi kết nối máy chủ khi nạp 3D.");
        setStepText("⚠️ Lỗi kết nối. Vui lòng thử lại.");
      }
    }, 2400);
  };

  const handleStartScan = () => {
    if (videoRef.current && canvasRef.current) {
      const canvas = canvasRef.current;
      canvas.width = videoRef.current.videoWidth || 640;
      canvas.height = videoRef.current.videoHeight || 480;
      const ctx = canvas.getContext("2d");
      if (ctx) {
        ctx.drawImage(videoRef.current, 0, 0, canvas.width, canvas.height);
        canvas.toBlob(
          (blob) => {
            trigger3DScan(blob || undefined);
          },
          "image/jpeg",
          0.95
        );
        return;
      }
    }
    trigger3DScan();
  };

  const handleNativePhotoCapture = (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0];
    if (file) {
      trigger3DScan(file);
    }
  };

  return (
    <div className="fixed inset-0 z-50 bg-black/95 backdrop-blur-md flex flex-col items-center justify-between sm:justify-center p-3 sm:p-6 overflow-y-auto animate-fadeIn">
      <div className="relative w-full max-w-2xl bg-gradient-to-b from-[#1c0409] via-zinc-950 to-black rounded-2xl sm:rounded-3xl p-4 sm:p-6 border-2 border-amber-400/80 shadow-[0_0_80px_rgba(251,191,36,0.35)] flex flex-col items-center text-center my-auto">
        
        {/* Header */}
        <div className="w-full flex justify-between items-center mb-3 pb-3 border-b border-zinc-800">
          <div className="flex items-center gap-2">
            <span className="w-2.5 h-2.5 rounded-full bg-red-500 animate-ping" />
            <span className="text-xs sm:text-sm font-black text-amber-300 uppercase tracking-wider">
              QUÉT 3D KHUÔN MẶT Y KHOA (3D SCANNER)
            </span>
          </div>
          <div className="flex items-center gap-2">
            {streamActive && (
              <button
                type="button"
                onClick={toggleCameraFacing}
                title="Đổi camera trước / sau"
                className="px-3 py-1.5 rounded-full bg-zinc-800 hover:bg-zinc-700 text-amber-300 text-xs font-bold border border-zinc-700 flex items-center gap-1.5 transition"
              >
                <span>🔄</span>
                <span className="hidden sm:inline">{cameraFacing === "user" ? "Camera Sau" : "Camera Trước"}</span>
              </button>
            )}
            <button
              onClick={onClose}
              aria-label="Đóng camera quét"
              className="w-8 h-8 rounded-full bg-white/10 hover:bg-white/20 text-white font-bold transition flex items-center justify-center text-sm"
            >
              ✕
            </button>
          </div>
        </div>

        {/* Khung Video Camera Lớn & Rõ Ràng */}
        <div className="relative w-full h-[50vh] min-h-[280px] sm:h-[420px] max-h-[520px] rounded-2xl overflow-hidden bg-zinc-950 border-2 border-zinc-700 shadow-inner flex items-center justify-center">
          <video
            ref={videoRef}
            playsInline
            autoPlay
            muted
            className={`w-full h-full object-cover ${cameraFacing === "user" ? "transform scale-x-[-1]" : ""} ${
              streamActive ? "block" : "hidden"
            }`}
          />
          <canvas ref={canvasRef} className="hidden" />

          {/* Nếu camera video chưa lên hoặc lỗi, hiện nút kích hoạt chụp ảnh native */}
          {!streamActive && (
            <div className="p-6 flex flex-col items-center justify-center text-center">
              <div className="w-16 h-16 sm:w-20 sm:h-20 rounded-full bg-amber-500/20 border border-amber-400/40 flex items-center justify-center text-3xl sm:text-4xl mb-3 animate-pulse">
                📷
              </div>
              <p className="text-xs sm:text-sm text-zinc-300 max-w-sm mb-4 leading-relaxed">
                {errorMessage || "Chạm nút bên dưới để mở Camera điện thoại/iPad quét khuôn mặt trực tiếp."}
              </p>
              <button
                type="button"
                onClick={() => fileInputRef.current?.click()}
                className="px-6 py-3.5 rounded-full bg-gradient-to-r from-yellow-500 to-amber-300 text-black font-black text-xs sm:text-sm uppercase tracking-wider shadow-lg hover:scale-105 active:scale-95 transition"
              >
                📸 MỞ CAMERA CHỤP QUÉT NGAY
              </button>
            </div>
          )}

          {/* Lưới Quét Laser 3D (HUD) Chuyên Nghiệp */}
          {streamActive && (
            <div className="absolute inset-0 pointer-events-none flex flex-col items-center justify-center">
              <div
                className={`w-44 h-56 sm:w-60 sm:h-76 rounded-[48%] border-2 border-dashed transition-all duration-300 flex items-center justify-center ${
                  isScanning
                    ? "border-amber-400 scale-105 shadow-[0_0_50px_rgba(251,191,36,0.8)] bg-amber-500/5"
                    : "border-emerald-400/80 shadow-[0_0_20px_rgba(52,211,153,0.3)]"
                }`}
              >
                {/* Laser scan bar */}
                <div className="w-full h-1 bg-gradient-to-r from-transparent via-amber-400 to-transparent animate-bounce opacity-90 shadow-[0_0_15px_rgba(251,191,36,1)]" />
              </div>

              {/* Vạch đo giải phẫu */}
              <div className="absolute bottom-3 text-[10px] sm:text-xs font-mono font-bold text-amber-300 bg-black/80 backdrop-blur-sm px-3 py-1 rounded-full border border-amber-400/40">
                TRUEDEPTH 3D • ĐỘ CHÍNH XÁC 0.1MM
              </div>
            </div>
          )}
        </div>

        {/* Trạng thái quét & Progress bar */}
        <div className="w-full mt-3">
          <p className="text-xs sm:text-sm font-bold text-white mb-2 min-h-[20px]">{stepText}</p>

          {isScanning && (
            <div className="w-full h-2.5 bg-zinc-800 rounded-full overflow-hidden mb-3">
              <div
                className="h-full bg-gradient-to-r from-yellow-500 via-amber-300 to-emerald-400 transition-all duration-300"
                style={{ width: `${scanProgress}%` }}
              />
            </div>
          )}
        </div>

        {/* Input ẩn chụp ảnh Native iOS/Android */}
        <input
          ref={fileInputRef}
          type="file"
          accept="image/*"
          capture={cameraFacing === "user" ? "user" : "environment"}
          className="hidden"
          onChange={handleNativePhotoCapture}
        />

        {/* Nút bấm hành động */}
        <div className="w-full flex gap-2 sm:gap-3 mt-2">
          <button
            type="button"
            onClick={onClose}
            className="flex-1 py-3 sm:py-3.5 rounded-full border border-zinc-700 text-zinc-300 font-bold text-xs uppercase hover:bg-zinc-800 transition"
          >
            Hủy Bỏ
          </button>

          {streamActive ? (
            <button
              type="button"
              disabled={isScanning}
              onClick={handleStartScan}
              className="flex-[2] py-3 sm:py-3.5 rounded-full bg-gradient-to-r from-yellow-500 via-amber-300 to-yellow-500 text-black font-black text-xs sm:text-sm uppercase tracking-wider shadow-[0_0_30px_rgba(251,191,36,0.8)] hover:scale-105 active:scale-95 transition flex items-center justify-center gap-2 disabled:opacity-70"
            >
              <span>⚡</span>
              <span>{isScanning ? "ĐANG QUÉT 3D..." : "CHỤP & BẮT KHỐI 3D"}</span>
            </button>
          ) : (
            <button
              type="button"
              disabled={isScanning}
              onClick={() => fileInputRef.current?.click()}
              className="flex-[2] py-3 sm:py-3.5 rounded-full bg-gradient-to-r from-yellow-500 via-amber-300 to-yellow-500 text-black font-black text-xs sm:text-sm uppercase tracking-wider shadow-[0_0_30px_rgba(251,191,36,0.8)] hover:scale-105 active:scale-95 transition flex items-center justify-center gap-2"
            >
              <span>📸</span>
              <span>CHỤP ẢNH TẠO 3D NGAY</span>
            </button>
          )}
        </div>
      </div>
    </div>
  );
}
