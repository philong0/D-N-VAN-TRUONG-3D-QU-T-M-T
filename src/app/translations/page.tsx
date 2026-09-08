import Link from 'next/link';

export default function TranslationsPage() {
  const files = [
    {
      title: "Phần I: Địa hình học Cơ cau mày (Corrugator Topography)",
      authors: "Jeffrey E. Janis, M.D., Ashkan Ghavami, M.D., Bahman Guyuron, M.D., et al.",
      journal: "Plastic and Reconstructive Surgery • 120(6): 1647-1653",
      desc: "Bản dịch tiếng Việt chuyên khoa Y khoa - Phẫu thuật Tạo hình. Bao gồm đầy đủ Hình 1, Hình 2, Hình 3 và Bảng 1, Bảng 2, Bảng 3.",
      file: "/translations/Giai_Phau_Co_Cau_May_Phan_1_Dia_Hinh_Co.pdf",
      size: "1.9 MB"
    },
    {
      title: "Phần II: Các dạng phân nhánh Thần kinh trên ổ mắt (Supraorbital Nerve Branching Patterns)",
      authors: "Jeffrey E. Janis, M.D., Ashkan Ghavami, M.D., Bahman Guyuron, M.D., et al.",
      journal: "Plastic and Reconstructive Surgery • 121(1): 233-240",
      desc: "Bản dịch tiếng Việt chuyên khoa Y khoa - Phẫu thuật Tạo hình. Bao gồm phân loại 4 Type (Type I đến Type IV), Hình 1 đến Hình 7 và Bảng 1.",
      file: "/translations/Giai_Phau_Co_Cau_May_Phan_2_Phan_Nhanh_TK_Tren_O_Mat.pdf",
      size: "5.2 MB"
    }
  ];

  return (
    <div className="min-h-screen bg-slate-950 text-slate-100 py-12 px-4 sm:px-6 lg:px-8">
      <div className="max-w-4xl mx-auto">
        <div className="text-center mb-10">
          <span className="inline-block px-3 py-1 text-xs font-semibold uppercase tracking-wider text-teal-400 bg-teal-950/80 border border-teal-800/60 rounded-full mb-3">
            Tài Liệu Y Khoa Dịch Thuật
          </span>
          <h1 className="text-3xl font-bold tracking-tight text-white sm:text-4xl">
            Tải Về 2 File PDF Giải Phẫu Cơ Cau Mày
          </h1>
          <p className="mt-3 text-base text-slate-400 max-w-2xl mx-auto">
            Bản dịch chuẩn y khoa đầy đủ hình ảnh minh họa độ phân giải cao, bảng số liệu và phân tích phẫu thuật giải ép thần kinh điều trị đau nửa đầu.
          </p>
        </div>

        <div className="space-y-6">
          {files.map((item, idx) => (
            <div key={idx} className="bg-slate-900 border border-slate-800 rounded-2xl p-6 shadow-xl hover:border-teal-500/40 transition">
              <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-4">
                <div className="space-y-2 flex-1">
                  <div className="flex items-center gap-2">
                    <span className="text-xs font-mono px-2 py-0.5 rounded bg-slate-800 text-teal-300 border border-slate-700">
                      PDF • {item.size}
                    </span>
                    <span className="text-xs text-slate-500 italic">{item.journal}</span>
                  </div>
                  <h2 className="text-xl font-semibold text-white">{item.title}</h2>
                  <p className="text-xs text-slate-400">{item.authors}</p>
                  <p className="text-sm text-slate-300">{item.desc}</p>
                </div>
                <div className="flex flex-row sm:flex-col gap-2 shrink-0">
                  <a
                    href={item.file}
                    download
                    className="inline-flex items-center justify-center px-5 py-2.5 text-sm font-medium rounded-xl text-slate-950 bg-teal-400 hover:bg-teal-300 font-semibold shadow-md transition"
                  >
                    📥 Tải File PDF
                  </a>
                  <a
                    href={item.file}
                    target="_blank"
                    rel="noopener noreferrer"
                    className="inline-flex items-center justify-center px-4 py-2 text-sm font-medium rounded-xl text-slate-200 bg-slate-800 hover:bg-slate-700 border border-slate-700 transition"
                  >
                    👁 Xem Trực Tiếp
                  </a>
                </div>
              </div>
            </div>
          ))}
        </div>

        <div className="mt-10 text-center text-xs text-slate-500">
          Dr. Văn Trương 3D Studio • Hệ thống dịch thuật tài liệu y khoa
        </div>
      </div>
    </div>
  );
}

