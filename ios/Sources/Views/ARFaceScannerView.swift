//
//  ARFaceScannerView.swift
//  DrVanTruongScanner (iOS Native TrueDepth Layer)
//

import SwiftUI
import ARKit

struct ARSCNViewContainer: UIViewRepresentable {
    let session: ARSession
    
    func makeUIView(context: Context) -> ARSCNView {
        let view = ARSCNView(frame: .zero)
        view.session = session
        view.automaticallyUpdatesLighting = true
        view.rendersContinuously = true
        return view
    }
    
    func updateUIView(_ uiView: ARSCNView, context: Context) {}
}

// MARK: - 4 Corner Brackets (Matches Apple Face ID Image 2)
struct CornerBracket: View {
    enum Corner { case topLeft, topRight, bottomLeft, bottomRight }
    let corner: Corner
    let length: CGFloat = 28
    let thickness: CGFloat = 4.5
    let radius: CGFloat = 10
    
    var body: some View {
        Path { path in
            switch corner {
            case .topLeft:
                path.move(to: CGPoint(x: 0, y: length))
                path.addLine(to: CGPoint(x: 0, y: radius))
                path.addQuadCurve(to: CGPoint(x: radius, y: 0), control: CGPoint(x: 0, y: 0))
                path.addLine(to: CGPoint(x: length, y: 0))
            case .topRight:
                path.move(to: CGPoint(x: 0, y: 0))
                path.addLine(to: CGPoint(x: length - radius, y: 0))
                path.addQuadCurve(to: CGPoint(x: length, y: radius), control: CGPoint(x: length, y: 0))
                path.addLine(to: CGPoint(x: length, y: length))
            case .bottomLeft:
                path.move(to: CGPoint(x: 0, y: 0))
                path.addLine(to: CGPoint(x: 0, y: length - radius))
                path.addQuadCurve(to: CGPoint(x: radius, y: length), control: CGPoint(x: 0, y: length))
                path.addLine(to: CGPoint(x: length, y: length))
            case .bottomRight:
                path.move(to: CGPoint(x: length, y: 0))
                path.addLine(to: CGPoint(x: length, y: length - radius))
                path.addQuadCurve(to: CGPoint(x: length - radius, y: length), control: CGPoint(x: length, y: length))
                path.addLine(to: CGPoint(x: 0, y: length))
            }
        }
        .stroke(Color.white, style: StrokeStyle(lineWidth: thickness, lineCap: .round, lineJoin: .round))
        .frame(width: length, height: length)
    }
}

//// MARK: - Apple Face ID Smile Glyph (Exact Match with Image 1)
struct FaceIDSmileGlyph: View {
    var body: some View {
        ZStack {
            // Face outline circle
            Circle()
                .stroke(Color(white: 0.65), lineWidth: 7)
                .frame(width: 140, height: 140)
            
            // Left eye
            Capsule()
                .fill(Color(white: 0.65))
                .frame(width: 8, height: 22)
                .offset(x: -26, y: -12)
            
            // Right eye
            Capsule()
                .fill(Color(white: 0.65))
                .frame(width: 8, height: 22)
                .offset(x: 26, y: -12)
            
            // Nose (L shape)
            Path { path in
                path.move(to: CGPoint(x: 0, y: -14))
                path.addLine(to: CGPoint(x: 0, y: 12))
                path.addLine(to: CGPoint(x: 9, y: 12))
            }
            .stroke(Color(white: 0.65), style: StrokeStyle(lineWidth: 6, lineCap: .round, lineJoin: .round))
            .offset(x: -3, y: 0)
            
            // Smile curve
            Path { path in
                path.addArc(
                    center: CGPoint(x: 0, y: 12),
                    radius: 28,
                    startAngle: .degrees(25),
                    endAngle: .degrees(155),
                    clockwise: false
                )
            }
            .stroke(Color(white: 0.65), style: StrokeStyle(lineWidth: 6, lineCap: .round))
        }
        .frame(width: 150, height: 150)
    }
}

// MARK: - Face ID 36-Tick Radial Ring (Matches Apple Face ID Image 1, 3, 4, 5)
struct FaceIDRadialRing: View {
    let ticks: [Bool]
    let totalTicks: Int = 36
    let radius: CGFloat = 146
    let tickLength: CGFloat = 17
    var isIntroMode: Bool = false
    
    var body: some View {
        ZStack {
            ForEach(0..<totalTicks, id: \.self) { index in
                let isFilled = (!isIntroMode && index < ticks.count) ? ticks[index] : false
                let angle = Double(index) * (360.0 / Double(totalTicks))
                
                Capsule()
                    .fill(isIntroMode ? Color(white: 0.72) : (isFilled ? Color(red: 0.19, green: 0.82, blue: 0.35) : Color(white: 0.38).opacity(0.45)))
                    .frame(width: isFilled ? 3.5 : (isIntroMode ? 3.0 : 2.5), height: isFilled ? tickLength + 4 : tickLength)
                    .shadow(color: isFilled ? Color(red: 0.19, green: 0.82, blue: 0.35).opacity(0.85) : Color.clear, radius: isFilled ? 5 : 0)
                    .offset(y: -radius)
                    .rotationEffect(.degrees(angle))
                    .animation(.spring(response: 0.2, dampingFraction: 0.6), value: isFilled)
            }
        }
    }
}

// MARK: - Apple Dynamic Ripple Glow Edge (Viền ánh sáng rung/gợn sóng quét theo góc quay đầu - Ảnh Người Dùng Gửi)
struct FaceIDRippleGlowEdge: View {
    let currentYaw: Float
    let isTracking: Bool
    @State private var ripplePhase: CGFloat = 0
    
    var body: some View {
        ZStack {
            // Vòng cung ánh sáng gợn sóng (Ripple Sonar Arc)
            ForEach(0..<5) { layer in
                let radiusOffset = CGFloat(layer) * 7.0
                let opacityVal = 0.85 - Double(layer) * 0.16
                
                Circle()
                    .stroke(
                        AngularGradient(
                            gradient: Gradient(colors: [
                                Color.clear,
                                Color(red: 0.2, green: 0.95, blue: 0.85).opacity(opacityVal),
                                Color(red: 0.1, green: 0.85, blue: 0.6).opacity(opacityVal * 0.8),
                                Color.clear
                            ]),
                            center: .center,
                            startAngle: .degrees(Double(currentYaw * 2.2) - 80),
                            endAngle: .degrees(Double(currentYaw * 2.2) + 80)
                        ),
                        lineWidth: 1.8
                    )
                    .frame(width: 258 - radiusOffset, height: 258 - radiusOffset)
                    .scaleEffect(1.0 + (ripplePhase * 0.02 * CGFloat(layer)))
            }
            
            // Hướng mũi tên mờ chỉ góc quay (Gợi ý quay đầu theo nhịp)
            if abs(currentYaw) > 12 {
                Image(systemName: currentYaw < 0 ? "arrow.left" : "arrow.right")
                    .font(.system(size: 38, weight: .bold))
                    .foregroundColor(Color.white.opacity(0.22))
                    .offset(x: currentYaw < 0 ? -45 : 45)
                    .animation(.easeInOut(duration: 0.8).repeatForever(autoreverses: true), value: currentYaw)
            }
        }
        .onAppear {
            withAnimation(.easeInOut(duration: 1.2).repeatForever(autoreverses: true)) {
                ripplePhase = 1.0
            }
        }
    }
}

// MARK: - Clinical Reticle Brackets for Rear Camera Mode
struct ClinicalReticleBrackets: View {
    var isAligned: Bool
    
    var body: some View {
        ZStack {
            RoundedRectangle(cornerRadius: 16)
                .stroke(isAligned ? Color.cyan : Color.white.opacity(0.4), style: StrokeStyle(lineWidth: 2, dash: [24, 12]))
                .frame(width: 250, height: 330)
            
            Image(systemName: "plus")
                .font(.title3)
                .foregroundColor(isAligned ? .cyan : .white.opacity(0.5))
        }
    }
}

public struct ARFaceScannerView: View {
    @ObservedObject var captureSession: ARFaceCaptureSession
    @Binding var isCompleted: Bool
    
    // Giai đoạn của Face ID Flow chuẩn Apple:
    // 0 = Màn hình Giới thiệu ban đầu (Ảnh 1 - Nền trắng, vòng xám, icon cười, nút Bắt đầu xanh)
    // 1 = Màn hình Định vị khuôn mặt (Ảnh 2 - Khung chữ nhật bo góc, 4 góc bracket trắng)
    // 2 = Màn hình Quét xoay đầu (Ảnh 3, 4, 5 - Vòng tròn nan quạt xanh lá neon + Viền ánh sáng rung)
    @State private var enrollmentPhase: Int = 0
    
    public init(captureSession: ARFaceCaptureSession, isCompleted: Binding<Bool>) {
        self.captureSession = captureSession
        self._isCompleted = isCompleted
    }
    
    public var body: some View {
        ZStack {
            // Nền trắng tinh khôi chuẩn Apple iOS Settings
            Color.white.edgesIgnoringSafeArea(.all)
            
            if captureSession.scannerMode == .faceIdSelfScan {
                // ====================================================
                // MODE 1: CHUẨN 100% APPLE FACE ID ENROLLMENT FLOW
                // ====================================================
                if enrollmentPhase == 0 {
                    // ----------------------------------------------------
                    // BƯỚC 1: MÀN HÌNH GIỚI THIỆU (GIỐNG 100% ẢNH 1 BẠN GỬI)
                    // ----------------------------------------------------
                    VStack(spacing: 0) {
                        // Top navigation
                        HStack {
                            Button {
                                captureSession.onScanCancelled?()
                                isCompleted = true
                            } label: {
                                Image(systemName: "chevron.left")
                                    .font(.system(size: 20, weight: .semibold))
                                    .foregroundColor(.black)
                                    .padding(10)
                                    .background(Color(white: 0.94), in: Circle())
                            }
                            Spacer()
                        }
                        .padding(.horizontal, 24)
                        .padding(.top, 20)
                        
                        Spacer(minLength: 20)
                        
                        // Center Face ID Graphic with Smile
                        ZStack {
                            FaceIDRadialRing(ticks: [], isIntroMode: true)
                            FaceIDSmileGlyph()
                        }
                        .frame(width: 320, height: 320)
                        
                        Spacer(minLength: 30)
                        
                        // Text: Tiêu đề và hướng dẫn chuẩn Apple Settings
                        VStack(spacing: 12) {
                            Text("Thiết lập quét 3D Thẩm mỹ")
                                .font(.system(size: 24, weight: .bold))
                                .foregroundColor(.black)
                            
                            Text("Đầu tiên, định vị khuôn mặt của bạn trong khung hình camera. Sau đó, di chuyển đầu của bạn theo hình tròn để hiển thị tất cả các góc cạnh trên khuôn mặt của bạn.")
                                .font(.system(size: 16, weight: .regular))
                                .foregroundColor(Color(white: 0.35))
                                .multilineTextAlignment(.center)
                                .lineSpacing(4)
                                .padding(.horizontal, 32)
                        }
                        
                        Spacer(minLength: 40)
                        
                        // Nút lớn bo tròn màu xanh dương Apple: [Bắt đầu]
                        Button {
                            withAnimation(.easeInOut(duration: 0.35)) {
                                enrollmentPhase = 1
                            }
                            captureSession.startSession()
                            captureSession.resetScan()
                            UIImpactFeedbackGenerator(style: .medium).impactOccurred()
                        } label: {
                            Text("Bắt đầu")
                                .font(.system(size: 17, weight: .bold))
                                .foregroundColor(.white)
                                .frame(maxWidth: .infinity)
                                .padding(.vertical, 16)
                                .background(Color(red: 0.0, green: 0.48, blue: 1.0))
                                .cornerRadius(28)
                        }
                        .padding(.horizontal, 28)
                        .padding(.bottom, 36)
                    }
                } else {
                    // ----------------------------------------------------
                    // BƯỚC 2 & 3: KHUNG VUÔNG NHẬN DIỆN & VÒNG TRÒN QUÉT
                    // ----------------------------------------------------
                    VStack(spacing: 0) {
                        // Top Bar: Back button
                        HStack {
                            Button {
                                captureSession.resetScan()
                                withAnimation { enrollmentPhase = 0 }
                            } label: {
                                Image(systemName: "chevron.left")
                                    .font(.system(size: 18, weight: .semibold))
                                    .foregroundColor(.black)
                                    .padding(10)
                                    .background(Color(white: 0.94), in: Circle())
                            }
                            Spacer()
                        }
                        .padding(.horizontal, 24)
                        .padding(.top, 16)
                        
                        Spacer(minLength: 15)
                        
                        // Card Camera: Chữ nhật bo góc ở Phase 1, Hình tròn 36 nấc ở Phase 2
                        ZStack {
                            // Viền đen bao quanh Card Camera
                            RoundedRectangle(
                                cornerRadius: enrollmentPhase == 2 ? 140 : 44,
                                style: .continuous
                            )
                            .fill(Color.black)
                            .frame(
                                width: 280,
                                height: enrollmentPhase == 2 ? 280 : 380
                            )
                            .shadow(color: Color.black.opacity(0.18), radius: 20, x: 0, y: 10)
                            
                            // Live Camera Feed
                            ARSCNViewContainer(session: captureSession.arSession)
                                .frame(
                                    width: 260,
                                    height: enrollmentPhase == 2 ? 260 : 360
                                )
                                .clipShape(
                                    RoundedRectangle(
                                        cornerRadius: enrollmentPhase == 2 ? 130 : 36,
                                        style: .continuous
                                    )
                                )
                                .overlay(
                                    Group {
                                        if enrollmentPhase == 1 {
                                            // 4 White Corner Brackets nhận diện khuôn mặt
                                            VStack {
                                                HStack {
                                                    CornerBracket(corner: .topLeft)
                                                    Spacer()
                                                    CornerBracket(corner: .topRight)
                                                }
                                                Spacer()
                                                HStack {
                                                    CornerBracket(corner: .bottomLeft)
                                                    Spacer()
                                                    CornerBracket(corner: .bottomRight)
                                                }
                                            }
                                            .padding(14)
                                        }
                                    }
                                )
                            
                            // Viền ánh sáng rung/gợn sóng và 36 nan quạt khi ở Phase 2 (Vòng tròn)
                            if enrollmentPhase == 2 {
                                FaceIDRippleGlowEdge(
                                    currentYaw: captureSession.currentYawDeg,
                                    isTracking: captureSession.isTracking
                                )
                                
                                // 36 nan quạt xoay quanh hình tròn (Ban đầu 100% xám, chỉ sáng xanh khi bắt đầu quét thật)
                                FaceIDRadialRing(ticks: captureSession.faceIdTicks, isIntroMode: false)
                                    .transition(.scale(scale: 0.88).combined(with: .opacity))
                            }
                        }
                        .animation(.spring(response: 0.55, dampingFraction: 0.75), value: enrollmentPhase)
                        
                        Spacer(minLength: 25)
                        
                        // Hướng dẫn tương tác
                        VStack(spacing: 12) {
                            if enrollmentPhase == 1 {
                                Text("Định vị khuôn mặt\ncủa bạn trong khung.")
                                    .font(.system(size: 22, weight: .bold))
                                    .foregroundColor(.black)
                                    .multilineTextAlignment(.center)
                                    .padding(.horizontal, 24)
                            } else if !captureSession.isScanningActive {
                                Text("Đã nhận diện khuôn mặt.\nBấm nút bên dưới để bắt đầu quét.")
                                    .font(.system(size: 19, weight: .bold))
                                    .foregroundColor(.black)
                                    .multilineTextAlignment(.center)
                                    .padding(.horizontal, 24)
                            } else {
                                Text("Di chuyển chậm đầu của bạn để hoàn thành vòng tròn.")
                                    .font(.system(size: 19, weight: .bold))
                                    .foregroundColor(.black)
                                    .multilineTextAlignment(.center)
                                    .padding(.horizontal, 24)
                                
                                // Hiển thị tiến trình quét nấc rõ ràng
                                Text("Đã quét: \(captureSession.faceIdFilledCount)/36 nấc (\(Int(Double(captureSession.faceIdFilledCount) / 36.0 * 100))%)")
                                    .font(.system(size: 15, weight: .bold))
                                    .foregroundColor(Color(red: 0.0, green: 0.48, blue: 1.0))
                                    .padding(.horizontal, 14)
                                    .padding(.vertical, 4)
                                    .background(Color(red: 0.0, green: 0.48, blue: 1.0).opacity(0.12))
                                    .cornerRadius(12)

                                // Dynamic Clinical Guidance nhỏ gọn
                                Text(captureSession.guidanceFeedback)
                                    .font(.system(size: 13, weight: .semibold))
                                    .foregroundColor(Color(red: 0.12, green: 0.65, blue: 0.28))
                                    .padding(.horizontal, 14)
                                    .padding(.vertical, 5)
                                    .background(Color(red: 0.19, green: 0.82, blue: 0.35).opacity(0.14))
                                    .cornerRadius(12)
                            }
                        }
                        
                        Spacer(minLength: 25)
                        
                        // Nút tương tác
                        VStack(spacing: 12) {
                            if enrollmentPhase == 2 && !captureSession.isScanningActive {
                                // Nút lớn xanh dương BẮT ĐẦU QUÉT (Chủ động bấm mới được quét!)
                                Button {
                                    captureSession.startActiveSweep()
                                    UIImpactFeedbackGenerator(style: .medium).impactOccurred()
                                } label: {
                                    Text("Bắt đầu quét")
                                        .font(.system(size: 17, weight: .bold))
                                        .foregroundColor(.white)
                                        .frame(maxWidth: .infinity)
                                        .padding(.vertical, 16)
                                        .background(Color(red: 0.0, green: 0.48, blue: 1.0))
                                        .cornerRadius(28)
                                }
                            } else {

                                // Nút Bắt đầu lại
                                Button {
                                    if enrollmentPhase == 1 {
                                        captureSession.resetScan()
                                    } else {
                                        captureSession.startActiveSweep()
                                    }
                                } label: {
                                    Text("Bắt đầu lại")
                                        .font(.system(size: 15, weight: .semibold))
                                        .foregroundColor(.black.opacity(0.75))
                                        .frame(maxWidth: .infinity)
                                        .padding(.vertical, 13)
                                        .background(Color(white: 0.94))
                                        .cornerRadius(28)
                                }
                            }
                        }
                        .padding(.horizontal, 28)
                        .padding(.bottom, 36)
                    }
                    .onChange(of: captureSession.isFaceInFramingRect) { inFraming in
                        // Khi khuôn mặt lọt vào tâm -> Nhảy sang Vòng tròn Face ID và TỰ ĐỘNG BẮT ĐẦU QUÉT NGAY
                        if inFraming {
                            if enrollmentPhase == 1 {
                                withAnimation(.spring(response: 0.65, dampingFraction: 0.75)) {
                                    enrollmentPhase = 2
                                }
                            }
                            if !captureSession.isScanningActive {
                                captureSession.startActiveSweep()
                            }
                            UIImpactFeedbackGenerator(style: .medium).impactOccurred()
                        }
                    }
                }
            } else {
                // ====================================================
                // MODE 2: ĐIỀU DƯỠNG QUÉT (CAMERA SAU 48MP)
                // ====================================================
                ZStack {
                    ARSCNViewContainer(session: captureSession.arSession)
                        .edgesIgnoringSafeArea(.all)
                    
                    VStack {
                        // Top Bar: Close, Mode Toggle, 5-Step Bar
                        VStack(spacing: 10) {
                            HStack {
                                Button {
                                    captureSession.onScanCancelled?()
                                    isCompleted = true
                                } label: {
                                    Image(systemName: "xmark.circle.fill")
                                        .font(.title2)
                                        .foregroundColor(.white.opacity(0.85))
                                }
                                
                                Spacer()
                                
                                HStack(spacing: 4) {
                                    Button {
                                        captureSession.setScannerMode(.faceIdSelfScan)
                                    } label: {
                                        Text("Tự Quét (Face ID)")
                                            .font(.system(size: 11, weight: .bold))
                                            .padding(.horizontal, 10)
                                            .padding(.vertical, 6)
                                            .foregroundColor(.white.opacity(0.6))
                                    }
                                    
                                    Button {
                                        captureSession.setScannerMode(.rearClinicalAssistant)
                                    } label: {
                                        Text("Điều Dưỡng (Camera Sau)")
                                            .font(.system(size: 11, weight: .bold))
                                            .padding(.horizontal, 10)
                                            .padding(.vertical, 6)
                                            .background(Color.blue)
                                            .foregroundColor(.white)
                                            .cornerRadius(12)
                                    }
                                }
                                .padding(3)
                                .background(Color.white.opacity(0.15))
                                .cornerRadius(15)
                                
                                Spacer()
                                
                                Text("\(captureSession.capturedFrames.count)/5")
                                    .font(.system(size: 14, weight: .black))
                                    .foregroundColor(.cyan)
                            }
                            .padding(.horizontal)
                            
                            // 5 Clinical angle progress bars
                            HStack(spacing: 6) {
                                ForEach(ScanAngleStep.allCases) { step in
                                    VStack(spacing: 2) {
                                        RoundedRectangle(cornerRadius: 3)
                                            .fill(captureSession.capturedFrames[step] != nil ? Color.green : (step == captureSession.currentStep ? Color.cyan : Color.gray.opacity(0.4)))
                                            .frame(height: 5)
                                        
                                        Text(step.title.components(separatedBy: ":").last?.trimmingCharacters(in: .whitespaces) ?? "")
                                            .font(.system(size: 8, weight: .medium))
                                            .foregroundColor(.white.opacity(0.8))
                                            .lineLimit(1)
                                    }
                                }
                            }
                            .padding(.horizontal)
                        }
                        .padding(.vertical, 10)
                        .background(Color.black.opacity(0.82))
                        .cornerRadius(16)
                        .padding(.horizontal)
                        .padding(.top, 40)
                        
                        Spacer()
                        
                        // Clinical Reticle Framing
                        ZStack {
                            ClinicalReticleBrackets(isAligned: captureSession.isPoseAligned)
                            
                            VStack {
                                Text(captureSession.currentStep.title)
                                    .font(.system(size: 13, weight: .bold))
                                    .foregroundColor(.white)
                                    .padding(.horizontal, 12)
                                    .padding(.vertical, 6)
                                    .background(Color.blue.opacity(0.85))
                                    .cornerRadius(12)
                                    .padding(.top, 16)
                                Spacer()
                            }
                            .frame(width: 250, height: 330)
                        }
                        
                        Spacer()
                        
                        // Bottom Controls: Instruction & Shutter
                        VStack(spacing: 12) {
                            Text(captureSession.guidanceFeedback)
                                .font(.system(size: 13, weight: .bold))
                                .foregroundColor(captureSession.isPoseAligned ? .cyan : .yellow)
                                .padding(.horizontal, 18)
                                .padding(.vertical, 8)
                                .background(Color.black.opacity(0.8))
                                .cornerRadius(16)
                            
                            HStack(spacing: 20) {
                                if captureSession.capturedFrames.count > 0 {
                                    Button {
                                        captureSession.retakePreviousStep()
                                    } label: {
                                        Image(systemName: "arrow.uturn.backward.circle.fill")
                                            .font(.system(size: 38))
                                            .foregroundColor(.orange)
                                    }
                                }
                                
                                Button(action: {
                                    if captureSession.capturedFrames.count >= 5 {
                                        captureSession.triggerPackageUpload { _ in }
                                    } else {
                                        try? captureSession.captureCurrentStep()
                                    }
                                }) {
                                    HStack(spacing: 8) {
                                        Image(systemName: captureSession.capturedFrames.count >= 5 ? "arrow.up.circle.fill" : "camera.fill")
                                            .font(.title3)
                                        Text(captureSession.capturedFrames.count >= 5 ? "GỬI TẢI LÊN (5/5)" : "BẤM CHỤP (\(captureSession.capturedFrames.count)/5)")
                                            .font(.system(size: 14, weight: .black))
                                    }
                                    .foregroundColor(.white)
                                    .padding(.horizontal, 24)
                                    .padding(.vertical, 14)
                                    .background(captureSession.capturedFrames.count >= 5 ? Color.orange : Color.blue)
                                    .cornerRadius(28)
                                    .shadow(radius: 6)
                                }
                                
                                Button {
                                    captureSession.resetScan()
                                } label: {
                                    Image(systemName: "arrow.clockwise.circle.fill")
                                        .font(.system(size: 38))
                                        .foregroundColor(.white.opacity(0.85))
                                }
                            }
                            .padding(.bottom, 25)
                        }
                    }
                }
            }
            
            // 3. Uploading & AI 3D Reconstruction Overlay
            if captureSession.isUploading {
                ZStack {
                    Color.black.opacity(0.92).edgesIgnoringSafeArea(.all)
                    VStack(spacing: 20) {
                        ProgressView()
                            .scaleEffect(1.6)
                            .progressViewStyle(CircularProgressViewStyle(tint: Color(red: 0.18, green: 0.85, blue: 0.35)))
                        
                        Text("ĐANG DỰNG MÔ HÌNH 3D FULL-HEAD")
                            .font(.headline)
                            .bold()
                            .foregroundColor(.white)
                        
                        // Thanh tiến trình ngang mượt mà
                        VStack(spacing: 6) {
                            ProgressView(value: Double(max(0.08, captureSession.uploadProgress)), total: 1.0)
                                .progressViewStyle(LinearProgressViewStyle(tint: Color(red: 0.18, green: 0.85, blue: 0.35)))
                                .frame(width: 220)
                            
                            Text("\(Int(max(0.08, captureSession.uploadProgress) * 100))%")
                                .font(.system(size: 13, weight: .bold))
                                .foregroundColor(Color(red: 0.18, green: 0.85, blue: 0.35))
                        }
                        
                        Text(captureSession.lastErrorMessage == nil
                             ? (!captureSession.uploadStatusMessage.isEmpty ? captureSession.uploadStatusMessage : "Dữ liệu đang được gửi tới AI Engine để tạo mô hình 3D thực tế...")
                             : (captureSession.lastErrorMessage ?? ""))
                            .font(.footnote)
                            .foregroundColor(captureSession.lastErrorMessage == nil ? .white.opacity(0.9) : Color(red: 1.0, green: 0.4, blue: 0.4))
                            .multilineTextAlignment(.center)
                            .padding(.horizontal, 24)
                            .animation(.easeInOut(duration: 0.3), value: captureSession.uploadStatusMessage)
                        
                        if captureSession.lastErrorMessage != nil {
                            Button {
                                captureSession.isUploading = false
                                captureSession.startActiveSweep()
                                withAnimation { enrollmentPhase = 1 }
                            } label: {
                                Text("Quét lại")
                                    .font(.system(size: 15, weight: .bold))
                                    .foregroundColor(.white)
                                    .padding(.horizontal, 24)
                                    .padding(.vertical, 10)
                                    .background(Color(red: 0.0, green: 0.48, blue: 1.0))
                                    .cornerRadius(20)
                            }
                        } else {
                            Button {
                                captureSession.isUploading = false
                                captureSession.startActiveSweep()
                            } label: {
                                Text("Hủy")
                                    .font(.system(size: 14, weight: .regular))
                                    .foregroundColor(Color(white: 0.6))
                                    .padding(.top, 4)
                            }
                        }
                    }
                    .padding(28)
                    .background(Color.black.opacity(0.95))
                    .cornerRadius(24)
                    .overlay(
                        RoundedRectangle(cornerRadius: 24)
                            .stroke(Color(red: 0.18, green: 0.85, blue: 0.35).opacity(0.5), lineWidth: 1.5)
                    )
                }
            }
        }
        .onAppear {
            captureSession.startSession()
        }
        .onDisappear {
            captureSession.pauseSession()
        }
    }
}
