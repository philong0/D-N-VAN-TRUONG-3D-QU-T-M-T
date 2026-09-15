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

// MARK: - Face ID 36-Tick Radial Ring (Matches Apple Face ID Image 3, 4, 5)
struct FaceIDRadialRing: View {
    let ticks: [Bool]
    let totalTicks: Int = 36
    let radius: CGFloat = 146
    let tickLength: CGFloat = 17
    
    var body: some View {
        ZStack {
            ForEach(0..<totalTicks, id: \.self) { index in
                let isFilled = index < ticks.count ? ticks[index] : false
                let angle = Double(index) * (360.0 / Double(totalTicks))
                
                Capsule()
                    .fill(isFilled ? Color(red: 0.18, green: 0.85, blue: 0.35) : Color(white: 0.35).opacity(0.5))
                    .frame(width: isFilled ? 3.5 : 2.5, height: isFilled ? tickLength + 4 : tickLength)
                    .shadow(color: isFilled ? Color(red: 0.18, green: 0.85, blue: 0.35).opacity(0.85) : Color.clear, radius: isFilled ? 5 : 0)
                    .offset(y: -radius)
                    .rotationEffect(.degrees(angle))
                    .animation(.spring(response: 0.2, dampingFraction: 0.6), value: isFilled)
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
    
    // Apple Face ID Phase Transition:
    // false = Giai đoạn 1: Khung bo góc chữ nhật 4 góc trắng (Ảnh 2)
    // true  = Giai đoạn 2: Hút mặt vào vòng tròn 36 nan quạt xoay 60° (Ảnh 3, 4, 5)
    @State private var hasFaceLocked = false
    
    public init(captureSession: ARFaceCaptureSession, isCompleted: Binding<Bool>) {
        self.captureSession = captureSession
        self._isCompleted = isCompleted
    }
    
    public var body: some View {
        ZStack {
            // Apple Pure Dark Canvas
            Color.black.edgesIgnoringSafeArea(.all)
            
            if captureSession.scannerMode == .faceIdSelfScan {
                // ====================================================
                // MODE 1: CHUẨN 100% APPLE FACE ID ENROLLMENT FLOW
                // ====================================================
                VStack(spacing: 0) {
                    // Top Bar: Back & Mode Toggle
                    HStack {
                        Button {
                            captureSession.onScanCancelled?()
                            isCompleted = true
                        } label: {
                            Image(systemName: "chevron.left")
                                .font(.system(size: 18, weight: .bold))
                                .foregroundColor(.white)
                                .padding(10)
                                .background(Color.white.opacity(0.12), in: Circle())
                        }
                        
                        Spacer()
                        
                        // Mode Switcher (Tự Quét vs Điều Dưỡng)
                        HStack(spacing: 4) {
                            Button {
                                captureSession.setScannerMode(.faceIdSelfScan)
                            } label: {
                                HStack(spacing: 4) {
                                    Image(systemName: "faceid")
                                    Text("Tự Quét (Face ID)")
                                        .font(.system(size: 11, weight: .bold))
                                }
                                .padding(.horizontal, 12)
                                .padding(.vertical, 7)
                                .background(Color(red: 0.18, green: 0.85, blue: 0.35))
                                .foregroundColor(.black)
                                .cornerRadius(14)
                            }
                            
                            Button {
                                captureSession.setScannerMode(.rearClinicalAssistant)
                            } label: {
                                HStack(spacing: 4) {
                                    Image(systemName: "camera.viewfinder")
                                    Text("Điều Dưỡng")
                                        .font(.system(size: 11, weight: .bold))
                                }
                                .padding(.horizontal, 12)
                                .padding(.vertical, 7)
                                .background(Color.white.opacity(0.08))
                                .foregroundColor(.white.opacity(0.7))
                                .cornerRadius(14)
                            }
                        }
                        .padding(3)
                        .background(Color.white.opacity(0.12))
                        .cornerRadius(16)
                        
                        Spacer()
                        
                        Color.clear.frame(width: 38, height: 38)
                    }
                    .padding(.horizontal, 20)
                    .padding(.top, 45)
                    
                    Spacer(minLength: 20)
                    
                    // Center Viewport: Morphing from Rectangle (Ảnh 2) to Circle (Ảnh 3, 4, 5)
                    ZStack {
                        // Live Camera Feed inside Morphing Shape
                        ARSCNViewContainer(session: captureSession.arSession)
                            .frame(
                                width: hasFaceLocked ? 260 : 270,
                                height: hasFaceLocked ? 260 : 360
                            )
                            .clipShape(
                                RoundedRectangle(
                                    cornerRadius: hasFaceLocked ? 130 : 44,
                                    style: .continuous
                                )
                            )
                            .overlay(
                                Group {
                                    if !hasFaceLocked {
                                        // 4 White Corner Brackets (Exact Match with Image 2)
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
                                        .padding(12)
                                    } else {
                                        // Subtle Circle Border (Exact Match with Image 3)
                                        Circle()
                                            .stroke(Color.white.opacity(0.18), lineWidth: 1.5)
                                    }
                                }
                            )
                            .shadow(color: hasFaceLocked ? Color(red: 0.18, green: 0.85, blue: 0.35).opacity(0.35) : Color.black.opacity(0.5), radius: 24)
                            .animation(.spring(response: 0.65, dampingFraction: 0.75), value: hasFaceLocked)
                        
                        // 36-Tick Radial Ring around the Circle (Appears when Face Locked - Image 3, 4, 5)
                        if hasFaceLocked {
                            FaceIDRadialRing(ticks: captureSession.faceIdTicks)
                                .transition(.scale(scale: 0.85).combined(with: .opacity))
                        }
                    }
                    
                    Spacer(minLength: 25)
                    
                    // Instructions & Live Status Notifications (Exact Match with Apple)
                    VStack(spacing: 12) {
                        if !hasFaceLocked {
                            // Phase 1 Instruction (Ảnh 2)
                            Text("Định vị khuôn mặt\ncủa bạn trong khung.")
                                .font(.system(size: 21, weight: .bold))
                                .foregroundColor(.white)
                                .multilineTextAlignment(.center)
                                .padding(.horizontal, 24)
                            
                            Text("Giữ điện thoại cách mặt khoảng 35 - 50 cm")
                                .font(.system(size: 13, weight: .medium))
                                .foregroundColor(.white.opacity(0.6))
                        } else {
                            // Phase 2 Instruction (Ảnh 3, 4)
                            Text("Di chuyển chậm đầu của bạn để hoàn thành vòng tròn.")
                                .font(.system(size: 19, weight: .bold))
                                .foregroundColor(.white)
                                .multilineTextAlignment(.center)
                                .padding(.horizontal, 28)
                            
                            // Dynamic Clinical Angle Guidance (Báo cho khách biết góc nghiêng sống mũi)
                            Text(captureSession.guidanceFeedback)
                                .font(.system(size: 14, weight: .bold))
                                .foregroundColor(Color(red: 0.18, green: 0.85, blue: 0.35))
                                .padding(.horizontal, 16)
                                .padding(.vertical, 6)
                                .background(Color(red: 0.18, green: 0.85, blue: 0.35).opacity(0.12))
                                .cornerRadius(14)
                            
                            // Progress pill
                            HStack(spacing: 6) {
                                Image(systemName: "checkmark.circle.fill")
                                    .foregroundColor(Color(red: 0.18, green: 0.85, blue: 0.35))
                                Text("Đã quét: \(captureSession.faceIdFilledCount)/36 tia (\(Int(Double(captureSession.faceIdFilledCount) / 36.0 * 100))%)")
                                    .font(.system(size: 13, weight: .bold))
                                    .foregroundColor(.white)
                            }
                            .padding(.horizontal, 16)
                            .padding(.vertical, 6)
                            .background(Color.white.opacity(0.1))
                            .cornerRadius(16)
                        }
                    }
                    
                    Spacer(minLength: 20)
                    
                    // Bottom Button: "Bắt đầu lại" (Exact Match with Apple Face ID Image 2, 3, 4, 5)
                    Button {
                        captureSession.resetScan()
                        withAnimation { hasFaceLocked = false }
                    } label: {
                        Text("Bắt đầu lại")
                            .font(.system(size: 16, weight: .semibold))
                            .foregroundColor(.white)
                            .frame(maxWidth: .infinity)
                            .padding(.vertical, 14)
                            .background(Color.white.opacity(0.12))
                            .cornerRadius(28)
                    }
                    .padding(.horizontal, 36)
                    .padding(.bottom, 35)
                }
                .onChange(of: captureSession.isTracking) { isTracking in
                    if isTracking && !hasFaceLocked {
                        withAnimation(.spring(response: 0.65, dampingFraction: 0.75)) {
                            hasFaceLocked = true
                        }
                        UIImpactFeedbackGenerator(style: .medium).impactOccurred()
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
                            .scaleEffect(1.8)
                            .progressViewStyle(CircularProgressViewStyle(tint: Color(red: 0.18, green: 0.85, blue: 0.35)))
                        
                        Text("ĐANG DỰNG MÔ HÌNH 3D FULL-HEAD")
                            .font(.headline)
                            .bold()
                            .foregroundColor(.white)
                        
                        Text("Dữ liệu đang được gửi tới AI Engine để tạo mô hình 3D thực tế (tai, tóc, mắt, sống mũi)...")
                            .font(.footnote)
                            .foregroundColor(.white.opacity(0.8))
                            .multilineTextAlignment(.center)
                            .padding(.horizontal, 32)
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
