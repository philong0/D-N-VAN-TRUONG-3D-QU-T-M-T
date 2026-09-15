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

// MARK: - Face ID 36-Tick Radial Ring (Mockup Exact Match)
struct FaceIDRadialRing: View {
    let ticks: [Bool]
    let totalTicks: Int = 36
    let radius: CGFloat = 144
    let tickLength: CGFloat = 16
    
    var body: some View {
        ZStack {
            ForEach(0..<totalTicks, id: \.self) { index in
                let isFilled = index < ticks.count ? ticks[index] : false
                let angle = Double(index) * (360.0 / Double(totalTicks))
                
                Capsule()
                    .fill(isFilled ? Color(red: 0.0, green: 0.95, blue: 0.6) : Color(red: 0.3, green: 0.45, blue: 0.5).opacity(0.4))
                    .frame(width: isFilled ? 3.5 : 2.5, height: isFilled ? tickLength + 4 : tickLength)
                    .shadow(color: isFilled ? Color(red: 0.0, green: 0.95, blue: 0.6).opacity(0.85) : Color.clear, radius: isFilled ? 5 : 0)
                    .offset(y: -radius)
                    .rotationEffect(.degrees(angle))
                    .animation(.spring(response: 0.2, dampingFraction: 0.6), value: isFilled)
            }
        }
    }
}

// MARK: - Stylized Cyan Face Silhouette Overlay (Matches Mockup)
struct FaceSilhouetteOverlay: View {
    var isTracking: Bool
    
    var body: some View {
        ZStack {
            // Head contour outline
            Ellipse()
                .stroke(isTracking ? Color(red: 0.2, green: 0.85, blue: 1.0).opacity(0.65) : Color.white.opacity(0.3), lineWidth: 1.5)
                .frame(width: 140, height: 185)
                .shadow(color: isTracking ? Color.cyan.opacity(0.4) : Color.clear, radius: 6)
            
            // Eyes alignment markers
            HStack(spacing: 38) {
                Capsule()
                    .stroke(isTracking ? Color(red: 0.2, green: 0.85, blue: 1.0).opacity(0.55) : Color.white.opacity(0.25), lineWidth: 1.5)
                    .frame(width: 24, height: 9)
                Capsule()
                    .stroke(isTracking ? Color(red: 0.2, green: 0.85, blue: 1.0).opacity(0.55) : Color.white.opacity(0.25), lineWidth: 1.5)
                    .frame(width: 24, height: 9)
            }
            .offset(y: -14)
            
            // Nose bridge marker
            Path { p in
                p.move(to: CGPoint(x: 0, y: -10))
                p.addLine(to: CGPoint(x: 0, y: 15))
                p.addLine(to: CGPoint(x: 5, y: 18))
            }
            .stroke(isTracking ? Color(red: 0.2, green: 0.85, blue: 1.0).opacity(0.55) : Color.white.opacity(0.25), lineWidth: 1.5)
            
            // Mouth guide
            Capsule()
                .stroke(isTracking ? Color(red: 0.2, green: 0.85, blue: 1.0).opacity(0.55) : Color.white.opacity(0.25), lineWidth: 1.5)
                .frame(width: 28, height: 7)
                .offset(y: 40)
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
    
    public init(captureSession: ARFaceCaptureSession, isCompleted: Binding<Bool>) {
        self.captureSession = captureSession
        self._isCompleted = isCompleted
    }
    
    public var body: some View {
        ZStack {
            // Dark futuristic background gradient matching mockup
            LinearGradient(
                gradient: Gradient(colors: [Color(red: 0.05, green: 0.09, blue: 0.14), Color(red: 0.02, green: 0.04, blue: 0.07)]),
                startPoint: .top,
                endPoint: .bottom
            )
            .edgesIgnoringSafeArea(.all)
            
            if captureSession.scannerMode == .faceIdSelfScan {
                // ====================================================
                // MODE 1: 100% EXACT MATCH VỚI MOCKUP FACE ID VIP
                // ====================================================
                VStack(spacing: 16) {
                    // Top Bar: Back, Title
                    HStack {
                        Button {
                            captureSession.onScanCancelled?()
                            isCompleted = true
                        } label: {
                            Image(systemName: "chevron.left")
                                .font(.system(size: 18, weight: .semibold))
                                .foregroundColor(.white)
                                .padding(8)
                        }
                        
                        Spacer()
                        
                        Text("Dr. Văn Trường 3D Studio")
                            .font(.system(size: 16, weight: .bold))
                            .foregroundColor(.white)
                        
                        Spacer()
                        
                        // Invisible balance spacer
                        Color.clear.frame(width: 36, height: 36)
                    }
                    .padding(.horizontal, 16)
                    .padding(.top, 45)
                    
                    // Segmented Mode Switcher (Pill Style with Cyan Glow)
                    HStack(spacing: 4) {
                        Button {
                            captureSession.setScannerMode(.faceIdSelfScan)
                        } label: {
                            VStack(spacing: 1) {
                                Text("Tự Quét")
                                    .font(.system(size: 13, weight: .bold))
                                Text("(Face ID)")
                                    .font(.system(size: 10, weight: .regular))
                            }
                            .frame(maxWidth: .infinity)
                            .padding(.vertical, 8)
                            .background(
                                RoundedRectangle(cornerRadius: 18)
                                    .fill(Color(red: 0.12, green: 0.28, blue: 0.38).opacity(0.85))
                                    .overlay(
                                        RoundedRectangle(cornerRadius: 18)
                                            .stroke(Color.cyan.opacity(0.7), lineWidth: 1.5)
                                    )
                                    .shadow(color: Color.cyan.opacity(0.5), radius: 8)
                            )
                            .foregroundColor(.white)
                        }
                        
                        Button {
                            captureSession.setScannerMode(.rearClinicalAssistant)
                        } label: {
                            VStack(spacing: 1) {
                                Text("Điều Dưỡng Quét")
                                    .font(.system(size: 13, weight: .medium))
                                Text("(Camera Sau)")
                                    .font(.system(size: 10, weight: .regular))
                            }
                            .frame(maxWidth: .infinity)
                            .padding(.vertical, 8)
                            .foregroundColor(.white.opacity(0.6))
                        }
                    }
                    .padding(4)
                    .background(Color.white.opacity(0.08))
                    .cornerRadius(22)
                    .padding(.horizontal, 24)
                    
                    Spacer()
                    
                    // Central Circular Face ID Viewport with Cyan Silhouette
                    ZStack {
                        // 1. Live Camera Feed Masked cleanly into Circle (Diameter 260pt)
                        ARSCNViewContainer(session: captureSession.arSession)
                            .frame(width: 260, height: 260)
                            .clipShape(Circle())
                            .overlay(
                                Circle()
                                    .stroke(Color.white.opacity(0.2), lineWidth: 1.5)
                            )
                            .shadow(color: Color.cyan.opacity(0.3), radius: 20)
                        
                        // 2. Stylized Neon Face Mask Overlay inside the circle
                        FaceSilhouetteOverlay(isTracking: captureSession.isTracking)
                        
                        // 3. 36-Tick Radial Nan Quạt around the circle border
                        FaceIDRadialRing(ticks: captureSession.faceIdTicks)
                    }
                    
                    Spacer()
                    
                    // Distance Warning if held too close (< 32cm)
                    if captureSession.currentDistanceMeters < 0.32 && captureSession.isTracking {
                        Text("⚠️ Hãy giữ máy cách mặt 35 - 50 cm để mặt vừa khung")
                            .font(.system(size: 12, weight: .bold))
                            .foregroundColor(.yellow)
                            .padding(.horizontal, 14)
                            .padding(.vertical, 6)
                            .background(Color.black.opacity(0.7))
                            .cornerRadius(12)
                            .transition(.opacity)
                    }
                    
                    // Main Instruction Text
                    Text("Xoay nhẹ đầu theo vòng tròn tự nhiên")
                        .font(.system(size: 16, weight: .medium))
                        .foregroundColor(.white)
                        .padding(.top, 4)
                    
                    // Live Yaw & Pitch Telemetry Badges (Mockup Exact Match)
                    HStack(spacing: 12) {
                        Text(String(format: "Yaw %.2f°", captureSession.currentYawDeg))
                            .font(.system(size: 12, weight: .semibold))
                            .foregroundColor(.white.opacity(0.85))
                            .padding(.horizontal, 14)
                            .padding(.vertical, 7)
                            .background(Color(red: 0.12, green: 0.22, blue: 0.3).opacity(0.75))
                            .cornerRadius(14)
                        
                        Text(String(format: "Pitch %.2f°", captureSession.currentPitchDeg))
                            .font(.system(size: 12, weight: .semibold))
                            .foregroundColor(.white.opacity(0.85))
                            .padding(.horizontal, 14)
                            .padding(.vertical, 7)
                            .background(Color(red: 0.12, green: 0.22, blue: 0.3).opacity(0.75))
                            .cornerRadius(14)
                    }
                    
                    // Progress & Reset Link
                    HStack {
                        Text("Đã quét: \(captureSession.faceIdFilledCount)/36 tia")
                            .font(.system(size: 12, weight: .bold))
                            .foregroundColor(Color(red: 0.0, green: 0.95, blue: 0.6))
                        
                        Spacer()
                        
                        Button {
                            captureSession.resetScan()
                        } label: {
                            Image(systemName: "arrow.clockwise")
                                .font(.system(size: 13))
                                .foregroundColor(.white.opacity(0.6))
                        }
                    }
                    .padding(.horizontal, 36)
                    .padding(.bottom, 30)
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
                            .progressViewStyle(CircularProgressViewStyle(tint: Color(red: 0.0, green: 0.95, blue: 0.6)))
                        
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
                            .stroke(Color(red: 0.0, green: 0.95, blue: 0.6).opacity(0.5), lineWidth: 1.5)
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
