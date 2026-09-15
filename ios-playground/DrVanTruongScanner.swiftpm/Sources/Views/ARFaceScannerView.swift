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

// MARK: - Face ID 36-Tick Radial Ring
struct FaceIDRadialRing: View {
    let ticks: [Bool]
    let totalTicks: Int = 36
    let radius: CGFloat = 148
    let tickLength: CGFloat = 16
    
    var body: some View {
        ZStack {
            ForEach(0..<totalTicks, id: \.self) { index in
                let isFilled = index < ticks.count ? ticks[index] : false
                let angle = Double(index) * (360.0 / Double(totalTicks))
                
                Capsule()
                    .fill(isFilled ? Color(red: 0.19, green: 0.82, blue: 0.35) : Color.white.opacity(0.28))
                    .frame(width: isFilled ? 3.5 : 2.5, height: isFilled ? tickLength + 4 : tickLength)
                    .shadow(color: isFilled ? Color(red: 0.19, green: 0.82, blue: 0.35).opacity(0.8) : Color.clear, radius: isFilled ? 4 : 0)
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
    
    public init(captureSession: ARFaceCaptureSession, isCompleted: Binding<Bool>) {
        self.captureSession = captureSession
        self._isCompleted = isCompleted
    }
    
    public var body: some View {
        ZStack {
            Color.black.edgesIgnoringSafeArea(.all)
            
            if captureSession.scannerMode == .faceIdSelfScan {
                // ==========================================
                // MODE 1: CHUẨN 100% APPLE FACE ID SELF-SCAN
                // ==========================================
                VStack(spacing: 20) {
                    // Top Bar: Close & Mode Switcher
                    HStack {
                        Button {
                            captureSession.onScanCancelled?()
                            isCompleted = true
                        } label: {
                            Image(systemName: "xmark.circle.fill")
                                .font(.title2)
                                .foregroundColor(.white.opacity(0.75))
                        }
                        
                        Spacer()
                        
                        // Mode Switcher Pill
                        HStack(spacing: 4) {
                            Button {
                                captureSession.setScannerMode(.faceIdSelfScan)
                            } label: {
                                HStack(spacing: 4) {
                                    Image(systemName: "faceid")
                                    Text("Tự Quét (Face ID)")
                                        .font(.system(size: 11, weight: .bold))
                                }
                                .padding(.horizontal, 10)
                                .padding(.vertical, 6)
                                .background(Color(red: 0.19, green: 0.82, blue: 0.35))
                                .foregroundColor(.black)
                                .cornerRadius(12)
                            }
                            
                            Button {
                                captureSession.setScannerMode(.rearClinicalAssistant)
                            } label: {
                                HStack(spacing: 4) {
                                    Image(systemName: "camera.viewfinder")
                                    Text("Điều Dưỡng")
                                        .font(.system(size: 11, weight: .bold))
                                }
                                .padding(.horizontal, 10)
                                .padding(.vertical, 6)
                                .background(Color.white.opacity(0.1))
                                .foregroundColor(.white)
                                .cornerRadius(12)
                            }
                        }
                        .padding(3)
                        .background(Color.white.opacity(0.12))
                        .cornerRadius(15)
                        
                        Spacer()
                        
                        // Empty placeholder for symmetry
                        Color.clear.frame(width: 32, height: 32)
                    }
                    .padding(.horizontal, 20)
                    .padding(.top, 45)
                    
                    // Main Instruction Text (Apple Face ID Style)
                    VStack(spacing: 6) {
                        Text(captureSession.isTracking ? "Xoay nhẹ đầu theo vòng tròn" : "Đưa khuôn mặt vào vòng tròn")
                            .font(.system(size: 20, weight: .bold))
                            .foregroundColor(.white)
                            .multilineTextAlignment(.center)
                        
                        Text(captureSession.isTracking ? "Nghiêng đầu nhẹ mọi góc để phủ kín 36 nan quạt" : "Nhìn thẳng vào màn hình để bắt đầu")
                            .font(.system(size: 13, weight: .medium))
                            .foregroundColor(.white.opacity(0.65))
                            .multilineTextAlignment(.center)
                    }
                    .padding(.horizontal)
                    
                    Spacer()
                    
                    // Central Circular Face ID Viewport
                    ZStack {
                        // 1. Live Camera Feed Masked by Circle (Diameter 270pt)
                        ARSCNViewContainer(session: captureSession.arSession)
                            .frame(width: 270, height: 270)
                            .clipShape(Circle())
                            .overlay(
                                Circle()
                                    .stroke(captureSession.isTracking ? Color(red: 0.19, green: 0.82, blue: 0.35).opacity(0.6) : Color.white.opacity(0.2), lineWidth: 2)
                            )
                            .shadow(color: captureSession.isTracking ? Color(red: 0.19, green: 0.82, blue: 0.35).opacity(0.25) : Color.clear, radius: 16)
                        
                        // 2. 36-Tick Radial Nan Quạt around the Circle
                        FaceIDRadialRing(ticks: captureSession.faceIdTicks)
                        
                        // Subtle center guide crosshair
                        if !captureSession.isTracking {
                            Image(systemName: "face.dashed")
                                .font(.system(size: 60))
                                .foregroundColor(.white.opacity(0.35))
                        }
                    }
                    
                    Spacer()
                    
                    // Bottom Progress & Live Tick Counter
                    VStack(spacing: 12) {
                        HStack(spacing: 6) {
                            Image(systemName: "checkmark.circle.fill")
                                .foregroundColor(Color(red: 0.19, green: 0.82, blue: 0.35))
                            Text("Đã quét: \(captureSession.faceIdFilledCount)/36 tia (\(Int(Double(captureSession.faceIdFilledCount) / 36.0 * 100))%)")
                                .font(.system(size: 15, weight: .bold))
                                .foregroundColor(.white)
                        }
                        .padding(.horizontal, 16)
                        .padding(.vertical, 8)
                        .background(Color.white.opacity(0.1))
                        .cornerRadius(20)
                        
                        Button {
                            captureSession.resetScan()
                        } label: {
                            HStack(spacing: 6) {
                                Image(systemName: "arrow.clockwise")
                                Text("Quét lại vòng tròn")
                            }
                            .font(.system(size: 12, weight: .medium))
                            .foregroundColor(.white.opacity(0.6))
                        }
                    }
                    .padding(.bottom, 35)
                }
            } else {
                // ==========================================
                // MODE 2: ĐIỀU DƯỠNG QUÉT (CAMERA SAU 48MP)
                // ==========================================
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
                                        HStack(spacing: 4) {
                                            Image(systemName: "faceid")
                                            Text("Tự Quét")
                                                .font(.system(size: 11, weight: .bold))
                                        }
                                        .padding(.horizontal, 10)
                                        .padding(.vertical, 6)
                                        .background(Color.white.opacity(0.1))
                                        .foregroundColor(.white)
                                        .cornerRadius(12)
                                    }
                                    
                                    Button {
                                        captureSession.setScannerMode(.rearClinicalAssistant)
                                    } label: {
                                        HStack(spacing: 4) {
                                            Image(systemName: "camera.viewfinder")
                                            Text("Điều Dưỡng (Camera Sau)")
                                                .font(.system(size: 11, weight: .bold))
                                        }
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
                    Color.black.opacity(0.9).edgesIgnoringSafeArea(.all)
                    VStack(spacing: 20) {
                        ProgressView()
                            .scaleEffect(1.8)
                            .progressViewStyle(CircularProgressViewStyle(tint: Color(red: 0.19, green: 0.82, blue: 0.35)))
                        
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
                            .stroke(Color(red: 0.19, green: 0.82, blue: 0.35).opacity(0.5), lineWidth: 1.5)
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
