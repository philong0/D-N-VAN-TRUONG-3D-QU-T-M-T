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
    let radius: CGFloat = 145
    let tickLength: CGFloat = 16
    
    var body: some View {
        ZStack {
            ForEach(0..<totalTicks, id: \.self) { index in
                let isFilled = index < ticks.count ? ticks[index] : false
                let angle = Double(index) * (360.0 / Double(totalTicks))
                
                Capsule()
                    .fill(isFilled ? Color(red: 0.0, green: 0.95, blue: 0.6) : Color.white.opacity(0.3))
                    .frame(width: isFilled ? 3.5 : 2.5, height: isFilled ? tickLength + 4 : tickLength)
                    .shadow(color: isFilled ? Color(red: 0.0, green: 0.95, blue: 0.6).opacity(0.8) : Color.clear, radius: isFilled ? 4 : 0)
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
            
            // Central focus crosshair
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
            // 1. AR Camera Viewport (Runs TrueDepth front OR rear ARWorldTracking)
            ARSCNViewContainer(session: captureSession.arSession)
                .edgesIgnoringSafeArea(.all)
            
            // 2. Clinical HUD & Guidance Overlay
            VStack {
                // Top Header: Close, Mode Toggle, Step Status
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
                        
                        // Dual Mode Segmented Switcher
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
                                .background(captureSession.scannerMode == .faceIdSelfScan ? Color(red: 0.0, green: 0.8, blue: 0.5) : Color.clear)
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
                                .background(captureSession.scannerMode == .rearClinicalAssistant ? Color.blue : Color.clear)
                                .foregroundColor(.white)
                                .cornerRadius(12)
                            }
                        }
                        .padding(3)
                        .background(Color.white.opacity(0.15))
                        .cornerRadius(14)
                        
                        Spacer()
                        
                        Text("\(captureSession.capturedFrames.count)/5")
                            .font(.system(size: 14, weight: .black))
                            .foregroundColor(captureSession.scannerMode == .faceIdSelfScan ? Color(red: 0.0, green: 0.95, blue: 0.6) : Color.cyan)
                    }
                    .padding(.horizontal)
                    
                    // 5 Progress segments (Visual Angle Feedback)
                    HStack(spacing: 6) {
                        ForEach(ScanAngleStep.allCases) { step in
                            VStack(spacing: 2) {
                                RoundedRectangle(cornerRadius: 3)
                                    .fill(captureSession.capturedFrames[step] != nil ? Color.green : (step == captureSession.currentStep ? (captureSession.scannerMode == .faceIdSelfScan ? Color.yellow : Color.cyan) : Color.gray.opacity(0.4)))
                                    .frame(height: 5)
                                
                                Text(step.title.components(separatedBy: ":").last?.trimmingCharacters(in: .whitespaces) ?? "")
                                    .font(.system(size: 8, weight: .medium))
                                    .foregroundColor(.white.opacity(0.7))
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
                
                // Center Scanner Interface (Mode 1: Face ID Radial Sweep vs Mode 2: Rear Clinical Reticle)
                if captureSession.scannerMode == .faceIdSelfScan {
                    // MODE 1: FACE ID 36-TICK SWEEP
                    ZStack {
                        // 36-tick Apple Face ID radial sweep ring
                        FaceIDRadialRing(ticks: captureSession.faceIdTicks)
                        
                        // Inner Guide Oval
                        Ellipse()
                            .stroke(captureSession.isPoseAligned ? Color(red: 0.0, green: 0.95, blue: 0.6) : Color.white.opacity(0.45), lineWidth: captureSession.isPoseAligned ? 3 : 1.5)
                            .frame(width: 220, height: 290)
                        
                        // Center crosshair / direction marker
                        Image(systemName: "plus")
                            .font(.title3)
                            .foregroundColor(captureSession.isPoseAligned ? Color(red: 0.0, green: 0.95, blue: 0.6) : Color.white.opacity(0.35))
                        
                        // Live nan quạt progress count in center badge
                        VStack(spacing: 2) {
                            Spacer()
                            Text("\(captureSession.faceIdFilledCount)/36")
                                .font(.system(size: 13, weight: .heavy))
                                .foregroundColor(Color(red: 0.0, green: 0.95, blue: 0.6))
                                .padding(.horizontal, 10)
                                .padding(.vertical, 4)
                                .background(Color.black.opacity(0.7))
                                .cornerRadius(12)
                                .padding(.bottom, 20)
                        }
                        .frame(width: 220, height: 290)
                    }
                } else {
                    // MODE 2: REAR CAMERA CLINICAL ASSISTANT
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
                }
                
                Spacer()
                
                // Bottom Feedback & Capture Controls
                VStack(spacing: 12) {
                    // Guidance Pill
                    VStack(spacing: 4) {
                        Text(captureSession.guidanceFeedback)
                            .font(.system(size: 14, weight: .black))
                            .foregroundColor(captureSession.isPoseAligned ? (captureSession.scannerMode == .faceIdSelfScan ? Color(red: 0.0, green: 0.95, blue: 0.6) : .cyan) : .yellow)
                            .multilineTextAlignment(.center)
                        
                        if captureSession.scannerMode == .faceIdSelfScan {
                            Text("Xoay nhẹ đầu theo vòng tròn để phủ kín 36 nan quạt Face ID")
                                .font(.system(size: 11, weight: .medium))
                                .foregroundColor(.white.opacity(0.75))
                        } else {
                            Text(captureSession.currentStep.instruction)
                                .font(.system(size: 11, weight: .medium))
                                .foregroundColor(.white.opacity(0.75))
                        }
                    }
                    .padding(.horizontal, 18)
                    .padding(.vertical, 9)
                    .background(Color.black.opacity(0.85))
                    .cornerRadius(18)
                    
                    // Live Telemetry (Yaw, Distance, Pitch)
                    if captureSession.scannerMode == .faceIdSelfScan {
                        HStack(spacing: 16) {
                            HStack(spacing: 4) {
                                Image(systemName: "gyroscope")
                                    .font(.caption2)
                                Text(String(format: "Yaw %.0f°", captureSession.currentYawDeg))
                                    .font(.caption)
                                    .bold()
                            }
                            .foregroundColor(.white)
                            
                            HStack(spacing: 4) {
                                Image(systemName: "arrow.up.and.down")
                                    .font(.caption2)
                                Text(String(format: "Pitch %.0f°", captureSession.currentPitchDeg))
                                    .font(.caption)
                                    .bold()
                            }
                            .foregroundColor(.white)
                            
                            HStack(spacing: 4) {
                                Image(systemName: "ruler.fill")
                                    .font(.caption2)
                                Text(String(format: "%.2fm", captureSession.currentDistanceMeters))
                                    .font(.caption)
                                    .bold()
                            }
                            .foregroundColor(.white)
                        }
                        .padding(.horizontal, 12)
                        .padding(.vertical, 4)
                        .background(Color.black.opacity(0.6))
                        .cornerRadius(10)
                    }
                    
                    // Bottom Buttons: Retake, Shutter/Trigger, Reset
                    HStack(spacing: 18) {
                        // Retake Previous Step Button
                        if captureSession.capturedFrames.count > 0 {
                            Button {
                                captureSession.retakePreviousStep()
                            } label: {
                                Image(systemName: "arrow.uturn.backward.circle.fill")
                                    .font(.system(size: 36))
                                    .foregroundColor(.orange)
                            }
                            .accessibilityLabel("Chụp lại góc trước")
                        }
                        
                        // Big Shutter / Manual Capture Button
                        Button(action: {
                            if captureSession.capturedFrames.count >= 5 {
                                captureSession.triggerPackageUpload { _ in }
                            } else {
                                do {
                                    try captureSession.captureCurrentStep()
                                } catch {
                                    captureSession.guidanceFeedback = error.localizedDescription
                                }
                            }
                        }) {
                            HStack(spacing: 8) {
                                Image(systemName: captureSession.capturedFrames.count >= 5 ? "arrow.up.circle.fill" : (captureSession.scannerMode == .faceIdSelfScan ? "faceid" : "camera.fill"))
                                    .font(.title3)
                                Text(captureSession.isUploading ? "ĐANG TẢI LÊN..." : (captureSession.capturedFrames.count >= 5 ? "GỬI TẢI LÊN (5/5)" : (captureSession.scannerMode == .faceIdSelfScan ? "CHỤP GÓC (\(captureSession.capturedFrames.count)/5)" : "BẤM CHỤP (\(captureSession.capturedFrames.count)/5)")))
                                    .font(.system(size: 14, weight: .black))
                            }
                            .foregroundColor(.white)
                            .padding(.horizontal, 24)
                            .padding(.vertical, 14)
                            .background(captureSession.capturedFrames.count >= 5 ? Color.orange : (captureSession.scannerMode == .faceIdSelfScan ? Color(red: 0.0, green: 0.7, blue: 0.45) : Color.blue))
                            .cornerRadius(28)
                            .shadow(color: Color.black.opacity(0.5), radius: 6, x: 0, y: 3)
                        }
                        .disabled(captureSession.isUploading)
                        
                        // Reset Scan Button
                        Button {
                            captureSession.resetScan()
                        } label: {
                            Image(systemName: "arrow.clockwise.circle.fill")
                                .font(.system(size: 36))
                                .foregroundColor(.white.opacity(0.85))
                        }
                        .accessibilityLabel("Quét lại từ đầu")
                    }
                    .padding(.bottom, 25)
                }
            }
            
            // 3. Uploading & 3D Reconstruction Overlay
            if captureSession.isUploading {
                ZStack {
                    Color.black.opacity(0.88).edgesIgnoringSafeArea(.all)
                    VStack(spacing: 20) {
                        ProgressView()
                            .scaleEffect(1.8)
                            .progressViewStyle(CircularProgressViewStyle(tint: Color(red: 0.0, green: 0.95, blue: 0.6)))
                        
                        Text(captureSession.scannerMode == .rearClinicalAssistant ? "ĐANG TẢI GÓI ẢNH LÂM SÀNG 48MP" : "ĐANG TẢI DỮ LIỆU TRUEDEPTH FACE ID")
                            .font(.headline)
                            .bold()
                            .foregroundColor(.white)
                        
                        Text("Dữ liệu đang được gửi tới AI Engine để dựng hình 3D chuẩn xác thực khuôn mặt (tai, mắt, tóc, da)...")
                            .font(.footnote)
                            .foregroundColor(.white.opacity(0.8))
                            .multilineTextAlignment(.center)
                            .padding(.horizontal, 32)
                    }
                    .padding(28)
                    .background(Color.black.opacity(0.92))
                    .cornerRadius(24)
                    .overlay(
                        RoundedRectangle(cornerRadius: 24)
                            .stroke(Color(red: 0.0, green: 0.95, blue: 0.6).opacity(0.5), lineWidth: 1.5)
                    )
                }
            }
            
            // 4. Quality Gate / Upload Error Overlay
            if let errorMsg = captureSession.lastErrorMessage {
                ZStack {
                    Color.black.opacity(0.85).edgesIgnoringSafeArea(.all)
                    VStack(spacing: 16) {
                        Image(systemName: "exclamationmark.triangle.fill")
                            .font(.system(size: 44))
                            .foregroundColor(.yellow)
                        
                        Text("Thông Báo Quét")
                            .font(.title3)
                            .bold()
                            .foregroundColor(.white)
                        
                        Text(errorMsg)
                            .font(.subheadline)
                            .foregroundColor(.white.opacity(0.9))
                            .multilineTextAlignment(.center)
                            .padding(.horizontal, 20)
                        
                        Button(action: {
                            captureSession.resetScan()
                        }) {
                            HStack(spacing: 8) {
                                Image(systemName: "arrow.clockwise")
                                Text("Quét Lại Ngay")
                                    .bold()
                            }
                            .foregroundColor(.white)
                            .padding(.horizontal, 28)
                            .padding(.vertical, 14)
                            .background(Color.blue)
                            .cornerRadius(24)
                        }
                        .padding(.top, 8)
                    }
                    .padding(28)
                    .background(Color.black.opacity(0.95))
                    .cornerRadius(24)
                    .overlay(
                        RoundedRectangle(cornerRadius: 24)
                            .stroke(Color.yellow.opacity(0.5), lineWidth: 1.5)
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
