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

public struct ARFaceScannerView: View {
    @ObservedObject var captureSession: ARFaceCaptureSession
    @Binding var isCompleted: Bool
    
    public init(captureSession: ARFaceCaptureSession, isCompleted: Binding<Bool>) {
        self.captureSession = captureSession
        self._isCompleted = isCompleted
    }
    
    public var body: some View {
        ZStack {
            // 1. AR Camera Viewport
            ARSCNViewContainer(session: captureSession.arSession)
                .edgesIgnoringSafeArea(.all)
            
            // 2. Clinical HUD & Guidance Overlay
            VStack {
                // Top Header: Close, Title, Camera Switch, Progress
                VStack(spacing: 8) {
                    HStack {
                        Button {
                            captureSession.onScanCancelled?()
                            isCompleted = true
                        } label: {
                            Image(systemName: "xmark.circle.fill")
                                .font(.title2)
                                .foregroundColor(.white.opacity(0.85))
                        }
                        
                        Text(captureSession.currentStep.title)
                            .font(.system(size: 16, weight: .bold))
                            .foregroundColor(.white)
                        
                        Spacer()
                        
                        // Switch Front / Back Camera Button
                        Button {
                            captureSession.switchCamera()
                        } label: {
                            HStack(spacing: 4) {
                                Image(systemName: "camera.rotate.fill")
                                    .font(.system(size: 14, weight: .semibold))
                                Text(captureSession.cameraPosition == .front ? "Cam Trước" : "Cam Sau")
                                    .font(.system(size: 12, weight: .bold))
                            }
                            .foregroundColor(.white)
                            .padding(.horizontal, 10)
                            .padding(.vertical, 6)
                            .background(Color.blue.opacity(0.8))
                            .cornerRadius(12)
                        }
                        
                        Text("\(captureSession.capturedFrames.count)/5")
                            .font(.subheadline)
                            .bold()
                            .foregroundColor(.green)
                            .padding(.leading, 4)
                    }
                    .padding(.horizontal)
                    
                    // 5 Progress segments
                    HStack(spacing: 6) {
                        ForEach(ScanAngleStep.allCases) { step in
                            RoundedRectangle(cornerRadius: 3)
                                .fill(captureSession.capturedFrames[step] != nil ? Color.green : (step == captureSession.currentStep ? Color.yellow : Color.gray.opacity(0.4)))
                                .frame(height: 6)
                        }
                    }
                    .padding(.horizontal)
                }
                .padding(.vertical, 12)
                .background(Color.black.opacity(0.75))
                .cornerRadius(16)
                .padding(.horizontal)
                .padding(.top, 40)
                
                Spacer()
                
                // Center Face Reticle Oval with Hold Progress Ring
                ZStack {
                    // Outer guide oval
                    Ellipse()
                        .stroke(captureSession.isPoseAligned ? Color.green : Color.white.opacity(0.5), lineWidth: captureSession.isPoseAligned ? 4 : 2)
                        .frame(width: 260, height: 350)
                    
                    // Circular hold progress indicator when aligned
                    if captureSession.isPoseAligned && captureSession.holdProgress > 0.0 {
                        Circle()
                            .trim(from: 0.0, to: CGFloat(captureSession.holdProgress))
                            .stroke(Color.green, style: StrokeStyle(lineWidth: 6, lineCap: .round))
                            .frame(width: 90, height: 90)
                            .rotationEffect(.degrees(-90))
                            .animation(.linear(duration: 0.1), value: captureSession.holdProgress)
                    }
                    
                    // Crosshair Alignment Mark
                    Image(systemName: "plus")
                        .font(.title2)
                        .foregroundColor(captureSession.isPoseAligned ? .green : .white.opacity(0.4))
                }
                
                Spacer()
                
                // Bottom Feedback and Capture Controls
                VStack(spacing: 14) {
                    // Guidance Pill
                    VStack(spacing: 4) {
                        Text(captureSession.guidanceFeedback)
                            .font(.system(size: 15, weight: .black))
                            .foregroundColor(captureSession.isPoseAligned ? .green : .yellow)
                            .multilineTextAlignment(.center)
                        
                        if !captureSession.isPoseAligned && captureSession.isTracking {
                            Text(captureSession.currentStep.instruction)
                                .font(.system(size: 12, weight: .medium))
                                .foregroundColor(.white.opacity(0.8))
                        }
                    }
                    .padding(.horizontal, 20)
                    .padding(.vertical, 10)
                    .background(Color.black.opacity(0.85))
                    .cornerRadius(20)
                    
                    // Angle & Distance Telemetry
                    HStack(spacing: 24) {
                        HStack(spacing: 4) {
                            Image(systemName: "gyroscope")
                                .font(.caption)
                            Text(String(format: "Góc: %.1f°", captureSession.currentYawDeg))
                                .font(.caption)
                                .bold()
                        }
                        .foregroundColor(.white)
                        
                        HStack(spacing: 4) {
                            Image(systemName: "ruler.fill")
                                .font(.caption)
                            Text(String(format: "Cự ly: %.2fm", captureSession.currentDistanceMeters))
                                .font(.caption)
                                .bold()
                        }
                        .foregroundColor(.white)
                    }
                    
                    // Bottom Buttons Row: Retake, Big Shutter, Reset
                    HStack(spacing: 16) {
                        // Retake Previous Step Button
                        if captureSession.capturedFrames.count > 0 {
                            Button {
                                captureSession.retakePreviousStep()
                            } label: {
                                Image(systemName: "arrow.uturn.backward.circle.fill")
                                    .font(.system(size: 32))
                                    .foregroundColor(.orange)
                            }
                            .accessibilityLabel("Chụp lại góc trước")
                        }
                        
                        // Big Shutter Button
                        Button(action: {
                            do {
                                try captureSession.captureCurrentStep()
                                if captureSession.capturedFrames.count >= 5 {
                                    captureSession.triggerPackageUpload { _ in
                                        isCompleted = true
                                    }
                                }
                            } catch {
                                print("Capture error:", error)
                            }
                        }) {
                            HStack(spacing: 8) {
                                Image(systemName: "camera.fill")
                                    .font(.title3)
                                Text(captureSession.capturedFrames.count >= 5 ? "HOÀN TẤT & TẢI LÊN" : "BẤM CHỤP (\(captureSession.capturedFrames.count)/5)")
                                    .font(.system(size: 15, weight: .black))
                            }
                            .foregroundColor(.white)
                            .padding(.horizontal, 26)
                            .padding(.vertical, 15)
                            .background(captureSession.isPoseAligned ? Color.green : Color.blue)
                            .cornerRadius(30)
                            .shadow(color: Color.black.opacity(0.5), radius: 8, x: 0, y: 4)
                        }
                        .disabled(!captureSession.isTracking || captureSession.isUploading)
                    }
                    .padding(.bottom, 25)
                }
            }
            
            // 3. Uploading & 3D Reconstruction Overlay
            if captureSession.isUploading {
                ZStack {
                    Color.black.opacity(0.85).edgesIgnoringSafeArea(.all)
                    VStack(spacing: 20) {
                        ProgressView()
                            .scaleEffect(1.8)
                            .progressViewStyle(CircularProgressViewStyle(tint: .green))
                        
                        Text("ĐANG TẢI DỮ LIỆU TRUEDEPTH 3D")
                            .font(.headline)
                            .bold()
                            .foregroundColor(.white)
                        
                        Text("Hệ thống đang nén gói dữ liệu LiDAR và gửi lên AI Engine để dựng hình...")
                            .font(.footnote)
                            .foregroundColor(.white.opacity(0.8))
                            .multilineTextAlignment(.center)
                            .padding(.horizontal, 32)
                    }
                    .padding(28)
                    .background(Color.black.opacity(0.9))
                    .cornerRadius(24)
                    .overlay(
                        RoundedRectangle(cornerRadius: 24)
                            .stroke(Color.green.opacity(0.5), lineWidth: 1.5)
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
