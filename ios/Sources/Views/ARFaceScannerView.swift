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
                // Top Progress Bar
                VStack(spacing: 8) {
                    HStack {
                        Text(captureSession.currentStep.title)
                            .font(.headline)
                            .foregroundColor(.white)
                        Spacer()
                        Text("\(captureSession.capturedFrames.count) / 5 Đạt")
                            .font(.subheadline)
                            .bold()
                            .foregroundColor(.green)
                    }
                    .padding(.horizontal)
                    
                    HStack(spacing: 6) {
                        ForEach(ScanAngleStep.allCases) { step in
                            RoundedRectangle(cornerRadius: 3)
                                .fill(captureSession.capturedFrames[step] != nil ? Color.green : (step == captureSession.currentStep ? Color.blue : Color.gray.opacity(0.4)))
                                .frame(height: 6)
                        }
                    }
                    .padding(.horizontal)
                }
                .padding(.vertical, 12)
                .background(Color.black.opacity(0.65))
                .cornerRadius(16)
                .padding(.horizontal)
                .padding(.top, 40)
                
                Spacer()
                
                // Center Face Reticle Oval
                ZStack {
                    Ellipse()
                        .stroke(captureSession.isPoseAligned ? Color.green : Color.white.opacity(0.6), lineWidth: captureSession.isPoseAligned ? 4 : 2)
                        .frame(width: 260, height: 350)
                    
                    // Crosshair Alignment Mark
                    Image(systemName: "plus")
                        .font(.title2)
                        .foregroundColor(captureSession.isPoseAligned ? .green : .white.opacity(0.4))
                }
                
                Spacer()
                
                // Bottom Feedback and Capture Controls
                VStack(spacing: 16) {
                    // Guidance Pill
                    Text(captureSession.guidanceFeedback)
                        .font(.system(size: 15, weight: .bold))
                        .foregroundColor(captureSession.isPoseAligned ? .green : .yellow)
                        .multilineTextAlignment(.center)
                        .padding(.horizontal, 20)
                        .padding(.vertical, 10)
                        .background(Color.black.opacity(0.75))
                        .cornerRadius(20)
                    
                    // Angle Metrics
                    HStack(spacing: 20) {
                        Text(String(format: "Góc: %.1f°", captureSession.currentYawDeg))
                            .font(.caption)
                            .bold()
                            .foregroundColor(.white)
                        Text(String(format: "Cự ly: %.2fm", captureSession.currentDistanceMeters))
                            .font(.caption)
                            .bold()
                            .foregroundColor(.white)
                    }
                    
                    // Big Shutter Button
                    Button(action: {
                        do {
                            try captureSession.captureCurrentStep()
                            if captureSession.capturedFrames.count >= 5 {
                                isCompleted = true
                            }
                        } catch {
                            print("Capture error:", error)
                        }
                    }) {
                        HStack(spacing: 8) {
                            Image(systemName: "camera.fill")
                                .font(.title3)
                            Text("BẤM CHỤP GÓC NÀY (\(captureSession.capturedFrames.count)/5)")
                                .font(.system(size: 14, weight: .black))
                        }
                        .foregroundColor(.white)
                        .padding(.horizontal, 24)
                        .padding(.vertical, 14)
                        .background(captureSession.isTracking ? Color.green : Color.gray)
                        .cornerRadius(30)
                        .shadow(color: Color.black.opacity(0.4), radius: 8, x: 0, y: 4)
                    }
                    .disabled(!captureSession.isTracking)
                    .padding(.bottom, 25)
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
