//
//  ScanSummaryUploadView.swift
//  DrVanTruongScanner (iOS Native TrueDepth Layer)
//

import SwiftUI

public struct ScanSummaryUploadView: View {
    @ObservedObject var captureSession: ARFaceCaptureSession
    @ObservedObject var apiClient: BackendAPIClient
    let patientId: String
    let sessionId: String
    let onReset: () -> Void
    
    public var body: some View {
        NavigationView {
            VStack(spacing: 20) {
                // Header status
                VStack(spacing: 6) {
                    Image(systemName: "checkmark.seal.fill")
                        .font(.system(size: 48))
                        .foregroundColor(.green)
                    Text("Đã Thu Thập Đủ 5 Góc Y Khoa")
                        .font(.title2)
                        .bold()
                    Text("Dữ liệu ARFaceGeometry 1220 đỉnh + Độ sâu TrueDepth đã được ghi nhận.")
                        .font(.caption)
                        .foregroundColor(.secondary)
                        .multilineTextAlignment(.center)
                }
                .padding(.top)
                
                // 5 Angles Thumbnail Gallery
                ScrollView(.horizontal, showsIndicators: false) {
                    HStack(spacing: 12) {
                        ForEach(ScanAngleStep.allCases) { step in
                            VStack {
                                if let frame = captureSession.capturedFrames[step],
                                   let uiImage = UIImage(data: frame.rgbData) {
                                    Image(uiImage: uiImage)
                                        .resizable()
                                        .scaledToFill()
                                        .frame(width: 80, height: 110)
                                        .cornerRadius(12)
                                        .clipped()
                                } else {
                                    RoundedRectangle(cornerRadius: 12)
                                        .fill(Color.gray.opacity(0.2))
                                        .frame(width: 80, height: 110)
                                }
                                Text(step.title)
                                    .font(.system(size: 10, weight: .bold))
                                    .lineLimit(1)
                            }
                        }
                    }
                    .padding(.horizontal)
                }
                
                // Upload Progress & Messages
                if apiClient.isUploading || apiClient.studioURL != nil {
                    VStack(spacing: 8) {
                        ProgressView(value: apiClient.uploadProgress)
                            .progressViewStyle(LinearProgressViewStyle())
                            .padding(.horizontal)
                        Text(apiClient.uploadStatusMessage)
                            .font(.caption)
                            .bold()
                            .foregroundColor(.blue)
                    }
                    .padding()
                    .background(Color.blue.opacity(0.1))
                    .cornerRadius(16)
                    .padding(.horizontal)
                }
                
                Spacer()
                
                // Action Buttons
                VStack(spacing: 12) {
                    if let studio = apiClient.studioURL {
                        Link(destination: studio) {
                            HStack {
                                Image(systemName: "cube.transparent.fill")
                                Text("MỞ 3D STUDIO TƯ VẤN VIP →")
                                    .bold()
                            }
                            .frame(maxWidth: .infinity)
                            .padding()
                            .background(Color.green)
                            .foregroundColor(.white)
                            .cornerRadius(16)
                        }
                        .padding(.horizontal)
                    } else {
                        Button(action: {
                            apiClient.uploadScanPackage(
                                patientId: patientId,
                                sessionId: sessionId,
                                frames: captureSession.capturedFrames
                            ) { result in
                                print("Upload result:", result)
                            }
                        }) {
                            HStack {
                                Image(systemName: "arrow.up.doc.fill")
                                Text(apiClient.isUploading ? "ĐANG TẢI VÀ TÁI TẠO 3D..." : "TẢI LÊN & TÁI TẠO BASELINE 3D")
                                    .bold()
                            }
                            .frame(maxWidth: .infinity)
                            .padding()
                            .background(apiClient.isUploading ? Color.gray : Color.blue)
                            .foregroundColor(.white)
                            .cornerRadius(16)
                        }
                        .disabled(apiClient.isUploading)
                        .padding(.horizontal)
                    }
                    
                    Button(action: onReset) {
                        Text("Quét lại từ đầu")
                            .font(.subheadline)
                            .foregroundColor(.secondary)
                    }
                }
                .padding(.bottom, 20)
            }
            .navigationTitle("Kết Quả Quét")
            .navigationBarTitleDisplayMode(.inline)
        }
    }
}
