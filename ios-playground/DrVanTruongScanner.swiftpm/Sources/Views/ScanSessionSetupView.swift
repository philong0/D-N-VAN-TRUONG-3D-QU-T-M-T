//
//  ScanSessionSetupView.swift
//  DrVanTruongScanner (iOS Native TrueDepth Layer)
//

import SwiftUI

public struct ScanSessionSetupView: View {
    @ObservedObject var apiClient: BackendAPIClient
    @Binding var patientId: String
    @Binding var sessionId: String
    @Binding var isScanningActive: Bool

    @State private var hasTrueDepth = ARFaceCaptureSession().isTrueDepthSupported
    @State private var patients: [PatientSummaryDTO] = []
    @State private var isLoadingPatients = false
    @State private var loadError: String? = nil
    @State private var isCreatingSession = false
    @State private var createError: String? = nil
    @State private var showManualEntry = false

    public init(apiClient: BackendAPIClient, patientId: Binding<String>, sessionId: Binding<String>, isScanningActive: Binding<Bool>) {
        self.apiClient = apiClient
        self._patientId = patientId
        self._sessionId = sessionId
        self._isScanningActive = isScanningActive
    }

    public var body: some View {
        NavigationView {
            Form {
                Section(header: Text("Trạng Thái Thiết Bị")) {
                    HStack {
                        Image(systemName: hasTrueDepth ? "checkmark.circle.fill" : "exclamationmark.triangle.fill")
                            .foregroundColor(hasTrueDepth ? .green : .red)
                        Text(hasTrueDepth ? "TrueDepth Camera Sẵn Sàng (60 FPS)" : "Thiết bị không hỗ trợ TrueDepth")
                            .font(.subheadline)
                    }
                }

                Section(header: Text("Máy Chủ Dr. Văn Trương Studio")) {
                    TextField("Địa chỉ máy chủ (Host:Port)", text: $apiClient.serverBaseURL)
                        .autocapitalization(.none)
                        .disableAutocorrection(true)
                    Button(action: loadPatients) {
                        HStack {
                            if isLoadingPatients {
                                ProgressView().padding(.trailing, 4)
                            }
                            Text(isLoadingPatients ? "Đang tải danh sách..." : "Tải Danh Sách Bệnh Nhân")
                        }
                    }
                    .disabled(isLoadingPatients)
                    if let loadError = loadError {
                        Text(loadError).font(.caption).foregroundColor(.red)
                    }
                }

                if !patients.isEmpty {
                    Section(header: Text("Chọn Bệnh Nhân (\(patients.count))")) {
                        ForEach(patients) { patient in
                            Button(action: { selectPatient(patient) }) {
                                HStack {
                                    VStack(alignment: .leading) {
                                        Text(patient.fullName).font(.body).foregroundColor(.primary)
                                        Text(patient.phone).font(.caption).foregroundColor(.secondary)
                                    }
                                    Spacer()
                                    if patientId == patient.id {
                                        Image(systemName: "checkmark.circle.fill").foregroundColor(.blue)
                                    }
                                    if isCreatingSession && patientId == patient.id {
                                        ProgressView()
                                    }
                                }
                            }
                            .disabled(isCreatingSession)
                        }
                    }
                }

                if let createError = createError {
                    Section {
                        Text(createError).font(.caption).foregroundColor(.red)
                    }
                }

                Section {
                    Button(action: { showManualEntry.toggle() }) {
                        Text(showManualEntry ? "Ẩn Nhập Mã Thủ Công" : "Nhập Mã Thủ Công (Nâng Cao)")
                            .font(.caption)
                    }
                    if showManualEntry {
                        TextField("Mã Bệnh Nhân (Patient UUID)", text: $patientId)
                            .autocapitalization(.none)
                            .disableAutocorrection(true)
                        TextField("Mã Phiên Quét (Session UUID)", text: $sessionId)
                            .autocapitalization(.none)
                            .disableAutocorrection(true)
                        Button(action: { isScanningActive = true }) {
                            HStack {
                                Spacer()
                                Text("BẮT ĐẦU QUÉT KHUÔN MẶT 3D")
                                    .bold()
                                    .foregroundColor(.white)
                                Spacer()
                            }
                            .padding(.vertical, 8)
                        }
                        .listRowBackground(patientId.isEmpty || sessionId.isEmpty ? Color.gray : Color.blue)
                        .disabled(patientId.isEmpty || sessionId.isEmpty)
                    }
                }
            }
            .navigationTitle("Dr. Văn Trương 3D")
            .onAppear {
                if patients.isEmpty { loadPatients() }
            }
        }
    }

    private func loadPatients() {
        isLoadingPatients = true
        loadError = nil
        apiClient.fetchPatients { result in
            isLoadingPatients = false
            switch result {
            case .success(let list):
                patients = list
            case .failure(let error):
                loadError = "Không tải được danh sách bệnh nhân: \(error.localizedDescription)"
            }
        }
    }

    private func selectPatient(_ patient: PatientSummaryDTO) {
        patientId = patient.id
        createError = nil
        isCreatingSession = true
        apiClient.createScanSession(patientId: patient.id) { result in
            isCreatingSession = false
            switch result {
            case .success(let newSessionId):
                sessionId = newSessionId
                isScanningActive = true
            case .failure(let error):
                createError = "Không tạo được phiên quét cho \(patient.fullName): \(error.localizedDescription)"
            }
        }
    }
}
