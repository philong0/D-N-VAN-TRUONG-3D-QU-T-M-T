//
//  DepthDataProcessor.swift
//  DrVanTruong3DStudio (iOS Native TrueDepth Layer)
//
//  Extracts and normalizes calibrated Float32 millimeter depth data from AVDepthData / CVPixelBuffer.
//

import Foundation
import AVFoundation
import CoreVideo

public final class DepthDataProcessor {
    
    public struct DepthMapOutput {
        public let width: Int
        public let height: Int
        public let rawData: Data // Float32 array of meters
        public let minDepthMm: Float
        public let maxDepthMm: Float
        public let validPointCount: Int
    }
    
    /// Converts AVDepthData to calibrated Float32 metre buffer. ARKit and
    /// AVFoundation report DepthFloat32 in metres; retaining those units
    /// prevents a 1,000× reconstruction-scale error downstream.
    public static func processDepthData(_ depthData: AVDepthData) -> DepthMapOutput? {
        // Convert to depth in meters (Float32)
        let convertedDepth: AVDepthData
        if depthData.depthDataType != kCVPixelFormatType_DepthFloat32 {
            convertedDepth = depthData.converting(toDepthDataType: kCVPixelFormatType_DepthFloat32)
        } else {
            convertedDepth = depthData
        }
        
        let depthPixelBuffer = convertedDepth.depthDataMap
        CVPixelBufferLockBaseAddress(depthPixelBuffer, .readOnly)
        defer { CVPixelBufferUnlockBaseAddress(depthPixelBuffer, .readOnly) }
        
        guard let baseAddress = CVPixelBufferGetBaseAddress(depthPixelBuffer) else {
            return nil
        }
        
        let width = CVPixelBufferGetWidth(depthPixelBuffer)
        let height = CVPixelBufferGetHeight(depthPixelBuffer)
        let bytesPerRow = CVPixelBufferGetBytesPerRow(depthPixelBuffer)
        let floatBuffer = baseAddress.assumingMemoryBound(to: Float32.self)
        
        var outputData = Data(count: width * height * MemoryLayout<Float32>.size)
        var minD: Float = Float.greatestFiniteMagnitude
        var maxD: Float = 0.0
        var validPoints = 0
        
        outputData.withUnsafeMutableBytes { outPtr in
            guard let outFloats = outPtr.baseAddress?.assumingMemoryBound(to: Float32.self) else { return }
            
            for y in 0..<height {
                let rowStart = baseAddress.advanced(by: y * bytesPerRow).assumingMemoryBound(to: Float32.self)
                for x in 0..<width {
                    let rawVal = rowStart[x]
                    let idx = y * width + x
                    
                    // Filter out invalid/NaN depth values
                    if rawVal.isFinite && rawVal > 0.15 && rawVal < 1.2 { // Valid face range: 15cm - 120cm
                        outFloats[idx] = rawVal
                        if rawVal < minD { minD = rawVal }
                        if rawVal > maxD { maxD = rawVal }
                        validPoints += 1
                    } else {
                        outFloats[idx] = 0.0 // 0 marks invalid/unmeasured background
                    }
                }
            }
        }
        
        return DepthMapOutput(
            width: width,
            height: height,
            rawData: outputData,
            minDepthMm: minD == Float.greatestFiniteMagnitude ? 0 : minD * 1000.0,
            maxDepthMm: maxD * 1000.0,
            validPointCount: validPoints
        )
    }
}
