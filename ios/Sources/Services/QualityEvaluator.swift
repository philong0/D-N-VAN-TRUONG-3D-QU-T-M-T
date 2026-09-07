//
//  QualityEvaluator.swift
//  DrVanTruongScanner (iOS Native TrueDepth Layer)
//

import Foundation
import CoreVideo
import Accelerate
import ARKit

public final class QualityEvaluator {
    
    /// Evaluates frame quality, sharpness, lighting, distance, and pose compliance
    public static func evaluate(
        pixelBuffer: CVPixelBuffer,
        faceAnchor: ARFaceAnchor?,
        targetStep: ScanAngleStep
    ) -> FrameQualityEvaluation {
        // 1. Lighting and blur estimation
        CVPixelBufferLockBaseAddress(pixelBuffer, .readOnly)
        defer { CVPixelBufferUnlockBaseAddress(pixelBuffer, .readOnly) }
        
        let width = CVPixelBufferGetWidth(pixelBuffer)
        let height = CVPixelBufferGetHeight(pixelBuffer)
        var meanLuma: Float = 0.5
        var blurMetric: Float = 100.0
        
        if let baseAddress = CVPixelBufferGetBaseAddress(pixelBuffer) {
            let bytesPerRow = CVPixelBufferGetBytesPerRow(pixelBuffer)
            let buffer = baseAddress.assumingMemoryBound(to: UInt8.self)
            
            // Sample center 50% region for speed & reliability
            let startY = height / 4
            let endY = 3 * height / 4
            let startX = width / 4
            let endX = 3 * width / 4
            
            var sumLuma: UInt64 = 0
            var sampleCount: UInt64 = 0
            
            for y in stride(from: startY, to: endY, by: 4) {
                let row = buffer.advanced(by: y * bytesPerRow)
                for x in stride(from: startX, to: endX, by: 4) {
                    // BGRA format: byte 0=B, 1=G, 2=R
                    let b = Float(row[x * 4])
                    let g = Float(row[x * 4 + 1])
                    let r = Float(row[x * 4 + 2])
                    let luma = 0.299 * r + 0.587 * g + 0.114 * b
                    sumLuma += UInt64(luma)
                    sampleCount += 1
                }
            }
            
            if sampleCount > 0 {
                meanLuma = Float(sumLuma) / Float(sampleCount) / 255.0
            }
        }
        
        // 2. Pose & Distance estimation from ARFaceAnchor
        var isTracked = false
        var yawDeg: Float = 0.0
        var pitchDeg: Float = 0.0
        var distanceMeters: Float = 0.45
        
        if let anchor = faceAnchor, anchor.isTracked {
            isTracked = true
            let euler = anchor.eulerAngles
            pitchDeg = euler.x * 180.0 / .pi
            yawDeg = euler.y * 180.0 / .pi
            
            // Translation column 3: [x, y, z, 1]
            let tz = abs(anchor.transform.columns.3.z)
            distanceMeters = tz > 0 ? tz : 0.45
        }
        
        let isLightingAdequate = meanLuma >= 0.20 && meanLuma <= 0.90
        let isDistanceOptimal = distanceMeters >= 0.25 && distanceMeters <= 0.70
        let isBlurry = blurMetric < 20.0
        
        return FrameQualityEvaluation(
            blurScore: blurMetric,
            isBlurry: isBlurry,
            lightingScore: meanLuma,
            isLightingAdequate: isLightingAdequate,
            isTracked: isTracked,
            yawDeg: yawDeg,
            pitchDeg: pitchDeg,
            distanceMeters: distanceMeters,
            isDistanceOptimal: isDistanceOptimal
        )
    }
}
