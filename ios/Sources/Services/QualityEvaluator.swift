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
        
        let isPlanar = CVPixelBufferIsPlanar(pixelBuffer)
        let width = isPlanar ? CVPixelBufferGetWidthOfPlane(pixelBuffer, 0) : CVPixelBufferGetWidth(pixelBuffer)
        let height = isPlanar ? CVPixelBufferGetHeightOfPlane(pixelBuffer, 0) : CVPixelBufferGetHeight(pixelBuffer)
        var meanLuma: Float = 0.5
        var blurMetric: Float = 0.0
        
        let baseAddress = isPlanar ? CVPixelBufferGetBaseAddressOfPlane(pixelBuffer, 0) : CVPixelBufferGetBaseAddress(pixelBuffer)
        let bytesPerRow = isPlanar ? CVPixelBufferGetBytesPerRowOfPlane(pixelBuffer, 0) : CVPixelBufferGetBytesPerRow(pixelBuffer)
        if let baseAddress {
            let buffer = baseAddress.assumingMemoryBound(to: UInt8.self)
            
            // Sample center 50% region for speed & reliability
            let startY = height / 4
            let endY = 3 * height / 4
            let startX = width / 4
            let endX = 3 * width / 4
            
            var sumLuma: UInt64 = 0
            var sampleCount: UInt64 = 0
            var laplacianEnergy: Double = 0
            var laplacianCount: UInt64 = 0
            
            for y in stride(from: startY, to: endY, by: 4) {
                let row = buffer.advanced(by: y * bytesPerRow)
                for x in stride(from: startX, to: endX, by: 4) {
                    let luma: Float
                    if isPlanar {
                        luma = Float(row[x])
                    } else {
                        // BGRA format: byte 0=B, 1=G, 2=R
                        let b = Float(row[x * 4])
                        let g = Float(row[x * 4 + 1])
                        let r = Float(row[x * 4 + 2])
                        luma = 0.299 * r + 0.587 * g + 0.114 * b
                    }
                    sumLuma += UInt64(luma)
                    sampleCount += 1

                    if x > startX && x + 1 < endX && y > startY && y + 1 < endY && isPlanar {
                        let center = Int(row[x])
                        let left = Int(row[x - 1])
                        let right = Int(row[x + 1])
                        let up = Int(buffer[(y - 1) * bytesPerRow + x])
                        let down = Int(buffer[(y + 1) * bytesPerRow + x])
                        let lap = 4 * center - left - right - up - down
                        laplacianEnergy += Double(lap * lap)
                        laplacianCount += 1
                    }
                }
            }
            
            if sampleCount > 0 {
                meanLuma = Float(sumLuma) / Float(sampleCount) / 255.0
            }
            if laplacianCount > 0 {
                blurMetric = Float(laplacianEnergy / Double(laplacianCount))
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
        // Variance of Laplacian is calculated from the actual camera luma
        // plane; a constant placeholder would let motion-blurred RGB enter
        // an otherwise accurate TrueDepth surface.
        let isBlurry = blurMetric < 45.0
        
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
