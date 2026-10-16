// Encodes a numbered PNG sequence (frame_0000.png, frame_0001.png, ...) to an H.264 .mp4 with the system's AVFoundation.
// Usage: encode <frames directory> <output.mp4> <frames per second>
import AVFoundation
import AppKit
import CoreVideo
import Foundation

let arguments = CommandLine.arguments
guard arguments.count == 4, let fps = Int32(arguments[3]), fps > 0 else {
    FileHandle.standardError.write("usage: encode <frames directory> <output.mp4> <fps>\n".data(using: .utf8)!)
    exit(2)
}
let directory = URL(fileURLWithPath: arguments[1])
let output = URL(fileURLWithPath: arguments[2])
let files = try FileManager.default.contentsOfDirectory(at: directory, includingPropertiesForKeys: nil)
    .filter { $0.pathExtension == "png" }
    .sorted { $0.lastPathComponent < $1.lastPathComponent }
guard let first = files.first, let firstImage = NSImage(contentsOf: first), let firstCG = firstImage.cgImage(forProposedRect: nil, context: nil, hints: nil) else {
    FileHandle.standardError.write("no frames found\n".data(using: .utf8)!)
    exit(1)
}
let width = firstCG.width
let height = firstCG.height
try? FileManager.default.removeItem(at: output)
let writer = try AVAssetWriter(outputURL: output, fileType: .mp4)
let settings: [String: Any] = [
    AVVideoCodecKey: AVVideoCodecType.h264,
    AVVideoWidthKey: width,
    AVVideoHeightKey: height,
    AVVideoCompressionPropertiesKey: [AVVideoAverageBitRateKey: 4_500_000, AVVideoProfileLevelKey: AVVideoProfileLevelH264HighAutoLevel],
]
let input = AVAssetWriterInput(mediaType: .video, outputSettings: settings)
let adaptor = AVAssetWriterInputPixelBufferAdaptor(
    assetWriterInput: input,
    sourcePixelBufferAttributes: [kCVPixelBufferPixelFormatTypeKey as String: kCVPixelFormatType_32ARGB, kCVPixelBufferWidthKey as String: width, kCVPixelBufferHeightKey as String: height]
)
writer.add(input)
writer.startWriting()
writer.startSession(atSourceTime: .zero)

func pixelBuffer(for url: URL) -> CVPixelBuffer? {
    guard let image = NSImage(contentsOf: url), let cg = image.cgImage(forProposedRect: nil, context: nil, hints: nil) else { return nil }
    var buffer: CVPixelBuffer?
    CVPixelBufferCreate(kCFAllocatorDefault, width, height, kCVPixelFormatType_32ARGB, nil, &buffer)
    guard let pixels = buffer else { return nil }
    CVPixelBufferLockBaseAddress(pixels, [])
    let context = CGContext(
        data: CVPixelBufferGetBaseAddress(pixels), width: width, height: height, bitsPerComponent: 8, bytesPerRow: CVPixelBufferGetBytesPerRow(pixels),
        space: CGColorSpaceCreateDeviceRGB(), bitmapInfo: CGImageAlphaInfo.noneSkipFirst.rawValue
    )
    context?.draw(cg, in: CGRect(x: 0, y: 0, width: width, height: height))
    CVPixelBufferUnlockBaseAddress(pixels, [])
    return pixels
}

for (index, file) in files.enumerated() {
    while !input.isReadyForMoreMediaData { Thread.sleep(forTimeInterval: 0.005) }
    guard let buffer = pixelBuffer(for: file) else {
        FileHandle.standardError.write("cannot read \(file.lastPathComponent)\n".data(using: .utf8)!)
        exit(1)
    }
    adaptor.append(buffer, withPresentationTime: CMTime(value: Int64(index), timescale: fps))
}
input.markAsFinished()
let done = DispatchSemaphore(value: 0)
writer.finishWriting { done.signal() }
done.wait()
if writer.status != .completed {
    FileHandle.standardError.write("encoding failed: \(String(describing: writer.error))\n".data(using: .utf8)!)
    exit(1)
}
print("wrote \(output.path): \(files.count) frames, \(width)x\(height), \(fps) fps")
