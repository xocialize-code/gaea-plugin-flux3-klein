import CoreGraphics
import Foundation
import ImageIO
func rgba(_ path: String) -> (Int, Int, [UInt8]) {
    let src = CGImageSourceCreateWithURL(URL(fileURLWithPath: path) as CFURL, nil)!
    let img = CGImageSourceCreateImageAtIndex(src, 0, nil)!
    let w = img.width, h = img.height
    var buf = [UInt8](repeating: 0, count: w * h * 4)
    let ctx = CGContext(data: &buf, width: w, height: h, bitsPerComponent: 8, bytesPerRow: w * 4,
                        space: CGColorSpace(name: CGColorSpace.sRGB)!, bitmapInfo: CGImageAlphaInfo.premultipliedLast.rawValue)!
    ctx.draw(img, in: CGRect(x: 0, y: 0, width: w, height: h))
    return (w, h, buf)
}
let a = rgba(CommandLine.arguments[1]), b = rgba(CommandLine.arguments[2])
guard a.0 == b.0, a.1 == b.1 else { print("size differs", a.0, a.1, b.0, b.1); exit(1) }
var maxd = 0, diff = 0
for i in 0..<a.2.count { let d = abs(Int(a.2[i]) - Int(b.2[i])); if d > 0 { diff += 1 }; maxd = max(maxd, d) }
print("\(a.0)x\(a.1): \(diff) of \(a.2.count) samples differ, max \(maxd)")
exit(diff == 0 ? 0 : 1)
