#!/usr/bin/swift
import CoreGraphics
import CoreText
import Foundation
import ImageIO

let args = CommandLine.arguments
guard args.count >= 4 else {
    print("Usage: text2png <text> <fontSize> <output.png> [color=white|gold|pink]")
    exit(1)
}

let text = args[1]
let fontSize = CGFloat(Double(args[2]) ?? 24)
let outputPath = args[3]
let colorStr = args.count > 4 ? args[4] : "white"

let color: CGColor
switch colorStr {
case "gold": color = CGColor(red: 1.0, green: 0.85, blue: 0.3, alpha: 1.0)
case "white": color = CGColor(red: 1.0, green: 1.0, blue: 1.0, alpha: 1.0)
case "pink": color = CGColor(red: 1.0, green: 0.4, blue: 0.6, alpha: 1.0)
default:     color = CGColor(red: 1.0, green: 1.0, blue: 1.0, alpha: 1.0)
}

// Create attributed string using CoreText keys
let font = CTFontCreateWithName("PingFangSC-Semibold" as CFString, fontSize, nil)
let attrs: [CFString: Any] = [
    kCTFontAttributeName: font,
    kCTForegroundColorAttributeName: color,
]
let attrStr = CFAttributedStringCreate(nil, text as CFString, attrs as CFDictionary)!

// Measure text
let line = CTLineCreateWithAttributedString(attrStr)
let bounds = CTLineGetImageBounds(line, nil)
let width = Int(ceil(bounds.width) + 60)
let height = Int(ceil(bounds.height + abs(bounds.origin.y)) + 30)

// Create bitmap
let colorSpace = CGColorSpace(name: CGColorSpace.sRGB)!
let ctx = CGContext(
    data: nil,
    width: width, height: height,
    bitsPerComponent: 8, bytesPerRow: width * 4,
    space: colorSpace,
    bitmapInfo: CGImageAlphaInfo.premultipliedLast.rawValue
)!

// White background with slight transparency (for overlay blending)
ctx.setFillColor(CGColor(red: 0, green: 0, blue: 0, alpha: 0))
ctx.fill(CGRect(x: 0, y: 0, width: CGFloat(width), height: CGFloat(height)))

// Draw text centered
ctx.textPosition = CGPoint(x: 30, y: 15)
CTLineDraw(line, ctx)

// Save PNG using ImageIO
let image = ctx.makeImage()!
let url = URL(fileURLWithPath: outputPath)
let dest = CGImageDestinationCreateWithURL(url as CFURL, "public.png" as CFString, 1, nil)!
CGImageDestinationAddImage(dest, image, nil)
CGImageDestinationFinalize(dest)

print("\(outputPath): \(width)x\(height)")
