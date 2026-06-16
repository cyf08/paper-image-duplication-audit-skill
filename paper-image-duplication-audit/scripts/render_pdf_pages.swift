import AppKit
import Foundation
import PDFKit

func fail(_ message: String) -> Never {
    FileHandle.standardError.write((message + "\n").data(using: .utf8)!)
    exit(1)
}

let args = CommandLine.arguments
if args.count < 3 {
    fail("Usage: swift render_pdf_pages.swift <input.pdf> <output-dir> [dpi] [page-list]")
}

let inputPath = args[1]
let outputDir = args[2]
let dpi = args.count >= 4 ? (Double(args[3]) ?? 180.0) : 180.0
let pageListArg = args.count >= 5 ? args[4] : ""
let scale = dpi / 72.0

let inputURL = URL(fileURLWithPath: inputPath)
guard let document = PDFDocument(url: inputURL) else {
    fail("Cannot open PDF: \(inputPath)")
}

let fileManager = FileManager.default
try? fileManager.createDirectory(
    at: URL(fileURLWithPath: outputDir),
    withIntermediateDirectories: true
)

func parsePages(_ spec: String, count: Int) -> [Int] {
    if spec.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty {
        return Array(0..<count)
    }

    var pages: [Int] = []
    for part in spec.split(separator: ",") {
        let piece = part.trimmingCharacters(in: .whitespaces)
        if piece.contains("-") {
            let bounds = piece.split(separator: "-", maxSplits: 1).compactMap { Int($0) }
            if bounds.count == 2 {
                let start = max(1, bounds[0])
                let end = min(count, bounds[1])
                if start <= end {
                    pages.append(contentsOf: (start...end).map { $0 - 1 })
                }
            }
        } else if let page = Int(piece), page >= 1, page <= count {
            pages.append(page - 1)
        }
    }
    return Array(Set(pages)).sorted()
}

let pages = parsePages(pageListArg, count: document.pageCount)
print("page_count\t\(document.pageCount)")

for pageIndex in pages {
    guard let page = document.page(at: pageIndex) else { continue }
    let bounds = page.bounds(for: .mediaBox)
    let pixelWidth = max(1, Int((bounds.width * scale).rounded()))
    let pixelHeight = max(1, Int((bounds.height * scale).rounded()))
    let bitmap = NSBitmapImageRep(
        bitmapDataPlanes: nil,
        pixelsWide: pixelWidth,
        pixelsHigh: pixelHeight,
        bitsPerSample: 8,
        samplesPerPixel: 4,
        hasAlpha: true,
        isPlanar: false,
        colorSpaceName: .deviceRGB,
        bytesPerRow: 0,
        bitsPerPixel: 0
    )

    guard let rep = bitmap else {
        fail("Cannot allocate bitmap for page \(pageIndex + 1)")
    }

    NSGraphicsContext.saveGraphicsState()
    let context = NSGraphicsContext(bitmapImageRep: rep)
    NSGraphicsContext.current = context

    guard let cgContext = context?.cgContext else {
        fail("Cannot create graphics context for page \(pageIndex + 1)")
    }

    cgContext.setFillColor(NSColor.white.cgColor)
    cgContext.fill(CGRect(x: 0, y: 0, width: pixelWidth, height: pixelHeight))
    cgContext.saveGState()
    cgContext.scaleBy(x: CGFloat(scale), y: CGFloat(scale))
    cgContext.translateBy(x: -bounds.origin.x, y: -bounds.origin.y)
    page.draw(with: .mediaBox, to: cgContext)
    cgContext.restoreGState()

    NSGraphicsContext.restoreGraphicsState()

    guard let png = rep.representation(using: .png, properties: [:]) else {
        fail("Cannot encode PNG for page \(pageIndex + 1)")
    }

    let outputPath = URL(fileURLWithPath: outputDir)
        .appendingPathComponent(String(format: "page-%03d.png", pageIndex + 1))
    try png.write(to: outputPath)
    print("rendered\t\(pageIndex + 1)\t\(outputPath.path)")
}
