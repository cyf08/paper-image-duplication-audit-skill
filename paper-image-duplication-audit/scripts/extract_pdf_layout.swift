import Foundation
import PDFKit

func fail(_ message: String) -> Never {
    FileHandle.standardError.write((message + "\n").data(using: .utf8)!)
    exit(1)
}

func jsonEscape(_ value: String) -> String {
    var out = ""
    for scalar in value.unicodeScalars {
        switch scalar {
        case "\"": out += "\\\""
        case "\\": out += "\\\\"
        case "\n": out += "\\n"
        case "\r": out += "\\r"
        case "\t": out += "\\t"
        default:
            if scalar.value < 0x20 {
                out += String(format: "\\u%04x", scalar.value)
            } else {
                out += String(scalar)
            }
        }
    }
    return out
}

let args = CommandLine.arguments
if args.count < 3 {
    fail("Usage: swift extract_pdf_layout.swift <input.pdf> <output.json>")
}

let inputPath = args[1]
let outputPath = args[2]
guard let document = PDFDocument(url: URL(fileURLWithPath: inputPath)) else {
    fail("Cannot open PDF: \(inputPath)")
}

let regexSpecs: [(String, String)] = [
    ("figure_title", "\\b(?:Supplementary\\s+)?Figure\\s+\\d+\\.?"),
    ("panel_caption", "\\b[A-Z](?:-[A-Z])?\\.")
]

var compiled: [(String, NSRegularExpression)] = []
for (kind, pattern) in regexSpecs {
    compiled.append((
        kind,
        try NSRegularExpression(pattern: pattern, options: [.caseInsensitive])
    ))
}

var pageJSON: [String] = []

for pageIndex in 0..<document.pageCount {
    guard let page = document.page(at: pageIndex) else { continue }
    let bounds = page.bounds(for: .mediaBox)
    let text = page.string ?? ""
    let nsText = text as NSString
    var matchJSON: [String] = []

    for (kind, regex) in compiled {
        let range = NSRange(location: 0, length: nsText.length)
        for match in regex.matches(in: text, options: [], range: range) {
            let matchText = nsText.substring(with: match.range)
            guard let selection = page.selection(for: match.range) else { continue }
            let rect = selection.bounds(for: page)
            matchJSON.append(String(format:
                "{\"kind\":\"%@\",\"text\":\"%@\",\"index\":%d,\"x\":%.3f,\"y\":%.3f,\"w\":%.3f,\"h\":%.3f}",
                jsonEscape(kind),
                jsonEscape(matchText),
                match.range.location,
                rect.origin.x,
                rect.origin.y,
                rect.width,
                rect.height
            ))
        }
    }

    pageJSON.append(String(format:
        "{\"page\":%d,\"width\":%.3f,\"height\":%.3f,\"text\":\"%@\",\"matches\":[%@]}",
        pageIndex + 1,
        bounds.width,
        bounds.height,
        jsonEscape(text),
        matchJSON.joined(separator: ",")
    ))
}

let json = "{\"page_count\":\(document.pageCount),\"pages\":[\(pageJSON.joined(separator: ","))]}"
try json.write(to: URL(fileURLWithPath: outputPath), atomically: true, encoding: .utf8)
print("layout\t\(outputPath)")
