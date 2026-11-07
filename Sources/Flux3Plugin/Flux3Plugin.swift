import Foundation
import GaeaPluginKit
import GaeaProtocol
import ImageIO
import MLXFlux3Klein
import MLXToolKit

// gaea-plugin-flux3-klein: Black Forest Labs' FLUX 3 Image Klein (FLUX Kommunity licence weights, MIT port) as a Gaea
// research plugin, on flux3-klein-swift through MLXServeEngine (Gaea AB-T-0374).
//
// One model, flux-3-klein, as mlx-image-runtime's catalog serves it: text-to-image and editing with up to 10 references
// in one 5.55B flow-map DiT on Qwen3-VL-4B, 16 steps, no guidance. Strong English text, bounding-box layout. It wants
// full scene captions, so BFL's prompt upsampler (int8) expands each prompt first unless a request turns it off.
//
// The weights are not permissive (LicenseRef-FLUX-Kommunity-1.0, AB-D-0151), so Gaea makes the model named-only and
// opt-in by itself; the offers say named-only too. Three lanes, one per DiT quant, all made at load from the bf16
// snapshot, with the package's footprints; bf16 is the default (the catalog's choice from 48 GB). BFL's safety checks
// are off, as in the catalog. Weights come from the models root Gaea sends in configure; the engine materializes a
// missing snapshot from the gated mirror (an HF token whose account accepted the licence).
//
// Past 2048² of area (0.2.0, AB-T-0464) the DiT draws a sixteenth of the area and FlowUpscaler (flux3-klein-swift
// 0.7.0, `metaData["latentUpscale"] = 2`) doubles its latents twice before the row-band decode, up to 4096²: the picture
// the smaller draw made, with finer detail (AB-R-0797). Up to 2048² it draws natively, as before. An edit is upscaled
// only when it asks for a size; otherwise it is drawn at the first image's size fitted to 2048², as before.

/// The package's envelope (`Flux3KleinEnvelope`): a native draw ≤ 2048² of area, sides ≥ 64 in multiples of 16, up to
/// 10 references; with a latent upscale the draw ≤ 1024² and the image ≤ 4096².
public let nativePixels = Flux3KleinEnvelope.maxOutputPixels
public let maxPixels = Flux3KleinEnvelope.maxUpscaledOutputPixels
public let upscaleDrawPixels = Flux3KleinEnvelope.maxLatentUpscaleDrawPixels
public let minSide = Flux3KleinEnvelope.minSide, grid = 16, maxReferences = Flux3KleinEnvelope.maxReferenceImages

public let modelID = "flux-3-klein", modelTitle = "FLUX 3 Klein"
/// Flux3Schedule.defaultSteps, and the catalog's GenerationDefaults.
public let defaultSteps = 16

/// The lanes, default first: bf16 (the catalog's quant from 48 GB), int8, and int4 with 8-bit attention.
public let quants: [Quant] = [.bf16, .int8, .int4]
public let defaultQuant = Quant.bf16

/// The package configuration for one lane, the catalog's default: the int8 upsampler, safety checks off. flux3-inproc
/// builds the same one.
public func configuration(quant: Quant) -> Flux3KleinConfiguration {
    Flux3KleinConfiguration(quant: quant, snapshotPath: "", upsampler: .int8, upsamplerPath: "", safetyChecks: false,
                            safetyClassifierPath: "")
}

/// The MES size a job asks for (`size`, or `aspect` with `megapixels`, Image profile §2), else `fallback`.
func requestedSize(_ job: Job, fallback: (Int, Int)) throws -> (Int, Int) {
    if let size = job.param("size") {
        guard let w = size["width"]?.intValue, let h = size["height"]?.intValue, w > 0, h > 0 else {
            throw GPPError(.invalidRequest, "size is {width, height} in pixels.")
        }
        return (w, h)
    }
    if let aspect = job.param("aspect")?.stringValue {
        let parts = aspect.split(separator: ":").compactMap { Double($0) }
        guard parts.count == 2, parts[0] > 0, parts[1] > 0 else { throw GPPError(.invalidRequest, "aspect is W:H, e.g. 16:9.") }
        let pixels = (job.param("megapixels")?.doubleValue ?? 1.0) * 1_000_000
        let height = (pixels * parts[1] / parts[0]).squareRoot()
        return (Int(height * parts[0] / parts[1]), Int(height))
    }
    return fallback
}

/// A PNG or JPEG's pixel size; nil when ImageIO can't read it. image.edit's first image sets the output size.
public func pixelSize(of url: URL) -> (Int, Int)? {
    guard let source = CGImageSourceCreateWithURL(url as CFURL, nil),
          let properties = CGImageSourceCopyPropertiesAtIndex(source, 0, nil) as? [CFString: Any],
          let width = properties[kCGImagePropertyPixelWidth] as? Int,
          let height = properties[kCGImagePropertyPixelHeight] as? Int else { return nil }
    return (width, height)
}

/// How an image size is made: the draw, and the FlowUpscaler passes (0, or 2) that double it on each side.
public struct SizePlan: Sendable {
    public var width: Int, height: Int
    public var drawWidth: Int, drawHeight: Int
    public var passes: Int
    public var warning: Warning?
}

/// Fits a size to the envelope and plans it: scaled down to 4096² of area (2048² when `upscale` is off), keeping the
/// aspect; drawn as it is up to 2048²; past that drawn at a sixteenth of the area and upscaled twice. Each side is
/// floored to a multiple of 16 (64 when upscaled). A change is reported as `size_snapped`. The package refuses rather
/// than clamps, so the worker fits it first.
public func plan(_ size: (Int, Int), upscale: Bool = true) throws -> SizePlan {
    let cap = upscale ? maxPixels : nativePixels
    var (w, h) = (Double(size.0), Double(size.1))
    if w * h > Double(cap) {
        let scale = (Double(cap) / (w * h)).squareRoot()
        w *= scale
        h *= scale
    }
    let passes = w * h <= Double(nativePixels) ? 0 : 2
    let unit = grid << passes
    let width = Int(w) / unit * unit, height = Int(h) / unit * unit
    let drawWidth = width >> passes, drawHeight = height >> passes
    guard min(drawWidth, drawHeight) >= minSide, drawWidth * drawHeight <= (passes == 0 ? nativePixels : upscaleDrawPixels) else {
        throw GPPError(.invalidRequest, "FLUX 3 Klein makes at least \(minSide) px a side and at most 4096² in area (past 2048² it "
            + "draws a sixteenth of the area, at least \(minSide) px a side, and upscales it); \(size.0)×\(size.1) doesn't fit.",
            detail: ["param": "size"])
    }
    let note = passes == 0 ? "multiples of 16, at most \(upscale ? "4096" : "2048")² in area"
        : "drawn at \(drawWidth)×\(drawHeight) and upscaled ×4, so multiples of 64, at most 4096² in area"
    let warning = (width, height) == size ? nil : Warning(
        code: "size_snapped", message: "Made \(width)×\(height) for \(size.0)×\(size.1): \(note).", param: "size")
    return SizePlan(width: width, height: height, drawWidth: drawWidth, drawHeight: drawHeight, passes: passes, warning: warning)
}

/// What one engine call carries, built the same way by the worker and flux3-inproc so the two send byte-identical
/// requests. No negative prompt and no guidance: the model has no CFG.
public struct Flux3Request: Sendable {
    public var prompt: String
    public var width: Int
    public var height: Int
    public var steps: Int
    public var seed: UInt64
    /// `Flux3KleinMetaKeys.upsample`: false sends the prompt as typed; nil (absent) runs the configured upsampler.
    public var upsample: Bool?
    /// `Flux3KleinMetaKeys.boxes`: layout boxes or box edits, passed through as the catalog's request key does.
    public var boxes: MetaValue?
    /// FlowUpscaler passes after the draw (`metaData["latentUpscale"]`); 0 draws at `width` × `height` as it is.
    public var latentUpscale: Int

    public init(prompt: String, width: Int, height: Int, steps: Int = defaultSteps, seed: UInt64, upsample: Bool? = nil,
                boxes: MetaValue? = nil, latentUpscale: Int = 0) {
        self.prompt = prompt
        self.width = width
        self.height = height
        self.steps = steps
        self.seed = seed
        self.upsample = upsample
        self.boxes = boxes
        self.latentUpscale = latentUpscale
    }

    /// The package's per-request keys, only those the request sets.
    public var metaData: MetaData {
        var meta: MetaData = [:]
        if let upsample { meta[Flux3KleinMetaKeys.upsample] = .bool(upsample) }
        if let boxes { meta[Flux3KleinMetaKeys.boxes] = boxes }
        if latentUpscale > 0 { meta["latentUpscale"] = .int(latentUpscale) }
        return meta
    }

    /// Image `index` of an image.generate, with seed + index. (The package keeps the seed's low 32 bits.)
    public func generate(index: Int = 0) -> T2IRequest {
        T2IRequest(prompt: prompt, width: width, height: height, steps: steps, seed: seed + UInt64(index), metaData: metaData)
    }

    /// An image.edit of `images`, the prompt as the instruction.
    public func edit(_ images: [Image]) -> IEditRequest {
        IEditRequest(images: images, prompt: prompt, width: width, height: height, steps: steps, seed: seed, metaData: metaData)
    }
}

/// An MES JSON value as package metaData.
func metaValue(_ value: JSONValue) -> MetaValue {
    switch value {
    case .null: .null
    case .bool(let b): .bool(b)
    case .int(let i): .int(Int(i))
    case .double(let d): .double(d)
    case .string(let s): .string(s)
    case .array(let a): .array(a.map(metaValue))
    case .object(let o): .object(o.mapValues(metaValue))
    }
}

/// The run's settings shared by generate and edit: steps, upsampling, boxes, seed, and the warnings for what the model
/// ignores.
struct Shared {
    var steps: Int
    var upsample: Bool?
    var boxes: JSONValue?
    var seed: UInt64
    var count: Int
    var warnings: [Warning] = []

    /// `counted`: image.generate takes MES `count`; image.edit has one output and no count.
    init(_ job: Job, counted: Bool = true) throws {
        steps = job.param("x-flux3.steps")?.intValue ?? defaultSteps
        guard (1...100).contains(steps) else { throw GPPError(.invalidRequest, "x-flux3.steps is 1–100.") }
        if let value = job.param("x-flux3.upsample") {
            guard case .bool(let b) = value else { throw GPPError(.invalidRequest, "x-flux3.upsample is true or false.") }
            upsample = b
        }
        if let value = job.param("x-flux3.boxes") {
            guard let items = value.arrayValue, items.allSatisfy({ $0.objectValue != nil }) else {
                throw GPPError(.invalidRequest, "x-flux3.boxes is an array of box objects.", detail: ["param": "x-flux3.boxes"])
            }
            boxes = value
        }
        count = counted ? job.param("count")?.intValue ?? 1 : 1
        guard (1...8).contains(count) else { throw GPPError(.invalidRequest, "count is 1–8.") }
        if let negative = job.param("negative_prompt")?.stringValue, !negative.isEmpty {
            warnings.append(Warning(code: "negative_prompt_ignored",
                                    message: "FLUX 3 Klein runs without CFG, so a negative prompt does nothing.",
                                    param: "negative_prompt"))
        }
        if job.param("prompt_enhance")?.stringValue == "on" {
            warnings.append(Warning(code: "prompt_enhance_ignored",
                                    message: "FLUX 3 Klein's own upsampler expands prompts (x-flux3.upsample); this plugin doesn't take prompt_enhance.",
                                    param: "prompt_enhance"))
        }
        // Image profile §2: a request without a seed still gets one, reported in effective.
        seed = job.seed ?? UInt64.random(in: 0..<(1 << 31))
    }

    func request(prompt: String, _ size: SizePlan) -> Flux3Request {
        Flux3Request(prompt: prompt, width: size.drawWidth, height: size.drawHeight, steps: steps, seed: seed, upsample: upsample,
                     boxes: boxes.map(metaValue), latentUpscale: size.passes)
    }

    func effective(_ size: SizePlan, counted: Bool = true) -> [String: JSONValue] {
        var values: [String: JSONValue] = [
            "size": ["width": .int(Int64(size.width)), "height": .int(Int64(size.height))],
            "seed": .int(Int64(seed)), "x-flux3.steps": .int(Int64(steps)), "x-flux3.upsample": .bool(upsample ?? true),
        ]
        if size.passes > 0 {
            values["x-flux3.upscaled"] = ["from": ["width": .int(Int64(size.drawWidth)), "height": .int(Int64(size.drawHeight))],
                                          "passes": .int(Int64(size.passes)), "model": "FlowUpscaler"]
        }
        if counted { values["count"] = .int(Int64(count)) }
        if let boxes { values["x-flux3.boxes"] = boxes }
        return values
    }
}

/// MES image.generate: one engine call per image, each with seed + i.
func generate() -> EngineOperation {
    EngineOperation("image.generate", outputNames: ["image": "images"]) { job in
        guard let prompt = try job.text("prompt"), !prompt.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty else {
            throw GPPError(.invalidRequest, "image.generate needs a prompt.")
        }
        let size = try plan(try requestedSize(job, fallback: (1024, 1024)))
        var shared = try Shared(job)
        if let warning = size.warning { shared.warnings.insert(warning, at: 0) }
        let request = shared.request(prompt: prompt, size)
        let effective = shared.effective(size)
        return (0..<shared.count).map { i in
            EngineCall(request.generate(index: i), effective: effective, warnings: shared.warnings)
        }
    }
}

/// MES image.edit: editing and composition from 1–10 references, named in the instruction in order ("the second
/// image"), drawn at the first image's size (Image profile §3). One output. The package fits one reference under 2048²
/// and several each under 1 MP itself.
func edit() -> EngineOperation {
    EngineOperation("image.edit", outputNames: ["image": "image"]) { job in
        guard let instruction = try job.text("instruction") ?? job.text("prompt"), !instruction.isEmpty else {
            throw GPPError(.invalidRequest, "image.edit needs an instruction.")
        }
        let name = job.inputs["images"] == nil ? "image" : "images"
        let references = job.inputs[name] ?? []
        guard (1...maxReferences).contains(references.count) else {
            throw GPPError(.invalidRequest, "image.edit takes 1–\(maxReferences) images; got \(references.count).",
                           detail: ["param": "images"])
        }
        guard let inputSize = pixelSize(of: references[0]) else {
            throw GPPError(.invalidRequest, "image.edit needs readable PNG or JPEG images.")
        }
        // Upscaled only when a size is asked: an edit sized by its first image keeps the native 2048² cap, as before.
        let asked = job.param("size") != nil || job.param("aspect") != nil
        let size = try plan(try requestedSize(job, fallback: inputSize), upscale: asked)
        var shared = try Shared(job, counted: false)
        if let warning = size.warning { shared.warnings.insert(warning, at: 0) }
        if job.param("strength") != nil {
            shared.warnings.append(Warning(code: "param_ignored",
                                           message: "FLUX 3 Klein edits by reference conditioning from noise, so strength does nothing.",
                                           param: "strength"))
        }
        let images = try references.indices.map { try job.image(name, index: $0) }
        let request = shared.request(prompt: instruction, size)
        return [EngineCall(request.edit(images), effective: shared.effective(size, counted: false),
                           warnings: shared.warnings)]
    }
}

public func lanes() -> [LaneDefinition] {
    quants.map { quant in
        LaneDefinition(id: quant.rawValue, default: quant == defaultQuant, configuration: configuration(quant: quant))
    }
}

/// The FLUX Kommunity licence (AB-D-0151): commercial_use false, since Gaea can't tell whether its operator qualifies
/// for the outputs exception; the notice names it, as pixel-agency-normalizer's `ImageNormalizer.notice` does.
let kommunity: JSONValue = [
    "spdx": "LicenseRef-FLUX-Kommunity-1.0", "commercial_use": false, "accepted_required": true,
    "terms_url": "https://huggingface.co/black-forest-labs/flux-3-image-klein/blob/main/LICENSE.md",
    "notice": .string("FLUX Kommunity licence: non-commercial use; commercial use of outputs only for businesses under "
        + "US$5M revenue that filter content and disclose AI generation; never use outputs to train another image model. "
        + "Read the terms before you use or share results."),
]
let limits = PluginManifest.Limits(maxPixels: maxPixels, minSide: minSide, sizeMultiple: grid)
let editLimits = PluginManifest.Limits(maxPixels: maxPixels, minSide: minSide, sizeMultiple: grid, maxImages: maxReferences)

let params: JSONValue = [
    "type": "object", "additionalProperties": false,
    "properties": [
        "x-flux3.steps": ["type": "integer", "minimum": 1, "maximum": 100, "default": .int(Int64(defaultSteps)),
                          "description": "Denoising steps; the model is tuned for 16.",
                          "x-mes-ui": ["widget": "slider", "label": "Steps", "advanced": true]],
        "x-flux3.upsample": [
            "type": "boolean", "default": true,
            "description": .string("Expand the prompt into a full scene caption with BFL's prompt upsampler first (about "
                + "10 s). Off sends the prompt as typed, for a caller that wrote the full caption."),
            "x-mes-ui": ["widget": "toggle", "label": "Prompt upsampler", "advanced": true],
        ],
        "x-flux3.boxes": [
            "type": "array",
            "description": .string("Layout boxes on 0–1 coordinates, {id, bbox: [y_min, x_min, y_max, x_max], desc}, or "
                + "box edits {id, from: ref_image_n | target, src_bbox, tgt_bbox (null removes), desc}."),
            "items": [
                "type": "object",
                "properties": [
                    "id": ["type": "string"], "desc": ["type": "string"],
                    "from": ["type": "string"],
                    "bbox": ["type": "array", "items": ["type": "number"], "minItems": 4, "maxItems": 4],
                    "src_bbox": ["type": "array", "items": ["type": "number"], "minItems": 4, "maxItems": 4],
                    "tgt_bbox": ["type": ["array", "null"], "items": ["type": "number"], "minItems": 4, "maxItems": 4],
                ],
            ],
            "x-mes-ui": ["label": "Layout boxes", "advanced": true],
        ],
    ],
]

func offerModel(edit: Bool) -> JSONValue {
    let use = edit
        ? "Editing and composition from 1–10 images, named in the instruction in order (\"the second image\"); describe the "
            + "whole result with the edit folded in. Drawn from noise at the first image's size; strength does nothing. "
            + "A size past 2048² is drawn at a sixteenth of the area and upscaled (FlowUpscaler), up to 4096². "
        : "Text-to-image, 16 steps without CFG: negative prompts do nothing. Strong English text and box layout. "
            + "Past 2048² it draws a sixteenth of the area and upscales it (FlowUpscaler), up to 4096²: the same picture, finer. "
    return [
        "id": .string(modelID), "title": .string(modelTitle), "license": kommunity,
        "memory_class": "large", "speed_class": "slow",
        "quality_tiers": ["standard", "best"], "named_only": true,
        "notes": .string("Research plugin, named-only and opt-in (FLUX Kommunity licence). " + use
            + "The model wants full scene captions; short prompts are fine, since its upsampler expands them (about 10 s). "
            + "Put text to render in quotes, e.g. a sign reading \"OPEN\"."),
    ]
}

public let flux3Plugin = GaeaPluginDefinition(
    id: "com.xocialize.gaea.flux3-klein",
    version: "0.2.0",
    minGaea: "1.2.0",
    team: "CFZS6RXUCP",
    repo: "https://github.com/xocialize-code/gaea-plugin-flux3-klein",
    title: "FLUX 3 Klein",
    summary: "Black Forest Labs' FLUX 3 Image Klein (FLUX Kommunity licence: non-commercial; commercial outputs only for "
        + "Qualifying Users): a 5.55B flow-map DiT on Qwen3-VL-4B, 16 steps. Text-to-image and editing from up to 10 "
        + "references, strong English text, bounding-box layout, and BFL's prompt upsampler. A research plugin: named-only "
        + "and opt-in.",
    license: "Apache-2.0",
    executable: "gaea-flux3-klein-worker",
    models: [
        .engine(Flux3KleinPackage.self, id: modelID, title: modelTitle, lanes: lanes(), operations: [generate(), edit()]),
    ],
    offers: [
        PluginManifest.Offer(operation: "image.generate", profile: "image", model: offerModel(edit: false), params: params,
                             defaults: ["count": 1], limits: limits),
        PluginManifest.Offer(operation: "image.edit", profile: "image", model: offerModel(edit: true), params: params,
                             limits: editLimits),
    ])
