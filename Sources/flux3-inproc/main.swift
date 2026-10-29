import Flux3Plugin
import Foundation
import MLXFlux3Klein
import MLXServeCore
import MLXToolKit

// The in-process reference for the plugin: one FLUX 3 Klein lane loaded by MLXServeEngine in this process, with the
// lane configuration the worker uses (Flux3Plugin) and the request it would build (Flux3Request, after the same size
// snapping). A cold run then a warm one; the first image is written to --out, and one JSON line of timings is printed.
// Not part of the plugin bundle.
//
//   flux3-inproc [--model flux-3-klein] [--lane bf16 | int8 | int4] [--prompt <text>] [--width 1024] [--height 1024]
//                [--steps 16] [--seed 42] [--no-upsample] [--image <png> ...] [--models-root <dir>] [--out <png>]
//                [--runs 2]
//
// --image (repeatable, 1–10) makes it an image.edit with the prompt as the instruction; without --width and --height
// the first image sets the size, as in the worker. --no-upsample is the worker's x-flux3.upsample false: the prompt
// goes as typed. Without it the request carries no upsample key, as the worker's without the param.

var args = Array(CommandLine.arguments.dropFirst())
func option(_ name: String) -> String? {
    guard let i = args.firstIndex(of: name), i + 1 < args.count else { return nil }
    defer { args.removeSubrange(i...(i + 1)) }
    return args[i + 1]
}
func options(_ name: String) -> [String] {
    var values: [String] = []
    while let value = option(name) { values.append(value) }
    return values
}
func flag(_ name: String) -> Bool {
    guard let i = args.firstIndex(of: name) else { return false }
    args.remove(at: i)
    return true
}
func seconds(since start: ContinuousClock.Instant) -> Double {
    let d = ContinuousClock.now - start
    return Double(d.components.seconds) + Double(d.components.attoseconds) / 1e18
}
func fail(_ message: String) -> Never {
    FileHandle.standardError.write(Data((message + "\n").utf8))
    exit(2)
}

let started = ContinuousClock.now
guard (option("--model") ?? modelID) == modelID else { fail("unknown --model; the plugin has one: \(modelID)") }
let quant = Quant(rawValue: option("--lane") ?? defaultQuant.rawValue) ?? defaultQuant
let prompt = option("--prompt") ?? "A photorealistic lighthouse on a rocky coast at dusk, warm light in the lantern room"
let width = Int(option("--width") ?? ""), height = Int(option("--height") ?? "")
let steps = Int(option("--steps") ?? "") ?? defaultSteps
let seed = UInt64(option("--seed") ?? "") ?? 42
let upsample: Bool? = flag("--no-upsample") ? false : nil
let imagePaths = options("--image")
let outPath = option("--out")
let runs = Int(option("--runs") ?? "") ?? 2
let modelsRoot = URL(fileURLWithPath: option("--models-root")
                     ?? NSHomeDirectory() + "/Library/Application Support/MLXEngine/Models", isDirectory: true)
let isEdit = !imagePaths.isEmpty
guard imagePaths.count <= maxReferences else { fail("--image takes 1–\(maxReferences) images") }

// The worker's size: the asked one, else 1024² (generate) or the first image's (edit), snapped to the envelope.
let asked: (Int, Int)
if let width, let height {
    asked = (width, height)
} else if isEdit {
    guard let size = pixelSize(of: URL(fileURLWithPath: imagePaths[0])) else { fail("can't read \(imagePaths[0])") }
    asked = size
} else {
    asked = (1024, 1024)
}
let size = try snapped(asked)
let request = Flux3Request(prompt: prompt, width: size.width, height: size.height, steps: steps, seed: seed, upsample: upsample)

var result: [String: Any] = ["model": modelID, "lane": quant.rawValue, "width": size.width, "height": size.height,
                             "steps": steps, "seed": seed, "upsample": upsample ?? true, "references": imagePaths.count]

// The budget Gaea sends a worker on a large Mac (wiresWeights off, the automatic pool cap).
let engine = MLXServeEngine(workerBudget: WorkerBudget(ceilingBytes: 64_000_000_000, wiresWeights: false))
await engine.useModelStore(ModelStore(root: modelsRoot))
let id = PackageID(modelID)
_ = try await engine.register(Flux3KleinPackage.registration, configuration: configuration(quant: quant), id: id)

var clock = ContinuousClock.now
_ = try await engine.prepare(isEdit ? .imageEdit : .textToImage, package: id)
result["prepare_s"] = seconds(since: clock)

let images = try imagePaths.map { path in
    Image(format: path.lowercased().hasSuffix(".png") ? .png : .jpeg, data: try Data(contentsOf: URL(fileURLWithPath: path)))
}
// The worker's request, built by the same function (Flux3Plugin's generate and edit).
let engineRequest: any CapabilityRequest = isEdit ? request.edit(images) : request.generate()

for run in 1...max(1, runs) {
    clock = ContinuousClock.now
    let response = try await engine.run(engineRequest, package: id)
    result["run\(run)_s"] = seconds(since: clock)
    let image = (response as? T2IResponse)?.image ?? (response as? IEditResponse)?.image
    if run == 1, let image, let outPath { try image.data.write(to: URL(fileURLWithPath: outPath)) }
}
result["cold_s"] = seconds(since: started) - (runs > 1 ? result["run\(runs)_s"] as! Double : 0)

let data = try JSONSerialization.data(withJSONObject: result, options: [.sortedKeys])
print(String(decoding: data, as: UTF8.self))
