// swift-tools-version: 6.2
import PackageDescription

// gaea-plugin-flux3-klein: Black Forest Labs' FLUX 3 Image Klein (FLUX Kommunity licence) as a Gaea research plugin.
// One worker executable built on GaeaPluginKit over flux3-klein-swift, packaged as a signed .gaeaplugin by
// scripts/build-plugin.sh.
//
// flux3-inproc is not part of the plugin: it runs the same package in-process through MLXServeEngine, with the same
// dependency resolution, as the reference a run through Gaea is compared against.
//
// GAEA_PLUGIN_KIT_PATH points the kit at a local checkout while you work on both.
let kitPath = Context.environment["GAEA_PLUGIN_KIT_PATH"]

let package = Package(
    name: "gaea-plugin-flux3-klein",
    platforms: [.macOS(.v26)], // mlx-engine-swift's MLXToolKit requires macOS 26
    dependencies: [
        kitPath.map { .package(path: $0) }
            ?? .package(url: "https://github.com/xocialize-code/gaea-plugin-kit.git", from: "0.3.0"),
        .package(url: "https://github.com/xocialize/flux3-klein-swift.git", exact: "0.6.1"),
        .package(url: "https://github.com/xocialize/mlx-engine-swift.git", from: "0.72.0"),
    ],
    targets: [
        .target(
            name: "Flux3Plugin",
            dependencies: [
                .product(name: "GaeaPluginKit", package: "gaea-plugin-kit"),
                .product(name: "MLXFlux3Klein", package: "flux3-klein-swift"),
                .product(name: "MLXToolKit", package: "mlx-engine-swift"),
            ]
        ),
        .executableTarget(
            name: "gaea-flux3-klein-worker",
            dependencies: [
                "Flux3Plugin",
                .product(name: "GaeaPluginKit", package: "gaea-plugin-kit"),
            ]
        ),
        .executableTarget(
            name: "flux3-inproc",
            dependencies: [
                "Flux3Plugin",
                .product(name: "MLXFlux3Klein", package: "flux3-klein-swift"),
                .product(name: "MLXToolKit", package: "mlx-engine-swift"),
                .product(name: "MLXServeCore", package: "mlx-engine-swift"),
            ],
            swiftSettings: [.swiftLanguageMode(.v5)]
        ),
    ]
)
