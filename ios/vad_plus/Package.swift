// swift-tools-version: 5.9
// The swift-tools-version declares the minimum version of Swift required to build this package.

import PackageDescription

let package = Package(
    name: "vad_plus",
    platforms: [
        .iOS("15.0")
    ],
    products: [
        // Dynamic, so the @_cdecl FFI functions (looked up by name from Dart,
        // never referenced by the app) are neither dead-stripped nor dropped.
        .library(name: "vad-plus", type: .dynamic, targets: ["vad_plus"])
    ],
    dependencies: [
        .package(
            url: "https://github.com/microsoft/onnxruntime-swift-package-manager",
            .upToNextMinor(from: "1.20.0")
        )
    ],
    targets: [
        .target(
            name: "vad_plus",
            dependencies: [
                .product(name: "onnxruntime", package: "onnxruntime-swift-package-manager")
            ],
            resources: [
                .copy("Resources/silero_vad_v6.onnx")
            ]
        )
    ]
)
