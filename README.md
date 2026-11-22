# Detect_Rust

High-performance, modular computer vision and real-time object tracking engine written in **Rust**.

Designed for ultra-low latency inference, deterministic frame processing, and scalability across architectures — from embedded ARM/Linux single-board computers to x86_64 workstations and GPU servers.

---

## Key Features

- **Modular Architecture**: Clean separation between core tracking algorithms, compute backends, and platform adapters via Cargo Workspace.
- **Ultra-Low Latency**: Zero-allocation hot path, efficient ring buffers, and cache-friendly data layouts.
- **Pluggable Inference Engines**: Flexible abstraction supporting ONNX Runtime, Tract (`no_std` / pure Rust), and native SIMD pipelines.
- **Fast Correlation Tracking**: High-framerate tracking filter (KCF / MOSSE) capable of running at 200+ FPS on embedded CPUs.
- **Motion Estimation & Filtering**: Integrated digital IIR filtering and PID controllers for smooth trajectory tracking.
- **Cross-Platform**: First-class support for Linux (x86_64, aarch64), Windows, and embedded targets.

---

## Workspace Structure

```
Detect_Rust/
├── crates/
│   ├── detect-core/        # Pure algorithmic engine (tracking, filters, math)
│   ├── detect-compute/     # Hardware acceleration and inference abstraction
│   └── detect-ffi/         # C-compatible FFI bindings for external runtimes
└── apps/
    └── desktop-bench/      # Profiling and benchmarking suite
```

---

## Quick Start

### Prerequisites
- Rust 1.85+ (Edition 2024 / 2021)
- Cargo

### Building

```bash
# Build all workspace crates in release mode
cargo build --release

# Run unit and integration tests
cargo test
```

---

## License

Dual-licensed under MIT or Apache 2.0.
