<div align="center">
  <h1>Open Tile Intermediate Representation (OTIR)</h1>
  <p>
    <strong>A Hardware-Agnostic, Tile-Centric Intermediate Representation for AI Accelerators</strong>
  </p>
  <p>
    <a href="SPEC_v0.1.0.md"><strong>Read the Specification (v0.1.0-draft) »</strong></a>
    <br />
    <br />
    <a href="#vision">Vision</a>
    ·
    <a href="#why-otir">Why OTIR?</a>
    ·
    <a href="#roadmap">Roadmap</a>
    ·
    <a href="CONTRIBUTING.md">Contributing</a>
  </p>
</div>

---

## Vision

The landscape of AI acceleration is fragmented. While the "Tile" (or Block) has emerged as the universal atomic unit of computation for modern workloads like Transformers and ConvNets, the programming models remains deeply coupled to specific vendor architectures.

**Open Tile IR (OTIR)** aims to bridge this gap.

OTIR is an open standard intermediate representation designed to express high-performance parallel computation at the **Tile level**. It decouples the *intent* of an algorithm (e.g., "matrix multiply two 128x128 blocks") from the *implementation* details of the underlying hardware (e.g., Warp scheduling, Tensor Core instructions, or DMA barriers).

Our goal is simple: **Write once, tile everywhere.** Whether targeting NVIDIA GPUs, AMD GPUs, Ascend NPUs, or RISC-V Vector processors, OTIR provides a unified abstraction layer for compilers and library developers.

## Why OTIR?

Current intermediate representations (IRs) often operate at either too high a level (Graph level) or too low a level (SIMT/SIMD threads).

*   **Explicit Memory Hierarchy:** Unlike traditional graph IRs, OTIR exposes explicit memory spaces (`Global`, `Workgroup`, `Private`), allowing compilers to manage on-chip SRAM efficiently—a critical requirement for NPUs and spatial architectures.
*   **Asynchrony as a Primitive:** OTIR treats data movement as a first-class asynchronous citizen, enabling aggressive latency hiding essential for high-performance compute.
*   **Vendor Neutral:** Designed from the ground up to support both SIMT (GPU) and Dataflow (NPU/TPU) architectures without bias.

## Core Concepts

OTIR is built upon a few key pillars, defined formally in the [Specification](SPEC_v0.1.0.md):

1.  **Tiles, Not Threads:** The fundamental data type is the multi-dimensional `!otir.tile`.
2.  **Logical Single Thread:** Code describes the behavior of a single "Workgroup" operating on tiles, abstracting away thread-level synchronization.
3.  **Layout Agnosticism:** Internal data layouts can be opaque, allowing backends to choose optimal swizzling or blocking formats (e.g., `NC1HWC0`) transparently.

## Roadmap

- [x] **v0.1.0 (Draft):** Initial Specification release. Definition of Type System, Memory Model, and Core Ops.
- [ ] **MLIR Dialect Definition:** Reference implementation of the OTIR Dialect in MLIR (TableGen).
- [ ] **Reference Verifier:** Static analysis tools to validate OTIR correctness.
- [ ] **CPU Reference Backend:** Lowering OTIR to LLVM IR (Vector/SCF) for correctness verification on CPUs.
- [ ] **NVPTX Backend:** Lowering OTIR to NVIDIA PTX (via NVGPU Dialect) to demonstrate performance parity with CUDA.
- [ ] **Hardware Backends:** Collaboration with vendor-specific compiler teams (Ascend, ROCm, RISC-V).

## Repository Structure

*   [`SPEC_v0.1.0.md`](SPEC.md): The normative specification document.
*   `rfc/`: Request for Comments documents for proposed changes.
*   `examples/`: (Planned) Example OTIR snippets demonstrating common kernels (GEMM, LayerNorm, FlashAttention).

## Community & Governance

OTIR is an open project governed by the community. We welcome participation from hardware vendors, compiler engineers, and framework developers.

*   **Discussion:** Please use GitHub Issues for specification discussions.
*   **Chat:** [Link to Discord/Slack/Matrix if you have one]

## License

This project is licensed under the **Apache License 2.0** with LLVM Exceptions. See [LICENSE](LICENSE) for details.
