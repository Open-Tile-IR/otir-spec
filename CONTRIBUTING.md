# Contributing to Open Tile IR

Thank you for your interest in contributing to the Open Tile IR (OTIR) specification!

OTIR is an open standard, and its success depends on diverse input from the community—especially those with expertise in different hardware architectures (GPU, NPU, DSP, CPU). This document outlines the process for contributing changes to the specification and related tooling.

## Code of Conduct

We are committed to providing a friendly, safe, and welcoming environment for all, regardless of level of experience, gender identity and expression, sexual orientation, disability, personal appearance, body size, race, ethnicity, age, religion, nationality, or other similar characteristic.

Please maintain a professional and respectful tone in all discussions.

## How to Contribute

### 1. Request for Comments (RFCs)

Major changes to the specification (e.g., adding new instructions, changing the memory model, or modifying the type system) must go through the **RFC process**.

1.  **Draft:** Create a new markdown file in the `rfc/` directory (e.g., `rfc/001-sparse-tiles.md`).
2.  **Describe:** The RFC should detail:
    *   **Motivation:** Why is this change necessary? What use case does it enable?
    *   **Proposal:** Technical details of the change.
    *   **Impact:** How does this affect existing backends or frontends?
    *   **Alternatives:** What other approaches were considered?
3.  **Pull Request:** Submit a PR with your RFC. The community will discuss and refine the proposal.
4.  **Adoption:** Once consensus is reached, the RFC is merged, and the changes are integrated into the main `SPEC_vX.X.X.md` document.

### 2. Minor Improvements and Fixes

For typos, clarifications, or non-normative changes (e.g., adding examples or improving the wording):

1.  Fork the repository.
2.  Make your changes directly to the relevant document.
3.  Submit a Pull Request with a clear description of the fix.

### 3. Reporting Issues

If you find an ambiguity, contradiction, or error in the specification, please open a **GitHub Issue**.
*   Use a clear title (e.g., "Ambiguity in `otir.dma_copy_async` return type").
*   Reference the specific section number (e.g., "Section 3.2").
*   Suggest a correction if possible.

## Style Guide for Specification

*   **Language:** English (US).
*   **Tone:** Formal, precise, and technical. Avoid colloquialisms.
*   **Terminology:** Use standard compiler and architecture terminology (e.g., "SIMT", "DRAM", "Barrier"). When introducing OTIR-specific terms (e.g., "Workgroup Space"), define them clearly upon first use.
*   **Formatting:** Use Markdown. Code snippets should be highlighted (e.g., `mlir`).

## License

By contributing to this repository, you agree that your contributions will be licensed under the [Apache License 2.0 with LLVM Exceptions](LICENSE).
