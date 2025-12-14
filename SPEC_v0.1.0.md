# **Open Tile Intermediate Representation (OTIR) - Core Specification**

---

## **1. Scope**

### **1.1 Problem Statement**
Modern High-Performance Computing (HPC) exists at a delicate balancing point. On one hand, the complexity of hardware architectures—such as the multi-generational evolution of Tensor Cores in NVIDIA GPUs or the Systolic Arrays in Google TPUs—has surpassed the ability of most programmers to manage effectively through low-level assembly (e.g., PTX/SASS) or intrinsics. On the other hand, the "abstraction penalty" introduced by high-level abstractions like PyTorch's eager mode remains unacceptable in scenarios demanding ultimate performance. Existing intermediate representations (IRs) are either too close to the physical implementation (e.g., LLVM IR), lacking domain-specific semantics, or too high-level (e.g., TensorFlow Graph), unable to express fine-grained memory movement and computation pipelines.

### **1.2 OTIR's Position: A "Controlled Hardware Exposure"**
The core philosophy of OTIR is not to create a utopian IR that completely hides hardware details. Such attempts (for example, early OpenCL abstraction models) often end up sacrificing performance. Instead, OTIR aims to provide a **"Controlled Hardware Exposure."**

It abstracts away details that vary greatly across different architectures and are tedious to optimize (e.g., warp scheduling, register bank conflicts, instruction cache management). At the same time, it **explicitly exposes** hardware features that are critical to performance and logically common across architectures: a **hierarchical memory structure** and **asynchronous data transfer capabilities**. OTIR believes that any serious performance engineer must think about the flow of data between different memory levels at the algorithmic level; this is something that cannot and should not be fully automated. Therefore, OTIR is not an "auto-parallelization" tool, but rather a precise language designed for experts to **explicitly construct dataflow pipelines**.

## **2. Normative References**
The following documents contain provisions which, through reference in this text, constitute provisions of this standard.
*   IEEE Std 754-2019, IEEE Standard for Floating-Point Arithmetic.
*   ISO/IEC 14882:2020, Programming languages — C++ (specifically regarding memory ordering semantics).

## **3. Terms and Definitions**
For the purposes of this standard, the following terms and definitions apply.
### **3.1 Tile**
A multi-dimensional, immutable array of scalar values that serves as an atomic operand for computation instructions. A tile resides in a specific memory space and has a statically determined shape.
### **3.2 Workgroup**
The fundamental unit of scheduling and resource allocation. A workgroup consists of a single logical control flow that operates on tiles. Workgroups are executed independently and can share on-chip memory resources.
### **3.3 Grid**
The collection of all workgroups launched for a single kernel execution.
### **3.4 Memory Space**
A logical partition of memory with distinct visibility, latency, and lifetime characteristics.
### **3.5 Layout**
The mapping from a multi-dimensional logical index space to a linear physical address space.

---

## **4. Abstract Machine Model**

### **4.1.3 Theoretical Foundation of Logical Single-Thread (LST)**
The choice of the LST model over the traditional SIMT model is based on a deep understanding of modern compiler optimization theory. While the SIMT model intuitively maps to the physical execution of GPUs, it imposes a significant burden on the compiler, especially when handling intra-warp divergence and complex data exchanges (shuffles). The static analyzer must expend considerable effort to prove that threads within a warp behave uniformly before it can generate efficient vectorized code.

The LST model completely inverts this problem. It starts from an **a priori, aggregate perspective**, declaring that an operation on a `Tile` is semantically data-parallel. This provides the compiler with a powerful foundation for **Alias Analysis** and **Dependence Analysis**. The compiler no longer needs to "reconstruct" parallelism from a tangle of scalar operations. Instead, it begins from a naturally parallel starting point and only needs to consider how to efficiently "lower" it onto the physical execution units of the target hardware. This "aggregate-to-disperse" lowering path is algorithmically far simpler and more robust than the "disperse-to-aggregate" reconstruction path.

### **4.2 Memory Model: A Compromise to Physical Reality**
Abandoning the abstraction of unified memory is one of OTIR's most controversial yet central design decisions. While the unified memory model simplifies programming, it is essentially an "optimistic guess" that relies on complex hardware/driver paging mechanisms to migrate data on demand. In predictable, data-intensive HPC workloads, this guesswork is often suboptimal. A well-designed algorithm almost always knows when and what data is needed better than an operating system's paging mechanism.

OTIR's hierarchical memory space model (`#global`, `#workgroup`, `#private`) is a direct mapping of physical reality. It forces the programmer or a higher-level DSL compiler to think about **data locality** and **data reuse**. The `#workgroup` space is semantically a **user-managed scratchpad**, not an automatic cache. The theoretical basis for this design choice is that for streaming computations, an explicitly managed scratchpad always outperforms a general-purpose cache in terms of performance, as it avoids cache pollution and unnecessary write-back overhead.

### **4.3 Memory Consistency: Borrowing from C++ Instead of Hardware**
OTIR's memory consistency model does not reinvent the wheel but directly adopts the atomic operations and memory ordering semantics (`relaxed`, `acquire`, `release`) from the ISO C++20 standard. This decision is based on the following considerations:

1.  **Industry-Proven:** The C++ memory model has been vetted by decades of debate and practice, striking a precise balance between programmer intent and hardware capabilities.
2.  **Compiler-Friendly:** Mainstream compilers (LLVM/GCC) have mature optimizations and implementations for the C++ memory model. Aligning OTIR with this model greatly simplifies the complexity of mapping to `atomicrmw` instructions in LLVM IR.
3.  **Avoiding Hardware Pitfalls:** Directly exposing the weak memory model of specific hardware (e.g., early models of POWER or ARMv7) would create a significant portability disaster for higher levels. Adopting the C++ model provides a sufficiently powerful and portable abstraction layer.

The semantics of `otir.barrier` are strictly defined to provide Sequentially Consistent guarantees **only within the `#workgroup` space**. It does not affect the memory order of the `#global` space. Consistency guarantees for the global space must be explicitly declared through the `memory_order` parameter of atomic operations. This distinction is crucial as it avoids the unrealistic hardware requirement of implementing a cheap global barrier in distributed-memory or multi-chip systems.

---

## **5. Type System**

### **5.2 Tile Type Design Trade-offs**

#### **5.2.1 The Necessity of Static Shapes**
Requiring `Tile`s to have a static shape (a compile-time constant shape) is another seemingly strict but vital design decision. While dynamic shapes are flexible, they are a disaster for the compiler. A compiler's ability to optimize performance is directly proportional to its ability to statically predict memory access patterns.

With a statically-shaped `Tile`, the compiler can:
*   **Fully unroll loops:** Completely expand operations on the Tile into a series of scalar or vector instructions.
*   **Optimize register allocation:** Pre-calculate all virtual registers needed for an operation, performing precise liveness analysis and coloring.
*   **Statically plan pipelines:** Accurately calculate the instruction cycles for data movement (DMA) and computation at compile time, generating non-blocking software pipelines.

Dynamic shapes invalidate nearly all of the above optimizations, forcing the compiler to generate "slow path" code that relies on runtime checks and dynamic allocation. OTIR's philosophy is that dynamism should be confined to higher-level control flow (e.g., grid launch parameters) and should not permeate the `Tile` definitions at the core of the computation.

#### **5.2.4 The Strategic Value of Layout Opacity**
Layout opacity is key to achieving cross-hardware portability in OTIR. NVIDIA GPU Tensor Cores require operands to be arranged in registers in a specific swizzled layout, whereas Google TPU Systolic Arrays prefer a data stream ordered by wavefront. If the IR mandated a single layout (e.g., row-major), the backend would have to insert numerous expensive `shuffle` or `transpose` instructions for data reordering when mapping to this hardware.

By allowing the layout of `#workgroup` and `#private` spaces to be opaque, OTIR gives the responsibility and freedom of layout selection entirely to the backend, which knows the hardware best. The frontend only needs to describe the logical matrix multiplication. The backend can then silently choose the optimal physical layout for intermediate `Tile`s and perform a layout conversion only once when finally writing back to the `#global` space. This is a **"Deferred Layout Decision"** strategy, a key technique for achieving "zero-cost abstraction."

---

## **6. Instruction Set Architecture (ISA)**

This section details the operational semantics of the OTIR core instruction set. All instructions must satisfy the constraints on the type, shape, and memory space of their operands; otherwise, the program is ill-formed.

### **6.1 Context Intrinsics**
These instructions provide access to the abstract machine's execution environment.

#### **`otir.program_id`**
*   **Syntax:** `%result = otir.program_id axis(integer_literal)`
*   **Description:** Returns the coordinate index of the current workgroup within the multi-dimensional grid, for the dimension specified by `axis`. The value of `axis` must be a compile-time constant and less than the rank of the grid.
*   **Return Value:** An `index` type.

### **6.2 Memory Management and Transfer Operations**

#### **`otir.alloc`**
*   **Syntax:** `%tile = otir.alloc() : !otir.tile<...>`
*   **Description:** Allocates a block of uninitialized memory in the specified address space (typically `#workgroup` or `#private`). The returned `Tile` object is a reference to this memory region. The lifetime of this memory is determined by its scope and is typically released automatically upon function exit.
*   **Constraint:** `alloc` is not permitted in the `#global` space.

#### **`otir.dma_copy_async`**
*   **Syntax:** `%token = otir.dma_copy_async %src, %dst, [%mask] : (SrcTile, DstTile, [MaskTile]) -> !otir.token`
*   **Description:** Initiates an asynchronous data copy from a source `Tile` (`%src`) to a destination `Tile` (`%dst`). This instruction is non-blocking and immediately returns a `!otir.token`.
*   **Constraints:**
    1.  `%src` must typically be in the `#global` space, and `%dst` in the `#workgroup` space.
    2.  The element types of `%src` and `%dst` must be the same.
    3.  The optional `%mask` operand is a `Tile` of `i1` type used to implement a predicated copy, transferring only the elements where the mask is true.
*   **Rationale:** Making DMA operations explicit is a cornerstone of building high-performance software pipelines.

#### **`otir.load` and `otir.store`**
*   **Syntax:** `%val = otir.load %ptr`; `otir.store %val, %ptr`
*   **Description:** Synchronously loads or stores data between different memory levels. A typical use case is loading data from `#workgroup` space to `#private` space, or vice versa.
*   **Constraint:** `load` and `store` are synchronous operations; their completion is guaranteed before the program proceeds.

#### **`otir.atomic_rmw`**
*   **Syntax:** `%old_val = otir.atomic_rmw %ptr, %val, kind(atomic_op), memory_order(order)`
*   **Description:** Performs an atomic read-modify-write operation on a memory address in the `#global` space.
*   `kind`: Defines the type of atomic operation, such as `add`, `max`, `xor`.
*   `memory_order`: Defines the memory ordering, which must be one of `relaxed`, `acquire`, `release`, `acq_rel`. The semantics strictly align with C++20 atomic operations.

### **6.3 Arithmetic and Logical Operations (Element-wise)**
These instructions apply a function independently to each element of a `Tile`.

*   **Binary Ops:** `otir.add`, `otir.sub`, `otir.mul`, `otir.div_s`, `otir.div_u`, `otir.rem_s`, `otir.rem_u`, `otir.and`, `otir.or`, `otir.xor`, `otir.shl`, `otir.shr_s`, `otir.shr_u`.
*   **Unary Ops:** `otir.neg`, `otir.not`, `otir.abs`, `otir.sqrt`, `otir.exp`, `otir.log`.
*   **Comparison Ops:** `otir.cmp kind(predicate)`, where `predicate` includes `eq`, `ne`, `slt`, `sgt`, etc. Returns a `Tile` of `i1` type.
*   **Broadcast Semantics:** If the shapes of the operands do not match but are broadcast-compatible, the lower-rank operand will be broadcast to match the higher-rank operand.

### **6.4 Matrix Operations**

#### **`otir.matmul`**
*   **Syntax:** `%D = otir.matmul %A, %B, %C : (TypeA, TypeB, TypeC) -> TypeD`
*   **Semantics:** Performs a fused matrix multiply-add operation: $D = A \times B + C$.
*   **Constraints:**
    1.  **Memory Space:** All operands `%A`, `%B`, and `%C` must reside in the `#private` space.
    2.  **Shape:** Must satisfy the matrix multiplication shape constraint of `(M, K) x (K, N) -> (M, N)`.
    3.  **Precision:** The accumulator `%C` is allowed to have a higher precision than the inputs `%A` and `%B`.
*   **Implementation-defined:** The summation order of the inner products is implementation-defined. This may lead to bit-level differences in floating-point results across different implementations or optimization levels. Implementations requiring bit-exact results must constrain this behavior through additional specifications.

### **6.5 Transformation and Reshaping Operations**

#### **`otir.reshape`**
*   **Syntax:** `%reshaped = otir.reshape %input`
*   **Description:** Changes the logical shape of a `Tile` without altering the total number of elements or the physical layout. This is a zero-cost metadata operation.

#### **`otir.transpose`**
*   **Syntax:** `%transposed = otir.transpose %input, permutation(...)`
*   **Description:** Reorders the dimensions of a `Tile` according to the provided dimension permutation vector. This may trigger an actual data movement.

#### **`otir.extract_slice` / `otir.insert_slice`**
*   **Description:** Extracts a smaller `Tile` from or inserts it into a larger `Tile`. These are the core instructions for implementing tiling algorithms and register blocking.

### **6.6 Control Flow & Synchronization**

#### **`otir.barrier`**
*   **Syntax:** `otir.barrier`
*   **Description:** Executes a workgroup-level synchronization barrier. Ensures that all writes to the `#workgroup` space before the barrier are visible to all reads after the barrier.

#### **`otir.wait`**
*   **Syntax:** `otir.wait [%token1, %token2, ...]`
*   **Description:** Blocks the current execution flow until one or more asynchronous operations associated with the `!otir.token`s have all completed.

#### **Structured Control Flow (SCF)**
OTIR does not define its own control flow instructions but reuses the `scf` dialect from MLIR.
*   **`scf.for`:** Used to express loops with a fixed number of iterations.
*   **`scf.if`:** Used to express conditional branches.
*   **Constraint:** The use of unstructured jumps (e.g., `goto`) is prohibited, as they severely hinder dependency analysis and automatic pipeline optimization.

---

## **7. Conformance**

A conforming OTIR implementation (whether a frontend generator or a backend compiler) must satisfy all of the following conditions:

**7.1 Syntax and Type Checking**
The implementation must be able to correctly parse and validate OTIR code that adheres to sections 5.0 and 6.0 of this specification. Any program that violates type, shape, or memory space constraints must be diagnosed as ill-formed.

**7.2 Semantic Emulation**
At execution time, the observable behavior of the implementation (the final writes to the `#global` space) must be consistent with the abstract machine model defined in section 4.0 of this specification. In particular, the semantics of memory consistency and synchronization instructions must be respected.

**7.3 Documentation of Implementation-Defined Behavior**
For behaviors marked as "implementation-defined" in this specification (such as the summation order of `otir.matmul`), a conforming implementation must clearly document the specific behavior it adopts.

---
