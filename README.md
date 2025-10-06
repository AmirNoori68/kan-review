# kan-review

Curated, **table-first** guide to Kolmogorov–Arnold Networks (KANs): papers & code, grouped by topic.  
Each table lists only the **core** items (1–2 line description + links). For full context, see the review paper (PDF to be added).

## Quick Nav

- 4. Bridging KANs and MLPs
- 6. Basis Functions (coming next)
- 7. Accuracy Improvement (coming next)
- 8. Efficiency Improvement (coming next)
- 9. Sparsity & Regularization (coming next)
- 10. Convergence & Scaling Laws (coming next)

---
## Review and Survey Papers on KANs

| Title | Citation | Paper | 
|---|---|---|
| The first two months of Kolmogorov-Arnold Networks (KANs): A survey of the state-of-the-art | [Dutta25_review] | [Paper](https://link.springer.com/content/pdf/10.1007/s11831-025-10328-2.pdf) |
| KAT to KANs: A review of Kolmogorov–Arnold Networks and the neural leap forward | [Basina24_interp_review] | [Paper](https://arxiv.org/abs/2411.10622) |
| Scientific machine learning with Kolmogorov–Arnold Networks | [Faroughi25_review] | [Paper](https://arxiv.org/abs/2507.22959) |
| Kolmogorov-Arnold Networks: Overview of Architectures and Use Cases | [Essahraui25] | [Paper](https://ieeexplore.ieee.org/abstract/document/11135248) |
| Kolmogorov–Arnold Networks for interpretable and efficient function approximation | [Andrade25] | [Paper](https://www.preprints.org/manuscript/202504.1742/v1) |
| Scalable and interpretable function-based architectures: A survey of Kolmogorov–Arnold Networks | [Beatrize25] | [Paper](https://engrxiv.org/preprint/view/4515) |
| Convolutional Kolmogorov–Arnold Networks: A survey | [Kilani25] | [Paper](https://hal.science/hal-05177765/document) |
| Convolutional Kolmogorov–Arnold Networks | [Bonder25] | [Paper](https://arxiv.org/abs/2406.13155) |
| A survey on Kolmogorov–Arnold Networks | [Somvanshi25] | [Paper](https://arxiv.org/abs/2411.06078) |
| Kolmogorov-Arnold Networks: A Critical Assessment of Claims, Performance, and Practical Viability | [Hou25] | [Paper](https://arxiv.org/abs/2407.11075) |

---
## 📚 Representative Repositories

| Repository | Description | Citation |
|-------------|-------------|-----------|
| [KindXiaoming/pykan](https://github.com/KindXiaoming/pykan) | Official PyKAN for “KAN” and “KAN 2.0”. | [Liu24b] |
| [afrah/pinn_learnable_activation](https://github.com/afrah/pinn_learnable_activation) | Compares various KAN bases vs. MLP on PDEs. | [Farea25_BasisComp] |
| [1ssb/torchkan](https://github.com/1ssb/torchkan) | Simplified PyTorch KAN with multiple variants. | [TorchKAN] |
| [mintisan/awesome-kan](https://github.com/mintisan/awesome-kan) | Curated list of KAN resources, projects, and papers. | [awesomeKAN] |
| [sidhu2690/Deep-KAN](https://github.com/sidhu2690/Deep-KAN) | Spline-KAN examples and PyPI package. | [DeepKAN] |
| [sidhu2690/RBF-KAN](https://github.com/sidhu2690/RBF-KAN) | Gaussian RBF-based KAN implementation. | [RBFKAN] |
| [yu-rp/KANbeFair](https://github.com/yu-rp/KANbeFair) | Fair benchmarking of KANs vs. MLPs. | [Yu24_fairer] |
| [Blealtan/efficient-kan](https://github.com/Blealtan/efficient-kan) | Efficient PyTorch implementation of KAN. | [EfficientKAN] |
| [srigas/jaxKAN](https://github.com/srigas/jaxKAN) | JAX-based KAN with grid extension support. | [pde_Rigas24] |
| [ZiyaoLi/fast-kan](https://github.com/ZiyaoLi/fast-kan) | FastKAN using RBFs for acceleration. | [Li24] |
| [AthanasiosDelis/faster-kan](https://github.com/AthanasiosDelis/faster-kan) | Uses reflectional switch activations. | [Athanasios2024] |
| [Indoxer/LKAN](https://github.com/Indoxer/LKAN) | Lightweight KAN variants and experiments. | [liu2024kan] |
| [pnnl/neuromancer (fbkans branch)](https://github.com/pnnl/neuromancer/tree/feature/fbkans) | Partition of unity (FBKAN) for PDE solving. | [pde_fbkan_Howard24] |
| [quiqi/relu_kan](https://github.com/quiqi/relu_kan) | Minimal ReLU-KAN example. | [Qiu24] |
| [OSU-STARLAB/MatrixKAN](https://github.com/OSU-STARLAB/MatrixKAN) | Matrix-parallelized KAN implementation. | [Coffman25] |
| [Iri-sated/PowerMLP](https://github.com/Iri-sated/PowerMLP) | MLP-type network with KAN-level expressiveness. | [Qiu25] |
| [GistNoesis/FourierKAN](https://github.com/GistNoesis/FourierKAN) | Fourier-based KAN layer. | [FourierKAN] |
| [GistNoesis/FusedFourierKAN](https://github.com/GistNoesis/FusedFourierKAN) | Optimized FourierKAN with fused GPU kernels. | [FusedFourierKAN] |
| [alirezaafzalaghaei/fKAN](https://github.com/alirezaafzalaghaei/fKAN) | Fractional KAN using Jacobi functions. | [Aghaei24_fkan] |
| [alirezaafzalaghaei/rKAN](https://github.com/alirezaafzalaghaei/rKAN) | Rational KAN (Padé/Jacobi rational designs). | [Aghaei24_rkan] |
| [M-Wolff/CVKAN](https://github.com/M-Wolff/CVKAN) | Complex-valued KANs. | [Wolff25] |
| [DUCH714/SincKAN](https://github.com/DUCH714/SincKAN) | Sinc-based KAN for PINN applications. | [Yu24] |
| [SynodicMonth/ChebyKAN](https://github.com/SynodicMonth/ChebyKAN) | Chebyshev polynomial-based KAN. | [ChebyKAN] |
| [Boris-73-TA/OrthogPolyKANs](https://github.com/Boris-73-TA/OrthogPolyKANs) | Orthogonal polynomial-based KAN implementations. | [OrthogPolyKANs] |
| [kolmogorovArnoldFourierNetwork/kaf_act](https://github.com/kolmogorovArnoldFourierNetwork/kaf_act) | RFF-based activation library. | [kaf_act] |
| [kolmogorovArnoldFourierNetwork/KAF](https://github.com/kolmogorovArnoldFourierNetwork/KAF) | Kolmogorov–Arnold Fourier Networks. | [Zhang25] |
| [kelvinhkcs/HRKAN](https://github.com/kelvinhkcs/HRKAN) | Higher-order ReLU-KANs. | [KAN_pde_So24] |
| [yizheng-wang/KINN](https://github.com/yizheng-wang/Research-on-Solving-Partial-Differential-Equations-of-Solid-Mechanics-Based-on-PINN) | PIKAN for solid mechanics PDEs. | [pde_Wang24] |
| [Ali-Stanford/KAN_PointNet_CFD](https://github.com/Ali-Stanford/KAN_PointNet_CFD) | Jacobi-based KAN for CFD predictions. | [Kashefi25] |
| [Jinfeng-Xu/FKAN-GCF](https://github.com/Jinfeng-Xu/FKAN-GCF) | FourierKAN-GCF for graph filtering. | [Xu25_fourier] |
| [jdtoscano94/KKANs_PIML](https://github.com/jdtoscano94/KKANs_PIML) | Kurkova-KANs combining MLP with basis functions. | [Toscano24_kkan] |
| [Zhangyanbo/MLP-KAN](https://github.com/Zhangyanbo/MLP-KAN) | MLP-augmented KAN activations. | [MLP-KAN] |
| [Adamdad/kat](https://github.com/Adamdad/kat) | Kolmogorov–Arnold Transformer. | [Yang25_transformer] |
| [YihongDong/FAN](https://github.com/YihongDong/FAN) | Fourier Analysis Network (FAN). | [Dong25_FAN] |
| [seydi1370/Basis_Functions](https://github.com/seydi1370/Basis_Functions) | Polynomial bases for KANs. | [Seydi24a] |
| [zavareh1/Wav-KAN](https://github.com/zavareh1/Wav-KAN) | Wavelet-based KANs. | [Bozorgasl24] |
| [Jim137/qkan](https://github.com/Jim137/qkan) | Quantum-inspired KAN variants and pruning. | [Jiang25_quantum] |
| [liouvill/KAN-Converge](https://github.com/liouvill/KAN-Converge) | Additive & hybrid KANs for convergence-rate experiments. | [Liu25_convergence] |


---
## 4. Bridging KANs and MLPs

| Brief result | Citation | Paper | Code |
|---|---|---|---|
| Equivalence: ReLU^k MLP ↔ B-spline KAN. | [Wang25] | [Paper](https://arxiv.org/abs/2410.01803) | - |
| Piecewise-linear KAN = ReLU MLP. | [Schoots25] | [Paper](https://arxiv.org/abs/2503.01702) | - |
| Adaptive spline KANs mimic MLPs with data-driven capacity. | [Actor25] | [Paper](https://arxiv.org/abs/2505.18131) | - |
| NTK view: richer KAN bases reduce spectral bias vs MLP. | [Gao25] | [Paper](https://arxiv.org/abs/2410.08041) | - |


---


## 6. Basis Functions

| Name | Support | Equation | Grid | Type | Citation | Paper | Code |
|---|---|---|---|---|---|---|---|
| B-spline | Local | $\sum_n c_nB_n(x)$ | Yes | B-spline | [Liu24] | [Paper](https://arxiv.org/abs/2404.19756) [Paper](https://arxiv.org/abs/2408.10205) | [Code](https://github.com/KindXiaoming/pykan) |
| Chebyshev | Global | $\sum_k c_kT_k(\tanh x)$ | No | Chebyshev + tanh | [SS24] | [Paper](https://arxiv.org/abs/2405.07200) | - |
| Stabilized Chebyshev | Global | $\tanh\big(\sum_k c_kT_k(\tanh x)\big)$ | No | Chebyshev + linear head | [Daryakenari25] | [Paper](https://arxiv.org/abs/2504.07379) | - |
| Chebyshev (grid) | Global | $\sum_k c_kT_k\Big(\tfrac{1}{m}\sum_i \tanh(w_i x+b_i)\Big)$ | Yes | Chebyshev + tanh | [Toscano24_kkan] | [Paper](https://arxiv.org/abs/2412.16738) | [Code](https://github.com/jdtoscano94/KKANs_PIML) |
| ReLU-KAN | Local | $\sum_i w_iR_i(x)$ | Yes | Squared ReLU | [Qiu24] | [Paper](link) | [Code](https://github.com/quiqi/relu_kan) |
| HRKAN | Local | $\sum_i w_i\big[\mathrm{ReLU}(x)\big]^m$ | Yes | Polynomial ReLU | [KAN_pde_So24] | [Paper](https://arxiv.org/abs/2409.14248) | [Code](https://github.com/kelvinhkcs/HRKAN) |
| Adaptive ReLU-KAN | Local | $\sum_i w_iv_i(x)$ | Yes | Adaptive ReLU | [pde_Rigas24] | [Paper](https://arxiv.org/abs/2407.17611) | [Code](https://github.com/srigas/jaxKAN) |
| fKAN (Jacobi) | Global | $\sum_n c_nP_n(x)$ | No | Jacobi | [Aghaei24_fkan] | [Paper](https://arxiv.org/abs/2406.07456) | [Code](https://github.com/alirezaafzalaghaei/fKAN) |
| rKAN (Padé/Jacobi) | Global | $\dfrac{\sum_i a_iP_i(x)}{\sum_j b_jP_j(x)}$ | No | Rational + Jacobi | [Aghaei24_rkan] | [Paper](https://arxiv.org/abs/2406.14495) | [Code](https://github.com/alirezaafzalaghaei/rKAN) |
| Jacobi-KAN | Global | $\sum_i c_iP_i(\tanh x)$ | No | Jacobi + tanh | [Kashefi25] | [Paper](https://arxiv.org/abs/2408.02950) | [Code](https://github.com/Ali-Stanford/KAN_PointNet_CFD) |
| FourierKAN | Global | $\sum_k a_k\cos(kx)+b_k\sin(kx)$ | No | Fourier | [Xu25_fourier] | [Paper](https://arxiv.org/abs/2406.01034v3) | [Code](https://github.com/Jinfeng-Xu/FKAN-GCF) |
| KAF | Global | $\alpha\mathrm{GELU}(x)+\sum_j \beta_j\psi_j(x)$ | No | RFF + GELU | [Zhang25] | [Paper](https://arxiv.org/abs/2502.06018) | [Code](https://github.com/kolmogorovArnoldFourierNetwork/KAF) |
| Gaussian (FastKAN) | Local | $\sum_i w_i\exp\Big(-\big(\tfrac{x-g_i}{\varepsilon}\big)^2\Big)$ | Yes | Gaussian RBF | [Li24] | [Paper](https://arxiv.org/abs/2405.06721) | [Code](https://github.com/ZiyaoLi/fast-kan) |
| RSWAF-KAN | Local | $\sum_i w_i\left(s_i-\tanh^2\big(\tfrac{x-c_i}{h_i}\big)\right)$ | Yes | Switch (tanh$^2$) | [Athanasios2024] | - | [Code](https://github.com/AthanasiosDelis/faster-kan) |
| CVKAN | Local | $\sum_{u,v} w_{uv}\exp\big(-\lvert z-g_{uv}\rvert^2\big)$ | Yes | Complex Gaussian | [Wolff25] | [Paper](https://arxiv.org/abs/2502.02417) | [Code](https://github.com/M-Wolff/CVKAN) |
| BSRBF-KAN | Local | $\sum_i a_i B_i(x)+\sum_j b_j\exp\big(-\tfrac{(x-g_j)^2}{\varepsilon^2}\big)$ | Yes | B-spline + Gaussian | [Ta24] | [Paper](https://arxiv.org/abs/2406.11173) | [Code](https://github.com/hoangthangta/BSRBF_KAN) |
| Wav-KAN | Local | $\sum_{j,k} c_{j,k}\psi\big(\tfrac{x-u_{j,k}}{s_j}\big)$ | No | Wavelet | [Bozorgasl24] | [Paper](https://arxiv.org/abs/2405.12832) | [Code](https://github.com/zavareh1/Wav-KAN}{zavareh1/Wav-KAN) |
| FBKAN | Local | $\sum_j \omega_j(x)K_j(x)$ | Yes | PoU + B-spline | [pde_fbkan_Howard24] | [Paper](https://arxiv.org/abs/2406.19662) | [Code](https://github.com/pnnl/neuromancer/tree/feature/fbkans) |
| SincKAN | Global | $\sum_{i=-N}^{N} c_i\mathrm{sinc}\left(\frac{\pi}{h}(x - i h)\right)$ | Yes | Sinc | [Yu24] | [Paper](https://arxiv.org/abs/2410.04096) | [Code](https://github.com/DUCH714/SincKAN}{DUCH714/SincKAN) |
| Poly-KAN | Global | $\sum_i w_iP_i(x)$ | No | Polynomial | [Seydi24a] | [Paper](https://arxiv.org/abs/2406.02583) | [Code](https://github.com/seydi1370/Basis_Functions) |


---

## 7. Accuracy Improvement

### 7.1 Physics & Loss Design

| Brief result | Citation | Paper | Code |
|---|---|---|---|
| Physics-informed KAN (cPIKAN): residual attention, entropy-viscosity. | [KAN_pde_Shukla24] | [Paper](https://arxiv.org/abs/2406.02917) | - |
| KAN-PINN for strongly nonlinear PDEs (actuator deflection). | [Zhang25_comp] | [Paper](https://doi.org/10.1016/j.engappai.2025.110126) | - |
| Attention-guided KAN with NSE residuals + BC losses. | [Yang25] | [Paper](https://doi.org/10.1016/j.jcp.2025.113846) | - |
| Residual physics + sparse regression (variable-coeff. PDEs). | [Guo25] | [Paper](https://doi.org/10.1016/j.physd.2025.134689) | -|
| Self-scaled residual reweighting (ssRBA). | [Toscano24_kkan] | [Paper](https://arxiv.org/abs/2412.16738) | [Code](https://github.com/jdtoscano94/KKANs_PIML) |
| Augmented-Lagrangian PINN–KAN (learnable multipliers). | [pde_Zhang24] | [Paper](https://doi.org/10.1038/s41598-025-92900-1) | - |
| Velocity–vorticity loss for turbulence reconstruction. | [Toscano25_aivt] | [Paper](https://doi.org/10.1126/sciadv.ads5236) | - |
| Fractional/integro-diff. operators in KAN. | [Aghaei24_kantorol] | [Paper](https://arxiv.org/abs/2409.06649) | - |

### 7.2 Adaptive Sampling & Grids

| Brief result | Citation | Paper | Code |
|---|---|---|---|
| Multilevel knots (coarse→fine) for nested spline spaces. | [Actor25] | [Paper](https://arxiv.org/abs/2505.18131) | - |
| Free-knot KAN (trainable knots via cumulative softmax). | [Actor25] | [Paper](https://arxiv.org/abs/2505.18131) | - |
| Grid extension with optimizer state transition. | [pde_Rigas24] | [Paper](https://arxiv.org/abs/2407.17611) | [Code](https://github.com/srigas/jaxKAN) |
| Residual-adaptive sampling (RAD). | [pde_Rigas24] | [Paper](https://arxiv.org/abs/2407.17611) | [Code](https://github.com/srigas/jaxKAN) |
| Multi-resolution sampling schedule for cPIKAN. | [Yang25_multiScale] | [Paper](https://arxiv.org/abs/2507.19888) | - |

### 7.3 Domain Decomposition

| Brief result | Citation | Paper | Code |
|---|---|---|---|
| Finite-basis KAN (FBKAN) with PoU blending of local KANs. | [pde_fbkan_Howard24] | [Paper](https://arxiv.org/abs/2406.19662) | [Code](https://github.com/pnnl/neuromancer/tree/feature/fbkans) |
| Temporal subdomains to improve NTK conditioning. | [Faroughi25] | [Paper](https://arxiv.org/abs/2506.07958) | - |

### 7.4 Function Decomposition

| Brief result | Citation | Paper | Code |
|---|---|---|---|
| Multi-fidelity KAN (freeze LF, learn HF linear + nonlinear heads). | [pde_Howard24] | [Paper](https://arxiv.org/abs/2410.14764) | [Code](https://github.com/pnnl/neuromancer/tree/feature/fbkans) |
| Separable PIKAN (sum of products of 1D KAN factors). | [pde_jacob24] | [Paper](https://arxiv.org/abs/2411.06286) | - |
| KAN-SR: recursive simplification for symbolic discovery. | [Buhler25_regression] | [Paper](https://arxiv.org/abs/2509.10089) | - |

### 7.5 Hybrid / Ensemble & Data

| Brief result | Citation | Paper | Code |
|---|---|---|---|
| MLP–KAN mixture of experts. | [He24] | [Paper](https://arxiv.org/abs/2410.03027) | [Code](https://github.com/DLYuanGod/MLP-KAN) |
| Parallel KAN ∥ MLP branches with learnable fusion. | [Xu25] | [Paper](https://arxiv.org/abs/2503.23289) | - |
| KKAN: per-dim MLP features + explicit basis expansion. | [Toscano24_kkan] | [Paper](https://arxiv.org/abs/2412.16738) | [Code](https://github.com/jdtoscano94/KKANs_PIML) |

### 7.6 Sequence / Attention Hybrids

| Brief result | Citation | Paper | Code |
|---|---|---|---|
| FlashKAT: group-rational KAN blocks in Transformers. | [Raffel25] | [Paper](https://arxiv.org/abs/2505.13813) | - |
| GINN-KAN: interpretable growth + KAN in PINNs. | [pde_Ranasinghe24] | [Paper](https://arxiv.org/abs/2408.14780) | - |
| KAN-ODE: KAN as $\dot u$ model (adjoint training). | [pde_Koeing24] | [Paper](https://arxiv.org/abs/2407.04192) | - |
| AAKAN-WGAN: adaptive KAN + GAN for data augmentation. | [Shen25] | [Paper](https://doi.org/10.1016/j.mtcomm.2025.113198) | - |
| Attention-KAN-PINN for battery SOH forecasting. | [Wei26_battery] | [Paper](https://doi.org/10.1016/j.eswa.2025.128969) | - |

### 7.7 Discontinuities & Sharp Gradients

| Brief result | Citation | Paper | Code |
|---|---|---|---|
| SincKAN for kinks/boundary layers. | [Yu24] | [Paper](https://arxiv.org/abs/2410.04096) | [Code](https://github.com/DUCH714/SincKAN}{DUCH714/SincKAN) |
| rKAN (rational bases) for asymptotics/jumps. | [Aghaei24_rkan] | [Paper](https://arxiv.org/abs/2406.14495) | [Code](https://github.com/alirezaafzalaghaei/rKAN) |
| DKAN: $\tanh$ jump gate + spline background. | [Lei25] | [Paper](https://arxiv.org/abs/2507.08338) | - |
| KINN for singularities/stress concentrations. | [pde_Wang24] | [Paper](https://doi.org/10.1016/j.cma.2024.117518) | [Code](https://github.com/yizheng-wang/Research-on-Solving-Partial-Differential-Equations-of-Solid-Mechanics-Based-on-PINN) |
| Two-phase PINN–KAN for saturation fronts. | [Kalesh25] | [Paper](https://link.springer.com/chapter/10.1007/978-3-031-97596-7_30) | - |

### 7.8 Optimization & Adaptive Training

| Brief result | Citation | Paper | Code |
|---|---|---|---|
| Adam/RAdam warmup → (L-)BFGS refinement. | [Mostajeran25], [Daryakenari25], [KAN_pde_Zeng24] | [Paper](https://arxiv.org/abs/2501.02762) [Paper](https://arxiv.org/abs/2504.07379) [Paper](https://arxiv.org/abs/2408.07906)| - |
| Hybrid optimizers for sharp fronts. | [Kalesh25] | [Paper](https://link.springer.com/chapter/10.1007/978-3-031-97596-7_30) | - |
| Bayesian hyperparameter tuning for KANs. | [Lin25_geo] | [Paper](https://doi.org/10.1016/j.jrmge.2025.02.023) | - |
| Bayesian PINN–KAN (variational + KL) for UQ. | [pde_bayesian_Giroux24] | [Paper](https://arxiv.org/abs/2410.01687) | [Code](https://github.com/wmdataphys/Bayesian-HR-KAN) |
| NTK perspective: conditioning ↔ convergence. | [Faroughi25] | [Paper](https://arxiv.org/abs/2506.07958) | - |


---

## 8. Efficiency Improvement

### 8.1 Parallelism, GPU, and JAX Engineering

| Brief result | Citation | Paper | Code |
|---|---|---|---|
| ReLU^m activations replace splines (CUDA-friendly). | [Qiu25] | [Paper](https://arxiv.org/abs/2412.13571) | [Code](https://github.com/Iri-sated/PowerMLP) |
| Spline→matmul CUDA kernels (GEMM fusion). | [Qiu24], [KAN_pde_So24] | [Paper](https://arxiv.org/abs/2406.02075) [Paper](https://arxiv.org/abs/2409.14248) | [Code](https://github.com/quiqi/relu_kan) |
| Matrix B-spline evaluation fused on GPU. | [Coffman25] | [Paper](https://arxiv.org/abs/2502.07176) | [Code](https://github.com/OSU-STARLAB/MatrixKAN) |
| Dual-matrix merge + trainable RFF for scaling. | [Zhang25] | [Paper](https://arxiv.org/abs/2502.06018) | [Code](https://github.com/kolmogorovArnoldFourierNetwork/KAF) |
| Custom GPU backward for KAN attention blocks. | [Raffel25] | [Paper](https://arxiv.org/abs/2505.13813) | - |
| Parallel KAN ∥ MLP branches (stream/layer parallelism). | [Xu25] | [Paper](https://arxiv.org/abs/2503.23289) | - |
| Domain decomposition parallelism (multi-GPU, PoU/separable). | [KAN_pde_Shukla24], [pde_fbkan_Howard24], [pde_jacob24] | [Paper](https://arxiv.org/abs/2406.02917) [paper](https://arxiv.org/abs/2406.19662) [Paper](https://arxiv.org/abs/2411.06286)| [Code](https://github.com/pnnl/neuromancer/tree/feature/fbkans) |
| JAX/XLA: `jit`/`vmap`/`pmap`, fusion, memory-aware. | [Daryakenari25], [pde_Rigas24] | [Paper](https://arxiv.org/abs/2407.17611) [Paper](https://arxiv.org/abs/2504.07379) | [Code](https://github.com/srigas/jaxKAN) |

### 8.2 Matrix Optimization & Parameter-Efficient Bases

| Brief result | Citation | Paper | Code |
|---|---|---|---|
| ReLU-power vs B-splines: fewer params, vectorized polynomials. | [Qiu25], [Qiu24], [KAN_pde_So24] | [Paper](https://arxiv.org/abs/2412.13571) [Paper](https://arxiv.org/abs/2406.02075) [Paper](https://arxiv.org/abs/2409.14248) | [Code](https://github.com/quiqi/relu_kan) [Code](https://github.com/kelvinhkcs/HRKAN)|
| Orthogonal polynomials with cheap recurrences. | [KAN_pde_Shukla24], [Guo24], [Mostajeran24], [Mostajeran25], [pde_Wang24] | [Paper](https://arxiv.org/abs/2406.02917) [Paper](https://arxiv.org/abs/2410.10897) [Paper](https://doi.org/10.1016/j.cma.2024.117518) [Paper](https://arxiv.org/abs/2501.02762)| [Code](https://github.com/yizheng-wang/Research-on-Solving-Partial-Differential-Equations-of-Solid-Mechanics-Based-on-PINN) |
| Compact RBF bases (local Gaussians). | [Lin25_geo], [pde_Koeing24] | [Paper](https://doi.org/10.1016/j.jrmge.2025.02.023) [Paper](https://arxiv.org/abs/2407.04192) | - |
| Wavelets for multi-resolution and sparse coeffs. | [pde_Patra24] | [Paper](https://arxiv.org/abs/2407.18373) | - |
| Dual-matrix + RFF compression to cut memory traffic. | [Zhang25] | [Paper](https://arxiv.org/abs/2502.06018) | [Code](https://github.com/kolmogorovArnoldFourierNetwork/KAF) |
| Sparsity regularization (ℓ1/group) with pruning. | [Guo25] | [Paper](https://doi.org/10.1016/j.physd.2025.134689) | - |
| Hierarchical channel-wise refinement (shared params). | [Actor25] | [Paper](https://arxiv.org/abs/2505.18131) | - |
| DEKAN: connectivity via Differential Evolution. | [Li25_DEKAN] | [Paper](https://doi.org/10.1109/CEC65147.2025.11043029) | - |
| Mix spectral (derivatives) + spatial (coeffs) sparsity for operators. | [Lee25_operator] | [Paper](https://arxiv.org/abs/2509.16825) | - |



---

## 9. Sparsity & Regularization

### 9.1 ℓ1 sparsity with entropy balancing

| Brief result | Citation | Paper | Code |
|---|---|---|---|
| Layerwise ℓ1 on edge activations + entropy balance. | [Liu24] | [Paper](https://arxiv.org/abs/2404.19756) | [Code](https://github.com/KindXiaoming/pykan) |
| EfficientKAN: direct ℓ1 on weights (simple, practical). | [EfficientKAN] | - | [Code](https://github.com/Blealtan/efficient-kan) |
| Sparse symbolic discovery with ℓ1 + entropy. | [Wang25] | [Paper](https://arxiv.org/abs/2410.01803) | - |
| PDE KAN: ℓ1 + smoothness penalty to denoise coefficients. | [Guo25] | [Paper](https://doi.org/10.1016/j.physd.2025.134689) | - |
| Post-training pruning with layerwise ℓ1. | [pde_Koeing24] | [Paper](https://arxiv.org/abs/2407.04192) | - |
| KAN-SR: magnitude + entropy at subunit level (+ℓ1 on bases). | [Buhler25_regression] | [Paper](https://arxiv.org/abs/2509.10089) | - |

### 9.2 ℓ2 weight decay and extensions

| Brief result | Citation | Paper | Code |
|---|---|---|---|
| AAKAN: ℓ2 + temporal smoothing + MI regularizer. | [Shen25] | [Paper](https://doi.org/10.1016/j.mtcomm.2025.113198) | - |
| Small ℓ2 (e.g., 1e−5) improves stability in PINNs/DeepOKAN. | [KAN_pde_Shukla24], [Toscano25_aivt] | [Paper](https://arxiv.org/abs/2406.02917) [paper](https://doi.org/10.1126/sciadv.ads5236) | - |

### 9.3 Implicit and dropout-style regularizers

| Brief result | Citation | Paper | Code |
|---|---|---|---|
| Nested activations (e.g., tanh∘tanh) for bounded outputs & smooth grads. | [Daryakenari25] | [Paper](https://arxiv.org/abs/2504.07379) | - |
| DropKAN: post-activation masking (noise after spline eval). | [Altarabichi24] | [Paper](link) | - |


---

## 10. Convergence & Scaling Laws (KAN/PIKAN only)

### 10.1 Approximation & Sample Complexity

| Brief result | Citation | Paper | Code |
|---|---|---|---|
| Depth-rate $L^{-2s/d}$ for spline KAN (fixed width, $G = 2$). | [Wang25] | [Paper](https://arxiv.org/abs/2410.01803) | — |
| Minimax sample rate for spline KANs; additive case is dimension-free. | [Liu25_convergence] | [Paper](https://arxiv.org/abs/2502.04072) | [Code](https://github.com/liouvill/KAN-Converge) |

### 10.2 Optimization Dynamics & Spectral Bias
| Brief result | Citation | Paper | Code |
|---|---|---|---|
| NTK view: learnable bases flatten spectra; less spectral bias than MLP. | [Gao25] | [Paper](https://arxiv.org/abs/2410.08041) | — |
| Basis/grid enrichment widens NTK spectrum; speeds high-freq learning. | [Farea25_BasisComp] | [Paper](https://github.com/afrah/pinn_learnable_activation) | — |
| KAN vs MLP: earlier capture of high frequencies under NTK dynamics. | [Wang25] | [Paper](https://arxiv.org/abs/2410.01803) | — |

### 10.3 Empirical Power Laws

| Brief result | Citation | Paper | Code |
|---|---|---|---|
| Error follows $\ell \propto P^{-\alpha}$; increasing grid $G$ boosts accuracy. | [Liu24] | [Paper](https://arxiv.org/abs/2405.04354) | — |
| Depth (via dyadic grids) improves accuracy consistent with theory. | [Wang25] | [Paper](https://arxiv.org/abs/2410.01803) | — |


---


