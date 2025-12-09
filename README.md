# kan-review
A structured companion to our [KAN review paper](https://www.researchgate.net/publication/397082000_A_Practitioner's_Guide_to_Kolmogorov-Arnold_Networks).

We **welcome corrections, discussions, and new contributions** — The updates below come from recent communications with researchers and newly released studies.  
If you notice any missing or misattributed references, kindly contact [amir_noori@hkbu.edu.hk](mailto:amir_noori@hkbu.edu.hk) so they can be added in the next GitHub update and preprint revision.

## Quick Nav
1. [Citation](#1-citation)
2. [History of Kolmogorov Superposition Theorem](#2-history-of-kolmogorov-superposition-theorem)
3. [Review and Survey Papers on KANs](#3-review-and-survey-papers-on-kans)
4. [Representative Repositories](#4-representative-repositories-for-regression-function-approximation-and-pde-solving)
5. [Bridging KANs and MLPs](#5-bridging-kans-and-mlps)
6. [Basis Functions](#6-basis-functions)
7. [Accuracy Improvement](#7-accuracy-improvement)
   - [7.1 Physics & Loss Design](#71-physics--loss-design)
   - [7.2 Adaptive Sampling & Grids](#72-adaptive-sampling--grids)
   - [7.3 Domain Decomposition](#73-domain-decomposition)
   - [7.4 Function Decomposition](#74-function-decomposition)
   - [7.5 Hybrid / Ensemble & Data](#75-hybrid--ensemble--data)
   - [7.6 Sequence / Attention Hybrids](#76-sequence--attention-hybrids)
   - [7.7 Discontinuities & Sharp Gradients](#77-discontinuities--sharp-gradients)
   - [7.8 Optimization & Adaptive Training](#78-optimization--adaptive-training)
8. [Efficiency Improvement](#8-efficiency-improvement)
   - [8.1 Parallelism, GPU, and JAX Engineering](#81-parallelism-gpu-and-jax-engineering)
   - [8.2 Matrix Optimization & Parameter-Efficient Bases](#82-matrix-optimization--parameter-efficient-bases)
9. [Sparsity & Regularization](#9-sparsity--regularization)
   - [9.1 ℓ1 Sparsity with Entropy Balancing](#91-ℓ1-sparsity-with-entropy-balancing)
   - [9.2 ℓ2 Weight Decay and Extensions](#92-ℓ2-weight-decay-and-extensions)
   - [9.3 Implicit and Dropout-Style Regularizers](#93-implicit-and-dropout-style-regularizers)
10. [Convergence & Scaling Laws](#10-convergence--scaling-laws-kanpikan-only)
    - [10.1 Approximation & Sample Complexity](#101-approximation--sample-complexity)
    - [10.2 Optimization Dynamics & Spectral Bias](#102-optimization-dynamics--spectral-bias)
    - [10.3 Empirical Power Laws](#103-empirical-power-laws)



---

## 1.  Citation
Paper and repository reference information:

```bibtex
@misc{GuideToKAN, 
  title     = {A Practitioner's Guide to Kolmogorov-Arnold Networks}, 
  author    = {Amir Noorizadegan and Sifan Wang and Leevan Ling},
  year      = {2025},
  eprint    = {2510.25781},
  archivePrefix = {arXiv},
  primaryClass  = {cs.LG},
  url       = {https://arxiv.org/abs/2510.25781}
}

@misc{GuideToKAN_GitHub,
  author    = {Amir Noorizadegan},
  title     = {KAN Review: Companion Repository for "A Practitioner's Guide to Kolmogorov-Arnold Networks"},
  year      = {2025},
  howpublished = {\url{https://github.com/AmirNoori68/kan-review}},
  note      = {Accessed: 2025-11-01}
}
```
---
## 2.  History of Kolmogorov Superposition Theorem
| Year | Reference | Key Contribution |
|------|-----------|------------------|
| 1900 | [Hilbert](#) | Poses Hilbert's 13th problem |
| 1956 | [Kolmogorov](#) | Preliminary idea of superpositions; first hint toward the theorem |
| 1957 | [Arnol'd](#) | First explicit 3-variable construction (9 terms); counterexample to Hilbert 13 |
| 1957 | [Kolmogorov](#) | Full Kolmogorov Superposition Theorem; first general \(n\)-D proof |
| 1958 | [Arnol'd](#) | Supplies missing lemmas; completes Kolmogorov’s proof |
| 1962 | [Lorentz](#) | Simplified canonical form with a **single outer function** |
| 1965 | [Sprecher](#) | First **single universal inner function**; reduces all \(\phi_{p,q}\) to one \(\psi\) |
| 1967 | [Fridman](#) | Shows universal inner functions can be taken **Lipschitz-1** |
| 1980 | [de Figueiredo](#) | First network-like interpretation; block diagram + learned outer function (Chebyshev basis) |
| 1987 | [Hecht–Nielsen](#) | First explicit **neural mapping theorem** based on KST |
| 1989 | [Girosi–Poggio](#) | First rigorous critique: inner functions must be non-smooth; outer functions non-parametric |
| 1989 | [Frisch et al.](#) | First computational implementation of Lorentz form; iterative outer-function learning |
| 1991 | [Kurková](#) | First approximation-theoretic reinterpretation; relates network size to modulus of continuity |
| 1992 | [Kurková](#) | Two-hidden-layer sigmoidal approximants; universal inner weights |
| 1993 | [Sprecher](#) | Single universal \(\psi\) valid for **all** \(n\) |
| 1993 | [Nakamura et al.](#) | First **fully constructive** version with guaranteed accuracy |
| 1994 | [Nees](#) | First piecewise-linear inner maps with geometric error decay; constructive algorithm |
| 1996 | [Sprecher](#) | First executable version of \(\psi\) with verified separation property |
| 1997 | [Sprecher](#) | Explicit constructive algorithm for the outer functions |
| 2002 | [Köppen](#) | Corrected continuous monotone inner function; first training-ready KST inner map |
| 2003 | [Igelnik–Parikh](#) | Kolmogorov Spline Network (KSN): trainable spline-based inner/outer functions |
| 2009 | [Braun–Griebel](#) | First **correct constructive** KST; repairs Sprecher’s scheme |
| 2019 | [Actor–Knepley](#) | Proves **\(C^1\) inner functions impossible**; smoothness obstruction |
| 2024 | [Liu et al.](#) | Introduces **KAN**, the first deep architecture inspired by the Kolmogorov–Arnold representation |

---

## 3. Review and Survey Papers on KANs

| Title |  Paper | 
|---|---|
| The first two months of Kolmogorov-Arnold Networks (KANs): A survey of the state-of-the-art |  [Dutta](https://link.springer.com/content/pdf/10.1007/s11831-025-10328-2.pdf) |
| KAT to KANs: A review of Kolmogorov–Arnold Networks and the neural leap forward | [Basina](https://arxiv.org/abs/2411.10622) |
| Scientific machine learning with Kolmogorov–Arnold Networks | [Faroughi](https://arxiv.org/abs/2507.22959) |
| Kolmogorov-Arnold Networks: Overview of Architectures and Use Cases | [Essahraui](https://ieeexplore.ieee.org/abstract/document/11135248) |
| Kolmogorov–Arnold Networks for interpretable and efficient function approximation | [Andrade](https://www.preprints.org/manuscript/202504.1742/v1) |
| Scalable and interpretable function-based architectures: A survey of Kolmogorov–Arnold Networks | [Beatrize](https://engrxiv.org/preprint/view/4515) |
| Convolutional Kolmogorov–Arnold Networks: A survey | [Kilani](https://hal.science/hal-05177765/document) |
| Convolutional Kolmogorov–Arnold Networks| [Bonder](https://arxiv.org/abs/2406.13155) |
| A survey on Kolmogorov–Arnold Networks | [Somvanshi](https://arxiv.org/abs/2411.06078) |
| Kolmogorov-Arnold Networks: A Critical Assessment of Claims, Performance, and Practical Viability | [Hou](https://arxiv.org/abs/2407.11075) |

---
## 4. Representative Repositories (for regression, function approximation, and PDE solving)

| Repository | Description |
|-------------|-------------|
| [.../pykan](https://github.com/KindXiaoming/pykan) | Official PyKAN for “KAN” and “KAN 2.0”. | 
| [.../pinn_learnable_activation](https://github.com/afrah/pinn_learnable_activation) | Compares various KAN bases vs. MLP on PDEs. | 
| [.../torchkan](https://github.com/1ssb/torchkan) | Simplified PyTorch KAN with multiple variants. | 
| [.../awesome-kan](https://github.com/mintisan/awesome-kan) | Curated list of KAN resources, projects, and papers. | 
| [.../Deep-KAN](https://github.com/sidhu2690/Deep-KAN) | Spline-KAN examples and PyPI package. | 
| [.../RBF-KAN](https://github.com/sidhu2690/RBF-KAN) | Gaussian RBF-based KAN implementation. | 
| [.../KANbeFair](https://github.com/yu-rp/KANbeFair) | Fair benchmarking of KANs vs. MLPs. | 
| [.../efficient-kan](https://github.com/Blealtan/efficient-kan) | Efficient PyTorch implementation of KAN. | 
| [.../jaxKAN](https://github.com/srigas/jaxKAN) | JAX-based KAN with grid extension support. | 
| [.../fast-kan](https://github.com/ZiyaoLi/fast-kan) | FastKAN using RBFs for acceleration. | 
| [.../faster-kan](https://github.com/AthanasiosDelis/faster-kan) | Uses reflectional switch activations. |
| [.../LKAN](https://github.com/Indoxer/LKAN) | Lightweight KAN variants and experiments. | 
| [.../neuromancer (fbkans branch)](https://github.com/pnnl/neuromancer/tree/feature/fbkans) | Partition of unity (FBKAN) for PDE solving. | 
| [.../relu_kan](https://github.com/quiqi/relu_kan) | Minimal ReLU-KAN example. | 
| [.../MatrixKAN](https://github.com/OSU-STARLAB/MatrixKAN) | Matrix-parallelized KAN implementation. | 
| [.../PowerMLP](https://github.com/Iri-sated/PowerMLP) | MLP-type network with KAN-level expressiveness. | 
| [.../FourierKAN](https://github.com/GistNoesis/FourierKAN) | Fourier-based KAN layer. | 
| [.../FusedFourierKAN](https://github.com/GistNoesis/FusedFourierKAN) | Optimized FourierKAN with fused GPU kernels. |
| [.../fKAN](https://github.com/alirezaafzalaghaei/fKAN) | Fractional KAN using Jacobi functions. | 
| [.../rKAN](https://github.com/alirezaafzalaghaei/rKAN) | Rational KAN (Padé/Jacobi rational designs). | 
| [.../CVKAN](https://github.com/M-Wolff/CVKAN) | Complex-valued KANs. | 
| [.../SincKAN](https://github.com/DUCH714/SincKAN) | Sinc-based KAN for PINN applications. |
| [.../ChebyKAN](https://github.com/SynodicMonth/ChebyKAN) | Chebyshev polynomial-based KAN. | 
| [.../OrthogPolyKANs](https://github.com/Boris-73-TA/OrthogPolyKANs) | Orthogonal polynomial-based KAN implementations. |
| [.../kaf_act](https://github.com/kolmogorovArnoldFourierNetwork/kaf_act) | RFF-based activation library. | 
| [.../KAF](https://github.com/kolmogorovArnoldFourierNetwork/KAF) | Kolmogorov–Arnold Fourier Networks. | 
| [.../HRKAN](https://github.com/kelvinhkcs/HRKAN) | Higher-order ReLU-KANs. | 
| [.../KINN](https://github.com/yizheng-wang/Research-on-Solving-Partial-Differential-Equations-of-Solid-Mechanics-Based-on-PINN) | PIKAN for solid mechanics PDEs. | 
| [.../KAN_PointNet_CFD](https://github.com/Ali-Stanford/KAN_PointNet_CFD) | Jacobi-based KAN for CFD predictions. | 
| [.../FKAN-GCF](https://github.com/Jinfeng-Xu/FKAN-GCF) | FourierKAN-GCF for graph filtering. | 
| [.../KKANs_PIML](https://github.com/jdtoscano94/KKANs_PIML) | Kurkova-KANs combining MLP with basis functions. | 
| [.../MLP-KAN](https://github.com/Zhangyanbo/MLP-KAN) | MLP-augmented KAN activations. | 
| [.../kat](https://github.com/Adamdad/kat) | Kolmogorov–Arnold Transformer. | 
| [.../FAN](https://github.com/YihongDong/FAN) | Fourier Analysis Network (FAN). |
| [.../Basis_Functions](https://github.com/seydi1370/Basis_Functions) | Polynomial bases for KANs. | 
| [.../Wav-KAN](https://github.com/zavareh1/Wav-KAN) | Wavelet-based KANs. | 
| [.../qkan](https://github.com/Jim137/qkan) | Quantum-inspired KAN variants and pruning. | 
| [.../KAN-Converge](https://github.com/liouvill/KAN-Converge) | Additive & hybrid KANs for convergence-rate experiments. | 
| [.../BSRBF_KAN](https://github.com/hoangthangta/BSRBF_KAN) | Combines B-spline and RBF bases. |
| [.../Bayesian-HR-KAN](https://github.com/wmdataphys/Bayesian-HR-KAN) | Bayesian higher-order ReLU-KANs with uncertainty quantification. |
| [.../Legend-KINN](https://github.com/zhang-zhuo001/Legend-KINN) | Legendre polynomial–based KAN for efficient PDE solving. |
| [.../DeepOKAN](https://github.com/DiabAbu/DeepOKAN) | Deep Operator Network based on KAN. |
| [.../LeanKAN](https://github.com/DENG-MIT/LeanKAN) | A memory-efficient Kolmogorov–Arnold Network. |
| [.../SPIKANs](https://github.com/pnnl/spikans) | A Separation-of-variables to decompose high-dimensional PDEs into smaller KANs. |
| [openkan.org](http://openkan.org/) | Features a non-spline KAN trained via Newton–Kaczmarz. |
| [.../Anant-Net](https://github.com/ParamIntelligence/Anant-Net) | High-dimensional PDE solver with tensor sweeps. |
| [.../RGA-KANs](https://github.com/srigas/RGA-KANs) | Deep cPIKANs with variance-preserving initialization. |
| [.../lmkan](https://github.com/schwallergroup/lmkan) | Lookup-based KAN for fast high-dimensional mappings. |
| [.../KAN_Initialization_Schemes](https://github.com/srigas/KAN_Initialization_Schemes) | Initialization schemes for spline-based KANs. |
| [.../mlp-kan](https://github.com/geoelements-dev/mlp-kan) | KAN vs. MLP for PDEs in DeepONet/GNS frameworks. |
| [.../KANQAS_code](https://github.com/Aqasch/KANQAS_code) | KANQAS: KAN for quantum architecture search. |
| [.../pkan](https://github.com/andrewpolar/pkan) | Probabilistic KAN via divisive data re-sorting. |
| [.../spikans](https://github.com/pnnl/spikans) | Separable PIKAN (SPIKAN) for high-dimensional PDEs. |



---
## 5. Bridging KANs and MLPs

| Brief result | Paper , Code | 
|---|---|
| Equivalence: ReLU^k MLP ↔ B-spline KAN. | [Wang](https://arxiv.org/abs/2410.01803) |
| Piecewise-linear KAN = ReLU MLP. | [Schoots](https://arxiv.org/abs/2503.01702) | 
| Adaptive spline KANs mimic MLPs with data-driven capacity. | [Actor](https://arxiv.org/abs/2505.18131) | 
| NTK view: richer KAN bases reduce spectral bias vs MLP.  | [Gao](https://arxiv.org/abs/2410.08041) | 


---


## 6. Basis Functions

| Name | Support | Equation | Grid | Type |  Paper , Code |
|---|---|---|---|---|------|
| B-spline | Local | $\sum_n c_nB_n(x)$ | Yes | B-spline | [Liu](https://arxiv.org/abs/2404.19756) & [Liu](https://arxiv.org/abs/2408.10205) , [Code](https://github.com/KindXiaoming/pykan) & [Actor](https://arxiv.org/abs/2505.18131) & [Basina](https://arxiv.org/abs/2411.10622) & [Coffman](https://arxiv.org/abs/2502.07176) & [Guo](https://doi.org/10.1016/j.physd.2025.134689) & [Kalesh](https://link.springer.com/chapter/10.1007/978-3-031-97596-7_30) & [Gao](https://doi.org/10.1109/TIT.2025.3588401) & [Zeng](https://arxiv.org/abs/2408.07906) & [Khedr](https://doi.org/10.21203/rs.3.rs-6743344/v1) & [Lei](https://arxiv.org/abs/2507.08338) & [Li](https://doi.org/10.1109/cec65147.2025.11043029) & [Lin](https://doi.org/10.1016/j.jrmge.2025.02.023) & [Pal](https://openreview.net/forum?id=yPE7S57uei) & [Howard](https://arxiv.org/abs/2410.14764) & [Jacob](https://arxiv.org/abs/2411.06286) , [Code](https://github.com/pnnl/spikans) & [Aghaei](https://arxiv.org/abs/2409.06649) & [Patra](https://arxiv.org/abs/2407.18373) & [Ranasinghe](https://arxiv.org/abs/2408.14780) & [Rigas](https://doi.org/10.21105/joss.07830) , [Code](https://github.com/srigas/jaxKAN) & [Shuai](https://arxiv.org/abs/2408.06650) & [Wang](https://doi.org/10.1016/j.cma.2024.117518) & [Zhang](https://doi.org/10.1038/s41598-025-92900-1) & [Raffel](https://arxiv.org/abs/2505.13813) & [Schoots](https://arxiv.org/abs/2503.01702) & [Wang](https://arxiv.org/abs/2410.01803) & [Wang](https://doi.org/10.1002/advs.202413805) & [Xu](https://arxiv.org/abs/2503.23289) & [Shen](https://doi.org/10.1016/j.mtcomm.2025.113198) & [Yang](https://doi.org/10.1016/j.jcp.2025.113846) & [Howard](https://arxiv.org/abs/2406.19662) , [Code](https://github.com/pnnl/neuromancer/tree/feature/fbkans/examples/KANs) & [Code](https://github.com/sidhu2690/Deep-KAN) & [Gong](https://arxiv.org/abs/2508.16999) & [Guo](https://doi.org/10.1145/3765904) & [Lee](https://arxiv.org/abs/2509.16825) & [Mallick](https://arxiv.org/abs/2509.09145) & [Sen](https://arxiv.org/abs/2509.18483)|
| Chebyshev | Global | $\sum_k c_kT_k(\tanh x)$ | No | Chebyshev + tanh | [Sidharth](https://arxiv.org/abs/2405.07200) , [Code](https://github.com/sidhu2690/Deep-KAN/tree/main/DEEP-KAN-pypi) & [Code](https://github.com/SynodicMonth/ChebyKAN) & [Yang](https://arxiv.org/abs/2507.19888) & [Mahmoud](https://doi.org/10.1109/ACCESS.2025.3566551) & [Guo](https://arxiv.org/abs/2411.04516) & [Faroughi](https://arxiv.org/abs/2506.07958) & [Yu](https://arxiv.org/abs/2410.04096), [Code](https://github.com/DUCH714/SincKAN) & [Rigas 2025](https://www.arxiv.org/abs/2510.23501) , [Code](https://github.com/srigas/RGA-KANs)|
| Stabilized Chebyshev | Global | $\tanh\big(\sum_k c_kT_k(\tanh x)\big)$ | No | Chebyshev + linear head  | [Daryakenari](https://arxiv.org/abs/2504.07379) |
| Chebyshev (grid) | Global | $\sum_k c_kT_k\Big(\tfrac{1}{m}\sum_i \tanh(w_i x+b_i)\Big)$ | Yes | Chebyshev + tanh | [Toscano](https://arxiv.org/abs/2412.16738) , [Code](https://github.com/jdtoscano94/KKANs_PIML) |
| ReLU-KAN | Local | $\sum_i w_iR_i(x)$ | Yes | Squared ReLU | [Qiu](https://arxiv.org/abs/2406.02075) , [Code](https://github.com/quiqi/relu_kan) |
| HRKAN | Local | $\sum_i w_i\big[\mathrm{ReLU}(x)\big]^m$ | Yes | Polynomial ReLU  | [So](https://arxiv.org/abs/2409.14248) , [Code](https://github.com/kelvinhkcs/HRKAN) |
| Adaptive ReLU-KAN | Local | $\sum_i w_iv_i(x)$ | Yes | Adaptive ReLU  | [Rigas](https://doi.org/10.21105/joss.07830) , [Code](https://github.com/srigas/jaxKAN) |
| fKAN (Jacobi) | Global | $\sum_n c_nP_n(x)$ | No | Jacobi | [Aghaei](https://arxiv.org/abs/2406.07456) , [Code](https://github.com/alirezaafzalaghaei/fKAN) |
| rKAN (Padé/Jacobi) | Global | $\dfrac{\sum_i a_iP_i(x)}{\sum_j b_jP_j(x)}$ | No | Rational + Jacobi | [Aghaei](https://arxiv.org/abs/2406.14495) , [Code](https://github.com/alirezaafzalaghaei/rKAN) |
| Jacobi-KAN | Global | $\sum_i c_iP_i(\tanh x)$ | No | Jacobi + tanh  | [Kashefi](https://arxiv.org/abs/2408.02950) , [Code](https://github.com/Ali-Stanford/KAN_PointNet_CFD) &  [Shukla](https://doi.org/10.1016/j.cma.2024.117290) & [Xiong](https://doi.org/10.1016/j.cnsns.2025.109414) & [Zhang](https://doi.org/10.1016/j.eswa.2025.129839) , [Code](https://github.com/zhang-zhuo001/Legend-KINN)|
| FourierKAN | Global | $\sum_k a_k\cos(kx)+b_k\sin(kx)$ | No | Fourier |  [Xu](https://arxiv.org/abs/2406.01034v3) , [Code](https://github.com/Jinfeng-Xu/FKAN-GCF) & [Code](https://github.com/YihongDong/FAN) & [Guo](https://doi.org/10.1007/s00477-025-03105-x) &  [Jiang](https://doi.org/10.48550/arXiv.2509.14026) , [Code](https://github.com/Jim137/qkan)|
| KAF | Global | $\alpha\mathrm{GELU}(x)+\sum_j \beta_j\psi_j(x)$ | No | Random Fourier + GELU | [Zhang](https://arxiv.org/abs/2502.06018) , [Code](https://github.com/kolmogorovArnoldFourierNetwork/KAF) |
| Gaussian | Local | $\sum_i w_i\exp\Big(-\big(\tfrac{x-g_i}{\varepsilon}\big)^2\Big)$ | Yes | Gaussian RBF  | [Li](https://arxiv.org/abs/2405.06721) , [Code](https://github.com/ZiyaoLi/fast-kan) & [Lee](https://doi.org/10.1109/JAS.2022.105743)  & [Abueidda](https://doi.org/10.1016/j.cma.2024.117699) , [Code](https://github.com/DiabAbu/Dee) & [Koenig](https://doi.org/10.1016/j.neunet.2025.107883) , [Code](https://github.com/DENG-MIT/LeanKAN) & [Ta](https://arxiv.org/abs/2406.11173) , [Code](https://github.com/hoangthangta/BSRBF_KAN) & [Buhler](https://arxiv.org/abs/2509.10089) & [Zhang](https://arxiv.org/abs/2508.03965)|
| RSWAF-KAN | Local | $\sum_i w_i\left(s_i-\tanh^2\big(\tfrac{x-c_i}{h_i}\big)\right)$ | Yes | Switch ($tanh^2$)  |  [Code](https://github.com/AthanasiosDelis/faster-kan) |
| CVKAN | Local | $\sum_{u,v} w_{uv}\exp\big(-\lvert z-g_{uv}\rvert^2\big)$ | Yes | Complex Gaussian | [Wolff](https://arxiv.org/abs/2502.02417) , [Code](https://github.com/M-Wolff/CVKAN) & [Che](https://doi.org/10.1007/978-3-032-05176-9_34)|
| BSRBF-KAN | Local | $\sum_i a_i B_i(x)+\sum_j b_j\exp\big(-\tfrac{(x-g_j)^2}{\varepsilon^2}\big)$ | Yes | B-spline + Gaussian | [Ta](https://arxiv.org/abs/2406.11173) , [Code](https://github.com/hoangthangta/BSRBF_KAN) |
| Wav-KAN | Local | $\sum_{j,k} c_{j,k}\psi\big(\tfrac{x-u_{j,k}}{s_j}\big)$ | No | Wavelet | [Bozorgasl](https://arxiv.org/abs/2405.12832) , [Code](https://github.com/zavareh1/Wav-KAN}{zavareh1/Wav-KAN) & [Patra](https://arxiv.org/abs/2407.18373) & [Pratyush](https://doi.org/10.1093/bioinformatics/btaf124) & [Seydi](https://arxiv.org/abs/2406.07869) & [Meshir 2025](https://arxiv.org/abs/2502.00280)|
| FBKAN | Local | $\sum_j \omega_j(x)K_j(x)$ | Yes | PoU + B-spline |  [Howard](https://arxiv.org/abs/2406.19662) , [Code](https://github.com/pnnl/neuromancer/tree/feature/fbkans) |
| SincKAN | Global | $\sum_{i=-N}^{N} c_i\mathrm{sinc}\left(\frac{\pi}{h}(x - i h)\right)$ | Yes | Sinc | [Yu](https://arxiv.org/abs/2410.04096) , [Code](https://github.com/DUCH714/SincKAN}{DUCH714/SincKAN) |
| Poly-KAN | Global | $\sum_i w_iP_i(x)$ | No | Polynomial | [Seydi](https://arxiv.org/abs/2406.02583) , [Code](https://github.com/seydi1370/Basis_Functions) & [Attouri 2025](https://doi.org/10.1016/j.asej.2025.103884)|


---

## 7. Accuracy Improvement

### 7.1 Physics & Loss Design

| Brief result | Paper , Code | 
|---|---|
| Physics-informed KAN (cPIKAN): residual attention, entropy-viscosity. | [Shukla](https://arxiv.org/abs/2406.02917) | 
| KAN-PINN for strongly nonlinear PDEs (actuator deflection). | [Zhang](https://doi.org/10.1016/j.engappai.2025.110126) | 
| Attention-guided KAN with NSE residuals + BC losses. |  [Yang](https://doi.org/10.1016/j.jcp.2025.113846) | 
| Residual physics + sparse regression (variable-coeff. PDEs). |  [Guo](https://doi.org/10.1016/j.physd.2025.134689) | 
| Self-scaled residual reweighting (ssRBA). | [Toscano](https://arxiv.org/abs/2412.16738) , [Code](https://github.com/jdtoscano94/KKANs_PIML) |
| Augmented-Lagrangian PINN–KAN (learnable multipliers). | [Zhang](https://doi.org/10.1038/s41598-025-92900-1) | 
| Velocity–vorticity loss for turbulence reconstruction. | [Toscano](https://arxiv.org/abs/2407.15727v2) | 
| Fractional/integro-diff. operators in KAN. | [Aghaei](https://arxiv.org/abs/2409.06649) | 
| Physics-informed KAN for high-index DAEs (dual-network structure). | [Lou](https://arxiv.org/abs/2504.15806) |
| Holomorphic KAN  for elliptic PDEs; trains only on boundary conditions. | [Clafa](https://arxiv.org/abs/2507.22678) , [Code](https://github.com/teocala/pihnn) |

### 7.2 Adaptive Sampling & Grids

| Brief result | Paper , Code | 
|---|---|
| Multilevel knots (coarse→fine) for nested spline spaces. | [Actor](https://arxiv.org/abs/2505.18131) | 
| Free-knot KAN (trainable knots via cumulative softmax).  | [Actor](https://arxiv.org/abs/2505.18131) | 
| Grid extension with optimizer state transition.  | [Rigas](https://doi.org/10.21105/joss.07830) , [Code](https://github.com/srigas/jaxKAN) |
| Residual-adaptive sampling (RAD). | [Rigas](https://doi.org/10.21105/joss.07830) , [Code](https://github.com/srigas/jaxKAN) |
| Multi-resolution sampling schedule for cPIKAN.  | [Yang](https://arxiv.org/abs/2507.19888) | 

### 7.3 Domain Decomposition

| Brief result | Paper , Code |
|---|---|
| Finite-basis KAN (FBKAN) with PoU blending of local KANs. | [Howard](https://arxiv.org/abs/2406.19662) , [Code](https://github.com/pnnl/neuromancer/tree/feature/fbkans) |
| Temporal subdomains to improve NTK conditioning. | [Faroughi](https://arxiv.org/abs/2506.07958) | 

### 7.4 Function Decomposition

| Brief result | Paper , Code | 
|---|---|
| Multi-fidelity KAN (freeze LF, learn HF linear + nonlinear heads). | [Howard](https://arxiv.org/abs/2410.14764) , [Code](https://github.com/pnnl/neuromancer/tree/feature/fbkans) |
| Separable PIKAN (sum of products of 1D KAN factors). | [Jacob](https://arxiv.org/abs/2411.06286) , [Code](https://github.com/pnnl/spikans)|
| KAN-SR: recursive simplification for symbolic discovery.  | [Buhler](https://arxiv.org/abs/2509.10089) | 

### 7.5 Hybrid / Ensemble & Data

| Brief result | Paper , Code | 
|---|---|
| MLP–KAN mixture of experts. | [He](https://arxiv.org/abs/2410.03027) , [Code](https://github.com/DLYuanGod/MLP-KAN) |
| Parallel KAN ∥ MLP branches with learnable fusion. | [Xu](https://arxiv.org/abs/2503.23289) | 
| KKAN: per-dim MLP features + explicit basis expansion. | [Toscano](https://arxiv.org/abs/2412.16738) , [Code](https://github.com/jdtoscano94/KKANs_PIML) |

### 7.6 Sequence / Attention Hybrids

| Brief result | Paper , Code | 
|---|---|
| FlashKAT: group-rational KAN blocks in Transformers. | [Raffel](https://arxiv.org/abs/2505.13813) | 
| GINN-KAN: interpretable growth + KAN in PINNs.| [Ranasinghe](https://arxiv.org/abs/2408.14780) | 
| KAN-ODE: KAN as $\dot u$ model (adjoint training).  | [Koeing](https://arxiv.org/abs/2407.04192) | 
| AAKAN-WGAN: adaptive KAN + GAN for data augmentation.  | [Shen](https://doi.org/10.1016/j.mtcomm.2025.113198) | 
| Attention-KAN-PINN for battery SOH forecasting. | [Wei](https://doi.org/10.1016/j.eswa.2025.128969) | 
| KANQAS: uses KAN Double Deep Q-Network for quantum architecture search. | [Kundu 2024](https://doi.org/10.1140/epjqt/s40507-024-00289-z) , [Code](https://github.com/Aqasch/KANQAS_code) |


### 7.7 Discontinuities & Sharp Gradients

| Brief result  | Paper , Code | 
|---|---|
| SincKAN for kinks/boundary layers. | [Yu](https://arxiv.org/abs/2410.04096) , [Code](https://github.com/DUCH714/SincKAN}{DUCH714/SincKAN) |
| rKAN (rational bases) for asymptotics/jumps.  | [Aghaei](https://arxiv.org/abs/2406.14495) , [Code](https://github.com/alirezaafzalaghaei/rKAN) |
| DKAN: $\tanh$ jump gate + spline background.  | [Lei](https://arxiv.org/abs/2507.08338) | 
| KINN for singularities/stress concentrations.  | [Wang](https://doi.org/10.1016/j.cma.2024.117518) , [Code](https://github.com/yizheng-wang/Research-on-Solving-Partial-Differential-Equations-of-Solid-Mechanics-Based-on-PINN) |
| Two-phase PINN–KAN for saturation fronts.  | [Kalesh](https://link.springer.com/chapter/10.1007/978-3-031-97596-7_30) | 

### 7.8 Optimization & Adaptive Training

| Brief result  | Paper , Code | 
|---|---|
| Adam/RAdam warmup → (L-)BFGS refinement. | [Mostajeran](https://arxiv.org/abs/2501.02762) & [Daryakenari](https://arxiv.org/abs/2504.07379) & [Zeng](https://arxiv.org/abs/2408.07906)|
| Hybrid optimizers for sharp fronts. | [Kalesh](https://link.springer.com/chapter/10.1007/978-3-031-97596-7_30) |
| Bayesian hyperparameter tuning for KANs. | [Lin](https://doi.org/10.1016/j.jrmge.2025.02.023) |
| Bayesian PINN–KAN (variational + KL) for UQ.  | [Giroux](https://arxiv.org/abs/2410.01687) , [Code](https://github.com/wmdataphys/Bayesian-HR-KAN) |
| NTK perspective: conditioning ↔ convergence. | [Faroughi](https://arxiv.org/abs/2506.07958) |


---

## 8. Efficiency Improvement

### 8.1 Parallelism, GPU, and JAX Engineering

| Brief result | Paper , Code | 
|---|---|
| ReLU^m activations replace splines (CUDA-friendly). | [Qiu](https://arxiv.org/abs/2412.13571) , [Code](https://github.com/Iri-sated/PowerMLP) |
| Spline→matmul CUDA kernels (GEMM fusion). |  [Qiu](https://arxiv.org/abs/2406.02075), [Code](https://github.com/quiqi/relu_kan) & [So](https://arxiv.org/abs/2409.14248), [Code](https://github.com/kelvinhkcs/HRKAN) |
| Matrix B-spline evaluation fused on GPU. |  [Coffman](https://arxiv.org/abs/2502.07176) , [Code](https://github.com/OSU-STARLAB/MatrixKAN) |
| Dual-matrix merge + trainable RFF for scaling. | [Zhang](https://arxiv.org/abs/2502.06018) , [Code](https://github.com/kolmogorovArnoldFourierNetwork/KAF) |
| Custom GPU backward for KAN attention blocks. |  [Raffel](https://arxiv.org/abs/2505.13813) |
| Parallel KAN ∥ MLP branches (stream/layer parallelism). |  [Xu](https://arxiv.org/abs/2503.23289) | 
| Domain decomposition parallelism (multi-GPU, PoU/separable). |  [Shukla](https://arxiv.org/abs/2406.02917) & [Howard](https://arxiv.org/abs/2406.19662) , [Code](https://github.com/pnnl/neuromancer/tree/feature/fbkans) & [Jacob](https://arxiv.org/abs/2411.06286) , [Code](https://github.com/pnnl/spikans)| 
| JAX/XLA: `jit`/`vmap`/`pmap`, fusion, memory-aware. | [Daryakenari](https://arxiv.org/abs/2407.17611) & [Rigas](https://doi.org/10.21105/joss.07830) , [Code](https://github.com/srigas/jaxKAN) | 
| lmKANs: multivariate spline lookup tables, CUDA-friendly. | [Michalkiewicz](https://arxiv.org/abs/2503.04057) , [Code]() |


| Brief result | Paper , Code |
|---|---|
| ReLU^m activations replace splines (CUDA-friendly). | [Qiu](https://arxiv.org/abs/2412.13571) , [Code]() |
| Spline→matmul CUDA kernels (GEMM fusion). | [Qiu](https://arxiv.org/abs/2406.02075) , [Code]() & [So](https://arxiv.org/abs/2409.14248) , [Code]() |
| Matrix B-spline evaluation fused on GPU. | [Coffman](https://arxiv.org/abs/2502.07176) , [Code]() |
| Dual-matrix merge + trainable RFF for scaling. | [Zhang](https://arxiv.org/abs/2502.06018) , [Code]() |
| Custom GPU backward for KAN attention blocks. | [Raffel](https://arxiv.org/abs/2505.13813) , [Code]() |
| Parallel KAN ∥ MLP branches (stream/layer parallelism). | [Xu](https://arxiv.org/abs/2503.23289) , [Code]() |
| Domain decomposition parallelism (multi-GPU, PoU/separable). | [Shukla](https://arxiv.org/abs/2406.02917) & [Howard](https://arxiv.org/abs/2406.19662) , [Code]() & [Jacob](https://arxiv.org/abs/2411.06286) , [Code]() |
| JAX/XLA acceleration: `jit`, `vmap`, `pmap`, fusion. | [Daryakenari](https://arxiv.org/abs/2407.17611) & [Rigas](https://doi.org/10.21105/joss.07830) , [Code]() |
| lmKANs: multivariate spline lookup tables, CUDA-friendly. | [Pozdnyakov 2025](https://arxiv.org/abs/2509.07103) , [Code](https://github.com/schwallergroup/lmkan) |


### 8.2 Matrix Optimization & Parameter-Efficient Bases

| Brief result | Paper , Code |
|---|---|
| ReLU-power vs B-splines: fewer params, vectorized polynomials.  | [Qiu](https://arxiv.org/abs/2412.13571) , [Code](https://github.com/Iri-sated/PowerMLP) & [Qiu](https://arxiv.org/abs/2406.02075) , [Code](https://github.com/quiqi/relu_kan) & [So](https://arxiv.org/abs/2409.14248) , [Code](https://github.com/kelvinhkcs/HRKAN)|
| Orthogonal polynomials with cheap recurrences. | [Shukla](https://arxiv.org/abs/2406.02917) & [Guo](https://arxiv.org/abs/2411.04516) & [Mostajeran](https://arxiv.org/abs/2410.10897) &  [Mostajeran](https://arxiv.org/abs/2501.02762) & [Wang](https://doi.org/10.1016/j.cma.2024.117518) ,  [Code](https://github.com/yizheng-wang/Research-on-Solving-Partial-Differential-Equations-of-Solid-Mechanics-Based-on-PINN) |
| Compact RBF bases (local Gaussians).  | [Lin](https://doi.org/10.1016/j.jrmge.2025.02.023) & [Koeing](https://arxiv.org/abs/2407.04192) , [Code](https://github.com/DENG-MIT/LeanKAN)| 
| Wavelets for multi-resolution and sparse coeffs.  | [Patra](https://arxiv.org/abs/2407.18373) | 
| Dual-matrix + RFF compression to cut memory traffic.  | [Zhang](https://arxiv.org/abs/2502.06018) , [Code](https://github.com/kolmogorovArnoldFourierNetwork/KAF) |
| Sparsity regularization (ℓ1/group) with pruning. | [Guo](https://doi.org/10.1016/j.physd.2025.134689) | 
| Hierarchical channel-wise refinement (shared params). | [Actor](https://arxiv.org/abs/2505.18131) | 
| DEKAN: connectivity via Differential Evolution. | [Li](https://doi.org/10.1109/CEC65147.2025.11043029) | 
| Mix spectral (derivatives) + spatial (coeffs) sparsity for operators. | [Lee](https://arxiv.org/abs/2509.16825) | 
| Tensor sweeps + selective differentiation for scalable high-D PDEs. | [Sidharth](https://doi.org/10.1016/j.cma.2025.118403) , [Code](https://github.com/ParamIntelligence/Anant-Net) |
| Operator-aware spectral–spatial mixing for near-diagonal matvecs. | [Lee](https://arxiv.org/abs/2509.16825) |



---

## 9. Sparsity & Regularization

### 9.1 ℓ1 sparsity with entropy balancing

| Brief result | Paper , Code |
|---|---|
| Layerwise ℓ1 on edge activations + entropy balance. | [Liu](https://arxiv.org/abs/2404.19756) , [Code](https://github.com/KindXiaoming/pykan) |
| EfficientKAN: direct ℓ1 on weights (simple, practical). |  [EfficientKAN](https://github.com/Blealtan/efficient-kan) |
| Sparse symbolic discovery with ℓ1 + entropy. | [Wang](https://arxiv.org/abs/2410.01803) |
| PDE KAN: ℓ1 + smoothness penalty to denoise coefficients. | [Guo](https://doi.org/10.1016/j.physd.2025.134689) | 
| Post-training pruning with layerwise ℓ1. | [Koeing](https://arxiv.org/abs/2407.04192) , [Code](https://github.com/DENG-MIT/LeanKAN)|
| KAN-SR: magnitude + entropy at subunit level (+ℓ1 on bases). | [Buhler](https://arxiv.org/abs/2509.10089) |

### 9.2 ℓ2 weight decay and extensions

| Brief result | Paper , Code | 
|---|---|
| AAKAN: ℓ2 + temporal smoothing + MI regularizer. | [Shen](https://doi.org/10.1016/j.mtcomm.2025.113198) | 
| Small ℓ2 (e.g., 1e−5) improves stability in PINNs/DeepOKAN. | [Shukla](https://arxiv.org/abs/2406.02917) & [Toscano](https://doi.org/10.1126/sciadv.ads5236) | 

### 9.3 Implicit and dropout-style regularizers

| Brief result  | Paper , Code |
|---|---|
| Nested activations (e.g., tanh∘tanh) for bounded outputs & smooth grads.  | [Daryakenari](https://arxiv.org/abs/2504.07379) | 
| DropKAN: post-activation masking (noise after spline eval). | [Altarabichi](https://arxiv.org/abs/2407.13044) , [Code](https://github.com/Ghaith81/dropkan) | 


---

## 10. Convergence & Scaling Laws (KAN/PIKAN only)

### 10.1 Approximation & Sample Complexity

| Brief result | Paper , Code | 
|---|---|
| Depth-rate $L^{-2s/d}$ for spline KAN (fixed width, $G = 2$). |  [Wang](https://arxiv.org/abs/2410.01803) | 
| Minimax sample rate for spline KANs; additive case is dimension-free. | [Liu](https://arxiv.org/abs/2509.19830) , [Code](https://github.com/liouvill/KAN-Converge) |

### 10.2 Optimization Dynamics & Spectral Bias
| Brief result | Paper , Code |
|---|---|
| NTK view: learnable bases flatten spectra; less spectral bias than MLP.  | [Gao](https://arxiv.org/abs/2410.08041) | 
| Basis/grid enrichment widens NTK spectrum; speeds high-freq learning. | [Farea](https://arxiv.org/abs/2411.15111) , [Code](https://github.com/afrah/pinn_learnable_activation) | 
| KAN vs MLP: earlier capture of high frequencies under NTK dynamics. |  [Wang](https://arxiv.org/abs/2410.01803) | 

### 10.3 Empirical Power Laws

| Brief result | Paper , Code | 
|---|---|
| Error follows $\ell \propto P^{-\alpha}$; increasing grid $G$ boosts accuracy. | [Liu](https://arxiv.org/abs/2509.19830) , [Code](https://github.com/liouvill/KAN-Converge) | 
| Depth (via dyadic grids) improves accuracy consistent with theory. | [Wang](https://arxiv.org/abs/2410.01803) |


---


