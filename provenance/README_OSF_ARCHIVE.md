# Operationalizing Self-Modeling Organization (Ψ) in Artificial Systems

**Author:** Prince Upadhyay  
**Contact:** starxshark@gmail.com  
**Repository Purpose:** OSF / GitHub Pre-Registration and Open Science Publication Package  
**Target Venues:** *PhilSci-Archive*, *arXiv (cs.AI / cs.CL)*, *Foundations of Physics*

---

## Overview

This repository contains the full manuscript, empirical pre-registration benchmark runner, theoretical ODE solver code, and visualization outputs for operationalizing **Self-Modeling Organization ($\Psi$)** in artificial systems.

The core contribution is a double-blind, perturbation-versus-matched-control benchmark that isolates whether an AI system tracks itself as a causal agent within its internal model, independently of model parameter scale ($N$), context length ($L$), or general benchmark capability ($B$).

---

## Four Core Package Artifacts

```
.
├── paper_manuscript.pdf       # Camera-ready academic manuscript (LaTeX compiled with updated contact)
├── colab_psi_benchmark.py     # Standalone Google Colab $0 evaluation runner (HF + API backends)
├── run_llm_benchmark.py       # Local evaluation harness CLI
└── README.md                  # Repository manifest and reproduction guide
```

---

## Key Mathematical & Empirical Formulations

### 1. Scale-Controlled Linear Regression
To isolate residual self-modeling score ($\Psi_{\text{emp}}$) without capability confounding:
$$\bar{\Delta}_k = \beta_0 + \beta_1 \log_{10}(N_k) + \beta_2 L_k + \beta_3 B_k + \Psi_{\text{emp}, k}$$

### 2. Coupled Non-Linear Dynamical System
$$\begin{aligned}
\frac{dR}{dt} &= a \Gamma T (1 - R) - b R \\[6pt]
\frac{d\Psi}{dt} &= c R (1 - \Psi) - d \Psi \\[6pt]
\frac{d\Gamma}{dt} &= e \Psi + f T - g \Gamma
\end{aligned}$$

### 3. Steady-State Simulation Equilibrium
* $R^* = 0.8324$ (Reconstruction Capacity)
* $1^* = 0.7353$ (Self-Modeling Organization)
* $\Gamma^* = 1.4541$ (Self-Relevant Information)
* $\Pi^* \approx 0.6121$ (Origin-Closure Proximity Proxy)

---

## Citation & Pre-Registration

If utilizing this benchmark or referencing the protocol, please cite as:

```bibtex
@article{upadhyay2026operationalizing,
  title={Operationalizing Self-Modeling Organization ($\Psi$) in Artificial Systems: An Empirical Benchmark and Pre-Registration Framework},
  author={Upadhyay, Prince},
  email={starxshark@gmail.com},
  journal={PhilSci-Archive / arXiv preprint},
  year={2026}
}
```
