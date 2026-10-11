# Project brief: broadband class J PA (CG2H40010F)

Status snapshot as of 2026-10-11. This file lists the specification, what has been built, and what the
simulations have shown. It deliberately contains observations only, not interpretations.

## Specification

Course: TTT4201 Radio System Design (NTNU), semester project. The assignment text is
`docs/SemesterProject2026.pdf`. The scope below was agreed with the professor and differs from the handout.

| Item | Value |
| --- | --- |
| Device | MACOM/Wolfspeed CG2H40010F, ADS model `CG2H40010F_v2` (MA_RFP_ADS_v2p7 kit) |
| Mode | Class J / continuous class J (let α vary with frequency) |
| Band | 2.2–3.6 GHz (2.4 GHz must be inside; the band was earlier 2.4–3.8 GHz) |
| Drain bias | 28 V |
| Gate bias | −3.0 V (handout: Id ≥ 50 mA, Vg ≥ about −3.0 V); gives Idq ≈ 60 mA |
| Power | Pout ≥ 39 dBm with Pin ≤ 27 dBm |
| Stability | Unconditionally stable (μ > 1) at all frequencies |
| Small-signal gain | Handout says ≥ 14 dB at 2.35–2.45 GHz; may be lower over the wide band |
| Measurement | Done where the amplifier performs best |
| IMD | Two-tone, 5 MHz spacing, 38 dBm PEP; low priority |
| Target spec | Max Pout or max PAE (handout) |
| Substrate | FR-4, H = 1.52 mm, εr = 4.6, T = 35 µm, Cond = 5.96e7, tanδ = 0.02 |
| Components | Johanson R14S capacitors and chip inductors (values in the handout), E12 resistors in 0603 |
| Board | 62.5 × 62.5 mm, SMA connectors at the edges |

## Literature and references in `docs/`

| File | What it is |
| --- | --- |
| `A_Methodology_for_Realizing_High_Efficiency_Class-J_in_a_Linear_and_Broadband_PA.pdf` | Wright, Lees, Benedikt, Tasker, Cripps, IEEE TMTT 2009. 10 W GaN class J PA, 1.4–2.6 GHz, 60–70 % DE. Clipping contours; second-harmonic load-pull; two output network iterations |
| `Broadband, High Efficiency, Class J Power Amplifier Design Method with Compensating Drain-Source Capacitance _ 2021-03-09.pdf` | Zhang, Cheng, Liu, Gao, Microwave Journal 2021. CGH40010F, 1.4–3.6 GHz, 60–68 % DE, 40–41.5 dBm. X_Cds/R_L kept in 1–2.5; compensation circuit (series line + shunt open stub); stepped-impedance matching |
| `Matthaei - (QLP) Proc IEEE 1964.pdf` | Matthaei, Tables of Chebyshev impedance-transforming networks of low-pass filter form |
| `CG2H40010.pdf` | Device datasheet |
| `paracitics.png` | Output parasitic model used for de-embedding (Cds = 1.398 pF, Rd = 0.9 Ω, Ld = 0.165 nH, Cbond = 0.383 pF, package SRL 0.56 nH) |
| `SemesterProject2026.pdf` | Assignment text |

Also used: the Keysight example workspace `../ClassJ_wrk` (narrowband class J design flow). Its steps are
loadline → parasitic tune → ideal-load HB with input match → optimized TLine network → verification.

## ADS cells (`radiodesign_lib`)

| Cell | Purpose |
| --- | --- |
| `s10_classj_loadline` | DCIV and the Keysight class J utility |
| `s20_sim_drain_bias` | S-parameters of the `drain_bias` subcircuit |
| `s30_parasitic_tune` | External load sweep vs intrinsic impedance (older flow) |
| `s40_fund_targets` | Target extraction: drain → taper → tee (bias line on branch) → per-harmonic load box |
| `s50_output_match` | Linear S-parameter optimization bench for the output network |
| `s60_classj_ideal_load` | Ideal-load HB with double-stub input match (formerly s40/s50) |
| `x_alpha_search` | Scratch version of the target-extraction bench |
| `drain_bias` | High-Z line (0.2 mm wide; 6 mm, 4 mm tested) + 100 nF GRM18 decoupling cross |
| `output_match` | Stepped-impedance output network subcircuit |

## Definitions used in the benches

- Intrinsic load: `ZLi = -X1.X1.vdsi/X1.X1.idi` per harmonic (Z1i, Z2i, Z3i = harmonics 1, 2, 3).
- From the s10 utility at α = 1: Ropt = 27.88 Ω, X1max = 32.53 Ω, X2max = 33.13 Ω.
- α = −Im(Z2i)/33.13; fundamental target Z1 = 27.88 + j·α·32.53.
- Load box harmonic entries are written as reflection coefficients re 50 Ω: Γ = exp(j·φ). φ = 0 is an open,
  ±π a short, +π/2 is +j50 Ω, −π/2 is −j50 Ω. φ2 is the 2f0 entry, φ3 the 3f0 entry.
- Reference plane for the targets is the tee's through port (2.83 mm line) on the load side.

## Observations from simulations

### Ideal loads (old s40, 2.4 GHz, Pin = 27 dBm)
Ideal load 13 + j6 Ω with 2f0 near open: 42.0 dBm, 68.5 % DE, 15.0 dB Gt, 834 mA, intrinsic Z1 = 29.8 + j14.2 Ω,
Z2 = 0.9 − j14.9 Ω. The first 9-section stepped-impedance low-pass (50→50 Ω) in place of the ideal load gave
41.5 dBm and 54.6 % DE, with intrinsic Z1 = 20 − j18 Ω and Z2 = 150 − j50 Ω.

### Harmonic balance setup
- Starting directly at high Pin (25 dBm) with HB order 5 failed to converge; a Pin sweep (0→25 dBm) or order 9 converged.
- Under an Optim controller, a frequency ParamSweep is skipped when the goal points to the HB analysis. When the
  goal points to the ParamSweep, the HB data cannot be resolved. Sweeping `rf_freq` inside the HB controller
  with fixed Pin works with Optim.
- A MeasEqn that references only another MeasEqn evaluates to NULL; it must sit in the same MeasEqn block.
- `^` is rejected in simulator expressions; use `x*x`.

### α vs frequency with 2f0 open at the tee (6 mm bias line, 2.83 mm tee)
α falls monotonically: about 0.57 at 2.2 GHz, 0.49 at 2.4, 0.34 at 2.8, 0.28 at 3.0, 0.22 at 3.2, 0.06 at 3.6 GHz.
Linear fit: α = −0.3714·f[GHz] + 1.3861. Re(Z2i) is slightly negative (−0.5 to −2.5 Ω).

### 3f0 with 2f0 open
| φ3 | 3f0 intrinsic behaviour over 2.2–3.6 GHz |
| --- | --- |
| 0 (open) | Z3i passes through an open near f0 ≈ 3.0 GHz (827 − j179 Ω) |
| 1.75 | Open/resonance near 3.5–3.6 GHz |
| 2.0 | Short arc around the short (−j17 Ω at 2.2 GHz, ≈0 at 3.5, +j19 Ω at 3.6); resonance just above 3.6 GHz, trace bends inward (resistive) near it |
| 2.25 | Resonance moved to about 3.65–3.8 GHz |
| π | No resonance in band; −j0.68 (normalized) at 2.2 GHz |

Widening the tee from 2.83 to 5 mm shrank the 3f0 loop toward the short and lowered α slightly (0.57 → 0.55 at
2.2 GHz; α zero crossing about 3.65 GHz). ADS warns that the MTEE width ratio exceeds its recommended maximum of 5.

### Fundamental target extraction (s40_fund_targets, φ2 = 0, φ3 = 2.25, Pin = 25 dBm, 6 mm bias line)
RL and XL are quadratic polynomials in df = (f − 2.9 GHz)/1 GHz, with coefficients optimized so that |Z1i − Z1 target| is
minimized. The result was err < 1.8 Ω across 2.2–3.6 GHz. The external trajectory is about 11–12 Ω real, with reactance
going from about +6.5 Ω (2.2 GHz) to about −3.5 Ω (3.6 GHz).

### α vs external 2f0 phase φ2 (α at 2.2 GHz / 3.6 GHz; R2i in Ω)
| φ2 | 6 mm bias line | 4 mm bias line |
| --- | --- | --- |
| −3 | — | resonance at 2.2 (Z2i ≈ −28 + j100) / 0.38 |
| −2 | — | 0.11 / resonance at 3.6 (Z2i ≈ 16 − j34) |
| −1 | — | 0.45 / −0.07 |
| 0 | 0.56 / 0.05 | 0.63 / 0.08 |
| 1 | 0.74 / 0.16 | 0.82 / 0.16 |
| 2 | 1.08 / 0.24 (R2i ≈ −3 to −4) | 1.16 / 0.22 |
| 3 | resonance at 2.2 (Z2i ≈ −92 − j163) / 0.37 | resonance at 2.2 (Z2i ≈ −92 − j182) / 0.33 |

### Matthaei transformer (r = 4, n = 8, w = 0.6, fm = 2.9 GHz, R0 = 50 Ω; 50 Ω end is C1, device end is L8)
g1–g4 = 0.716904, 0.880613, 1.88630, 0.734992; g5 = g4·r, g6 = g3/r, g7 = g2·r, g8 = g1/r.
C1 = 0.787 pF, L2 = 2.416 nH, C3 = 2.070 pF, L4 = 2.017 nH, C5 = 3.227 pF, L6 = 1.294 nH, C7 = 3.866 pF, L8 = 0.492 nH.

Ideal lumped ladder (12.5 Ω on the device side):
- |S11| ≤ 0.05 over about 2.0–3.8 GHz; 0.18 at 4.0 GHz, 0.80 at 4.4 GHz, 0.98 at 4.8 GHz. (n = 6 gave 0.74 at 4.4 GHz.)
- Seen from the device side (Γ re 50 Ω), the fundamental Zin is 12.1–13.0 Ω with X ≈ 0.
- 2f0 phase is −169° at 4.4 GHz, 159° at 5.8 GHz and 145° at 7.2 GHz (near short).
- 3f0 phase is 150° at 6.6 GHz, falling to 119° at 10.8 GHz.

| L8 | Zin at 2.2 / 2.9 / 3.6 GHz (Ω) | 2f0 phase at 4.4 / 5.8 / 7.2 GHz |
| --- | --- | --- |
| 0.49 nH | 13 − j0 / 12.5 − j0 / 12.1 − j0 | −169° / 159° / 145° |
| 1.0 nH | 13 + j7 / 12.5 + j9 / 12.1 + j11 | 159° / 122° / 104° |
| 2.0 nH | 13 + j21 / 12.5 + j27 / 12.1 + j34 | 107° / 76° / 62° |
| 3.0 nH | 13 + j35 / 12.5 + j46 / 12.1 + j57 | 75° / 53° / 42° |

A 12.5 Ω or 20 Ω line of 60–90° (at 5.5 GHz) between the tee and the ideal ladder changes the fundamental little
for 12.5 Ω (≈12.4–12.7 Ω), more for 20 Ω (14–22 Ω, +j5 to +j10). The 2f0 phase then passes through the open
region only in mid-band (e.g. 12.5 Ω, 60°: φ2 = +2.89 / −1.18 / −2.57 rad at 4.4 / 5.8 / 7.2 GHz).

### Microstrip on this FR-4 (closed-form estimates)
118 Ω ≈ 0.39 mm wide (0.25 mm ≈ 130 Ω). 20 Ω ≈ 10.3 mm, 15 Ω ≈ 14.6 mm, 10 Ω ≈ 23.2 mm. Estimated width-wise
(transverse) resonance: 10 Ω line about 2.9 GHz, 15 Ω about 4.6 GHz, 20 Ω about 6.4 GHz. With 118 Ω / 20 Ω lines the
Matthaei C7 and C5 sections come out at 81° and 67° at 2.9 GHz.

### First S-parameter optimization of the stepped-line network (s50, all low-Z sections 20 Ω, Random optimizer)
Fundamental error up to about 12 Ω. |S11| dips to about 0.7 near 6 GHz, about 0.62 near 7 GHz and about 0.52 near 10.4 GHz.
The 2f0 phase is about 150–180° over 4.4–5 GHz. With ideal lumped capacitors instead of lines the dips disappear
(|Γ| ≈ 1 from 5 GHz upward), but the 2f0 phase stays near a short.

## Not yet done
- Input matching network and gate bias network (an ideal quarter-wave feed is used now).
- Stabilization and a stability check (μ, μ').
- Output network meeting the fundamental targets and harmonic conditions together.
- Full-amplifier HB verification across the band, realization with real component values, EM check.
