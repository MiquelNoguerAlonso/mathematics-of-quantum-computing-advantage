THE MATHEMATICS OF QUANTUM COMPUTING ADVANTAGE
Foundations, measurement, and operational cost frontiers
Miquel Noguer i Alonso — 15 November 2026

START HERE
Quantum_Computing_Advantage.pdf is the complete, unified manuscript.
It includes the algorithmic foundations previously supplied separately,
new channel and statistical foundations, the revised operational theory,
and the supporting appendices. No companion text is required.

ORGANIZATION
I. Quantum experiments and statistical information: access models, state
   geometry, channels and instruments, binary discrimination, pure-state
   overlap, relative entropy, accessible information, and costed testing.
II. Algorithmic mechanisms: interference, Simon, phase estimation and Shor,
   Grover, amplitude estimation, Hamiltonian simulation, linear algebra,
   classical simulation boundaries, and error-control resources.
III. Operational cost frontiers: fixed-Y and adaptive measurement classes,
   certified one-angle optimization, strictly ordered information rates,
   exact noiseless zero-error cost, a two-sided noiseless resource law,
   a uniform joint noise-depth-confidence law, exact finite-confidence
   catalogue frontiers, and a calibrated amplitude-oracle example.
Appendices: decision transfer, identifiability, perturbation, workload
   allocation, numerical checks, and notation.

FINAL REVISION
- Added an exact finite-confidence frontier for a full likelihood-lattice
  family: parity and stopped drift give a matching converse, attained by
  mixing at most two threshold policies at every price and confidence.
- Proved the exact qubit realizability region of this family. An explicit
  full-rank noncommuting example has sharp readout-price threshold 6/5.
- Added exact rational optimality certificates, including global tail
  cutoffs, and independent interval Bellman checks. At equal prices the
  limiting cost reduction is exactly 1/6 for the example.
- Characterized general finite-catalogue minimax costs by finite policy
  programs with a proved convergence bound; uniform action separation
  improves the horizon bound to exponential convergence.
- Distinguished these constructions from the established LP, sensor-
  selection and comparison-of-experiments literature with precise sources.
- Added three substantive sections on sharper constants, costed experiments
  in arbitrary finite dimension, and memory with finite-confidence stopping.
- Proved a factor-27 fixed-depth envelope and improved the global constants
  to 1/70 and 42 (gap 2940, previously about 1.06 million).
- Proved joint KL/affinity occupation constraints and a feasible-dual
  certificate for multiple hypotheses with action-dependent prices.
- Added a full-rank noncommuting three-state catalogue, with interval-
  verified LP duals and attainable randomized-design upper bounds.
- Proved a collective-block entry-cost bound and the noninterchangeability
  of the confidence and memory-size limits; credited established quantum
  sequential testing and relative-entropy regularization.
- Proved a Bellman truncation bound and checked grid interpolation against
  exact finite-horizon policy envelopes.
- Added interval-certified finite readout gains with explicit measurement-
  price thresholds, plus a robust uncertainty-band guarantee.
- Led the abstract with the exact catalogue frontier and uniform resource
  law, retained the requested foundational treatment, and streamlined the
  operational argument.
- Moved three auxiliary bounds to Appendix G, retained the factor-three
  comparison as a remark, and merged the resource-regime tables.
- Upgraded all three noisy research-cost rows to outward-interval measured-rate
  lower bounds; added the corresponding small rate improvements.
- Reported the observed finite-grid band alongside the conservative
  uniform constants, without promoting it to a universal bound.
- Checked the Berta et al. Lemma 1/Theorem 2 and Flammia--O'Donnell
  Definition 2.26 citations against the primary papers.
- Added error and mean-cost guarantees under a certified conditional
  signal floor, allowing predictable visibility drift. The proof retains
  both absorbing boundaries to recover the exact logistic error bound.
- Expanded the stopped-tree truncation and generalized-eigenvalue
  calculations, and clarified the hypothesis-independent phase eigenstate.
- Kept the established verification suites; the supplied coarse angular
  grid does not satisfy its proposed residual tolerance. Added a separate
  matrix/drift suite with dynamic programs over adaptive signal choices.
- Standardized all bra/ket arguments with explicit braces and added
  spacing after downarrow commands; the original shorthand was valid TeX.
- Closed the intermediate depth-limited gap up to universal constants.
- Proved d_M(t) between (2/5) s_t^2 G_t and 3 s_t^2 G_t, sharpening
  the earlier (1/4, 7) bracket, where
  G_t = 1 + log(1 + s_t^2/(1-r_t^2)).
- Added exact repeated conclusive counting and a mixed-state affinity
  converse, proving the uniform finite-confidence law
  C_M,delta is comparable to A_eff [1 + log(1/delta)/G_eff], with explicit constants.
  These are uniform comparison bounds, not exact finite-confidence costs.
- Integrated the reviewed noiseless converse and its full stopped-affinity
  proof. The largest allowed depth minimizes the zero-error cost.
- Added the projective-measurement reduction with its literature source,
  a global variational-residual certificate, and I_Y < I_M < I_Q for
  every positive noise rate in the stated phase-restricted model.
- Explained and proved why Y can be one-copy minimax optimal while it
  remains asymptotically suboptimal for sequential information acquisition.
- Recomputed amplitude-case lower bounds using outward interval arithmetic.
  At gamma=.020, C_M,.01 >= 6658.96 > 5701.07 >= C_R,.01, a lower-bound
  margin of approximately 16.80% at the common stated block price.
- Corrected the small-noise readout antecedent and declared the real
  perturbation parameter in the quadratic example.
- Retained the title and author; no ORCID or exercises are included.

BUILD: STANDARD TEX LIVE OR OVERLEAF
  latexmk -pdf Quantum_Computing_Advantage.tex
Alternatively run pdfLaTeX, BibTeX, pdfLaTeX, pdfLaTeX.
The bibliography uses natbib/plainnat. The inputs foundational_tools.tex,
joint_frontier.tex, auxiliary_readout_bounds.tex, deep_extensions.tex,
sharp_characterization.tex, all generated table files, and all five PDF
figures are included.
Upload this ZIP to Overleaf and select Quantum_Computing_Advantage.tex
as the main document. Python is not required for compilation.

REPRODUCE ALL FOURTEEN VERIFICATION SUITES
Use Python 3.12 with the packages in requirements.txt, then run:
  python3 verify_examples.py
  python3 research_checks.py
  python3 verify_boost.py
  python3 verify_case_study.py
  python3 verify_measurements.py
  python3 verify_readout_patch.py
  python3 verify_operational_frontier.py
  python3 verify_interval_certificates.py
  python3 verify_foundational_tools.py
  python3 verify_joint_frontier.py
  python3 verify_refinements.py
  python3 verify_table_rates.py
  python3 verify_deep_extensions.py
  python3 verify_sharp_characterization.py
The same ordered sequence is available as make verify.
The interval script must follow the operational and case-study scripts:
it generates the final amplitude and operational-rate tables. Earlier
scripts intentionally regenerate their own intermediate table versions.
Run verify_table_rates.py after the earlier suites to restore the final
research-cost table; verify_deep_extensions.py produces its own two tables.
The adaptive measurement suite may take a few minutes. Random seeds are
fixed; numerical-library changes can alter the last floating-point digits.

RESULT FILES AND COMPATIBILITY
The original six JSON schemas retain legacy Istar, LQ/UQ, or L_Q/U_Q
names for fixed-Y quantities. Those keys do not mean optimal POVM costs.
The explicit I_Q and L_Q_all_readouts fields are quantum-information bounds.
The new operational_frontier_results.json records measured-rate estimates,
maximizing depths, and residual diagnostics for 348 allowed depths.
interval_certificate_results.json records outward rate and cost enclosures
for all 38 allowed depths in each of the three amplitude cases, including
the unique maximizing depths and the direct-sampler upper bound.
foundational_verification_results.json reports independent spectral checks
of channels, discrimination, information, and the noiseless cost scale.
joint_frontier_results.json records independent checks of the new bounds,
finite mixed-state history trees, absorbing Markov systems, and the full
parameter grid underlying joint_frontier.pdf. The explicit constants
come from the proof; they are not inferred from the grid.
refinement_verification_results.json records generalized eigenvalues,
dephasing composition, the coarse-versus-refined residual comparison, and
16 adversarial finite-horizon drift calculations.
The validation manifest records the runs, environment, hashes, and PDF QA.

INTERPRETATION
All comparisons use the specified interfaces and costs. The calibrated
block model permits classical adaptation and discards quantum memory
between blocks; all readouts share a common block price. The general
algorithmic chapters explicitly distinguish stronger coherent interfaces.
The examples are mathematical calculations, not hardware benchmarks.
Established foundational and sequential results are credited; the text
does not claim novelty priority for them. The intermediate positive-noise
depth-limited regime has matching uniform bounds with constants 1/70 and 42.
The extended sections supply a robust calibration-band guarantee, a priced
finite-dimensional multi-hypothesis converse, and collective-block memory
and Bellman stopping results. The lattice family has exact finite-confidence
minimax values within its two-readout catalogue. The general policy programs
converge to exact values
but need not be computationally efficient. Neither result provides a general
composite-state law, arbitrary persistent-memory protocols, or an
error-corrected implementation.
Interval arithmetic evaluates proved inequalities; it is not a
proof-assistant formalization.

RESEARCH-COST CERTIFICATES
table_rate_results.json records interval endpoints at all 136 allowed
depths across the Noisy, Costly, and Restarts cases. Exact decimal inputs
and downward rounding preserve the displayed cost inequalities.

DEEP EXTENSION VALIDATION
verify_deep_extensions.py checks 633 global parameter cases, 90 independent
measured-information optimizations, 57 interval-certified readout depths,
six interval-feasible multi-hypothesis duals, six certified attainable designs,
40 three-hypothesis adaptive trees, 20 asymmetric directional-policy trees,
six exact Bellman policy horizons, and three stopping lower certificates.
The original joint-frontier suite retains its weaker valid constants as an
independent historical check; the deeper suite verifies the new constants.
Its complete results and arithmetic qualifications are in
deep_extension_results.json.

EXACT FRONTIER VALIDATION
verify_sharp_characterization.py uses Fraction arithmetic to certify nine
exact optima and their barrier mixtures, dual multipliers, and infinite-tail
cutoffs. Decimal table entries are rounded outward; these are exact-value
certificates, not approximate optimizer outputs. A separate interval Bellman
recursion allows every adaptive action at horizon 96. Independent absorbing
Markov systems, 2942 enumerated trees, conditional minimax LPs, and small
three-hypothesis LP/dual comparisons check different parts of the argument.
The JSON includes all rational values, certificates and price-curve points.
No posterior-grid interpolation is used for the exact-frontier example.
