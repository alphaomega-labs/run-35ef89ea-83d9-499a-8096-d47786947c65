# Knowledge Notes

## Scope
This synthesis targets quantum vs classical reservoir comparison for PCA-encoded image classification, with explicit attention to entanglement, Hamiltonian dynamics, observables, and evaluation protocol choices.

## Source Families
- QRC/QELM core papers.
- Encoding and quantum-kernel perspective papers.
- Entanglement/simulability and observable-design papers.
- Classical RC/QML comparator framing papers.

## Cross-Source Synthesis
Most QRC papers agree that performance sensitivity is dominated by encoding map, measurable observables, and reservoir hyperparameters rather than a universal quantum advantage; this aligns with seed-source concerns about MNIST separability after PCA.
Entanglement is treated as a controllable mechanism that can enrich feature maps but also increase simulation cost and variance; papers with kernel framing caution that observed gains may be attributable to implicit feature-map geometry rather than intrinsically non-classical readout power.

## Equation/Claim Links
- arxiv:2602.14677: equation_seed=We consider an N -qubit model, with dimension D = 2N of its Hilbert space H, described by the | claim_seed=To demonstrate the effectiveness of this approach, we present numerical experiments on image classification and time series prediction tasks, including chaotic and strongly non-Mar
- arxiv:2509.06873: equation_seed=The complete initial quantum state of the system, composed of N = ⌈M/2⌉ qubits for a | claim_seed=Riera2 Institut de Fı́sica d’Altes Energies (IFAE) - The Barcelona Institute of Science and Technology (BIST), Campus UAB, 08193 Bellaterra (Barcelona), Spain 2 Qilimanjaro Quantum
- arxiv:2407.02553: equation_seed=Vjk = C/∥rj − rk ∥6 describes the van der Waals interactions between atoms. The detuning is split into the | claim_seed=Despite this promise, most contemporary quantum methods require significant resources for variational parameter optimization and face issues with vanishing gradients, leading to ex
- arxiv:2409.00998: equation_seed=vector ⃗x = [x1 , ..., xM ]T ∈ RM and N = ⌈M/2⌉ | claim_seed=We exploit a quantum extreme learning machine by taking advantage of its rich feature map provided by the quantum reservoir substrate.
- arxiv:2412.06758: equation_seed=Vjk = C/∥rj − rk ∥6 describes the interactions between | claim_seed=However, an immediate challenge is how to extract the relevant molecular features to improve the overall performance of the model, since the structure-property relationship for mol
- arxiv:2101.11020: equation_seed=of complex-valued 2n × 2n -dimensional matrices equipped with the Hilbert-Schmidt inner product ⟨ρ, σ⟩F = tr{ρ† σ} | claim_seed=The famous representer theorem uses this to show that “optimal models” (i.e., those that minimise the cost) can be written in terms of the quantum kernel as M M m=1 m=1 fopt (x) = 
- arxiv:2008.08605: equation_seed=for one-dimensional inputs x ∈ R: quantum models consisting of layers of trainable circuit blocks W = W (θ) and data | claim_seed=We show that one can naturally write a quantum model as a partial Fourier series in the data, where the accessible frequencies are determined by the nature of the data encoding gat
- arxiv:2601.00745: equation_seed=k=1 . In this approach, classical data are embedded into an n-qubit | claim_seed=We present a training-free, certified error bound for quantum regression derived directly from Pauli expectation values. Generalizing the heuristic of minimum accuracy from classif
- arxiv:2601.00921: equation_seed=none | claim_seed=• We benchmark practical quantum kernel models for low-dimensional tabular biomarkers, including quantum kernel ridge regression and a clustered quantum kernel feature approach bas
- arxiv:2601.04812: equation_seed=dx(t) = Ax(t)dt + Bu(t)dt, | claim_seed=We further characterise the simplest qWiener instantiation, consisting of concatenated quantum harmonic oscillators, and show the difference with respect to the classical case.
- arxiv:2601.08733: equation_seed=none | claim_seed=4” illustrates the steps in our approach to integrate Quantum Computing with RBM, which could improve the explainability.
- arxiv:2601.13808: equation_seed=Q+ (x) := | claim_seed=We show that the classification of these representations reduces to the finite case, as they all factorise through some finite quotient SO(3)p mod pk .
- arxiv:2601.16665: equation_seed=where each layer Ul consists of parametrized single-qubit rotations and entangling gates (Ry , Rx , and CN OT gate in this work), and θ = {θ | claim_seed=Our results show that algebraic learning converges significantly faster, escapes loss plateaus, and achieves lower final errors.
- arxiv:2601.17862: equation_seed=h′ = h + α · Wr z, | claim_seed=Domain Generalization with Quantum Enhancement for Medical Image Classification: A Lightweight Approach for Cross-Center Deployment Jingsong Xia∗1 and Siqi Wang†1 arXiv:2601.17862v
- arxiv:2601.18814: equation_seed=z = Wf + b, | claim_seed=The results demonstrate that lightweight quantum feature enhancement improves discrimination of positive lesions, particularly under class-imbalanced conditions.
- arxiv:2601.19721: equation_seed=of discrete time samples, and tk+1 = tk + ∆t. | claim_seed=Through theoretical analysis, we demonstrate and quantify a significant enhancement in QRC-based sensor performance for quantum-state classification, tomography, and feature-predic
- arxiv:2601.22194: equation_seed=i=1 , where yi ∈ {+1, −1}, the Support Vector Machine (SVM) solves the following | claim_seed=Experimental results demonstrate that the QSVM achieves competitive classification performance relative to classical SVM baselines while operating on substantially reduced feature 
- arxiv:2601.22253: equation_seed=which is a normalized (Tr(ρ) = 1) positive semi-definite (ρ ≥ 0) linear operator acting on a Hilbert | claim_seed=Through extensive numerical simulations across various quantum state families, we demonstrate that our model achieves high classification accuracy.
- arxiv:2601.22562: equation_seed=H(t) = f1 [W1 X(t) , W3 H(t−1) ] + B1,3 = f1 W1,3 [X(t) , H(t−1) ] + B1,3 | claim_seed=Under full-data conditions (400 000 samples), both architectures achieve accuracies above 99.97%.
- arxiv:2601.23084: equation_seed=line, with the equation w · x + b = 0. Here, w is termed | claim_seed=These results suggest that a revised approach is required to study the generalisation performance of current QML models.

## Similarities and Differences
- Similarity: nearly all studies use fixed train/test splits and linear readout on reservoir states.
- Difference: entangling depth, Hamiltonian family (including transverse Ising variants), and measurement subsets vary substantially and can invert benchmark rankings.
- Similarity: many papers report strong dependence on data encoding (angle/amplitude/reupload) and feature scaling.
- Difference: comparator strength varies from simple linear baselines to kernel SVM/CNN controls, affecting claims of advantage.

## Reusable Limitations
- Small system sizes and simulator-only settings can confound practical advantage claims.
- Dataset simplicity (especially MNIST+low-rank PCA) can mask method differences.
- Measurement/readout choices are often under-optimized, limiting fair conclusions about reservoir dynamics.

## Candidate Datasets Beyond MNIST
- Fashion-MNIST, KMNIST, EMNIST, and CIFAR-10 grayscale/PCA subsets to stress feature-map discrimination.

## Iteration 2 Acquisition Delta

- Added S037 (arXiv:2602.18377): PTM-based interpretability and explicit readout equations.
- Added S038 (arXiv:2603.21371): memory-vs-nonlinearity unification for QRC regimes.
- Added S039 (arXiv:2602.21544): TD-QELM update/readout equations and NARMA protocol context.

Cross-paper synthesis:
- S037/S039 share linear-observable readout framing for feature maps.
- S038 frames regime-dependent utility of entanglement rather than unconditional advantage.
