# Cycle six: fifteen distinct interventions

Six cycles are complete. Parents entering this cycle were H02, C17, H10, H04 and C01.
The evaluated roster contains these five unchanged models and the fifteen children
below. All 149 frozen ROIs, coarse labels, class IDs, physical spacings and
the equal #9+#19 RGB conversion remain unchanged. The primary utility remains
the mean of IoU minus 0.05 times each image's excess seconds above one second.
No fine reference is available to a prediction implementation.

| Child | Parent | Single change to the model | Falsifiable risk |
| --- | --- | --- | --- |
| J01 | H02 | Conductance of the appearance field follows original H&E optical-density edges | Cell/stain edges can become false barriers |
| J02 | H02 | Joint two-way model: constant class colours plus one shared spatial nuisance field | Genuine class-specific drift is excluded |
| J03 | H02 | A Dirichlet harmonic glass-access probability connects exterior white seeds to interior white spaces | Border adipose can still look like glass |
| J04 | C17 | Integrate the palette posterior's dominant-class mass rather than take its MAP fractions | Finite importance quadrature can miss narrow posterior modes |
| J05 | C17 | Couple inverse fractions on the full occupied-colour graph, not the spatial image graph | Neighbouring colours can belong to different tissues |
| J06 | C17 | Learn a log-fraction decoder from original coarse tile observations | Coarse labels can miscalibrate a good palette inverse |
| J07 | H10 | Analytically integrate squared hinge loss under the same Gaussian tile-mean error | Bounded growth can underfit truly difficult tiles |
| J08 | H10 | Recover class scatter after subtracting tile-mean measurement noise, then use quadratic Gaussian discrimination | Few tiles poorly identify latent class scatter |
| J09 | H10 | Learn additive nonlinear colour responses with fixed Gaussian cubature | Additivity misses interactions and finite cubature is approximate |
| J10 | H04 | Fit one proper rigid motion of the entire palette, preserving its pairwise distances | Real conversion distortion may require nonrigid calibration |
| J11 | H04 | Fit palette endmembers to tile means whose latent fractions obey their observed winner inequalities | Winner of mean fractions need not equal winner of pixel counts |
| J12 | H04 | Propagate endmember calibration covariance into expected palette reconstruction risk | Interior Gaussian covariance ignores active box constraints |
| J13 | C01 | Condition all rival Gaussian count comparisons on the same own-count variable | Independent Gaussian class counts ignore multinomial anticorrelation |
| J14 | C01 | Infer a latent uniform contamination probability for normalized composite tile-winner evidence | Noise and poor appearance fit are not identifiable |
| J15 | C01 | Learn an eight-unit nonlinear colour vote field under the tile-count likelihood | Nonconvex fitting can create unsupported colour islands |

These are separate models, not a voting ensemble or a sequence of combined
solvers. A new C++ Gaussian quadratic kernel supports J08; a dominance-simplex
kernel supports J11. Their ports and threading profiles do not count as extra
innovations. Tests must compare the kernels with independent numerical
references and one/two-thread output parity. Failed finite optimization must
remain explicit rather than be labelled a converged optimum.

J01/J02 retain H02's 8 µm field grid and 16 µm physical regularity. J01 changes
the graph conductance; J02 changes the shared generative structure and solves
one joint convex block system. J03 uses a 4 µm H&E grid, exterior white seeds
and stained foreground seeds. Only bright pixels with glass access above
0.95 can become zero. It is not an automatic deletion of every lumen.

J04 uses deterministic cyclically symmetrized, fixed-seed scrambled
Sobol/Dirichlet importance quadrature with 128 base points per class and
reports its finite scope. Tied quadrature nodes split their dominance mass
equally. It integrates the original Gaussian
palette energy with a uniform simplex base measure; dominant-class probability
is different from mean mixture fraction. J05 adds one colour-graph quadratic
to the palette objective and reports its projected-gradient residual. J06
uses mean log-fractions per known tile as its calibrated observation, retaining
an identity-centred decoder prior. No input conversion file is rewritten.

J07 changes the statistical loss, J08 changes the class-density geometry,
and J09 changes the function class. J09 uses 27 Gaussian-Hermite points and
four softplus knots per colour coordinate, with one convex finite objective.
J08 subtracts I/25 from whitened between-tile class scatter, projects negative
eigenvalues to zero, shrinks sparse classes toward pooled scatter and adds
pixel nuisance I back for prediction.

J10 alternates exact fraction fitting with an anchored proper Procrustes
update. Rotated latent vertices can lie outside the RGB cube; they are model
parameters, not a newly generated RGB image. J11's star-order projection pools
the observed winner with violating rivals, then projects onto the simplex.
Finite accelerated quadratic solves report residuals. J12 uses an interior
Gaussian approximation around H04's bounded palette fit; expected loss adds
the trace of endmember uncertainty, not an arbitrary extra ridge.

J13 uses 21 Gaussian-Hermite points to integrate one shared own count, with
analytic gradients. It is not an exact multinomial winner probability. J14
normalizes all composite winner scores before mixing with uniform corruption,
and bounds inferred noise below 0.49. J15 learns a residual tanh colour model
with eight hidden units and a linear initial model, using CPU arrays only.

Preflight must cover all new models, mathematical gradients and kernels.
Then freeze a source snapshot, run an eight-case engineering pilot, audit
exact unchanged-parent masks and actually inspect the pilot. Only then run
all 149. A cycle cannot complete before all 149 all-method sheets have been
viewed and five survivors selected. The delivered viewer must contain only
the signed five. Colour outlier proposals retain unknown biological identity;
all missed small regions and all deleted positive reference pixels are scored.

## Local numerical preflight

All 209 focused tests pass, including the 36 new model, analytic-gradient,
independent-integration, physical-background and C++ checks. The first palette
integration check found an absolute probability error of 0.035641 in an
unscrambled preliminary quadrature. Before any benchmark scoring, the scheme
was changed to the fixed-seed scrambled rule above. At the production binary
point count (256 after cyclic symmetry), independent adaptive integration
gives 0.866253 against the model's 0.871235, absolute error 0.004982. This is
one numerical reference case, not an accuracy bound for arbitrary tissues.

Gaussian scoring agrees with NumPy; dominance projection and quadratic
solutions agree with independent SLSQP references. Both C++ kernels produce
identical one/two-thread outputs. The small local timing workload shows no
threading benefit; larger and Chiamaka workloads must be measured rather
than assuming speedup. No benchmark prediction or accuracy gain is asserted
by these numerical tests. Numerical tests alone did not complete the cycle; full review is documented below.

## Engineering pilot and full execution

The frozen de01909 snapshot also passes all 209 focused tests on Chiamaka.
All 176 pilot masks succeed, with zero failed methods. Both remote and local
audits verify the 176 exact display rasters and 40 unchanged-parent native
comparisons. All eight all-method contact sheets and three native label
sheets were actually inspected, with hashes and explicit observations.

Native BCSS45 shows J01 largely correcting H02's vessel-to-fat stripe, while
J02/J08 retain that error. In BCSS26 J08/J15 still insert false fat, and J06
adds false violet loops and rims. J15 creates nested contours in the BCSS06
overview. The background child J03 deletes an additional 6,474 annotated
positive native pixels across the eight cases. Its BCSS06 native crop shows
extra deletion along an already incorrect white cut through annotated fat;
all models also miss small green stromal islands. These are explanatory
development observations, not independent validation or survivor selection.

Finite optimizer flags remain explicit: J15 is unsuccessful on six pilot
cases, C01/J09 on three each, J07/J13 on two, and H10/J06/J14 on one each.
Successful mask production does not mean every optimization converged.

Serial Chiamaka synthetic kernel profiles give one/two-thread medians of
0.05432/0.02739 seconds for the million-pixel Gaussian kernel (1.983x), and
0.02223/0.01165 seconds for 2,048 dominance quadratics (1.908x), with exact
thread-output parity. This is kernel evidence, not whole-mask acceleration.

The same frozen implementation completed all 149 ROIs without model changes after
pilot scoring or review. All3,278 masks succeed. Both native audits pass and
all 745 unchanged-parent comparisons are exact; all 176 pilot masks match the
full-run arrays. All149 overview sheets and six targeted native label sheets
were actually inspected.

The signed five are J01, H02, C17, J07 and H10. All cycle-six candidates remain
below one second on every measured ROI. J01 improves H02 by 0.27685 IoU percentage
points; J07 improves H10 by 0.10996 points. Both have documented regressions.
J03 adds positive-support deletion and is rejected. CYCLE_SIX_REVIEW.md contains
benefits and rejection reasons for every candidate.

Six of twenty cycles are complete. MainPID 898917 and posthocPID 903248 are terminal;
do not relaunch them. The goal remains active.
