# Numerical validation

All values below come from script-generated copper coupons and analytic excitations. They describe the indicated mathematical problem.

| Model | Independent comparison | Observed discrepancy |
|---|---|---:|
| Whole-board heat | Two-material strip with an off-grid material interface, uniform heat and fixed end temperatures. The exact solution integrates the flux through both materials. Maximum nodal error at 1, 0.5 and 0.25 mm partitions. | 7.883, 1.972 and 0.492 K, respectively. Finest error is 0.169% of the analytic peak rise. |
| DC and skin impedance | Rectangular strip resistance versus resistivity times length over area. Slab AC resistance versus an independent 301-point through-thickness diffusion solve at 40 MHz. | Below 1e-9% DC; 0.00371% AC. |
| Generated and received fields | Finite straight current, point charge with image, uniform-flux Faraday loop and broadside retarded filament integral. | Below 1e-9% for the first three, 0.00184% for the small-loop radiation example. |
| AC reference current | Two unequal foils driven by a longitudinal EMF and joined at their terminals. Closed-form complex branch split and parallel-plate longitudinal voltage. | Below 1e-9% branch amplitude; 0.000047% voltage. |
| Power-port impedance | Rectangular regularised filament compared with its closed integral; independent ngspice AC sweep of one and two mounted branches. | 0.391% loop inductance; below 1e-9% for the passive circuit magnitude. |

Additional checks exercise AC currents on isolated metal islands, magnetic-field invariance under translation, and parallel capacitors sharing the same mounting copper. The latter preserve both the common resistance and the common inductance. The integration test generates a new two-layer layout, exports its geometry, runs both thermal and electrical studies, replays a charging transient, and opens the generated result files for checks. If the optional simulator executable is configured, the same AC cases are checked there too.

Finishing these checks establishes numerical consistency over the tested geometries and frequencies. Absolute temperatures depend on actual package contacts, air temperature, heat-transfer coefficients and losses. Closed-loop path width and the three-dimensional current distribution can change installed inductance. External field amplitudes, dielectric edge fields, cable impedance, and return paths outside the modeled region require separate measurements.

## Transmission lines

The following quasi-static cross sections use 24 cells per local reference dimension and an exterior margin of 18 dielectric heights. The reference for thin coupled and uncoupled microstrips is a spectral boundary solution. Centered striplines use elliptic-integral solutions. The reference for the finite-thickness strip uses a separate finite-volume discretization.

| Section and mode | Calculated Ω | Reference Ω | Relative error |
|---|---:|---:|---:|
| Microstrip | 49.76326 | 49.94244 | −0.359% |
| Centered stripline | 49.92268 | 50.00000 | −0.155% |
| Coupled microstrip, even / odd | 61.08819 / 41.06749 | 61.34048 / 41.20379 | −0.411% / −0.331% |
| Coupled stripline, even / odd | 57.25088 / 41.57541 | 57.38409 / 41.76149 | −0.232% / −0.446% |
| Microstrip with 35 µm copper | 48.49963 | 48.63442 | −0.277% |

The maximum relative error in effective permittivity across these seven modes was 0.345%. At the lower-cost 12/7 resolution the maximum impedance error was 1.93%. For new geometries refine the mesh and increase the margin independently. The centered stripline coupon also checks the homogeneous-dielectric propagation velocity. [Stripline analytical standard](https://www.sonnetsoftware.com/support/downloads/publications/transnov94_striplinestandard.pdf)

The illustrative masked differential pair in the generator gave 89.98294 Ω over the uniform parallel section. An independently refined cross section gave 90.06386 Ω. The jog produced 10.04254 ps of skew, against 10.05426 ps with independently calculated velocities. The nominal 10 ps budget is deliberately exceeded. At 2 GHz differential insertion loss was 0.290136 dB, against 0.290394 dB with an independent modal cascade. Both calculations continue the ideal reference across the flagged 1.5 mm gap. Actual slot radiation and launch loss require a three-dimensional model or a measurement.

The five-frequency, four-port Touchstone files were read back with scikit-rf 2.2.0. Converting the single-ended matrix to differential and common modes reproduced every complex entry within 6.88e-13. The mixed-mode port order is differential-near, common-near, differential-far, common-far. With 50 Ω single-ended ports the respective references are 100, 25, 100 and 25 Ω. [Touchstone 2.0](https://ibis.org/touchstone_ver2.0/touchstone_ver2_0.pdf) · [scikit-rf conversion](https://scikit-rf.readthedocs.io/en/latest/api/generated/skrf.network.Network.se2gmm.html)

## Ground stitch placement

| Coupon | Reference | Result |
|---|---|---|
| Identical barrels with equipotential ends | Parallel branch resistance and port inductance scale as 1/N for one, two and three barrels | Relative checks within 1e-10. The coupon includes identical radial access terms. |
| Offset foil connection | Rebuild and independently factor the sheet network for every remaining site after each insertion | The nearest eligible contact is selected first. DC and port voltage ratios agree within 1e-9 and closed-loop inductance within 1e-8. |
| Rectangular closed current path | Regularized rectangular filament plus internal impedance | 10.5860 nH calculated against 10.5680 nH, a 0.17% difference. |

These are tests of the network and selection algorithm. A board-specific return field also depends on the excitation path, barrel return radius, terminal geometry and boundary. Recompute a fixed site on a finer mesh and in a larger window before interpreting its rank. Greedy ranking gives the largest modeled improvement at each step. A group of sites placed together can have a different ranking.
