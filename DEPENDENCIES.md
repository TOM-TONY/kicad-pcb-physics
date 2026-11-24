# Dependency licenses

This source distribution contains the package, an original copper-fixture generator, tests and documentation. It ships no board designs, simulator binaries, vendor device models, data books, fonts or reference artwork.

| Component | Interface and distribution |
|---|---|
| NumPy and SciPy | Imported numerical libraries. The primary projects use BSD-style terms. Individual files and bundled numerical runtimes have additional notices, including permissive and runtime-specific exceptions. Keep the notices of the actual wheels when redistributing them. [NumPy license](https://github.com/numpy/numpy/blob/main/LICENSE.txt) · [SciPy license](https://github.com/scipy/scipy/blob/main/LICENSE.txt) |
| PyAMG | MIT-licensed multigrid preconditioner, installed as a dependency of the geometry-conforming solver. [Project license](https://github.com/pyamg/pyamg/blob/main/LICENSE.txt) |
| Shapely and GEOS | Shapely is BSD-3-Clause. Its binary wheels may include GEOS, licensed LGPL-2.1-or-later. Redistribution of that native library has separate notice, source and relinking requirements. This source package declares Shapely as a dependency. [Shapely license](https://github.com/shapely/shapely/blob/main/LICENSE.txt) · [GEOS license](https://libgeos.org/) |
| KiCad and pcbnew | KiCad is GPL-3.0-or-later. An external interpreter runs the geometry bridge against the user's installed bindings and returns geometry data. The bridge's original source has the project's MIT terms, compatible with GPLv3. A distributor combining it with GPL code must also satisfy the GPL for that combination. [KiCad licenses](https://www.kicad.org/about/licenses/) |
| ngspice | Executed as a separate process. The distribution contains modified BSD and other component-specific terms. Read the COPYING file for any build you redistribute. No ngspice code, device library or binary is part of this package. [ngspice distribution](https://ngspice.sourceforge.io/download.html) |
| LTspice | Optional, separately installed proprietary simulator, called via a temporary passive circuit description. Its vendor license governs the installation. No simulator or vendor model is redistributed. [Vendor download and terms](https://www.analog.com/en/resources/design-tools-and-calculators/ltspice-simulator.html) |
| Python and build tools | The interpreter and setuptools are separately installed. Standard platform math and sparse backends are chosen by the installed numerical wheels. |

The MIT grant covers this repository's original files. Dependency licenses continue to govern the corresponding dependencies. Bundling a runtime, statically linking libraries, or shipping vendor models would require a separate distribution review.

Independent Touchstone readback used scikit-rf, distributed under the BSD 3-Clause license. It is a verification tool outside the runtime package. [scikit-rf license](https://github.com/scikit-rf/scikit-rf/blob/master/LICENSE.txt)
