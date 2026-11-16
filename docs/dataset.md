# Dataset Documentation & Provenance

**Course:** INT396 - Unsupervised Learning  
**Project:** PCA and UMAP-Driven Dimensionality Reduction of Hyperspectral Crop Data for Disease-Relevant Feature Preservation

---

## 1. Overview of Primary Dataset

The primary dataset employed in this project is an authentic, peer-reviewed hyperspectral research dataset released in December 2024:

- **Full Title:** *Hyperspectral imaging dataset of potato plants exposed to water-deficient and nematode induced stress*
- **Primary Authors:** Janez Lapajne, Nik Susič, Andrej Vončina, Uroš Žibrat
- **Affiliation:** Agricultural Institute of Slovenia, Plant Protection Department, Ljubljana, Slovenia
- **Archive DOI:** [10.5281/zenodo.14267877](https://doi.org/10.5281/zenodo.14267877)
- **Licensing:** Creative Commons Attribution 4.0 International (CC BY 4.0)

---

## 2. Experimental Setup and Agronomic Grounding

### Plant Material and Growth Conditions:
- **Host Crop:** Potato (*Solanum tuberosum* L., cultivar Désirée).
- **Environment:** Controlled glasshouse experiment conducted at the Agricultural Institute of Slovenia.
- **Biotic Stress Agent:** Potato cyst nematodes (PCN), comprising two quarantined microscopic soil parasites:
  - *Globodera pallida* (pathotype Pa3)
  - *Globodera rostochiensis* (pathotype Ro1)
- **Abiotic Stress Agent:** Controlled drought stress (water withholding 3-5 days prior to imaging).

### Inoculum Levels:
- **Low Dose (L):** 5 viable eggs + juvenile stage 2 (J2) per $\text{cm}^3$ of soil.
- **High Dose (H):** 30 viable eggs + J2 per $\text{cm}^3$ of soil.
- **Control (C):** Non-inoculated sterile substrate.

### Factorial Treatment Matrix (70 Potted Replicates):
| Group Code | Watering Regime | Parasite / Stressor | Inoculum Dose | Biological Significance |
| :--- | :--- | :--- | :--- | :--- |
| `WWC` | Well-Watered | None (Control) | None | Healthy baseline vegetative growth. |
| `WWGPL` | Well-Watered | *G. pallida* | Low (5 eggs/$\text{cm}^3$) | Early/mild biotic pest infection. |
| `WWGPH` | Well-Watered | *G. pallida* | High (30 eggs/$\text{cm}^3$) | Severe biotic pest infection. |
| `WWGRL` | Well-Watered | *G. rostochiensis* | Low (5 eggs/$\text{cm}^3$) | Mild biotic pest infection. |
| `WWGRH` | Well-Watered | *G. rostochiensis* | High (30 eggs/$\text{cm}^3$) | Severe biotic pest infection. |
| `WDC` | Water-Deficient | None (Control) | None | Pure abiotic drought stress. |
| `WDGPL` | Water-Deficient | *G. pallida* | Low | Compound biotic + abiotic stress. |
| `WDGPH` | Water-Deficient | *G. pallida* | High | Severe compound stress. |
| `WDGRL` | Water-Deficient | *G. rostochiensis* | Low | Compound biotic + abiotic stress. |
| `WDGRH` | Water-Deficient | *G. rostochiensis* | High | Severe compound stress. |

---

## 3. Imaging Sessions and Disease Lifecycle Stages

1. **Session 1 (DAI 10 - Day 10 post-inoculation):**
   - Nematode juveniles penetrate root tips; no foliar symptoms visible to the human eye.
   - Tests early pre-symptomatic spectral detection capability.
2. **Session 2 (DAI 40 - Day 40 post-inoculation):**
   - High nematode metabolic demand; syncytia feeding sites established in root vascular bundles.
   - Early reduction in stomatal conductance ($g_{sw}$) and photosynthetic quantum efficiency ($\Phi_{PSII}$).
3. **Session 3 (DAI 70 - Day 70 post-inoculation):**
   - Adult nematode females develop into brown cysts containing hundreds of eggs.
   - Pronounced root necrosis, shoot stunting, chlorophyll degradation, and severe foliar chlorosis.

---

## 4. Sensor Specifications & Wavelength Calibration

- **VNIR Camera:** HySpex VNIR-1600 (Norsk Elektro Optikk, Oslo, Norway)
  - Spectral bands: 160 continuous bands
  - Range: $409.8\text{ nm}$ to $988.4\text{ nm}$
  - Spectral sampling: $\approx 3.6\text{ nm}$
- **SWIR Camera:** HySpex SWIR-384 (Norsk Elektro Optikk)
  - Spectral bands: 288 continuous bands
  - Range: $950.0\text{ nm}$ to $2509.6\text{ nm}$
  - Spectral sampling: $\approx 5.4\text{ nm}$
- **Calibration Standard:** Spectralon 99% diffuse reflectance white reference panel; radiometric calibration and dark-current subtraction performed before each acquisition.

---

## 5. Secondary Benchmark: Indian Pines AVIRIS Dataset

- **Source:** Purdue University / NASA Jet Propulsion Laboratory (AVIRIS sensor).
- **Scene:** Agricultural field flightline over Northwestern Indiana.
- **Dimensions:** 200 calibrated spectral channels, covering 16 agricultural and land-cover categories.
- **Purpose in Project:** Demonstrates system generalization on larger-scale pixel-level agricultural classifications.
