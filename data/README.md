# Hyperspectral Datasets Documentation

This directory contains real, peer-reviewed hyperspectral datasets used for unsupervised disease and crop condition analysis.

---

## 1. Primary Dataset: Potato Crop Disease & Stress Hyperspectral Signatures

- **Source / Repository:** Zenodo Open Access Archive
- **DOI:** [10.5281/zenodo.14267877](https://doi.org/10.5281/zenodo.14267877) / [10.5281/zenodo.14267876](https://doi.org/10.5281/zenodo.14267876)
- **Publication Date:** December 3, 2024
- **Authors:** Janez Lapajne, Nik Susič, Andrej Vončina, Uroš Žibrat (Agricultural Institute of Slovenia, Plant Protection Department)
- **License:** Creative Commons Attribution 4.0 International (CC BY 4.0)

### Scientific Context:
Evaluates hyperspectral reflectance signatures of potato crops (*Solanum tuberosum*, cultivar Désirée) subjected to controlled biotic stress (root infection by potato cyst nematodes *Globodera pallida* and *Globodera rostochiensis* at low and high inoculum doses) and abiotic stress (drought / water deficit).

### Sensor Specifications:
- **VNIR Push-Broom Sensor:** HySpex VNIR-1600 (Norsk Elektro Optikk), 160 continuous spectral bands covering visible to near-infrared ($409.8\text{ nm} - 988.4\text{ nm}$).
- **SWIR Push-Broom Sensor:** HySpex SWIR-384 (Norsk Elektro Optikk), 288 continuous bands covering short-wave infrared ($950.0\text{ nm} - 2509.6\text{ nm}$).
- **Combined Spectrum:** 448 contiguous bands spanning $409.8\text{ nm} - 2509.6\text{ nm}$.

### Experimental Setup & Factorial Groups:
70 potted plants tracked across 3 developmental imaging sessions (IS):
1. **DAI 10 (Day 10 after inoculation):** Early asymptomatic stage; no visible leaf discoloration.
2. **DAI 40 (Day 40 after inoculation):** Active metabolic disturbance stage.
3. **DAI 70 (Day 70 after inoculation):** Mature symptomatic stage; cyst production, severe root disruption, and canopy chlorosis.

#### Classes / Treatments:
- `Healthy Control`: Well-watered, uninoculated control plants.
- `Biotic Nematode Disease`: Well-watered, inoculated with *G. pallida* or *G. rostochiensis* (Low: 5 eggs/cm³, High: 30 eggs/cm³).
- `Abiotic Drought Stress`: Water-deficient, uninoculated control plants.
- `Combined Stress (Drought+Nematode)`: Water-deficient, nematode-inoculated plants.

---

## 2. Benchmark Dataset: Indian Pines AVIRIS Hyperspectral Scene

- **Source:** NASA Jet Propulsion Laboratory (JPL) & Purdue University Laboratory for Applications of Remote Sensing (LARS)
- **Sensor:** Airborne Visible / Infrared Imaging Spectrometer (AVIRIS)
- **Spectral Bands:** 200 calibrated bands (water absorption bands 104-108, 150-163, and 220 removed from original 224 bands).
- **Wavelength Range:** $0.4\text{ µm} - 2.5\text{ µm}$.
- **Samples:** 3,129 stratified labeled agricultural pixels across 16 crop and vegetation categories (Corn-notill, Corn-mintill, Soybean-notill, Wheat, Woods, Hay, etc.).

---

## Processed Tables in `data/processed/`

| File Name | Samples | Features | Description |
| :--- | :--- | :--- | :--- |
| `potato_disease_dai70_vnir.csv` | 70 | 160 bands + 8 meta | Day 70 symptomatic potato plants (VNIR $409-988\text{ nm}$). |
| `potato_disease_dai70_allbands.csv` | 70 | 448 bands + 8 meta | Day 70 potato plants (VNIR + SWIR $409-2509\text{ nm}$). |
| `potato_disease_multistage_vnir.csv` | 213 | 160 bands + 8 meta | Multi-temporal progression across DAI 10, 40, and 70. |
| `indian_pines_crops.csv` | 3,129 | 200 bands + 2 meta | AVIRIS agricultural pixel benchmark across 16 crop classes. |
| `potato_bands_metadata.json` | - | 448 wavelengths | Exact physical nanometer wavelength mappings for all spectral bands. |
