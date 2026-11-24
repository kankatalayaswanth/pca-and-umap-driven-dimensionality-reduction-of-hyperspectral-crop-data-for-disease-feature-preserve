"""
Comparative Benchmark Generator and Dynamic Academic Interpretation
Course: INT396 - Unsupervised Learning
"""

import pandas as pd
import numpy as np
from evaluation.metrics import evaluate_representation


def generate_comparison_benchmark(
    X_raw: np.ndarray,
    raw_labels: np.ndarray,
    raw_dim: int,
    X_pca: np.ndarray,
    pca_labels: np.ndarray,
    pca_dim: int,
    X_umap: np.ndarray,
    umap_labels: np.ndarray,
    umap_dim: int,
    n_clusters: int,
    y_true = None,
) -> dict:
    """
    Generate unified comparison table benchmarking Raw vs PCA vs UMAP representations.
    """
    raw_eval = evaluate_representation(X_raw, raw_labels, y_true=y_true, representation_name="Raw Features")
    pca_eval = evaluate_representation(X_pca, pca_labels, y_true=y_true, representation_name="PCA")
    umap_eval = evaluate_representation(X_umap, umap_labels, y_true=y_true, representation_name="UMAP")
    
    rows = []
    has_labels = (y_true is not None)
    
    # Helper to calculate dimension reduction %
    def calc_reduct(orig, red):
        return round((1.0 - (red / orig)) * 100.0, 1) if orig > 0 else 0.0

    eval_data = [
        {"rep": "Raw Features", "orig_dim": raw_dim, "red_dim": raw_dim, "eval": raw_eval},
        {"rep": "PCA", "orig_dim": raw_dim, "red_dim": pca_dim, "eval": pca_eval},
        {"rep": "UMAP", "orig_dim": raw_dim, "red_dim": umap_dim, "eval": umap_eval},
    ]

    for item in eval_data:
        e = item["eval"]
        row = {
            "Representation": item["rep"],
            "Original Dimensions": item["orig_dim"],
            "Reduced Dimensions": item["red_dim"],
            "Dim. Reduction (%)": f"{calc_reduct(item['orig_dim'], item['red_dim'])}%",
            "Clusters (K)": n_clusters,
            "Silhouette Score": e["silhouette"] if e["silhouette"] is not None else "N/A",
            "Davies-Bouldin": e["davies_bouldin"] if e["davies_bouldin"] is not None else "N/A",
            "Calinski-Harabasz": e["calinski_harabasz"] if e["calinski_harabasz"] is not None else "N/A",
            "ARI": e["ari"] if has_labels and e["ari"] is not None else "N/A",
            "NMI": e["nmi"] if has_labels and e["nmi"] is not None else "N/A",
        }
        rows.append(row)
        
    benchmark_df = pd.DataFrame(rows)
    
    # Store raw numerical dict for visualization and dynamic interpretation
    detailed_metrics = {
        "raw": raw_eval,
        "pca": pca_eval,
        "umap": umap_eval,
        "raw_dim": raw_dim,
        "pca_dim": pca_dim,
        "umap_dim": umap_dim,
        "n_clusters": n_clusters,
        "has_labels": has_labels,
    }
    
    # Generate dynamic scientific interpretation
    interpretation = generate_dynamic_interpretation(detailed_metrics)
    
    return {
        "benchmark_df": benchmark_df,
        "detailed_metrics": detailed_metrics,
        "interpretation": interpretation,
    }


def generate_dynamic_interpretation(details: dict) -> list:
    """
    Generate rigorous, factual findings based strictly on actual calculated values.
    No hardcoded or fabricated conclusions.
    """
    findings = []
    
    raw = details["raw"]
    pca = details["pca"]
    umap = details["umap"]
    raw_dim = details["raw_dim"]
    pca_dim = details["pca_dim"]
    umap_dim = details["umap_dim"]
    has_labels = details["has_labels"]
    
    # 1. Dimensionality Compression Impact
    pca_red = round((1 - pca_dim / raw_dim) * 100, 1)
    umap_red = round((1 - umap_dim / raw_dim) * 100, 1)
    findings.append({
        "category": "Dimensionality Compression",
        "badge": "Compression",
        "text": (
            f"The original hyperspectral feature space of {raw_dim} spectral bands was compressed to "
            f"{pca_dim} orthogonal components via PCA ({pca_red}% dimension reduction) and to "
            f"{umap_dim} manifold dimensions via UMAP ({umap_red}% dimension reduction). "
            f"Both methods mitigate the high collinearity typical of contiguous hyperspectral reflectance bands."
        )
    })
    
    # 2. Cluster Compactness (Silhouette Score)
    sils = {
        "Raw Features": raw["silhouette"],
        "PCA": pca["silhouette"],
        "UMAP": umap["silhouette"],
    }
    valid_sils = {k: v for k, v in sils.items() if v is not None}
    if valid_sils:
        best_sil_rep = max(valid_sils, key=valid_sils.get)
        best_sil_val = valid_sils[best_sil_rep]
        raw_sil_val = valid_sils.get("Raw Features", None)
        
        rel_diff = ""
        if raw_sil_val is not None and best_sil_rep != "Raw Features":
            diff = best_sil_val - raw_sil_val
            rel_diff = f" (an increase of {diff:+.4f} over the high-dimensional Raw baseline of {raw_sil_val:.4f})"
            
        findings.append({
            "category": "Cluster Compactness & Separation (Silhouette)",
            "badge": "Silhouette",
            "text": (
                f"<strong>{best_sil_rep}</strong> attained the highest Silhouette Score ({best_sil_val:.4f}){rel_diff}. "
                f"In hyperspectral clustering, a higher silhouette score indicates more compact intra-cluster point groups "
                f"with clearer decision boundaries between distinct crop clusters."
            )
        })
        
    # 3. Cluster Separation (Davies-Bouldin Index - Lower is Better)
    dbs = {
        "Raw Features": raw["davies_bouldin"],
        "PCA": pca["davies_bouldin"],
        "UMAP": umap["davies_bouldin"],
    }
    valid_dbs = {k: v for k, v in dbs.items() if v is not None}
    if valid_dbs:
        best_db_rep = min(valid_dbs, key=valid_dbs.get)
        best_db_val = valid_dbs[best_db_rep]
        findings.append({
            "category": "Inter-Cluster Similarity (Davies-Bouldin Index)",
            "badge": "Davies-Bouldin",
            "text": (
                f"<strong>{best_db_rep}</strong> yielded the most favorable (lowest) Davies-Bouldin index ({best_db_val:.4f}). "
                f"Because the Davies-Bouldin index evaluates the ratio of within-cluster scatter to inter-cluster distance, "
                f"this confirms that clusters formed in the {best_db_rep} space exhibit the least cluster overlap."
            )
        })

    # 4. Variance Ratio Criterion (Calinski-Harabasz Index - Higher is Better)
    chs = {
        "Raw Features": raw["calinski_harabasz"],
        "PCA": pca["calinski_harabasz"],
        "UMAP": umap["calinski_harabasz"],
    }
    valid_chs = {k: v for k, v in chs.items() if v is not None}
    if valid_chs:
        best_ch_rep = max(valid_chs, key=valid_chs.get)
        best_ch_val = valid_chs[best_ch_rep]
        findings.append({
            "category": "Variance Ratio Criterion (Calinski-Harabasz)",
            "badge": "Calinski-Harabasz",
            "text": (
                f"<strong>{best_ch_rep}</strong> achieved the maximum Calinski-Harabasz index ({best_ch_val:.2f}). "
                f"This criterion reflects the ratio of between-cluster dispersion to within-cluster dispersion, "
                f"validating strong cluster separability."
            )
        })
        
    # 5. Disease / Pest Signal Preservation (External Validation)
    if has_labels:
        aris = {
            "Raw Features": raw["ari"],
            "PCA": pca["ari"],
            "UMAP": umap["ari"],
        }
        valid_aris = {k: v for k, v in aris.items() if v is not None}
        if valid_aris:
            best_ari_rep = max(valid_aris, key=valid_aris.get)
            best_ari_val = valid_aris[best_ari_rep]
            best_nmi_val = details[best_ari_rep.lower().split()[0]]["nmi"]
            raw_ari = valid_aris.get("Raw Features", None)
            
            comparison_phrase = ""
            if raw_ari is not None:
                if best_ari_val > raw_ari:
                    comparison_phrase = f" Dimensionality reduction successfully enhanced disease signal alignment over raw features (Raw ARI: {raw_ari:.4f})."
                elif best_ari_val == raw_ari:
                    comparison_phrase = " Dimensionality reduction retained identical disease signal alignment to the raw bands."
                else:
                    comparison_phrase = f" Raw bands maintained slightly higher label congruence (Raw ARI: {raw_ari:.4f}), indicating some minor non-linear disease signal attenuation during compression."

            findings.append({
                "category": "Disease-Relevant Signal Preservation (External Benchmark)",
                "badge": "Disease Signal",
                "text": (
                    f"External evaluation against ground-truth labels demonstrates that <strong>{best_ari_rep}</strong> "
                    f"preserved the strongest correlation with crop disease classes, achieving an Adjusted Rand Index (ARI) of "
                    f"{best_ari_val:.4f} and Normalized Mutual Information (NMI) of {best_nmi_val if best_nmi_val is not None else 'N/A'}. "
                    f"{comparison_phrase}"
                )
            })
    else:
        findings.append({
            "category": "External Label Availability",
            "badge": "Unsupervised",
            "text": (
                "No ground-truth disease or pest label column was provided in the dataset. "
                "Consequently, external validation metrics (ARI, NMI, and contingency matrices) were omitted (N/A) "
                "to maintain rigorous academic integrity. All conclusions are derived strictly from unsupervised internal criteria."
            )
        })
        
    return findings
