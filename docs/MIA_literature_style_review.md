# Medical Image Analysis literature and presentation review

Reviewed 16 October 2026 against the 17 supplied Medical Image Analysis PDFs. This is an editorial note for the Bora draft, not manuscript text.

## Presentation observed in the supplied articles

- Titles name the method or question and the imaging task. Abstracts move from a concrete problem to the method, evaluation design, quantitative findings and a bounded conclusion.
- Introductions distinguish the unmet task from adjacent methods and end with the study contribution. Related work discusses what each cited method actually solves.
- Dataset access locations and annotation or split limits are stated near the experimental design. Results report denominators, comparator, metric direction and failure cases; figures and tables have self-contained legends.
- Discussion distinguishes an observed result from proposed applications and gives limits on sampling, domain shift, resolution and generalization.

The Bora manuscript already follows the numbered Introduction–Related work–Methods–Experimental setup–Results–Discussion–Conclusion structure. This update adds explicit retrieval locations in Section 4.1, maintains captions beneath figures, updates stale code-availability wording and cites only articles that support a specific statement.

## Medical Image Analysis papers selected for citation

| Supplied article | Relevance to Bora | Manuscript placement |
|---|---|---|
| Barker et al. (2016), local representative tiles, doi:10.1016/j.media.2015.12.002 | Precedent for sampling large WSIs; its outcome is slide classification, not tissue-boundary refinement | Related work 2.1 |
| Hetz et al. (2024), multi-domain stain normalization, doi:10.1016/j.media.2024.103149 | Stain-domain variation may affect Bora's colour-based cost | Related work 2.4 |
| Neary-Zajiczek et al. (2023), minimum image resolution, doi:10.1016/j.media.2023.102891 | Motivates measured-spacing analyses; its endpoint is classification | Discussion |
| Möllers et al. (2026), continuous magnification sampling, doi:10.1016/j.media.2026.104327 | Shows magnification-dependent representation performance; Bora's boundary optimum remains untested | Discussion |
| Redekop et al. (2026), SPADE, doi:10.1016/j.media.2026.104197 | Paired histology and spatial-transcriptomics representations motivate a future cross-modal evaluation | Discussion |

Other supplied articles concern classification, retrieval, virtual staining, super-resolution, out-of-distribution detection or representation learning without an equally direct claim about Bora's current boundary-refinement experiment. They were not added as generic citations. The supplied PDFs were used to study presentation; manuscript prose and figures were not copied from them.
