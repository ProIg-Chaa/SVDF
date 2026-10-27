# Experiment source snapshots (2026-10-27)

Scientific source files are copied byte-for-byte from the existing experiments; `SOURCE_MANIFEST.json` records their source paths and SHA-256 hashes. The original experiment directories are unchanged.

- `20261009_rethinker_core4_theta05`: frozen SVDF generation/replay and initialization ablations.
- `20261015_openvl_rethinker_lead_vcd_all`: LEAD/VCD benchmark workers and adapters.
- `20261017_pai_three_models_full`: PAI benchmark workers and adapter.
- `20261019_r1_ablation_complete`: R1 component, feedback, direction and capture-position ablations.
- `20261013_bias_amplification78`, `20261013_same_position_fact_push`: attention/soft/difference candidate diagnostics.
- `20261013_vista_natural_trajectory_rank`, `20261014_coco500_natural_rank`: natural-trajectory token ranking.
- `20261014_difference_content_mechanism`, `20261014_pai_attention_error_statistics`: content and attention diagnostics.
- `20261023_mmvp_pair_rescore`, `20261023_official_rescore`: paired-MMVP and evaluation scripts.

These are archival workers, not a turnkey benchmark package. They retain original cluster paths, external dependency references and expected manifest/sample interfaces. Models, dataset samples, generated answers, labels, benchmark manifests, credentials and scheduler state are absent. Configure local inputs, output roots and external repositories before running a worker. Some scripts refer to historical experiment folders; consult their imports and path constants. Copying code does not resume jobs.

Use the root `run_svdf.py` for portable single-image inference. This export checked CPU syntax and provenance, not new GPU equivalence. Inclusion of a diagnostic or controller does not assert that it improved results or belongs to the final method. Failed-attempt and controller-backup directories are excluded. Third-party source attributions and the root LEAD MIT notice are retained.
