# Latent representation faithfulness: POD vs. convolutional autoencoder (Darcy flow)

Analysis code and result logs for "How to Choose a Surrogate? A Falsifiable Test of Representation Faithfulness" (JCP submission).

## Contents

- `analysis/make_full_tables.py`: rebuilds the per-seed tables for Experiments A and B from the job logs, and Experiment C (dose response) from JSON results.
- `analysis/aggregate_seeds.py`: per-(experiment, n) summary across seeds, with sign-consistency count and a LaTeX table. Produces `results/table1_seeds.tex` (Table 1).
- `analysis/plot_dose.py`: dose-response summary (`dose_summary.csv`), gap and η figures, crossover estimate.
- `analysis/pod_vs_ae_eta.py`: POD and convolutional AE reconstruction error on held-out wells.
- `logs/1136521_{0..9}_podae_seeds.out`: job logs for the 500-epoch, five-seed runs. Tasks 0–4 are Experiment A (fixed well), tasks 5–9 are Experiment B (variable well).
- `logs/seeds/results_{A,B}_seed{s}.json`: per-seed results rebuilt from the logs (see Provenance).
- `logs/dose/results_r{R}_seed{s}.json`: Experiment C, five translation ranges R (250, 500, 1000, 1500, 3000 m), seeds 0–2.

## Definitions

- `eta_POD`, `eta_AE`: relative reconstruction error on the held-out wells, ‖X − X̂‖_F / ‖X‖_F over the whole test matrix (one global value per model).
- Relative gap: `gap = 100 × (eta_POD − eta_AE) / eta_POD` (%). Positive values favor the AE; negative values favor POD.
- Well-level split: 850 training wells and 150 held-out wells (`held_out_frac = 0.15`), seed 0.
- POD subtracts the training mean. The AE standardizes each cell by its training mean and standard deviation.
- Reported standard deviations are sample standard deviations (ddof = 1) across the five seeds.
- All paper results use the 500-epoch, five-seed runs.

## Autoencoder training and limitations

Each autoencoder was trained for at most 500 epochs with early stopping (patience 20, minimum 40 epochs), and the checkpoint with the lowest validation loss was evaluated. Across the 40 autoencoder fits (five seeds, four latent sizes, two experiments), 18 reached the 500-epoch cap before early stopping triggered. In eight of these the best validation epoch was the final one. These fits had not converged, so their reported errors are likely larger than what longer training would give. Because the gap is measured against POD, undertraining biases the gap toward POD. Interpret the sign of the gap at n = 16 and n = 32 with this caveat.

Validation for early stopping uses a random subset of training snapshots, not a subset of wells. Test wells are never used for model selection.

The per-epoch flag in `pod_vs_ae_eta.py` does not check whether the epoch cap was reached, so it reports OK for these fits. It should not be read as evidence of convergence.

## Provenance

The 500-epoch JSON files in `logs/seeds/` were rebuilt from the job logs by `make_full_tables.py` (each contains `"rebuilt_from_log"`). η values in the logs are printed to five decimal places, so the gaps carry a rounding uncertainty of about ±0.3 percentage points, and standard deviations of η below 1e-5 are not resolved. The per-epoch diagnostics in these JSON files do not match the logs and should not be used; use the log lines instead.

## Reproducing the tables

The scripts currently contain cluster paths (`<DATA_ROOT>`, `<DATA_ROOT>`). Set them before running.

```bash
# Experiments A and B (500 epochs) and Experiment C from the logs
python3 analysis/make_full_tables.py          # set W in the script first

# Table 1
python3 analysis/aggregate_seeds.py --dir logs/seeds --out table1_seeds.tex

# Dose-response summary and figures
python3 analysis/plot_dose.py --seeds logs/seeds --dose logs/dose --ell 1200 --out figs
```

`--ell` is the drawdown radius in metres (1200 m).

## TODO before release

- [ ] Replace cluster paths with configurable paths.
- [ ] Add the solver, case template, and data-generation code, or state where they are archived.
- [ ] Add a data-availability statement for the raw simulation data (`ldo_data_*`) and trained models.
- [ ] Add the environment (`requirements.txt`) and Slurm settings.
- [ ] Remove author names and personal paths for double-blind review.

## License and citation

TODO.
