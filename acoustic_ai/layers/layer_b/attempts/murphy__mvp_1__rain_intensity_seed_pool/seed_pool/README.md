Rain intensity seed pool
========================

This MVP constrains rain generation to curated AudioLDM2 seeds from the
`a2_intensity_audit_20260607` listening audit. Runtime `seed` is used only as
deterministic entropy to pick a seed from the requested intensity bucket.

The raw listening WAVs are intentionally not committed here. The committed audit
CSV and summary files document the seed selection provenance.
