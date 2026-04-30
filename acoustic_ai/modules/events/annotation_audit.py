"""Module C — annotation CSV audit and preprocessing.

Validates downloaded A2O annotation CSVs, joins event rows to clip paths
and env features, audits label quality and coverage, and writes:
  - acoustic_ai/data/module_c/annotation_event_index.csv
  - acoustic_ai/data/module_c/annotation_label_report.md
  - acoustic_ai/data/module_c/activity_labels.csv

Run this before any event/species model work.
See .claude/context/ai_mvp_decision_log_and_new_architecture.md — Layer C pre-training.
"""
# TODO: implement annotation audit pipeline
