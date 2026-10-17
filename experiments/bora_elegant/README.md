# Elegant whole-field Bora experiments

The frozen plan is in PLAN.json. The nine user-selected ROIs are development
cases, not a test set. RGB conversions #9, #10, #15, #19 and equal 9+19 remain
byte-identical. Develop on equal 9+19 only, using arithmetic mean per-image
native macro IoU. Every method operates on original coarse labels and converted
RGB; completed unknown labels do not train class models. No fine-reference mask
is read by prediction code. Background support and scoring domain are unchanged.

The core question is whether soft, global evidence can replace fixed margins,
hard anchors and repeated expensive local optimization. Six different initial
principles are specified before evaluating any of their scores. Follow-up
hypotheses and their outcomes are appended to DEVELOPMENT.json. Final methods
are frozen before full-corpus scoring, including the other four RGB conversions.
