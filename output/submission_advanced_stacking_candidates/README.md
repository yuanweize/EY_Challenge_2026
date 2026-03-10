# Advanced Stacking Candidate Submissions

Primary recommendation: primary_adv_taec_drp_anchor70.csv

These candidates keep the strong TA/EC predictions from submission_advanced_stacking and only regularize uncertain targets, mainly DRP.

- primary_adv_taec_drp_anchor70: Primary guess: preserve advanced TA/EC, regularize DRP toward anchor specialist exact. DRP mean=30.208, DRP std=8.355
- alt_adv_taec_drp_anchor50_strict25: DRP tri-blend with anchor specialist as center and strict hybrid as stabilizer. DRP mean=28.754, DRP std=8.477
- alt_adv90_ec10_drp_anchor70: Slight EC regularization plus anchor-heavy DRP blend. DRP mean=30.208, DRP std=8.355
- alt_adv_taec_drp_median3: Coordinate-wise DRP median of advanced, anchor exact, and strict hybrid. DRP mean=25.798, DRP std=8.721
- alt_adv_ta_anchor_ec_drp_anchor85: More conservative EC and DRP using anchor family outputs. DRP mean=30.474, DRP std=9.809
- alt_adv_taec_drp_target30_anchor70: DRP only from older target-specific plus anchor specialist family. DRP mean=33.746, DRP std=7.092
