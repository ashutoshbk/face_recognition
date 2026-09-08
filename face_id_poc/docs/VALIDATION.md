# Validation Status

The reference is the five-minute Reception demo with the v0.1.1 chair fix, using
20 supplied enrollment photos. Its three proposed enrolled labels are model
predictions, not client-confirmed identity ground truth.

The prior full run processed 3,000 frames with 1080p output at 10 FPS, around
20 analysis FPS on TITAN X Pascal. Rendering/encoding was slightly slower than
real time. This is not an RTX 5000 Ada acceptance test.

107 model-free tests passed for the chair-fix archive. Prior chair checks covered
476 old chair-box observations across 13 fragmented tracks; none of those reviewed
chair detections remained. One overlapping observation was a real foreground head.
Some weak head-only detections were also filtered; this is not a zero-regression
or universal false-positive claim.

Clean-checkout installation, launcher, GPU reproduction and CPU smoke results will
be recorded here before the PR is opened.
