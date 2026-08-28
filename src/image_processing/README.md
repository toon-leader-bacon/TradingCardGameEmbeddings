# image_processing

Extracts structured text from card images for sources that only
provide a photo/scan, not structured data — a fan-made-card scrape
being the motivating case. This is an optional, lower-priority effort:
nothing else in the system depends on it existing, and any source with
an already-structured dump (Scryfall, etc.) skips it entirely. Its
output feeds into `data_refinement` as just one more raw source, with
no special status.

No implementation exists yet. The intended shape: a cheap deterministic
prefilter to reject unusable images before spending a model call, a
VLM-based structured extraction step, a self-consistency check (extract
twice, diff the results, flag low-agreement fields), and a resulting
per-card confidence score meant to be used as a training weight rather
than a flat trust discount by source tier. That confidence score is what
lets low-trust fan-made extractions still contribute to training without
being treated as equally reliable as an official structured dump.

This file grows once this container's first real feature goes through
`design-recipe-skeleton`, informed by (but not bound to) the shape above.
