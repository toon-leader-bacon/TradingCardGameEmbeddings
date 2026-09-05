# TODO

- Standardize downloader constructor/method naming conventions across
  src/data_retrieval/ sources (rate_limiter keyword-only vs positional,
  phase_1/phase_2 vs descriptive method names) — see
  play_gwent/downloader.py for the first deliberate deviation.
- Make sure all systems are using the download_utils.py centralised
  download logic. Several existing downloaders (e.g. play_gwent,
  spire_codex/run_downloader) still carry their own private retry-loop
  helper, predating download_utils.py's own retry support — migrate them
  onto the shared function rather than leaving the duplicate logic in
  place.
- Consider ways to deduplicate the two-phase (poll-an-id-list, then
  fetch-each-id) download approach shared by play_gwent and sts_gg. May
  be tricky since each data source's list/detail API shape is largely
  unique — possibly not worth a shared abstraction at all. Only revisit
  once there are at least 3 examples of this pattern, and more likely
  once there are 4-5.
