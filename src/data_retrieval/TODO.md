# TODO

- Make sure all systems are using the download_utils.py centralised
  download logic. Some existing downloaders (e.g. play_gwent) still
  carry their own private retry-loop helper, predating
  download_utils.py's own retry support — migrate them onto the shared
  function rather than leaving the duplicate logic in place.
  (spire_codex/run_downloader already migrated, onto the newly-public
  download_utils.call_with_retries — see its _fetch_page.)
- Double check the scripts/run_data_retrieval.py is capturing all the various data_retireval classes. At time of writing (sep 21 2026) we are at least missing the following
  - isotropic
  
