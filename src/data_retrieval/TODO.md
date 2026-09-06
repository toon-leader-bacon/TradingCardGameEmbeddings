# TODO

- Make sure all systems are using the download_utils.py centralised
  download logic. Several existing downloaders (e.g. play_gwent,
  spire_codex/run_downloader) still carry their own private retry-loop
  helper, predating download_utils.py's own retry support — migrate them
  onto the shared function rather than leaving the duplicate logic in
  place.
