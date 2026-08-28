# TODO

- Implement better training loops that consider the test loss to determine when to stop.
- Use the test loss to adjust the loss function temperature in an intelligent way.
- Refactor CardIngestionStage.ingest to remove source_game as an argument. Concrete implementations should know which game they ingest already (typically a one-to-one mapping with a downloader), so a call like `MtgCardIngestionStage.ingest(..., GameId.POKEMON)` should be an illegal statement, not just an unenforced convention.
