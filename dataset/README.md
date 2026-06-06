SonicMorph Dataset — v1

This directory stores dataset artifacts produced by the SonicMorph pipeline.

Structure overview:

- `raw/{artist_name}/` - downloaded original audio files
- `stems/{artist_name}/{song_id}/` - Demucs-separated stems
- `processed/` - processed clips for style/mel/voice models
- `metadata/` - CSV exports and metadata
- `manifests/` - JSON manifests for dataset releases
- `cache/` - fingerprint and temporary cache
- `reports/` - per-artist quality reports
- `logs/` - structured pipeline logs
- `sonicmorph.db` - SQLite database (pipeline state)

See the project README and `config/pipeline.yaml` for configuration details and versioning.
