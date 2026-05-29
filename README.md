# musicdb-to-json

**Extract Apple Music library data to JSON** — parse the macOS `Music Library.musicdb` SQLite database and export tracks, playlists, play counts, and metadata as structured JSON.

## What This Gives You

- **Track extraction** — all tracks with metadata (title, artist, album, genre, year)
- **Play history** — play counts, skip counts, last played timestamps
- **Playlists** — playlist hierarchies with track membership
- **JSON output** — clean, structured JSON ready for analysis
- **Offline** — reads the local SQLite file, no Apple API needed

## Quick Start

```bash
python -m musicdb_to_json --output library.json
```

Or specify a custom library path:
```bash
python -m musicdb_to_json --path ~/Music/Music\ Library.musicdb --output library.json
```

## How It Fits

Data extraction tool for the SuperInstance music analysis pipeline. Feeds track data into `counterpoint-engine` for harmonic analysis and `symplectic-music` for phase space mapping.

## License

MIT
