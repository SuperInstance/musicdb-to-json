#!/usr/bin/env python3
"""
musicdb_to_json_spectral.py — musicdb-to-json with optional spectral analysis
===============================================================================

Drop-in replacement for musicdb_to_json.py that adds --spectral-analysis flag.
When enabled, wraps parsing with spectral hooks and enriches JSON output.

Backward-compatible: without --spectral-analysis, behavior is identical to
the original musicdb_to_json.py.

Usage:
    python musicdb_to_json_spectral.py musicdb --decryption-key KEY [--spectral-analysis] [-o output.json]
"""

import argparse
import io
import json
import sys

from musicdb import (
    get_library_bytes, merge_in, parse_boma, parse_container,
    parse_hfma, parse_master, read_next_chunk
)
from spectral_enhancer import SpectralEnhancer


def parse_args():
    parser = argparse.ArgumentParser(
        description='Convert an Apple Music musicdb file to JSON with optional spectral analysis. '
                    'This project assumes a Little-Endian library storage format!',
        epilog="The decryption key is required but not included."
    )

    parser.add_argument('musicdb_file', help="The musicdb library file to convert.")
    parser.add_argument('--output-file', '-o',
                        help="The file to write the JSON output to, or 'library.json' if not provided.",
                        default="library.json")
    parser.add_argument('--raw-bytes-file',
                        help="If provided, write the raw decrypted and decompressed bytes to this file.")
    parser.add_argument('--decryption-key',
                        help='The iTunes/Apple Music library AES key, as text ("BHU.............").',
                        required=True)
    parser.add_argument('--spectral-analysis', action='store_true',
                        help='Enable spectral analysis of the binary format. Adds spectral metadata to JSON output.')
    parser.add_argument('--spectral-report', action='store_true',
                        help='Print spectral analysis report to stderr in addition to JSON output.')
    return parser.parse_args()


def parse_library(args):
    """Parse the library with optional spectral analysis."""
    use_spectral = args.spectral_analysis
    enhancer = SpectralEnhancer() if use_spectral else None

    library = {}
    library_bytes = get_library_bytes(args.musicdb_file, args.decryption_key)
    library_bytestream = io.BytesIO(library_bytes)

    while library_bytestream.readable():
        r = read_next_chunk(library_bytestream)
        if r is None:
            break
        chunk_type, chunk_bytes = r
        ct_str = chunk_type.decode('ascii', errors='replace')

        # Spectral pre-parse hook
        if enhancer:
            enhancer.spectral_pre_parse(ct_str, chunk_bytes)

        if chunk_type == b"hsma":
            # Separator chunk
            if enhancer:
                enhancer.spectral_post_parse(ct_str, chunk_bytes, {})
            continue

        elif chunk_type == b"hfma":
            hfma_data, library_file_data = parse_hfma(chunk_bytes)
            merge_in(library, library_file_data)
            if enhancer:
                enhancer.spectral_post_parse(ct_str, chunk_bytes, library_file_data)

        elif chunk_type == b"plma":
            plma_data, library_data = parse_master(chunk_bytes)
            if enhancer:
                enhancer.spectral_set_entity('library')

            subtypes_parsed = set()
            for i in range(plma_data["boma_sections"]):
                _, cb = read_next_chunk(library_bytestream)
                md, ld = parse_boma(cb)

                if enhancer:
                    enhancer.spectral_pre_parse('boma', cb)

                if md["boma_subtype"] in subtypes_parsed:
                    print(f"Duplicate subtype ({md['boma_subtype']}) for plma section!")
                subtypes_parsed.add(md["boma_subtype"])

                if ld is not None:
                    merge_in(library_data, ld)
                else:
                    print(f"Unknown library subtype ({md['boma_subtype']}) for plma section!")
                    if enhancer:
                        enhancer.record_unknown_chunk(md["boma_subtype"], cb)

                if enhancer:
                    enhancer.spectral_post_parse('boma', cb, ld)

            merge_in(library, library_data)
            if enhancer:
                enhancer.spectral_end_entity()

        elif chunk_type == b"lama":
            lama_data, albums_data = parse_master(chunk_bytes)
            merge_in(library.setdefault("album_data", {}), albums_data)

            for _ in range(lama_data["iama_sections"]):
                _, ccb = read_next_chunk(library_bytestream)
                iama_data, album_data = parse_container(ccb, b"iama")
                album_id = album_data.get("album_id", "unknown_album")

                if enhancer:
                    enhancer.spectral_pre_parse('iama', ccb)
                    enhancer.spectral_set_entity(f'album_{album_id}')

                subtypes_parsed = set()
                for i in range(iama_data["boma_sections"]):
                    _, cb = read_next_chunk(library_bytestream)
                    md, ad = parse_boma(cb)

                    if enhancer:
                        enhancer.spectral_pre_parse('boma', cb)

                    if md["boma_subtype"] in subtypes_parsed:
                        print(f"Duplicate subtype ({md['boma_subtype']}) for {album_data.get('album_id')}!")
                    subtypes_parsed.add(md["boma_subtype"])

                    if ad is not None:
                        merge_in(album_data, ad)
                    else:
                        print(f"Unknown album subtype ({md['boma_subtype']}) for {album_data.get('album_id')}!")
                        if enhancer:
                            enhancer.record_unknown_chunk(md["boma_subtype"], cb)

                    if enhancer:
                        enhancer.spectral_post_parse('boma', cb, ad)

                library.setdefault("album_data", {}).setdefault("albums", []).append(album_data)
                if enhancer:
                    enhancer.spectral_end_entity()

        elif chunk_type == b"lAma":
            lAma_data, artists_data = parse_master(chunk_bytes)
            merge_in(library.setdefault("artist_data", {}), artists_data)

            for _ in range(lAma_data["iAma_sections"]):
                _, ccb = read_next_chunk(library_bytestream)
                iAma_data, artist_data = parse_container(ccb, b"iAma")
                artist_id = artist_data.get("artist_id", "unknown_artist")

                if enhancer:
                    enhancer.spectral_pre_parse('iAma', ccb)
                    enhancer.spectral_set_entity(f'artist_{artist_id}')

                subtypes_parsed = set()
                for _ in range(iAma_data["boma_sections"]):
                    _, cb = read_next_chunk(library_bytestream)
                    md, artd = parse_boma(cb)

                    if enhancer:
                        enhancer.spectral_pre_parse('boma', cb)

                    if md["boma_subtype"] in subtypes_parsed:
                        print(f"Duplicate subtype ({md['boma_subtype']}) for {artist_data.get('artist_id')}!")
                    subtypes_parsed.add(md["boma_subtype"])

                    if artd is not None:
                        merge_in(artist_data, artd)
                    else:
                        print(f"Unknown artist subtype ({md['boma_subtype']}) for {artist_data.get('artist_id')}!")
                        if enhancer:
                            enhancer.record_unknown_chunk(md["boma_subtype"], cb)

                    if enhancer:
                        enhancer.spectral_post_parse('boma', cb, artd)

                library.setdefault("artist_data", {}).setdefault("artists", []).append(artist_data)
                if enhancer:
                    enhancer.spectral_end_entity()

        elif chunk_type == b"ltma":
            ltma_data, tracks_data = parse_master(chunk_bytes)
            merge_in(library.setdefault("track_data", {}), tracks_data)

            for _ in range(ltma_data["itma_sections"]):
                _, ccb = read_next_chunk(library_bytestream)
                itma_data, track_data = parse_container(ccb, b"itma")
                track_id = track_data.get("track_persistent_id", "unknown_track")

                if enhancer:
                    enhancer.spectral_pre_parse('itma', ccb)
                    enhancer.spectral_set_entity(f'track_{track_id}')

                subtypes_parsed = set()
                for _ in range(itma_data["boma_sections"]):
                    _, cb = read_next_chunk(library_bytestream)
                    md, td = parse_boma(cb)

                    if enhancer:
                        enhancer.spectral_pre_parse('boma', cb)

                    if md["boma_subtype"] in subtypes_parsed:
                        continue
                    subtypes_parsed.add(md["boma_subtype"])

                    if td is not None:
                        merge_in(track_data, td)
                    else:
                        print(f"Unknown track subtype ({md['boma_subtype']}) for {track_data.get('track_persistent_id')}!")
                        if enhancer:
                            enhancer.record_unknown_chunk(md["boma_subtype"], cb)

                    if enhancer:
                        enhancer.spectral_post_parse('boma', cb, td)

                library.setdefault("track_data", {}).setdefault("tracks", []).append(track_data)
                if enhancer:
                    enhancer.spectral_end_entity()

        elif chunk_type == b"lPma":
            lPma_data, playlists_data = parse_master(chunk_bytes)
            merge_in(library.setdefault("playlist_data", {}), playlists_data)

            for _ in range(lPma_data["lpma_sections"]):
                _, ccb = read_next_chunk(library_bytestream)
                lpma_data, playlist_data = parse_container(ccb, b"lpma")
                playlist_id = playlist_data.get("playlist_id", "unknown_playlist")

                if enhancer:
                    enhancer.spectral_pre_parse('lpma', ccb)
                    enhancer.spectral_set_entity(f'playlist_{playlist_id}')

                for _ in range(lpma_data["boma_sections"]):
                    _, cb = read_next_chunk(library_bytestream)
                    md, pd = parse_boma(cb)

                    if enhancer:
                        enhancer.spectral_pre_parse('boma', cb)

                    if pd is not None:
                        merge_in(playlist_data, pd)
                    else:
                        print(f"Unknown playlist subtype ({md['boma_subtype']}) for {playlist_data.get('playlist_id')}!")
                        if enhancer:
                            enhancer.record_unknown_chunk(md["boma_subtype"], cb)

                    if enhancer:
                        enhancer.spectral_post_parse('boma', cb, pd)

                library.setdefault("playlist_data", {}).setdefault("playlists", []).append(playlist_data)
                if enhancer:
                    enhancer.spectral_end_entity()

        else:
            print(f"Skipping unexpected chunk: {chunk_type}!")
            if enhancer:
                enhancer.spectral_post_parse(ct_str, chunk_bytes, None)

    # Build spectral metadata
    if enhancer:
        spectral_meta = enhancer.generate_output_metadata()
        library["_spectral_analysis"] = spectral_meta["spectral_analysis"]

        if args.spectral_report:
            spec = spectral_meta["spectral_analysis"]
            print(f"\n{'='*60}", file=sys.stderr)
            print(f"SPECTRAL ANALYSIS REPORT", file=sys.stderr)
            print(f"{'='*60}", file=sys.stderr)
            print(f"  Chunks parsed:       {spec['total_chunks_parsed']}", file=sys.stderr)
            print(f"  Conservation score:  {spec['conservation_score']:.4f}", file=sys.stderr)
            print(f"  Format version:      {spec['format_version']}", file=sys.stderr)
            print(f"  Format similarity:   {spec['format_similarity']:.4f}", file=sys.stderr)
            print(f"  Corruption detected: {spec['corruption_detected']}", file=sys.stderr)
            print(f"  Anomalies:           {spec['anomaly_count']}", file=sys.stderr)
            print(f"  Entities analyzed:   {spec['entity_count']}", file=sys.stderr)
            print(f"  Unknown subtypes:    {spec['unknown_chunks']['count']}", file=sys.stderr)

            if spec['anomaly_list']:
                print(f"\n  Top anomalies:", file=sys.stderr)
                for a in spec['anomaly_list'][:10]:
                    print(f"    [{a['severity']:.2f}] {a['message']}", file=sys.stderr)
                    print(f"           → {a['recovery_suggestion']}", file=sys.stderr)

            if spec['low_conservation_entities']:
                print(f"\n  Low conservation entities:", file=sys.stderr)
                for e in spec['low_conservation_entities'][:10]:
                    print(f"    {e['entity_id']}: {e['score']:.4f}", file=sys.stderr)

            if spec['unknown_chunks']['reports']:
                print(f"\n  Unknown chunk exploration:", file=sys.stderr)
                for r in spec['unknown_chunks']['reports'][:10]:
                    print(f"    {r['subtype']}: nearest={r['nearest_category']}, "
                          f"suggested={r['suggested_field_name']}", file=sys.stderr)

            print(f"{'='*60}\n", file=sys.stderr)

    return library


def main():
    args = parse_args()

    library = parse_library(args)

    # Write raw bytes
    if args.raw_bytes_file:
        library_bytes = get_library_bytes(args.musicdb_file, args.decryption_key)
        with open(args.raw_bytes_file, "wb") as f:
            f.write(library_bytes)

    # Write JSON
    with open(args.output_file, "w", encoding="utf8") as f:
        json.dump(library, f, indent=2)

    print(f"Library written to {args.output_file}")


if __name__ == '__main__':
    main()
