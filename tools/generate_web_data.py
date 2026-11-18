import json
from pathlib import Path
from tempfile import TemporaryDirectory
from urllib.parse import quote

import click
import pyrootutils

pyrootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)


def generate_info_json(resources_dir: Path, output_json: Path):
    """Index chapters that have a corresponding non-empty audio file."""
    tracks = []
    unavailable = []
    for chapter in sorted((resources_dir / "books").glob("*.txt")):
        audio = resources_dir / "audios" / f"{chapter.stem}.mp3"
        if audio.is_file() and audio.stat().st_size:
            tracks.append({
                "title": chapter.stem,
                "src": f"resources/{quote(resources_dir.name, safe='')}/audios/{quote(audio.name, safe='')}",
            })
        else:
            unavailable.append(chapter.stem)

    with TemporaryDirectory(dir=output_json.parent) as staging:
        manifest = Path(staging) / output_json.name
        manifest.write_text(json.dumps({"audios": tracks, "unavailable": unavailable}, ensure_ascii=False, indent=4), encoding="utf-8")
        manifest.replace(output_json)


@click.command()
@click.option("--device", default=None, help="Inference device (defaults to CUDA if available, otherwise CPU)")
@click.option("--resources_dir", type=click.Path(exists=True, file_okay=False, path_type=Path),
              default="web/resources/leijun", show_default=True, help="Book directory under web/resources")
def main(resources_dir: Path, device: str | None):
    reference_audio = resources_dir / "reference.wav"
    reference_text_path = resources_dir / "reference.txt"
    chapters = sorted((resources_dir / "books").glob("*.txt"))
    if not reference_audio.is_file() or not reference_text_path.is_file():
        raise click.ClickException(f"Missing reference.wav or reference.txt in {resources_dir}")
    if not chapters:
        raise click.ClickException(f"No chapter text files in {resources_dir / 'books'}")

    reference_text = reference_text_path.read_text(encoding="utf-8")
    from tools.speech_inference import SpeechInference, bytes_to_audio

    model = SpeechInference(device=device)
    audio_dir = resources_dir / "audios"
    audio_dir.mkdir(exist_ok=True)
    failures = []
    with TemporaryDirectory(dir=audio_dir) as staging:
        for chapter in chapters:
            try:
                sample_rate, audio = model.generate(
                    text=chapter.read_text(encoding="utf-8"),
                    reference_audio=str(reference_audio), reference_text=reference_text,
                )
                output = Path(staging) / f"{chapter.stem}.mp3"
                bytes_to_audio(audio, sample_rate, str(output))
                if output.stat().st_size == 0:
                    raise ValueError("Encoder produced an empty MP3")
                output.replace(audio_dir / output.name)
                click.echo(f"Generated {output.name}")
            except Exception as exc:
                failures.append(chapter.name)
                click.echo(f"Failed to generate {chapter.name}: {exc}", err=True)

    generate_info_json(resources_dir, resources_dir / "info.json")
    if failures:
        raise click.ClickException(f"{len(failures)} of {len(chapters)} chapters failed")


if __name__ == "__main__":
    main()
