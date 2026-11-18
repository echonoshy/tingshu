"""Install the pinned GitHub release archive after checking its bytes."""

import argparse
import hashlib
from pathlib import Path, PurePosixPath
import tarfile
import tempfile
from urllib.request import urlopen

RELEASE_URL = "https://github.com/echonoshy/tingshu/releases/download/v1.0.0/weights.tar.gz"
RELEASE_SIZE = 917_777_070
RELEASE_SHA256 = "be450bb2e396dcbfa35ea72568ed10dc2ff38fd3a1ee162a24f9879732c3633a"
REQUIRED_FILES = {
    "config.json",
    "tokenizer.json",
    "tokenizer_config.json",
    "special_tokens_map.json",
    "model.pth",
    "firefly-gan-vq-fsq-8x1024-21hz-generator.pth",
}


def install_archive(archive: Path, destination: Path, *, expected_sha256: str, expected_size: int):
    if destination.exists() or destination.is_symlink():
        raise FileExistsError(f"已有权重目录，安装器不会覆盖：{destination}")
    if archive.stat().st_size != expected_size:
        raise ValueError("权重压缩包大小与发布记录不符。")
    with archive.open("rb") as source:
        if hashlib.file_digest(source, "sha256").hexdigest() != expected_sha256:
            raise ValueError("权重压缩包的 SHA-256 校验失败。")

    with tempfile.TemporaryDirectory(prefix=".tingshu-weights-", dir=destination.parent) as temp:
        stage = Path(temp)
        with tarfile.open(archive, "r:gz") as bundle:
            members = bundle.getmembers()
            names = set()
            for member in members:
                path = PurePosixPath(member.name)
                if (not path.parts or path.parts[0] != "weights" or ".." in path.parts
                        or not (member.isdir() or member.isfile()) or path in names):
                    raise ValueError(f"权重压缩包包含不支持的条目：{member.name}")
                names.add(path)
            bundle.extractall(stage, members=members, filter="data")
        extracted = stage / "weights"
        if not all((extracted / name).is_file() and (extracted / name).stat().st_size for name in REQUIRED_FILES):
            raise ValueError("权重压缩包缺少模型或分词器文件。")
        extracted.rename(destination)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive", type=Path, help="使用已有的 v1.0.0 权重压缩包")
    args = parser.parse_args()
    destination = Path.cwd() / "weights"
    try:
        if destination.exists() or destination.is_symlink():
            raise FileExistsError(f"已有权重目录，安装器不会覆盖：{destination}")
        if args.archive:
            install_archive(args.archive, destination, expected_sha256=RELEASE_SHA256, expected_size=RELEASE_SIZE)
        else:
            with tempfile.TemporaryDirectory(prefix="tingshu-download-") as temp:
                archive = Path(temp) / "weights.tar.gz"
                print(f"正在下载权重（{RELEASE_SIZE / 1_000_000:.0f} MB）…", flush=True)
                with urlopen(RELEASE_URL, timeout=60) as response, archive.open("wb") as output:
                    remaining = RELEASE_SIZE + 1
                    while remaining:
                        chunk = response.read(min(1024 * 1024, remaining))
                        if not chunk:
                            break
                        output.write(chunk)
                        remaining -= len(chunk)
                install_archive(archive, destination, expected_sha256=RELEASE_SHA256, expected_size=RELEASE_SIZE)
    except (OSError, ValueError, tarfile.TarError) as error:
        parser.exit(1, f"权重安装失败：{error}\n")
    print(f"已校验并安装权重：{destination}")


if __name__ == "__main__":
    main()
