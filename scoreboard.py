#!/usr/bin/env python3
"""Render the Sumabato top scoreboard with Python's standard library and FFmpeg."""

from __future__ import annotations

import argparse
import codecs
import json
import math
import re
import shutil
import subprocess
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path


ROOT = Path(__file__).resolve().parent
BOARD_WIDTH = 1264
BOARD_HEIGHT = 100
FADE = 0.2
ENCODERS = ("av1_nvenc", "hevc_nvenc", "h264_nvenc", "libx264")
VIDEO_BITRATE = "20000k"


def encoder_options(encoder: str) -> list[str]:
    options = [
        "-c:v", encoder,
        "-b:v", VIDEO_BITRATE, "-minrate", VIDEO_BITRATE,
        "-maxrate", VIDEO_BITRATE, "-bufsize", "30000k",
    ]
    if encoder == "libx264":
        options += ["-preset", "medium", "-x264-params", "nal-hrd=cbr:filler=1:vbv-init=1"]
    else:
        options += ["-preset", "p4", "-rc", "cbr"]
    if encoder == "hevc_nvenc":
        options += ["-tag:v", "hvc1"]
    return options


def select_encoder(video: Video, work: Path) -> str:
    """Check actual initialization/encoding/muxing, not just compiled encoders."""
    failures = []
    for encoder in ENCODERS:
        command = [
            "ffmpeg", "-hide_banner", "-loglevel", "error", "-nostdin", "-y",
            "-f", "lavfi", "-i", f"color=black:s=1920x1080:r={video.frame_rate}",
            "-frames:v", "3", "-an", *encoder_options(encoder),
            "-pix_fmt", "yuv420p", *video.color_options,
            str(work / f"probe-{encoder}.mp4"),
        ]
        try:
            result = subprocess.run(command, capture_output=True, text=True, timeout=30)
        except subprocess.TimeoutExpired:
            reason = "初期化・試し書きが30秒以内に完了しませんでした。"
        else:
            if result.returncode == 0:
                return encoder
            reason = next((line for line in result.stderr.splitlines() if line.strip()), f"終了コード {result.returncode}")
        failures.append(f"{encoder}: {reason}")
        print(f"  使用できません: {reason}", flush=True)
    raise ValueError("使用可能な映像エンコーダーがありません。\n" + "\n".join(failures))


def parse_time(value: object) -> float:
    if isinstance(value, bool):
        raise ValueError("時刻は秒数、HH:MM:SS.mmm、または MM:SS.mmm で指定してください。")
    if isinstance(value, (int, float)):
        seconds = float(value)
    elif isinstance(value, str):
        if re.fullmatch(r"\d+(?:\.\d+)?", value):
            seconds = float(value)
        else:
            normalized = f"00:{value}" if value.count(":") == 1 else value
            match = re.fullmatch(r"(\d+):([0-5]\d):([0-5]\d)(?:\.(\d{1,3}))?", normalized)
            if not match:
                raise ValueError(f"不正な時刻: {value!r}（秒数、HH:MM:SS.mmm、または MM:SS.mmm）")
            hours, minutes, secs, fraction = match.groups()
            seconds = int(hours) * 3600 + int(minutes) * 60 + int(secs)
            seconds += int((fraction or "0").ljust(3, "0")) / 1000
    else:
        raise ValueError("時刻は秒数、HH:MM:SS.mmm、または MM:SS.mmm で指定してください。")
    if not math.isfinite(seconds) or seconds < 0:
        raise ValueError("時刻は有限の非負数で指定してください。")
    return seconds


def integer(value: object, label: str, minimum: int = 0) -> int:
    if type(value) is not int or value < minimum:
        raise ValueError(f"{label} は {minimum} 以上の整数で指定してください。")
    return value


def single_line(value: object, label: str) -> str:
    if not isinstance(value, str) or any(c in value for c in "\r\n\x00"):
        raise ValueError(f"{label} は改行を含まない文字列で指定してください。")
    return value


@dataclass(frozen=True)
class Change:
    at: float
    left: int
    right: int


@dataclass(frozen=True)
class Match:
    event_name: str
    best_of: int
    round_name: str
    left_name: str
    right_name: str
    initial_left: int
    initial_right: int
    changes: tuple[Change, ...]

    @classmethod
    def load(cls, path: Path, video_duration: float) -> Match:
        data = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(data, dict):
            raise ValueError("設定のルートはJSONオブジェクトにしてください。")
        required = {"eventName", "bestOf", "round", "leftPlayerName", "rightPlayerName", "initialScore", "scoreChanges"}
        if set(data) != required:
            raise ValueError(f"設定項目を確認してください。必須: {', '.join(sorted(required))}")
        initial = data["initialScore"]
        if not isinstance(initial, dict) or set(initial) != {"left", "right"}:
            raise ValueError("initialScore に left と right を指定してください。")
        left = integer(initial["left"], "initialScore.left")
        right = integer(initial["right"], "initialScore.right")
        rows = data["scoreChanges"]
        if not isinstance(rows, list):
            raise ValueError("scoreChanges は配列で指定してください。")
        changes = []
        previous_at = -1.0
        previous_scores = [left, right]
        last_changes: list[float | None] = [None, None]
        for index, row in enumerate(rows, 1):
            if not isinstance(row, dict) or set(row) != {"at", "left", "right"}:
                raise ValueError(f"scoreChanges の {index} 件目に at・left・right を指定してください。")
            at = parse_time(row["at"])
            if at <= previous_at:
                raise ValueError("scoreChanges の時刻は重複せず、昇順にしてください。")
            if at >= video_duration:
                raise ValueError(f"変更時刻 {at:g} 秒は動画の長さ {video_duration:g} 秒以上です。")
            scores = [integer(row[side], f"scoreChanges[{index}].{side}") for side in ("left", "right")]
            for side, score in enumerate(scores):
                if score != previous_scores[side]:
                    last = last_changes[side]
                    if last is not None and at - last < 2 * FADE - 1e-9:
                        name = ("left", "right")[side]
                        raise ValueError(f"{name} の変更は0.4秒以上あけてください（{at:g} 秒）。")
                    last_changes[side] = at
            changes.append(Change(at, *scores))
            previous_at, previous_scores = at, scores
        return cls(
            single_line(data["eventName"], "eventName"),
            integer(data["bestOf"], "bestOf", 1),
            single_line(data["round"], "round"),
            single_line(data["leftPlayerName"], "leftPlayerName"),
            single_line(data["rightPlayerName"], "rightPlayerName"),
            left, right, tuple(changes),
        )


@dataclass(frozen=True)
class Video:
    duration: float
    frame_rate: str
    color_options: tuple[str, ...] = ()
    matrix: str = "bt709"
    color_range: str = "tv"


def probe(path: Path) -> Video:
    result = subprocess.run(
        ["ffprobe", "-v", "error", "-show_streams", "-show_format", "-of", "json", str(path)],
        check=True, capture_output=True, text=True, encoding="utf-8",
    )
    data = json.loads(result.stdout)
    stream = next((s for s in data["streams"] if s["codec_type"] == "video"), None)
    if stream is None:
        raise ValueError("入力ファイルに映像がありません。")
    if (stream.get("width"), stream.get("height")) != (1920, 1080):
        raise ValueError("このツールの入力映像は1920×1080にしてください。")
    if stream.get("sample_aspect_ratio", "1:1") not in {"1:1", "0:1", "N/A"}:
        raise ValueError("入力映像のピクセル縦横比は1:1にしてください。")
    rate = stream.get("avg_frame_rate", "0/0")
    numerator, denominator = map(int, rate.split("/"))
    if numerator <= 0 or denominator <= 0:
        raise ValueError("入力映像のフレームレートを取得できませんでした。")
    duration = float(stream.get("duration", data["format"].get("duration", "nan")))
    if not math.isfinite(duration) or duration <= 0:
        raise ValueError("入力映像の長さを取得できませんでした。")
    colors = []
    for key, option in (("color_space", "-colorspace"), ("color_transfer", "-color_trc"), ("color_primaries", "-color_primaries"), ("color_range", "-color_range")):
        value = stream.get(key)
        if value and value not in {"unknown", "unspecified", "reserved"}:
            colors.extend((option, value))
    matrices = {"bt709": "bt709", "bt470bg": "bt470bg", "smpte170m": "smpte170m", "smpte240m": "smpte240m", "fcc": "fcc", "bt2020nc": "bt2020", "bt2020c": "bt2020"}
    matrix = matrices.get(stream.get("color_space"), "bt709")
    color_range = "pc" if stream.get("color_range") == "pc" else "tv"
    return Video(duration, rate, tuple(colors), matrix, color_range)


def number(value: float) -> str:
    return f"{value:.9f}".rstrip("0").rstrip(".") or "0"


def filter_graph_file_option() -> str:
    """Return the filtergraph file option supported by the installed FFmpeg."""
    process = subprocess.Popen(
        ["ffmpeg", "-hide_banner", "-h", "full"],
        stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
    )
    help_text, _ = process.communicate()
    if re.search(rb"(?m)^\s*-filter_complex_script(?:\s|$)", help_text):
        return "-filter_complex_script"
    return "-/filter_complex"


def run_ffmpeg(command: list[str]) -> None:
    """Run FFmpeg interactively while relaying its UTF-8 diagnostics as text."""
    process = subprocess.Popen(command, stderr=subprocess.PIPE)
    assert process.stderr is not None
    decoder = codecs.getincrementaldecoder("utf-8")(errors="replace")
    try:
        while chunk := process.stderr.read1(4096):
            sys.stderr.write(decoder.decode(chunk))
            sys.stderr.flush()
        tail = decoder.decode(b"", final=True)
        if tail:
            sys.stderr.write(tail)
            sys.stderr.flush()
        returncode = process.wait()
    except BaseException:
        if process.poll() is None:
            process.terminate()
        process.wait()
        raise
    finally:
        process.stderr.close()
    if returncode != 0:
        raise subprocess.CalledProcessError(returncode, command)


def filter_path(path: Path) -> str:
    # These are FFmpeg filter escapes, independent of shell quoting.
    value = str(path.resolve())
    for character in ("\\", "'", ":", ",", ";", "[", "]", " "):
        value = value.replace(character, "\\" + character)
    # A second escaping layer is consumed by the filtergraph parser.
    return value.replace("\\", "\\\\").replace("'", "\\'")


def ease_out(progress: str) -> str:
    """Framer Motion easeOut: cubic-bezier(0, 0, 0.58, 1).

    Invert the x coordinate with 12 bisections, as in the source animation.
    FFmpeg evaluates this only once per active drawtext filter per frame.
    """
    step = (
        "st(3,(ld(1)+ld(2))/2);"
        "if(gt(1.74*ld(3)*ld(3)-0.74*ld(3)*ld(3)*ld(3),ld(0)),"
        "st(2,ld(3)),st(1,ld(3)));"
    )
    calculation = "st(1,0);st(2,1);" + step * 12 + "3*ld(3)*ld(3)-2*ld(3)*ld(3)*ld(3)"
    return f"st(0,{progress});if(lte(ld(0),0),0,if(gte(ld(0),1),1,{calculation}))"


@dataclass(frozen=True)
class Segment:
    score: int
    enter: float | None
    exit: float | None


def segments(match: Match, side: str) -> list[Segment]:
    score = match.initial_left if side == "left" else match.initial_right
    enter = None
    result = []
    for change in match.changes:
        following = getattr(change, side)
        if following != score:
            result.append(Segment(score, enter, change.at))
            score, enter = following, change.at + FADE
    result.append(Segment(score, enter, None))
    return result


def opacity(segment: Segment, clock: str) -> str:
    if segment.exit is None:
        expression = "1"
    else:
        end = number(segment.exit)
        gone = number(segment.exit + FADE)
        fading = ease_out(f"({clock}-{end})/{FADE}")
        expression = f"if(lt({clock},{end}),1,if(lt({clock},{gone}),1-({fading}),0))"
    if segment.enter is not None:
        start = number(segment.enter)
        arrived = number(segment.enter + FADE)
        fading = ease_out(f"({clock}-{start})/{FADE}")
        expression = f"if(lt({clock},{start}),0,if(lt({clock},{arrived}),{fading},{expression}))"
    return expression


def text_filter(
    work: Path, identifier: str, label: str, font: Path,
    size: int, color: str, x: float, y: float, width: float, height: float,
    alpha: str = "1", enable: str | None = None,
) -> str:
    path = work / f"{identifier}.txt"
    path.write_text(label, encoding="utf-8")
    # Center a consistent font line box rather than the particular glyph's ink.
    result = (
        f"drawtext=fontfile={filter_path(font)}:textfile={filter_path(path)}"
        f":expansion=none:fontsize={size}:fontcolor={color}"
        f":x='{number(x)}+({number(width)}-text_w)/2'"
        f":y='{number(y + height / 2)}-(font_a+font_d)/2'"
        f":y_align=font:alpha='{alpha}'"
    )
    if enable:
        result += f":enable='{enable}'"
    return result


def build_graph(match: Match, work: Path, start: float, duration: float, matrix: str = "bt709", color_range: str = "tv") -> str:
    medium = work / "NotoSansJP-Medium.otf"
    regular = work / "NotoSansJP-Regular.otf"
    heading = f"{match.event_name} / " if match.event_name else ""
    heading += f"Best of {match.best_of}"
    # drawtext on RGBA also changes the destination alpha during a fade,
    # causing overlay to apply the opacity a second time. Paint in RGB and
    # restore the original PNG's shape before compositing onto the video.
    filters = ["format=rgb24"]
    static = [
        ("event", heading, 20, 0.385 * BOARD_WIDTH, 0.02 * BOARD_HEIGHT, 290, 36),
        ("round", match.round_name.upper().replace("QUARTER FINALS", "QUARTERS"), 25, 0.386 * BOARD_WIDTH, 0.36 * BOARD_HEIGHT, 288, 47),
        ("left_name", match.left_name, 25, 0.013 * BOARD_WIDTH, 0, 386, 55),
        ("right_name", match.right_name, 25, BOARD_WIDTH * (1 - 0.014) - 386, 0, 386, 55),
    ]
    for identifier, label, size, x, y, width, height in static:
        if label:
            filters.append(text_filter(work, identifier, label, medium, size, "white", x, y, width, height))
    clock = f"(t+{number(start)})"
    for side in ("left", "right"):
        x = 0.326 * BOARD_WIDTH if side == "left" else BOARD_WIDTH * (1 - 0.326) - 73
        font = medium if side == "left" else regular
        for index, segment in enumerate(segments(match, side)):
            visible_from = segment.enter if segment.enter is not None else 0
            visible_until = segment.exit + FADE if segment.exit is not None else math.inf
            if visible_until <= start or visible_from >= start + duration:
                continue
            enable = f"gte({clock},{number(visible_from)})"
            if segment.exit is not None:
                enable += f"*lt({clock},{number(visible_until)})"
            filters.append(text_filter(
                work, f"{side}_{index}", str(segment.score), font,
                65, "0xFFD87F", x, -0.03 * BOARD_HEIGHT, 73, 92,
                opacity(segment, clock), enable,
            ))
    return (
        "[1:v]format=rgba,split[template][transparency];\n"
        "[transparency]alphaextract[mask];\n"
        "[template]" + ",\n".join(filters) + "[paint];\n"
        f"[paint][mask]alphamerge,scale=iw:ih:out_color_matrix={matrix}:out_range={color_range},format=yuva420p[board];\n"
        "[0:v][board]overlay=x=(main_w-overlay_w)/2:y=0:format=yuv420:shortest=1[v]\n"
    )


def render(
    source: Path, config: Path, output: Path,
    start: float = 0, duration: float | None = None,
) -> None:
    for binary in ("ffmpeg", "ffprobe"):
        if shutil.which(binary) is None:
            raise ValueError(f"{binary} が見つかりません。FFmpegをインストールしてください。")
    source, config, output = source.resolve(), config.resolve(), output.resolve()
    for path in (source, config, ROOT / "assets/sumabato_blue_score_board_top.png", ROOT / "assets/fonts/NotoSansJP-Regular.otf", ROOT / "assets/fonts/NotoSansJP-Medium.otf"):
        if not path.is_file() or path.stat().st_size == 0:
            raise ValueError(f"ファイルが見つからないか、空です: {path}")
    if output in (source, config):
        raise ValueError("出力先は入力動画・設定ファイルとは別にしてください。")
    if output.suffix.lower() != ".mp4":
        raise ValueError("出力ファイルの拡張子は .mp4 にしてください。")
    if not output.parent.is_dir():
        raise ValueError(f"出力先フォルダがありません: {output.parent}")
    video = probe(source)
    match = Match.load(config, video.duration)
    if not math.isfinite(start) or start < 0 or start >= video.duration:
        raise ValueError("--start は動画の長さ未満の非負の時刻にしてください。")
    if duration is not None and (not math.isfinite(duration) or duration <= 0):
        raise ValueError("--duration は0より大きい秒数にしてください。")
    length = min(duration if duration is not None else video.duration - start, video.duration - start)
    # Put scratch files under a simple system-generated path, never interpolate
    # user text into a shell or FFmpeg expression. Text expansion is disabled.
    with tempfile.TemporaryDirectory(prefix="sumabato-") as temporary:
        work = Path(temporary)
        encoder = select_encoder(video, work)
        # Copy into the scratch directory so all filter filenames are safe even
        # when the project directory contains quotes, spaces, colons or commas.
        for name in ("NotoSansJP-Regular.otf", "NotoSansJP-Medium.otf"):
            shutil.copyfile(ROOT / "assets/fonts" / name, work / name)
        graph = build_graph(match, work, start, length, video.matrix, video.color_range)
        graph_path = work / "filters.txt"
        graph_path.write_text(graph, encoding="utf-8")
        command = [
            "ffmpeg", "-hide_banner",
            "-ss", number(start), "-noautorotate", "-i", str(source),
            "-loop", "1", "-framerate", video.frame_rate, "-i", str(ROOT / "assets/sumabato_blue_score_board_top.png"),
            filter_graph_file_option(), str(graph_path),
            "-map", "[v]", "-map", "0:a?", "-map_metadata", "0",
            "-t", number(length), "-fps_mode", "passthrough",
            *encoder_options(encoder),
            "-pix_fmt", "yuv420p", *video.color_options,
            "-c:a", "copy", "-movflags", "+faststart", str(output),
        ]
        print(f"出力: {output}\n映像: {encoder} / {VIDEO_BITRATE.removesuffix('k')}kbps CBR\n対象区間: {start:g} ～ {start + length:g} 秒", flush=True)
        run_ffmpeg(command)


def time_argument(value: str) -> float:
    try:
        return parse_time(value)
    except ValueError as error:
        raise argparse.ArgumentTypeError(str(error)) from error


def main() -> int:
    parser = argparse.ArgumentParser(description="スコアボードをFFmpegで動画に重ねます。")
    parser.add_argument("--input", required=True, type=Path, help="入力動画（1920x1080）")
    parser.add_argument("--config", required=True, type=Path, help="固定テキストとスコア変更のJSON")
    parser.add_argument("--output", required=True, type=Path, help="出力MP4")
    parser.add_argument("--start", default=0, type=time_argument, help="切り出し開始時刻。省略時は動画先頭")
    parser.add_argument("--duration", type=time_argument, help="切り出す長さ。省略時は動画末尾まで")
    args = parser.parse_args()
    try:
        render(args.input, args.config, args.output, args.start, args.duration)
    except (ValueError, OSError, subprocess.CalledProcessError) as error:
        if isinstance(error, subprocess.CalledProcessError):
            detail = error.stderr or f"FFmpeg/ffprobeが終了コード {error.returncode} で失敗しました。"
        else:
            detail = str(error)
        print(f"エラー: {detail}", file=sys.stderr)
        return 1
    except KeyboardInterrupt:
        return 130
    return 0


if __name__ == "__main__":
    sys.exit(main())
