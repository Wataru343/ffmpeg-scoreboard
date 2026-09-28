"""Validate input errors and actual encoded scoreboard frames, not graph strings."""

import contextlib
import io
import json
import shutil
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch
from pathlib import Path

import scoreboard


def sample_config():
    return {
        "eventName": "スマバト",
        "bestOf": 5,
        "round": "Losers Quarter Finals",
        "leftPlayerName": "日本語 'A' 100%",
        "rightPlayerName": '選手 "B" %{n}',
        "initialScore": {"left": 0, "right": 0},
        "scoreChanges": [
            {"at": "00:01.000", "left": 1, "right": 0},
            {"at": "00:02", "left": 1, "right": 1},
            {"at": 2.8, "left": 1, "right": 1},
        ],
    }


class ConfigTests(unittest.TestCase):
    def load(self, data):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "match.json"
            path.write_text(json.dumps(data, ensure_ascii=False), encoding="utf-8")
            return scoreboard.Match.load(path, 10)

    def test_time_formats(self):
        for value, expected in (("00:02:13.500", 133.5), ("02:13.500", 133.5), ("03:31.224", 211.224), ("02:03.4", 123.4), ("59:59.999", 3599.999), ("00:00", 0), ("01:02:03.4", 3723.4), ("00:00:00", 0), (133.5, 133.5), ("133.5", 133.5)):
            with self.subTest(value=value):
                self.assertEqual(scoreboard.parse_time(value), expected)
        for value in (True, -1, float("nan"), float("inf"), "00:60:00", "00:00:60", "1:2:3", "00:00:00.1234", "60:00", "00:60", "02:13.1234", "-02:13.500"):
            with self.subTest(value=value), self.assertRaises(ValueError):
                scoreboard.parse_time(value)

    def test_invalid_order_and_video_end(self):
        for times in ((2, 1), (1, 1), (1, 10)):
            data = sample_config()
            data["scoreChanges"] = [{"at": at, "left": index, "right": 0} for index, at in enumerate(times)]
            with self.subTest(times=times), self.assertRaises(ValueError):
                self.load(data)


    def test_overlapping_changes_only_rejected_on_changed_side(self):
        data = sample_config()
        data["scoreChanges"] = [
            {"at": 1, "left": 1, "right": 0},
            {"at": 1.1, "left": 1, "right": 1},
            {"at": 1.2, "left": 1, "right": 1},
            {"at": 1.4, "left": 0, "right": 1},
        ]
        self.load(data)
        data["scoreChanges"][-1]["at"] = 1.39
        with self.assertRaisesRegex(ValueError, "0.4秒"):
            self.load(data)

    def test_score_and_text_validation(self):
        for score in (-1, 1.5, True, "1"):
            data = sample_config()
            data["initialScore"]["left"] = score
            with self.subTest(score=score), self.assertRaises(ValueError):
                self.load(data)
        for value in (None, "名前\n名前", "名前\x00"):
            data = sample_config()
            data["leftPlayerName"] = value
            with self.subTest(value=value), self.assertRaises(ValueError):
                self.load(data)


class EncoderSelectionTests(unittest.TestCase):
    def test_priority_and_each_fallback_position(self):
        expected = ("av1_nvenc", "hevc_nvenc", "h264_nvenc", "libx264")
        for successful_index, encoder in enumerate(expected):
            attempted = []
            def run(command, **kwargs):
                selected = command[command.index("-c:v") + 1]
                attempted.append(selected)
                status = 0 if selected == encoder else 1
                return subprocess.CompletedProcess(command, status, "", "NVENC unavailable")
            with self.subTest(encoder=encoder), tempfile.TemporaryDirectory() as folder, patch.object(scoreboard.subprocess, "run", side_effect=run), contextlib.redirect_stdout(io.StringIO()):
                selected = scoreboard.select_encoder(scoreboard.Video(10, "60/1"), Path(folder))
            self.assertEqual(selected, encoder)
            self.assertEqual(attempted, list(expected[:successful_index + 1]))

    def test_no_encoder_is_available(self):
        failed = subprocess.CompletedProcess([], 1, "", "unavailable")
        with tempfile.TemporaryDirectory() as folder, patch.object(scoreboard.subprocess, "run", return_value=failed) as run, contextlib.redirect_stdout(io.StringIO()):
            with self.assertRaisesRegex(ValueError, "使用可能な映像エンコーダーがありません"):
                scoreboard.select_encoder(scoreboard.Video(10, "60/1"), Path(folder))
            self.assertEqual(run.call_count, 4)

    def test_initialization_timeout_falls_back(self):
        results = [subprocess.TimeoutExpired("ffmpeg", 30), subprocess.CompletedProcess([], 0, "", "")]
        with tempfile.TemporaryDirectory() as folder, patch.object(scoreboard.subprocess, "run", side_effect=results), contextlib.redirect_stdout(io.StringIO()):
            self.assertEqual(scoreboard.select_encoder(scoreboard.Video(10, "60/1"), Path(folder)), "hevc_nvenc")


class FfmpegCompatibilityTests(unittest.TestCase):
    def test_filtergraph_file_option(self):
        with patch.object(scoreboard.subprocess, "Popen") as popen:
            popen.return_value.communicate.return_value = (b"  -filter_complex_script filename\n", None)
            self.assertEqual(scoreboard.filter_graph_file_option(), "-filter_complex_script")
        with patch.object(scoreboard.subprocess, "Popen") as popen:
            popen.return_value.communicate.return_value = (b"  -filter_complex graph_description\n", None)
            self.assertEqual(scoreboard.filter_graph_file_option(), "-/filter_complex")


@unittest.skipUnless(shutil.which("ffmpeg") and shutil.which("ffprobe"), "FFmpegが必要です")
class EncodedVideoTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temporary = tempfile.TemporaryDirectory(prefix="scoreboard-test-")
        cls.work = Path(cls.temporary.name)
        cls.source = cls.work / "source '日本語', test.mp4"
        cls.config = cls.work / "match '設定'.json"
        cls.config.write_text(json.dumps(sample_config(), ensure_ascii=False), encoding="utf-8")
        cls.full = cls.work / "full.mp4"
        cls.clip = cls.work / "clip.mp4"
        subprocess.run([
            "ffmpeg", "-v", "error", "-f", "lavfi", "-i", "color=0x235aaf:s=1920x1080:r=60:d=3.5",
            "-f", "lavfi", "-i", "sine=frequency=440:sample_rate=48000:duration=3.5",
            "-c:v", "libx264", "-preset", "ultrafast", "-crf", "24", "-c:a", "aac",
            "-colorspace", "bt709", "-color_trc", "bt709", "-color_primaries", "bt709", str(cls.source),
        ], check=True, capture_output=True)
        # Hide normal FFmpeg progress, while keeping failure details available.
        original_run = scoreboard.subprocess.run
        def quiet_run(*args, **kwargs):
            if not kwargs.get("capture_output"):
                kwargs["capture_output"] = True
            return original_run(*args, **kwargs)
        with patch.object(scoreboard.subprocess, "run", side_effect=quiet_run), patch.object(scoreboard, "select_encoder", return_value="libx264"), contextlib.redirect_stdout(io.StringIO()):
            scoreboard.render(cls.source, cls.config, cls.full)
            args = ["scoreboard.py", "--input", str(cls.source), "--config", str(cls.config), "--output", str(cls.clip), "--start", "00:02.250", "--duration", "00:00.500"]
            with patch.object(scoreboard.sys, "argv", args):
                if scoreboard.main() != 0:
                    raise RuntimeError("MM:SS.mmmでのCLI出力に失敗しました。")
        cls.left = cls.crop_frames(cls.full, 744, 8, 64, 78)
        cls.right = cls.crop_frames(cls.full, 1112, 8, 64, 78)

    @classmethod
    def tearDownClass(cls):
        cls.temporary.cleanup()

    @staticmethod
    def crop_frames(path, x, y, width, height):
        raw = subprocess.run([
            "ffmpeg", "-v", "error", "-i", str(path), "-vf", f"crop={width}:{height}:{x}:{y}",
            "-fps_mode", "passthrough", "-pix_fmt", "rgb24", "-f", "rawvideo", "pipe:1",
        ], check=True, capture_output=True).stdout
        size = width * height * 3
        return [raw[index:index + size] for index in range(0, len(raw), size)]

    @staticmethod
    def distance(first, second):
        return sum(abs(a - b) for a, b in zip(first, second)) / len(first)

    def test_first_frame_and_unchanged_side(self):
        self.assertEqual(len(self.left), 210)
        # No entrance fade: the first frame matches the fully displayed state.
        self.assertLess(self.distance(self.left[0], self.left[30]), 1)
        self.assertLess(self.distance(self.right[0], self.right[84]), 1)
        self.assertGreater(self.distance(self.left[0], self.left[72]), 10)
        # A repeated 1–1 row at 2.8s must not blank or fade either digit.
        self.assertLess(self.distance(self.left[168], self.left[174]), 1)
        self.assertLess(self.distance(self.right[168], self.right[174]), 1)

    def test_ease_out_and_sequential_fade(self):
        blank = self.left[72]  # 1.2s: old is gone, new has opacity zero.
        old_full = self.distance(self.left[0], blank)
        new_full = self.distance(self.left[84], blank)
        old_half = self.distance(self.left[66], blank) / old_full
        new_half = self.distance(self.left[78], blank) / new_full
        # CSS easeOut(0.5) = approximately 0.68464, rather than linear 0.5.
        self.assertAlmostEqual(old_half, 1 - 0.68464, delta=0.035)
        self.assertAlmostEqual(new_half, 0.68464, delta=0.035)
        self.assertLess(self.distance(self.left[60], self.left[0]), 1)
        # Right-only update must leave the already updated left digit stable.
        self.assertLess(self.distance(self.left[120], self.left[132]), 1)
        self.assertGreater(self.distance(self.right[120], self.right[132]), 10)

    def test_clip_keeps_original_timeline(self):
        left = self.crop_frames(self.clip, 744, 8, 64, 78)
        right = self.crop_frames(self.clip, 1112, 8, 64, 78)
        self.assertEqual(len(left), 30)
        self.assertLess(self.distance(left[0], self.left[135]), 1.5)
        self.assertLess(self.distance(right[0], self.right[135]), 1.5)
        self.assertLess(self.distance(right[9], self.right[144]), 1.5)

    def test_video_and_audio_properties(self):
        original = self.crop_frames(self.source, 16, 1000, 16, 16)[0]
        composed = self.crop_frames(self.full, 16, 1000, 16, 16)[0]
        self.assertLess(self.distance(original, composed), 2)
        for path, length in ((self.full, 3.5), (self.clip, 0.5)):
            data = json.loads(subprocess.run([
                "ffprobe", "-v", "error", "-show_streams", "-show_format", "-of", "json", str(path),
            ], check=True, capture_output=True, text=True).stdout)
            video, audio = data["streams"]
            self.assertEqual((video["width"], video["height"]), (1920, 1080))
            self.assertEqual(video["avg_frame_rate"], "60/1")
            self.assertEqual(video["color_space"], "bt709")
            self.assertEqual(video["codec_name"], "h264")
            self.assertAlmostEqual(float(video["bit_rate"]), 15_000_000, delta=150_000)
            self.assertEqual(audio["codec_name"], "aac")
            self.assertAlmostEqual(float(video["duration"]), length, delta=1 / 60)
            self.assertAlmostEqual(float(audio["duration"]), length, delta=0.04)
            self.assertLess(abs(float(video["start_time"]) - float(audio["start_time"])), 0.025)

    def test_simultaneous_change_reset_and_silent_video(self):
        source = self.work / "silent.mp4"
        subprocess.run([
            "ffmpeg", "-v", "error", "-i", str(self.source), "-an", "-c:v", "copy", str(source),
        ], check=True, capture_output=True)
        data = sample_config()
        data["scoreChanges"] = [
            {"at": 0.5, "left": 2, "right": 2},
            {"at": 1.5, "left": 0, "right": 0},
        ]
        config = self.work / "simultaneous.json"
        config.write_text(json.dumps(data, ensure_ascii=False), encoding="utf-8")
        output = self.work / "simultaneous.mp4"
        command = [
            sys.executable, str(scoreboard.ROOT / "scoreboard.py"), "--input", str(source),
            "--config", str(config), "--output", str(output), "--duration", "2.1",
        ]
        subprocess.run(command, check=True, capture_output=True)
        for x in (744, 1112):
            frames = self.crop_frames(output, x, 8, 64, 78)
            self.assertGreater(self.distance(frames[0], frames[60]), 3)
            # Inter-frame compression can quantize a later frame differently
            # from the first I frame, even when both show the same digit.
            self.assertLess(self.distance(frames[0], frames[120]), 2.5)

    def overwrite_attempt(self, filename, answer):
        output = self.work / filename
        previous = b"existing output must survive refusal\n"
        output.write_bytes(previous)
        original_run = subprocess.run
        stderr = []
        def run(command, **kwargs):
            if command[0] == "ffmpeg":
                kwargs.update(input=answer, capture_output=True, text=True, encoding="utf-8", errors="replace", timeout=15)
                try:
                    result = original_run(command, **kwargs)
                except subprocess.CalledProcessError as error:
                    stderr.append(error.stderr)
                    raise
                stderr.append(result.stderr)
                return result
            return original_run(command, **kwargs)
        with patch.object(scoreboard.subprocess, "run", side_effect=run), patch.object(scoreboard, "select_encoder", return_value="libx264"), contextlib.redirect_stdout(io.StringIO()):
            if answer == "n\n":
                # FFmpeg versions differ in the exit status when declining.
                # The essential behavior is that the existing file survives.
                try:
                    scoreboard.render(self.source, self.config, output, duration=0.1)
                except subprocess.CalledProcessError:
                    pass
                self.assertEqual(output.read_bytes(), previous)
            else:
                scoreboard.render(self.source, self.config, output, duration=0.1)
                self.assertNotEqual(output.read_bytes(), previous)
                self.assertAlmostEqual(scoreboard.probe(output).duration, 0.1, delta=1 / 60)
        return "\n".join(stderr)

    def test_overwrite_no_keeps_existing_file(self):
        self.assertIn("Overwrite? [y/N]", self.overwrite_attempt("refused.mp4", "n\n"))

    def test_overwrite_yes_replaces_existing_file(self):
        self.assertIn("Overwrite? [y/N]", self.overwrite_attempt("accepted.mp4", "y\n"))


if __name__ == "__main__":
    unittest.main()
