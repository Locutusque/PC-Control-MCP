"""Frame geometry on HiDPI displays (Retina Macs, fractional scaling on Linux).

Two coordinate spaces are in play and they are not the same on these displays:

* the **input coordinate space** (points) that mouse events are reported in;
* the **pixel buffer** that the screen grabber actually returns.

Grounding clicks against the wrong one mislabels every click by the scale
factor, and feeding ffmpeg the wrong one desynchronises the raw byte stream.
Neither failure raises on its own, so both are pinned here.
"""

from __future__ import annotations

import pytest

from gui_agent.capture.screen import Frame, ScreenGrabber, SegmentWriter, VideoUnavailable


class _FakeShot:
    def __init__(self, width, height):
        self.width, self.height = width, height
        self.raw = bytearray(width * height * 4)


class _FakeSct:
    """mss reports monitor geometry in points and grabs at true pixel density."""

    def __init__(self, points, ratio):
        self.monitors = [None, {"width": points[0], "height": points[1], "top": 0, "left": 0}]
        self.ratio = ratio

    def grab(self, monitor):
        return _FakeShot(int(monitor["width"] * self.ratio), int(monitor["height"] * self.ratio))

    def close(self):
        pass


def _grabber(points=(1512, 982), ratio=2.0, scale=1.0) -> ScreenGrabber:
    grabber = ScreenGrabber(scale=scale)
    grabber._sct = _FakeSct(points, ratio)
    grabber._monitor = grabber._sct.monitors[1]
    return grabber


class TestRetinaGeometry:
    def test_size_is_the_input_coordinate_space(self):
        # Mouse events arrive in points, so this is what clicks ground against.
        assert _grabber().size == (1512, 982)

    def test_frame_size_is_the_pixel_buffer(self):
        # A 2x panel returns four times the bytes the monitor dict implies.
        assert _grabber().frame_size == (3024, 1964)

    def test_pixel_ratio_is_reported(self):
        assert _grabber(ratio=2.0).pixel_ratio == pytest.approx(2.0)
        assert _grabber(ratio=1.0).pixel_ratio == pytest.approx(1.0)

    def test_frame_size_is_probed_once_and_cached(self):
        grabber = _grabber()
        assert grabber.frame_size == grabber.frame_size
        assert grabber._frame_size is not None

    def test_output_size_scales_the_pixel_buffer_not_the_points(self):
        # Halving a 2x Retina buffer lands back at the logical resolution.
        assert _grabber(scale=0.5).output_size == (1512, 982)

    def test_output_dimensions_are_even(self):
        assert all(v % 2 == 0 for v in _grabber(points=(1001, 667), scale=0.5).output_size)

    def test_non_hidpi_display_is_unaffected(self):
        grabber = _grabber(points=(1920, 1080), ratio=1.0)
        assert grabber.size == grabber.frame_size == (1920, 1080)


class TestWriterGuards:
    def test_mismatched_frame_size_is_rejected_on_the_first_frame(self):
        # ffmpeg reads a headerless stream: a size mismatch does not error, it
        # silently desynchronises and every later frame is garbage.
        writer = SegmentWriter("/tmp/unused.mkv", width=1512, height=982, fps=15)
        writer._proc = type("P", (), {"stdin": type("S", (), {"write": lambda self, b: None})()})()
        retina_frame = Frame(bytes(3024 * 1964 * 4), 3024, 1964, 0.0)
        with pytest.raises(VideoUnavailable, match="HiDPI"):
            writer.write(retina_frame)

    def test_matching_frame_is_accepted(self):
        writer = SegmentWriter("/tmp/unused.mkv", width=8, height=4, fps=15)
        writer._proc = type("P", (), {"stdin": type("S", (), {"write": lambda self, b: None})()})()
        assert writer.write(Frame(bytes(8 * 4 * 4), 8, 4, 0.0)) == 0

    def test_scaling_is_off_when_output_matches_input(self):
        assert not SegmentWriter("/tmp/x.mkv", 100, 100, 15).is_scaling

    def test_scaling_is_on_when_output_differs(self):
        assert SegmentWriter("/tmp/x.mkv", 100, 100, 15, output_width=50, output_height=50).is_scaling


class TestCaptureScaleIsWiredUp:
    def test_scale_reaches_the_encoder_command(self, monkeypatch, tmp_path):
        """capture_scale used to be computed and then dropped on the floor."""
        captured = {}

        class _Proc:
            stdin = stderr = None

            def wait(self, timeout=None):
                return 0

        def fake_popen(cmd, **kwargs):
            captured["cmd"] = cmd
            return _Proc()

        monkeypatch.setattr("subprocess.Popen", fake_popen)
        monkeypatch.setattr(SegmentWriter, "ffmpeg_path", staticmethod(lambda: "ffmpeg"))

        SegmentWriter(
            tmp_path / "seg.mkv", width=3024, height=1964, fps=15,
            output_width=1512, output_height=982,
        ).open()

        cmd = captured["cmd"]
        assert "-s" in cmd and "3024x1964" in cmd      # raw input at true pixels
        assert "-vf" in cmd
        assert "scale=1512:982" in cmd[cmd.index("-vf") + 1]
