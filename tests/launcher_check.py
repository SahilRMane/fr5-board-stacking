"""Launcher lifecycle checks with local stand-ins. No ROS, models or hardware."""
import argparse
import contextlib
import io
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import tempfile
import time
import unittest
from unittest.mock import MagicMock, patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))
import stack


class LauncherCheck(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix="stack test ")
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        for name in ("install/setup.sh", "web/dist/index.html", "web/.local/server.crt"):
            path = self.root / name
            path.parent.mkdir(parents=True, exist_ok=True)
            path.touch()
        (self.root / 'web/.local/session.json').write_text(json.dumps({'token': 'fixture-token'}))
        # Exercise the real supervisor and lock with harmless stand-ins for ROS/models.
        self.harness = self.root / "harness.py"
        self.harness.write_text(f'''
import sys, time
from pathlib import Path
sys.path.insert(0, {str(Path(stack.__file__).parent)!r})
import stack
stack.ROOT = Path(__file__).parent
def run(self):
    self.start("monitor", sys.executable, "-u", "-c", "import time; print('monitor ready', flush=True); time.sleep(60)")
    self.start("motion", sys.executable, "-u", "-c", "import time; print('motion ready', flush=True); time.sleep(60)")
    (self.root / "ready").write_text(str(self.children["motion"].pid))
    if self.args.mode == "demo":
        self.wait_for("fixture", 0.2, lambda: False)
    while True:
        self.check(startup=False)
        time.sleep(0.02)
stack.Stack.run = run
if __name__ == "__main__":
    try:
        sys.exit(stack.main())
    except RuntimeError as error:
        print(error, flush=True)
        sys.exit(1)
''')

    def launch(self, *args):
        path = self.root / f"launcher-{time.monotonic_ns()}.log"
        with path.open("w") as log:
            p = subprocess.Popen([sys.executable, str(self.harness), "up", "--sim", *args],
                                 stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
        self.addCleanup(self.finish, p)
        return p, path

    @staticmethod
    def finish(p):
        if p.poll() is None:
            p.terminate()
            p.wait(timeout=15)

    def ready(self):
        end = time.monotonic() + 5
        while time.monotonic() < end:
            if (self.root / "ready").exists():
                return int((self.root / "ready").read_text())
            time.sleep(0.02)
        self.fail("Fixture did not start")

    def down(self):
        return subprocess.run([sys.executable, str(self.harness), "down"],
                              text=True, capture_output=True, timeout=15)

    def test_down_stops_owned_processes_and_is_idempotent(self):
        unrelated = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(60)"], start_new_session=True)
        self.addCleanup(self.finish, unrelated)
        p, _ = self.launch()
        motion = self.ready()
        result = self.down()
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(p.wait(timeout=5), 0)
        self.assertFalse(stack.group_alive(motion))
        self.assertIsNone(unrelated.poll())
        self.assertIn("No stack is running", self.down().stdout)
        (self.root / "ready").unlink()
        second, _ = self.launch()
        self.ready()
        self.assertIsNone(second.poll())
        self.assertEqual(self.down().returncode, 0)

    def test_duplicate_start_preserves_running_stack(self):
        p, _ = self.launch()
        self.ready()
        duplicate, log = self.launch()
        self.assertEqual(duplicate.wait(timeout=5), 1)
        self.assertIn("already running", log.read_text())
        self.assertIsNone(p.poll())
        self.assertEqual(self.down().returncode, 0)

    def test_readiness_failure_cleans_up_and_keeps_logs(self):
        p, log = self.launch("demo")
        motion = self.ready()
        self.assertEqual(p.wait(timeout=10), 1)
        self.assertIn("NOT ready", log.read_text())
        self.assertFalse(stack.group_alive(motion))
        self.assertIn("motion ready", (self.root / "live_logs/current/processes/motion.log").read_text())

    def test_signals_stop_the_stack(self):
        for sig in (signal.SIGINT, signal.SIGTERM, signal.SIGHUP):
            with self.subTest(signal=sig):
                (self.root / "ready").unlink(missing_ok=True)
                p, _ = self.launch()
                motion = self.ready()
                p.send_signal(sig)
                self.assertEqual(p.wait(timeout=10), 0)
                self.assertFalse(stack.group_alive(motion))

    def test_runtime_crash_is_reported_without_stopping_healthy_processes(self):
        p, log = self.launch()
        motion = self.ready()
        os.killpg(motion, signal.SIGKILL)
        end = time.monotonic() + 5
        while "motion exited" not in log.read_text() and time.monotonic() < end:
            time.sleep(0.02)
        self.assertIn("motion exited", log.read_text())
        self.assertIsNone(p.poll())
        self.assertNotIn("Stopping monitor", log.read_text())
        self.assertEqual(self.down().returncode, 0)
        self.assertIn("Stopping monitor", log.read_text())

    def test_real_start_sequence_forces_simulation_isolation(self):
        args = argparse.Namespace(sim=True, mode="debug", rviz=False, no_gui=False, perception='yolo26')
        with patch.dict(os.environ, ROS_LOCALHOST_ONLY="0", WORKCELL_REAL="1"):
            s = stack.Stack(self.root, args)
        self.addCleanup(s.control.close)
        with patch.object(s, "start") as start, patch.object(s, "ros", return_value=(
                "/motion/get_state\n/task/list_jobs\nPublisher count: 1\n")), \
                patch.object(s, "url", return_value=True) as urls, patch("stack.socket.socket"), \
                patch("stack.time.sleep", side_effect=lambda _: s.request_stop()), \
                patch.dict(os.environ, DISPLAY=":0"), \
                patch("stack.subprocess.run", side_effect=FileNotFoundError("xdg-open")), \
                contextlib.redirect_stdout(io.StringIO()) as output:
            with self.assertRaises(KeyboardInterrupt):
                s.run()
        self.assertIn("All started", output.getvalue())
        urls.assert_any_call('http://localhost:8766/health', ('ready', 'asr_ready'))
        self.assertIn("Open http://localhost:8088 in a browser", output.getvalue())
        names = [call.args[0] for call in start.call_args_list]
        self.assertEqual(names, ["monitor", "llm", "motion", "fake", "world", "skills", "ctrl",
                                 "bridge", "demo-ui", "speech", "chat", "web"])
        self.assertEqual(s.env["ROS_LOCALHOST_ONLY"], "1")
        self.assertEqual(s.env["WORKCELL_REAL"], "0")
        self.assertIn("use_fake_hardware:=true", start.call_args_list[2].args)

    def test_http_success_does_not_imply_model_readiness(self):
        for value, expected in (({}, False), ({'ready': True}, False),
                                ({'ready': True, 'asr_ready': False}, False),
                                ({'ready': True, 'asr_ready': 'false'}, False),
                                ({'ready': True, 'asr_ready': 1}, False),
                                ([], False), ({'ready': True, 'asr_ready': True}, True)):
            response = MagicMock(status=200)
            response.__enter__.return_value = response
            response.read.return_value = json.dumps(value).encode()
            with self.subTest(value=value), patch('stack.urllib.request.urlopen', return_value=response):
                self.assertEqual(stack.Stack.url('http://fixture/health', ('ready', 'asr_ready')), expected)
        response.read.return_value = b'not JSON'
        with patch('stack.urllib.request.urlopen', return_value=response):
            self.assertFalse(stack.Stack.url('http://fixture/health', ('ready',)))
        with patch('stack.urllib.request.urlopen', side_effect=OSError('not listening')):
            self.assertFalse(stack.Stack.url('http://fixture/health', ('ready',)))

    def test_web_readiness_uses_authenticated_state_and_waits_for_session(self):
        args = argparse.Namespace(sim=True, mode='debug', rviz=False, no_gui=True, perception='yolo26')
        s = stack.Stack(self.root, args)
        self.addCleanup(s.control.close)
        with patch.object(s, 'url', return_value=False) as probe:
            self.assertFalse(s.web_ready())
        request, flags = probe.call_args.args
        self.assertEqual(request.full_url, 'http://localhost:8088/api/state')
        self.assertEqual(request.get_header('X-workcell-token'), 'fixture-token')
        self.assertEqual(flags, ('ready',))
        session = self.root / 'web/.local/session.json'
        for content in (None, 'not JSON', '{}', '[]', '{"token": null}', '{"token": ""}'):
            if content is None:
                session.unlink()
            else:
                session.write_text(content)
            with self.subTest(content=content), patch.object(s, 'url') as probe:
                self.assertFalse(s.web_ready())
                probe.assert_not_called()

    def test_hardware_launch_forwards_either_camera_model_without_starting_two_sources(self):
        for backend in ('yolo26','yolo11'):
            args = argparse.Namespace(sim=False,mode='debug',rviz=False,no_gui=True,perception=backend)
            s = stack.Stack(self.root,args)
            try:
                with patch.object(s,'start') as start,patch.object(s,'ros',return_value=(
                        '/motion/get_state\n/task/list_jobs\nPublisher count: 1\n')), \
                        patch.object(s,'url',return_value=True),patch('stack.socket.socket'), \
                        patch('stack.time.sleep',side_effect=lambda _:s.request_stop()), \
                        contextlib.redirect_stdout(io.StringIO()):
                    with self.assertRaises(KeyboardInterrupt):s.run()
                calls = [c for c in start.call_args_list if c.args[0]=='vision']
                self.assertEqual(len(calls),1)
                self.assertIn('backend:='+backend,calls[0].args)
                self.assertFalse(any(c.args[0]=='fake' for c in start.call_args_list))
            finally:
                s.control.close()


if __name__ == "__main__":
    unittest.main()
