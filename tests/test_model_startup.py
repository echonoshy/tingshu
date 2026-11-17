import queue
import tempfile
import threading
import unittest
from concurrent.futures import Future
from pathlib import Path
from unittest.mock import Mock, patch

from tools.llama import generate


class ModelStartupTests(unittest.TestCase):
    def start(self, checkpoint="weights"):
        result = Future()

        def call():
            try:
                result.set_result(generate.launch_thread_safe_queue(checkpoint, "cpu", None))
            except BaseException as error:
                result.set_exception(error)

        # A worker startup regression must fail this test without hanging the runner.
        threading.Thread(target=call, daemon=True).start()
        return result.result(timeout=5)

    def test_checkpoint_failure_reaches_caller(self):
        with tempfile.TemporaryDirectory() as checkpoint:
            with self.assertRaises(FileNotFoundError) as raised:
                self.start(checkpoint)
            self.assertEqual(Path(raised.exception.filename), Path(checkpoint) / "config.json")

    def test_cache_failure_reaches_caller(self):
        model = Mock()
        model.parameters.return_value = iter([Mock()])
        model.setup_caches.side_effect = RuntimeError("cache allocation")
        with patch.object(generate, "load_model", return_value=(model, None)):
            with self.assertRaisesRegex(RuntimeError, "cache allocation"):
                self.start()

    def test_initialized_worker_serves_requests(self):
        model = Mock()
        model.parameters.return_value = iter([Mock()])
        response = generate.GenerateResponse(action="next")
        with patch.object(generate, "load_model", return_value=(model, None)), \
             patch.object(generate, "generate_long", return_value=[response]):
            requests = self.start()
            try:
                model.setup_caches.assert_called_once()
                replies = queue.Queue()
                requests.put(generate.GenerateRequest(request={}, response_queue=replies))
                reply = replies.get(timeout=5)
                self.assertEqual(reply.status, "success")
                self.assertIs(reply.response, response)
            finally:
                requests.put(None)
