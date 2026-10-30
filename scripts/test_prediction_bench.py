#!/usr/bin/env python3
"""Deterministic tests for benchmark cache-preparation helpers."""
import importlib.util
import json
import pathlib
import tempfile
import unittest


SCRIPT = pathlib.Path(__file__).with_name("prediction-bench.py")
SPEC = importlib.util.spec_from_file_location("prediction_bench", SCRIPT)
BENCH = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(BENCH)


class CachePreparationTests(unittest.TestCase):
    def test_probe_reports_page_counts(self):
        with tempfile.TemporaryDirectory(dir="/tmp/opencode") as directory:
            path = pathlib.Path(directory) / "probe.bin"
            path.write_bytes(b"x" * 8192)
            probe = BENCH._probe_residency(path)
        self.assertIsNone(probe["error"])
        self.assertEqual(probe["pages"], 2)
        self.assertEqual(probe["resident_pages"] + probe["nonresident_pages"], 2)

    def test_missing_source_fails_without_becoming_cold(self):
        missing = pathlib.Path("/tmp/opencode/ramforge-definitely-missing-model.gguf")
        _, meta = BENCH.prepare_cache_condition(
            missing, "isolated-nonresident", "/tmp/opencode/copies"
        )
        self.assertFalse(meta["preparation_ok"])
        self.assertFalse(meta["controlled"])
        self.assertNotEqual(meta["actual"], "controlled-nonresident")

    def test_copy_failure_invalidates_cold_classification(self):
        with tempfile.TemporaryDirectory(dir="/tmp/opencode") as directory:
            source = pathlib.Path(directory) / "model.gguf"
            source.write_bytes(b"x" * 4096)
            copy_dir = pathlib.Path(directory) / "not-a-directory"
            copy_dir.write_text("occupied")
            _, meta = BENCH.prepare_cache_condition(
                source, "isolated-nonresident", copy_dir
            )
        self.assertFalse(meta["preparation_ok"])
        self.assertFalse(meta["controlled"])
        self.assertEqual(meta["actual"], "cache-preparation-failed")

    def test_warm_is_never_reported_as_controlled_cold(self):
        with tempfile.TemporaryDirectory(dir="/tmp/opencode") as directory:
            source = pathlib.Path(directory) / "model.gguf"
            source.write_bytes(b"x" * 8192)
            _, meta = BENCH.prepare_cache_condition(
                source, "warm", pathlib.Path(directory) / "copies"
            )
        self.assertFalse(meta["controlled"])
        self.assertTrue(meta["preparation_ok"])
        self.assertIn(meta["actual"], {"warm", "warm-or-unknown"})

    def test_profile_identity_is_rewritten_only_for_copy(self):
        with tempfile.TemporaryDirectory(dir="/tmp/opencode") as directory:
            directory = pathlib.Path(directory)
            source = directory / "model.gguf"
            source.write_bytes(b"x" * 4096)
            profile = directory / "profile.json"
            profile.write_text(
                json.dumps(
                    {
                        "identity": {
                            "model_path": str(source),
                            "model_size_bytes": 4096,
                            "model_modified_unix_ns": source.stat().st_mtime_ns,
                        },
                        "profile": {"model_id": str(source), "access": {}},
                    }
                )
            )
            _, meta = BENCH.prepare_cache_condition(
                source, "isolated-nonresident", directory / "copies"
            )
            prepared = BENCH.prepare_artifact_identity(
                profile, meta["model"], directory / "prepared", "profile"
            )
            self.assertNotEqual(prepared, str(profile))
            rewritten = json.loads(pathlib.Path(prepared).read_text())
            self.assertEqual(rewritten["identity"]["model_path"], meta["model"])
            original = json.loads(profile.read_text())
            self.assertEqual(original["identity"]["model_path"], str(source))

    def test_isolated_copy_does_not_modify_source(self):
        with tempfile.TemporaryDirectory(dir="/tmp/opencode") as directory:
            source = pathlib.Path(directory) / "model.gguf"
            payload = b"model-data" * 1024
            source.write_bytes(payload)
            _, meta = BENCH.prepare_cache_condition(
                source, "isolated-nonresident", pathlib.Path(directory) / "copies"
            )
            self.assertNotEqual(meta["model"], str(source))
            self.assertEqual(source.read_bytes(), payload)


if __name__ == "__main__":
    unittest.main()
