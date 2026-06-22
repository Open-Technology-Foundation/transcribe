#!/usr/bin/env python3
"""
Unit tests for transcribe_pkg.utils.cache.CacheManager.

These tests are hermetic: they touch only an isolated temporary cache
directory and never perform network, LLM, Whisper, torch, or pydub I/O.
"""
import os
import pickle
import shutil
import tempfile
import threading
import unittest
from unittest import mock

from transcribe_pkg.utils.cache import CacheManager


class CacheManagerTestBase(unittest.TestCase):
  """Base class providing an isolated, self-cleaning cache directory."""

  def setUp(self):
    self.cache_dir = tempfile.mkdtemp(prefix="transcribe_cache_test_")

  def tearDown(self):
    shutil.rmtree(self.cache_dir, ignore_errors=True)

  def make_manager(self, **kwargs):
    kwargs.setdefault("cache_dir", self.cache_dir)
    return CacheManager(**kwargs)


class TestConcurrentMemoryCache(CacheManagerTestBase):
  """Finding (1): shared CacheManager mutated concurrently with no lock."""

  def test_concurrent_eviction_through_api_does_not_raise(self):
    # The eviction in _add_to_memory is a non-atomic check-then-act:
    #   oldest = min(self.memory_timestamp.keys(), key=lambda k: ...get(k, 0))
    #   self._remove_from_memory(oldest)
    # min() iterates the LIVE dict_keys view and calls the key function for each
    # element. If ANOTHER CacheManager operation mutates memory_timestamp
    # between two iterator steps, CPython raises:
    #   RuntimeError: dictionary changed size during iteration
    #
    # We drive the race through the public/internal API so the lock fix is what
    # closes it. memory_timestamp is swapped for a dict subclass that overrides
    # get() (the key function). On the first call during min()'s scan, thread A
    # (holding the lock under the fix) launches thread B which calls another
    # CacheManager method that also mutates the memory dicts. Under the fix,
    # B blocks on the lock until A finishes, so no concurrent mutation occurs.
    # Without the lock, B mutates immediately and A's min() raises RuntimeError.
    # A does NOT join B (that would deadlock under the fix); it only gives B a
    # window to act, then continues iterating.
    manager = self.make_manager(
      max_memory_items=3,
      disk_cache_enabled=False,
      memory_cache_enabled=True,
    )

    errors = []
    b_done = threading.Event()

    def thread_b_mutation():
      # Goes through a CacheManager method that acquires the lock under the fix.
      try:
        manager.set("intruder", -1, cache_type="memory")
      finally:
        b_done.set()

    class RacingDict(dict):
      """Dict whose get() (min's key func) launches a racing API mutation once."""

      def __init__(self, *a, **k):
        super().__init__(*a, **k)
        self.armed = False
        self.fired = threading.Event()

      def get(self, key, default=None):
        if self.armed and not self.fired.is_set():
          self.fired.set()
          th = threading.Thread(target=thread_b_mutation, daemon=True)
          th.start()
          # Give B a window to (try to) mutate. Under the fix B is blocked on
          # the lock and this sleep does not release it; without the fix B's
          # raw dict insert lands here, changing the size mid-iteration.
          b_done.wait(timeout=0.2)
        return super().get(key, default)

    racing = RacingDict()
    manager.memory_timestamp = racing
    # Fill to capacity so the next add triggers the eviction (min) path.
    for i in range(manager.max_memory_items):
      manager.set(f"seed-{i}", i, cache_type="memory")
    racing.armed = True

    def adder():
      try:
        manager.set("trigger", 999, cache_type="memory")
      except Exception as exc:  # noqa: BLE001 - capture the race for assertion
        errors.append(exc)

    t = threading.Thread(target=adder)
    t.start()
    t.join(timeout=5.0)
    self.assertFalse(t.is_alive(), "adder thread deadlocked")
    # Let any blocked intruder thread drain so teardown is clean.
    b_done.wait(timeout=2.0)

    self.assertEqual(
      errors, [],
      msg=f"Eviction raced with a concurrent API mutation (no lock): {errors!r}",
    )

  def test_has_lock_attribute(self):
    # A lock must exist and be a usable (re-entrant) lock object.
    manager = self.make_manager()
    self.assertTrue(hasattr(manager, "_lock"))
    # RLock allows re-entrant acquisition; acquiring twice must not deadlock.
    self.assertTrue(manager._lock.acquire(blocking=False))
    self.assertTrue(manager._lock.acquire(blocking=False))
    manager._lock.release()
    manager._lock.release()


class TestAtomicWriteCleanup(CacheManagerTestBase):
  """Finding (2): orphaned .tmp on write failure."""

  def test_set_removes_tmp_file_on_write_failure(self):
    manager = self.make_manager(memory_cache_enabled=False)

    # Force the pickle.dump to fail AFTER the tmp file has been opened/created,
    # simulating a crash mid-write. The orphaned .tmp must be cleaned up.
    real_dump = pickle.dump

    def boom(*args, **kwargs):
      raise RuntimeError("simulated write crash")

    with mock.patch("transcribe_pkg.utils.cache.pickle.dump", side_effect=boom):
      manager.set("some-key", {"a": 1}, cache_type="disk")

    leftovers = [f for f in os.listdir(self.cache_dir) if f.endswith(".tmp")]
    self.assertEqual(
      leftovers, [],
      msg=f"Orphaned .tmp file(s) left after write failure: {leftovers}",
    )
    # Sanity: real_dump still works once we stop patching (cache still usable).
    self.assertIs(real_dump, pickle.dump)


class TestClearRemovesTmpFiles(CacheManagerTestBase):
  """Finding (2): clear() only deletes *.cache, leaving *.tmp behind."""

  def test_clear_removes_both_cache_and_tmp_files(self):
    manager = self.make_manager(memory_cache_enabled=False)

    # A normal entry produces a .cache file.
    manager.set("good-key", "value", cache_type="disk")
    # Simulate a leftover .tmp from a previously-crashed write.
    orphan = os.path.join(self.cache_dir, "deadbeef.cache.tmp")
    with open(orphan, "wb") as fh:
      fh.write(b"partial")

    self.assertTrue(
      any(f.endswith(".cache") for f in os.listdir(self.cache_dir)))
    self.assertTrue(
      any(f.endswith(".tmp") for f in os.listdir(self.cache_dir)))

    manager.clear(cache_type="disk")

    remaining = os.listdir(self.cache_dir)
    self.assertEqual(
      remaining, [],
      msg=f"clear() left files behind (expected none): {remaining}",
    )


class TestCachedDecoratorRemoved(unittest.TestCase):
  """Finding (3): the unused cached() decorator must be removed."""

  def test_cached_decorator_is_gone(self):
    import transcribe_pkg.utils.cache as cache_mod
    self.assertFalse(
      hasattr(cache_mod, "cached"),
      msg="Unused cached() decorator should be removed from cache.py",
    )


class TestBasicCacheBehaviour(CacheManagerTestBase):
  """Regression guard: core get/set/clear behaviour stays intact."""

  def test_set_get_roundtrip_memory_and_disk(self):
    manager = self.make_manager()
    manager.set("k", {"hello": "world"})
    self.assertEqual(manager.get("k"), {"hello": "world"})

  def test_disk_only_roundtrip_survives_new_manager(self):
    m1 = self.make_manager(memory_cache_enabled=False)
    m1.set("persist", [1, 2, 3], cache_type="disk")
    m2 = self.make_manager(memory_cache_enabled=False)
    self.assertEqual(m2.get("persist", cache_type="disk"), [1, 2, 3])

  def test_invalidate_removes_entry(self):
    manager = self.make_manager()
    manager.set("gone", "soon")
    manager.invalidate("gone")
    self.assertIsNone(manager.get("gone"))


if __name__ == "__main__":
  unittest.main()

#fin
