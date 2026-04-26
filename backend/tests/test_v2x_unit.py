import unittest
from unittest.mock import MagicMock, patch
import sys
import os

class TestV2XBusTail(unittest.TestCase):
    def setUp(self):
        # We need to add backend to sys.path to import v2x
        sys.path.append(os.path.join(os.getcwd(), 'backend'))

        # Use patch.dict for sys.modules to prevent global pollution
        self.modules_patcher = patch.dict('sys.modules', {
            'numpy': MagicMock(),
            'simulator': MagicMock()
        })
        self.modules_patcher.start()

        from v2x import V2XBus

        self.mock_sim = MagicMock()
        self.mock_sim.tls = {}
        self.bus = V2XBus(self.mock_sim, max_log=10)

    def tearDown(self):
        self.modules_patcher.stop()
        if os.path.join(os.getcwd(), 'backend') in sys.path:
            sys.path.remove(os.path.join(os.getcwd(), 'backend'))

    def test_tail_empty_log(self):
        """Test tail when the log is empty."""
        self.assertEqual(self.bus.tail(5), [])

    def test_tail_fewer_than_n_messages(self):
        """Test tail when there are fewer messages than requested."""
        for i in range(3):
            self.bus._push("TEST", "SRC", {"i": i})

        result = self.bus.tail(5)
        self.assertEqual(len(result), 3)
        self.assertEqual(result[0]['p']['i'], 0)
        self.assertEqual(result[-1]['p']['i'], 2)

    def test_tail_more_than_n_messages(self):
        """Test tail when there are more messages than requested."""
        for i in range(8):
            self.bus._push("TEST", "SRC", {"i": i})

        result = self.bus.tail(5)
        self.assertEqual(len(result), 5)
        # Should return messages 3, 4, 5, 6, 7
        self.assertEqual(result[0]['p']['i'], 3)
        self.assertEqual(result[-1]['p']['i'], 7)

    def test_tail_default_n(self):
        """Test tail with default value for n (50)."""
        # Since our max_log is 10, we can only test up to 10
        for i in range(10):
            self.bus._push("TEST", "SRC", {"i": i})

        result = self.bus.tail()
        self.assertEqual(len(result), 10)

    def test_tail_at_max_log(self):
        """Test tail when the log has reached max_log."""
        for i in range(15):
            self.bus._push("TEST", "SRC", {"i": i})

        # log should only have the last 10 messages (5 to 14)
        self.assertEqual(len(self.bus.log), 10)

        result = self.bus.tail(5)
        self.assertEqual(len(result), 5)
        # Should return messages 10, 11, 12, 13, 14
        self.assertEqual(result[0]['p']['i'], 10)
        self.assertEqual(result[-1]['p']['i'], 14)

if __name__ == '__main__':
    unittest.main()
