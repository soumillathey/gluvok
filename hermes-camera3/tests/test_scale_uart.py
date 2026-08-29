import unittest
import time
from unittest.mock import patch
from src.scale.scale_uart import ScaleUARTReader

class TestScaleUARTReader(unittest.TestCase):
    def setUp(self):
        self.reader = ScaleUARTReader()
        self.parsed_weights = []

    def _mock_processed_weight(self, val: float):
        self.parsed_weights.append(val)

    def test_complete_packet(self):
        with patch('src.scale.scale_uart.handle_scale_char_processed', side_effect=self._mock_processed_weight):
            self.reader.handle_scale_char("26500MN\r\n")
            self.assertEqual(len(self.parsed_weights), 1)
            self.assertEqual(self.parsed_weights[0], 26500.0)

    def test_split_packet(self):
        with patch('src.scale.scale_uart.handle_scale_char_processed', side_effect=self._mock_processed_weight):
            # First chunk: partial packet without terminator
            self.reader.handle_scale_char("265")
            self.assertEqual(len(self.parsed_weights), 0)

            # Second chunk: remaining packet with CR terminator
            self.reader.handle_scale_char("00MN\r")
            self.assertEqual(len(self.parsed_weights), 1)
            self.assertEqual(self.parsed_weights[0], 26500.0)

    def test_inter_char_timeout_flush(self):
        with patch('src.scale.scale_uart.handle_scale_char_processed', side_effect=self._mock_processed_weight):
            # Send packet without terminator
            self.reader.handle_scale_char("26500MN")
            self.assertEqual(len(self.parsed_weights), 0)

            # Before timeout (no sleep), check timeout should not flush
            self.reader._check_timeout_flush()
            self.assertEqual(len(self.parsed_weights), 0)

            # Wait > 300ms (0.35s) for inter-character timeout
            time.sleep(0.35)
            self.reader._check_timeout_flush()
            self.assertEqual(len(self.parsed_weights), 1)
            self.assertEqual(self.parsed_weights[0], 26500.0)

    def test_terminators_stx_etx(self):
        with patch('src.scale.scale_uart.handle_scale_char_processed', side_effect=self._mock_processed_weight):
            # Test STX / ETX terminators
            self.reader.handle_scale_char(b"\x0205000MN\x03")
            self.assertEqual(len(self.parsed_weights), 1)
            self.assertEqual(self.parsed_weights[0], 5000.0)

    def test_flexible_float_parsing(self):
        with patch('src.scale.scale_uart.handle_scale_char_processed', side_effect=self._mock_processed_weight):
            # Standard float with decimal
            self.reader.handle_scale_char("ST,GS,+05000.5kg\r\n")
            self.assertIn(5000.5, self.parsed_weights)

            # Negative float
            self.reader.handle_scale_char("-12.5\r\n")
            self.assertIn(-12.5, self.parsed_weights)

if __name__ == '__main__':
    unittest.main()
