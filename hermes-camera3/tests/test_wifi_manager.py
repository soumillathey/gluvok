import unittest
from unittest.mock import MagicMock, patch

from src.network.wifi_manager import (
    connect_to_wifi,
    is_hotspot_active,
    is_wifi_connected,
    start_emergency_hotspot,
    start_wifi_watchdog,
    stop_emergency_hotspot,
    stop_wifi_watchdog,
)


class TestWiFiManager(unittest.TestCase):
    def tearDown(self):
        stop_wifi_watchdog()
        stop_emergency_hotspot()

    @patch("src.network.wifi_manager.is_nmcli_available", return_value=True)
    @patch("src.network.wifi_manager.subprocess.run")
    def test_is_wifi_connected_true(self, mock_run, mock_nmcli):
        mock_res = MagicMock()
        mock_res.returncode = 0
        mock_res.stdout = "wifi:connected:Office_5G\nethernet:connected:Wired 1\n"
        mock_run.return_value = mock_res

        self.assertTrue(is_wifi_connected())

    @patch("src.network.wifi_manager.is_nmcli_available", return_value=True)
    @patch("src.network.wifi_manager.subprocess.run")
    def test_is_wifi_connected_false_when_disconnected(self, mock_run, mock_nmcli):
        mock_res = MagicMock()
        mock_res.returncode = 0
        mock_res.stdout = "wifi:disconnected:\nethernet:connected:Wired 1\n"
        mock_run.return_value = mock_res

        self.assertFalse(is_wifi_connected())

    @patch("src.network.wifi_manager.is_nmcli_available", return_value=True)
    @patch("src.network.wifi_manager.subprocess.run")
    def test_is_wifi_connected_false_when_in_hotspot_mode(self, mock_run, mock_nmcli):
        mock_res = MagicMock()
        mock_res.returncode = 0
        mock_res.stdout = "wifi:connected:Gluvok-Hotspot\n"
        mock_run.return_value = mock_res

        # If connected connection is the hotspot itself, it shouldn't count as normal wifi
        self.assertFalse(is_wifi_connected())

    @patch("src.network.wifi_manager.is_nmcli_available", return_value=True)
    @patch("src.network.wifi_manager.subprocess.run")
    def test_start_and_stop_emergency_hotspot(self, mock_run, mock_nmcli):
        mock_res = MagicMock()
        mock_res.returncode = 0
        mock_run.return_value = mock_res

        # Start hotspot
        success = start_emergency_hotspot("Gluvok-Setup", "gluvok1234")
        self.assertTrue(success)
        self.assertTrue(is_hotspot_active())

        # Stop hotspot
        stopped = stop_emergency_hotspot()
        self.assertTrue(stopped)
        self.assertFalse(is_hotspot_active())

    @patch("src.network.wifi_manager.is_nmcli_available", return_value=True)
    @patch("src.network.wifi_manager.subprocess.run")
    def test_connect_to_wifi_success(self, mock_run, mock_nmcli):
        mock_res = MagicMock()
        mock_res.returncode = 0
        mock_res.stdout = "Device 'wlan0' successfully activated with 'Office_5G'."
        mock_run.return_value = mock_res

        success, msg = connect_to_wifi("Office_5G", "password123")
        self.assertTrue(success)
        self.assertIn("Successfully connected", msg)

    @patch("src.network.wifi_manager.is_nmcli_available", return_value=True)
    @patch("src.network.wifi_manager.subprocess.run")
    def test_connect_to_wifi_failure(self, mock_run, mock_nmcli):
        mock_res = MagicMock()
        mock_res.returncode = 1
        mock_res.stderr = "Error: Secrets were required, but not provided."
        mock_run.return_value = mock_res

        success, msg = connect_to_wifi("Office_5G", "wrongpassword")
        self.assertFalse(success)
        self.assertIn("Failed to connect", msg)

    def test_connect_to_wifi_empty_ssid(self):
        success, msg = connect_to_wifi("", "password")
        self.assertFalse(success)
        self.assertIn("cannot be empty", msg)

    def test_watchdog_start_stop(self):
        start_wifi_watchdog(interval=0.5)
        stop_wifi_watchdog()


if __name__ == "__main__":
    unittest.main()
