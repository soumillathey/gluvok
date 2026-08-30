import json
import time
import unittest
import urllib.error
import urllib.request

from src.config.config_manager import config
from src.web.server import FallbackWebServer


class TestFallbackWebServer(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # Start test web server on non-standard test port
        cls.test_port = 8899
        cls.server = FallbackWebServer(host="127.0.0.1", port=cls.test_port)
        cls.server.start()
        time.sleep(0.3)  # Allow socket to bind

    @classmethod
    def tearDownClass(cls):
        cls.server.stop()

    def test_get_index_html(self):
        url = f"http://127.0.0.1:{self.test_port}/"
        with urllib.request.urlopen(url, timeout=3.0) as res:
            self.assertEqual(res.status, 200)
            content_type = res.headers.get("Content-Type", "")
            self.assertIn("text/html", content_type)
            html_body = res.read().decode("utf-8")
            self.assertIn("Gluvok Hermes", html_body)
            self.assertIn("tailwindcss", html_body)
            self.assertIn("NO_PLATE_DETECTED", html_body)
            self.assertIn("REJECTED_HUMAN_DETECTED", html_body)

    def test_subsystem_page_routes(self):
        for route in ("/scale", "/anpr", "/cloud", "/wifi", "/telemetry"):
            url = f"http://127.0.0.1:{self.test_port}{route}"
            with urllib.request.urlopen(url, timeout=3.0) as res:
                self.assertEqual(res.status, 200)
                content_type = res.headers.get("Content-Type", "")
                self.assertIn("text/html", content_type)




    def test_get_api_status(self):
        url = f"http://127.0.0.1:{self.test_port}/api/status"
        with urllib.request.urlopen(url, timeout=3.0) as res:
            self.assertEqual(res.status, 200)
            data = json.loads(res.read().decode("utf-8"))
            self.assertEqual(data["status"], "healthy")
            self.assertIn("scale", data)
            self.assertIn("argus", data)
            self.assertIn("supabase", data)
            self.assertIn("events", data)

    def test_post_api_wifi_success(self):
        url = f"http://127.0.0.1:{self.test_port}/api/wifi"
        payload = json.dumps({"ssid": "TestRouter_5G", "password": "SecretPassword123"}).encode("utf-8")
        req = urllib.request.Request(url, data=payload, headers={"Content-Type": "application/json"})

        with urllib.request.urlopen(req, timeout=3.0) as res:
            self.assertEqual(res.status, 200)
            data = json.loads(res.read().decode("utf-8"))
            self.assertTrue(data.get("success"))

        self.assertEqual(config.wifi_ssid, "TestRouter_5G")
        self.assertEqual(config.wifi_password, "SecretPassword123")

    def test_post_api_wifi_empty_ssid_error(self):
        url = f"http://127.0.0.1:{self.test_port}/api/wifi"
        payload = json.dumps({"ssid": "", "password": "password"}).encode("utf-8")
        req = urllib.request.Request(url, data=payload, headers={"Content-Type": "application/json"})

        with self.assertRaises(urllib.error.HTTPError) as ctx:
            urllib.request.urlopen(req, timeout=3.0)
        self.assertEqual(ctx.exception.code, 400)

    def test_post_api_wifi_clear(self):
        url = f"http://127.0.0.1:{self.test_port}/api/wifi/clear"
        req = urllib.request.Request(url, data=b"{}", headers={"Content-Type": "application/json"})

        with urllib.request.urlopen(req, timeout=3.0) as res:
            self.assertEqual(res.status, 200)
            data = json.loads(res.read().decode("utf-8"))
            self.assertTrue(data.get("success"))

        self.assertEqual(config.wifi_ssid, "")
        self.assertEqual(config.wifi_password, "")

    def test_404_not_found(self):
        url = f"http://127.0.0.1:{self.test_port}/unknown_endpoint"
        with self.assertRaises(urllib.error.HTTPError) as ctx:
            urllib.request.urlopen(url, timeout=3.0)
        self.assertEqual(ctx.exception.code, 404)


if __name__ == "__main__":
    unittest.main()
