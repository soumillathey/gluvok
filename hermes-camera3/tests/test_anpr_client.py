import unittest
from unittest.mock import patch, MagicMock
import requests

from src.camera.anpr_client import send_frame_to_anpr_server, get_highest_frequency_plate


class TestANPRClient(unittest.TestCase):
    def test_send_frame_empty_bytes(self):
        result = send_frame_to_anpr_server(b"")
        self.assertIsNone(result)

        result_none = send_frame_to_anpr_server(None)
        self.assertIsNone(result_none)

    @patch("src.camera.anpr_client.requests.post")
    def test_send_frame_argus_success(self, mock_post):
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "success": True,
            "rejected": False,
            "status": "success",
            "status_message": "License plate successfully detected and recognized on car via docling.",
            "vehicle_detected": True,
            "vehicle_type": "car",
            "human_detected": False,
            "filename": "frame.jpg",
            "provider": "docling",
            "results": [
                {
                    "plate": "RJ09GA0165",
                    "state": "Rajasthan",
                    "raw_text": "RJ09GA0165"
                }
            ],
            "execution_time_ms": 115.4
        }
        mock_post.return_value = mock_response

        fake_img = b"\xff\xd8\xff\xe0\x00\x10JFIF"
        result = send_frame_to_anpr_server(fake_img, server_url="http://127.0.0.1:8000/recognize")

        self.assertEqual(result, "RJ09GA0165")
        mock_post.assert_called_once()
        args, kwargs = mock_post.call_args
        self.assertEqual(args[0], "http://127.0.0.1:8000/recognize")
        self.assertIn("file", kwargs["files"])

    @patch("src.camera.anpr_client.requests.post")
    def test_send_frame_argus_rejected_human(self, mock_post):
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "success": False,
            "rejected": True,
            "status": "rejected_human_detected",
            "status_message": "Pre-screening policy rejected: Human presence detected.",
            "vehicle_detected": True,
            "vehicle_type": "car",
            "human_detected": True,
            "filename": "frame.jpg",
            "provider": "docling",
            "results": [],
            "execution_time_ms": 42.1
        }
        mock_post.return_value = mock_response

        result = send_frame_to_anpr_server(b"fake-bytes")
        self.assertIsNone(result)

    @patch("src.camera.anpr_client.requests.post")
    def test_send_frame_argus_no_plate(self, mock_post):
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "success": False,
            "rejected": False,
            "status": "no_plate_detected",
            "status_message": "No readable license plate characters recognized.",
            "vehicle_detected": True,
            "vehicle_type": "truck",
            "human_detected": False,
            "filename": "frame.jpg",
            "provider": "docling",
            "results": [],
            "execution_time_ms": 85.0
        }
        mock_post.return_value = mock_response

        result = send_frame_to_anpr_server(b"fake-bytes")
        self.assertIsNone(result)

    @patch("src.camera.anpr_client.requests.post")
    def test_send_frame_fallback_flat_json(self, mock_post):
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "plate": "MH12AB1234",
            "confidence": 0.95
        }
        mock_post.return_value = mock_response

        result = send_frame_to_anpr_server(b"fake-bytes")
        self.assertEqual(result, "MH12AB1234")

    @patch("src.camera.anpr_client.requests.post")
    def test_send_frame_timeout(self, mock_post):
        mock_post.side_effect = requests.exceptions.Timeout("Read timeout")
        result = send_frame_to_anpr_server(b"fake-bytes")
        self.assertIsNone(result)

    @patch("src.camera.anpr_client.requests.post")
    def test_send_frame_connection_error(self, mock_post):
        mock_post.side_effect = requests.exceptions.ConnectionError("Connection refused")
        result = send_frame_to_anpr_server(b"fake-bytes")
        self.assertIsNone(result)

    def test_highest_frequency_voting(self):
        samples = ["MH12AB1234", "MH12AB1234", "MH12AB1234", "MH12AB1235", "DL01AB9999"]
        winner = get_highest_frequency_plate(samples)
        self.assertEqual(winner, "MH12AB1234")

    def test_highest_frequency_empty_list(self):
        winner = get_highest_frequency_plate([])
        self.assertEqual(winner, "UNKNOWN_PLATE")

        winner_none = get_highest_frequency_plate([None, "", "   "])
        self.assertEqual(winner_none, "UNKNOWN_PLATE")


if __name__ == "__main__":
    unittest.main()
