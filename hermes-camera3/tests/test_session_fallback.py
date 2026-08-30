import unittest
from unittest.mock import patch

from src.camera.session_manager import SessionPhase, WeighbridgeSessionManager


class TestSessionErrorFallback(unittest.TestCase):
    def setUp(self):
        self.sm = WeighbridgeSessionManager()

    def tearDown(self):
        self.sm.reset_session()

    @patch("src.camera.session_manager.send_frame_to_anpr_server")
    @patch("src.camera.session_manager.fetch_image_bytes")
    def test_session_forwards_error_code_and_includes_truck_image(self, mock_fetch, mock_send):
        fake_truck_frame = b"\xff\xd8\xff\xe0\x00\x10JFIF_TRUCK_FRAME"
        mock_fetch.return_value = fake_truck_frame
        # Argus rejected due to human detected
        mock_send.return_value = (None, "REJECTED_HUMAN_DETECTED")

        self.sm.start_session()
        self.assertEqual(self.sm.phase, SessionPhase.PHASE_STABILIZING)

        # Simulate 2 ANPR frames captured with rejection
        self.sm._cam1_frames.append(fake_truck_frame)
        self.sm._cam1_statuses.append("REJECTED_HUMAN_DETECTED")

        # Weight stabilized
        self.sm.on_weight_stabilized(36500.0)
        self.sm._post_stability_start_time = 0.0  # Force timeout expired

        pkg = self.sm.check_session_progress()
        self.assertIsNotNone(pkg)
        assert pkg is not None

        # 1. Verify exact error code is assigned as the plate string
        self.assertEqual(pkg["anpr_plate"], "REJECTED_HUMAN_DETECTED")
        self.assertEqual(pkg["weight"], 36500.0)

        # 2. Verify the truck image is still present in the final package
        self.assertEqual(pkg["cam1_final_image"], fake_truck_frame)

    @patch("src.camera.session_manager.send_frame_to_anpr_server")
    @patch("src.camera.session_manager.fetch_image_bytes")
    def test_session_prefers_valid_plate_over_transient_errors(self, mock_fetch, mock_send):
        fake_truck_frame = b"\xff\xd8\xff\xe0\x00\x10JFIF_TRUCK_FRAME"
        mock_fetch.return_value = fake_truck_frame

        self.sm.start_session()

        # Simulate 1 transient error frame and 2 valid plate frames
        self.sm._cam1_frames.append(fake_truck_frame)
        self.sm._cam1_statuses.append("NO_PLATE_DETECTED")

        self.sm._cam1_plates.append("RJ09GA0165")
        self.sm._cam1_statuses.append("SUCCESS")

        self.sm._cam1_plates.append("RJ09GA0165")
        self.sm._cam1_statuses.append("SUCCESS")

        self.sm.on_weight_stabilized(42000.0)
        self.sm._post_stability_start_time = 0.0

        pkg = self.sm.check_session_progress()
        self.assertIsNotNone(pkg)
        assert pkg is not None

        # Plate winner should be the valid plate
        self.assertEqual(pkg["anpr_plate"], "RJ09GA0165")
        self.assertEqual(pkg["cam1_final_image"], fake_truck_frame)


if __name__ == "__main__":
    unittest.main()
