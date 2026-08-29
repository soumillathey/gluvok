from app.schemas.plate import (
    PlateResult,
    ProviderEnum,
    RecognitionResponse,
    RecognitionStatusEnum,
)


def test_provider_enum_values():
    assert ProviderEnum.PLATERECOGNIZER.value == "platerecognizer"
    assert ProviderEnum.NVIDIA.value == "nvidia"
    assert ProviderEnum.DOCLING.value == "docling"


def test_recognition_status_enum():
    assert RecognitionStatusEnum.SUCCESS.value == "success"
    assert RecognitionStatusEnum.REJECTED_HUMAN_DETECTED.value == "rejected_human_detected"
    assert RecognitionStatusEnum.REJECTED_NO_FOUR_WHEELER.value == "rejected_no_four_wheeler"
    assert RecognitionStatusEnum.REJECTED_MULTIPLE_VEHICLES.value == "rejected_multiple_vehicles"
    assert RecognitionStatusEnum.NO_PLATE_DETECTED.value == "no_plate_detected"


def test_plate_result_valid():
    res = PlateResult(plate="RJ09GA0165", state="Rajasthan")
    assert res.plate == "RJ09GA0165"
    assert res.state == "Rajasthan"


def test_recognition_response_valid():
    resp = RecognitionResponse(
        success=True,
        status=RecognitionStatusEnum.SUCCESS,
        status_message="Plate detected",
        vehicle_detected=True,
        vehicle_type="car",
        human_detected=False,
        filename="test.jpg",
        provider=ProviderEnum.DOCLING,
        results=[PlateResult(plate="RJ09GA0165", state="Rajasthan")],
        execution_time_ms=123.45,
    )
    assert resp.success is True
    assert resp.status == RecognitionStatusEnum.SUCCESS
    assert len(resp.results) == 1
    assert resp.results[0].plate == "RJ09GA0165"
