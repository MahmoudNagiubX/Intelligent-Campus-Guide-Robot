from app.pipeline.arabic_normalizer import normalize_arabic_transcript, normalize_room_reference


def test_empty_text_remains_empty() -> None:
    assert normalize_arabic_transcript("") == ""


def test_tatweel_removed() -> None:
    assert normalize_arabic_transcript("مـحـمـد") == "محمد"


def test_diacritics_removed() -> None:
    assert normalize_arabic_transcript("مَكْتَب") == "مكتب"


def test_alef_normalization() -> None:
    assert normalize_arabic_transcript("أين") == "اين"


def test_spoken_variant_mapping() -> None:
    assert normalize_arabic_transcript("فين المكتب") == "اين المكتب"


def test_latin_substrings_are_preserved() -> None:
    assert normalize_arabic_transcript("فين room 214") == "اين room 214"


def test_room_reference_arabic() -> None:
    assert normalize_room_reference("اوضة 214") == "room 214"


def test_room_reference_r_prefix() -> None:
    assert normalize_room_reference("r214") == "room 214"


def test_room_reference_lab_prefix() -> None:
    assert normalize_room_reference("lab 214") == "room 214"


def test_room_reference_room_number_prefix() -> None:
    assert normalize_room_reference("room no. 214") == "room 214"
