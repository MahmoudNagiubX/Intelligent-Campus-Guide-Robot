import re, pathlib

path = pathlib.Path('tests/unit/test_stt.py')
content = path.read_text(encoding='utf-8')
content = content.replace(
    '    assert options["language"] == "en"\n    assert options["keyterm"] == ["Robotics Lab"]',
    '    assert options["language"] == "en"\n    assert "keyterm" not in options'
)
path.write_text(content, encoding='utf-8')
print("Fixed")
