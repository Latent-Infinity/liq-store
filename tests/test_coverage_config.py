from pathlib import Path

from coverage import Coverage


def test_typed_implementation_is_counted_and_standalone_stub_is_excluded(tmp_path: Path) -> None:
    source = tmp_path / "nonfinancial_coverage_control.py"
    source.write_text(
        "def count(values: tuple[int, ...]) -> int:\n"
        "    return len(values)\n"
        "\n"
        "class Interface:\n"
        "    def call(self):\n"
        "        ...\n"
    )
    config = Path(__file__).parents[1] / "pyproject.toml"
    _, statements, excluded, _, _ = Coverage(config_file=str(config)).analysis2(str(source))
    assert 2 in statements
    assert 2 not in excluded
    assert 6 in excluded
