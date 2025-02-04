from scan_to_paperless.code import _is_rectangular


def test_square():
    # Perfect square geometry
    geometry = [(0.0, 0.0), (0.0, 1.0), (1.0, 1.0), (1.0, 0.0)]
    assert _is_rectangular(geometry) is True


def test_rectangle():
    # Rectangle geometry
    geometry = [(0.0, 0.0), (0.0, 2.0), (1.0, 2.0), (1.0, 0.0)]
    assert _is_rectangular(geometry) is True


def test_not_rectangular():
    # Triangle geometry (3 points)
    geometry = [(0.0, 0.0), (1.0, 1.0), (0.0, 1.0)]
    assert _is_rectangular(geometry) is False

    # Irregular quadrilateral with non-90° angles
    geometry = [(0.0, 0.0), (0.0, 1.0), (1.0, 0.5), (1.0, 0.0)]
    assert _is_rectangular(geometry) is False


def test_empty_geometry():
    assert _is_rectangular([]) is False


def test_rotated_rectangle():
    # Rectangle rotated 45 degrees
    geometry = [(0.5, 0.0), (0.0, 0.5), (0.5, 1.0), (1.0, 0.5)]
    assert _is_rectangular(geometry) is True
