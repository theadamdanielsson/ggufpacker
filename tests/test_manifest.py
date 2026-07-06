

def test_format_tag_is_stable() -> None:
    """The on-disk format tag is part of the artifact, not the branding.

    It must never change with a project rename: existing .ggufpack stores
    carry this exact string, and changing it silently orphans them all
    (this regressed once, during the ggufpack -> ggufpacker rename).
    """
    from ggufpacker.manifest import FORMAT

    assert FORMAT == "ggufpack/0"
