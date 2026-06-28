import pytest


@pytest.fixture(autouse=True)
def clear_module_cache():
    """Clear module caches so each test gets a fresh import."""
    modules_to_clear = [
        "commands.generate",
        "commands.init",
        "commands.curate",
        "commands.prepare",
        "commands.upload",
        "commands.train",
        "commands.evaluate",
        "commands.fuse",
        "commands.chat",
        "data.synthetic.config",
        "data.synthetic.teacher",
        "data.synthetic.embedder",
        "data.synthetic.pipeline",
        "data.synthetic.io",
        "commands",
    ]
    for mod in modules_to_clear:
        import sys
        sys.modules.pop(mod, None)
    yield
    # After test, also clear these to keep isolation
    for mod in modules_to_clear:
        import sys
        sys.modules.pop(mod, None)
