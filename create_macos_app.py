"""
Deprecated: use create_app.py instead.

This file is kept for backwards compatibility and simply forwards all
arguments to create_app.py, which handles macOS, Linux, and Windows.
"""
import runpy, sys

if __name__ == "__main__":
    runpy.run_path("create_app.py", run_name="__main__")
