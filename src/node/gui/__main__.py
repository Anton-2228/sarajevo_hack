"""`python -m node.gui` -- handy on Windows, where the installed entry point is
bound to pythonw.exe and shows no console to debug in."""

from node.gui.app import main

raise SystemExit(main())
