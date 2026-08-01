"""Ways to drive the application.

An interface parses input, calls exactly one service, and presents the result.
Anything an interface would have to *decide* belongs in `services` instead —
that is what keeps the CLI and the web UI from drifting apart.
"""
