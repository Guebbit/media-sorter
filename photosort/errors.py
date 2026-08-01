"""Every failure this application raises on purpose.

Collected in one place so the layers above can catch by intent — "the user's
configuration is wrong", "an inference engine is unusable" — without importing
the module that happens to detect the problem. Anything not derived from
`PhotosortError` is a bug, and is meant to reach a traceback.
"""

from __future__ import annotations


class PhotosortError(Exception):
    """Base class for every expected, reportable failure."""


class ConfigError(PhotosortError, ValueError):
    """A setting is missing or nonsensical."""


class RuleError(PhotosortError, ValueError):
    """A ruleset is malformed or refers to something that does not exist."""


class OutputNotWritable(ConfigError):
    """The output folder cannot be written to.

    Raised once, before any photo is touched: the same fault stops every file,
    and one clear failure beats the same warning repeated per photo.
    """


class IncompatibleIndex(PhotosortError, RuntimeError):
    """The index on disk was written by a different schema version."""


class EngineError(PhotosortError, RuntimeError):
    """An external inference engine was unreachable or returned nonsense.

    The pipeline catches this rather than the concrete adapter's error, which is
    what lets a stage keep running while a specific engine is down.
    """


class EngineUnavailable(EngineError):
    """The engine's dependencies are not installed in this environment."""


class VisionError(EngineError):
    """The vision-language engine failed on a request."""
