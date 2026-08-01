"""photosort — an offline photo indexer and sorter.

The packages are layered, innermost first. Each may only import from the ones
above it in this list:

  errors, imaging, filesystem   cross-cutting basics
  domain/       rules, decisions, value types — pure, no I/O, no dependencies
  config/       what the user chose, grouped by concern
  storage/      the SQLite index, one repository per table
  scanning/     the filesystem, indexed incrementally
  detecting/    adapters over the two inference engines
  analyzing/
  actions/      what happens to a photo once a rule matches it
  pipeline/     how the heavy stages are driven, concurrently and resumably
  services/     one function per use case — the only layer a front end calls
  interfaces/   the CLI and the web UI: parsing and presentation only

Nothing in the first ten knows how it is being invoked, and nothing outside
`interfaces` knows how anything is displayed.
"""

__version__ = "0.2.0"
