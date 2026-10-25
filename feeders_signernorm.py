"""Import shim that keeps the configs' feeder path valid.

The signer-normalized configs (config/signernorm/) load `feeder: feeders_signernorm.SignerNormFeeder`;
main.import_class resolves dotted paths attribute by attribute from the top-level package, and
feeders/__init__.py does not import the feeder_signernorm submodule. Implementation:
feeders/feeder_signernorm.py.
"""
from feeders.feeder_signernorm import SignerNormFeeder  # noqa: F401
