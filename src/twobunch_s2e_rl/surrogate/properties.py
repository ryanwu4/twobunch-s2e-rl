"""Moved to facet2-twobunch-cnf: this module IS `twobunch_cnf.properties` (same module object, private names included)."""

import sys

import twobunch_cnf.properties as _target

sys.modules[__name__] = _target
