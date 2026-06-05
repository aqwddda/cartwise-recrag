"""Compatibility wrapper for query LLM adapters.

New code should import from ``cartwise.query.llm``. This module remains for
legacy tests and scripts during the service-boundary migration.
"""

from cartwise.query.llm import *  # noqa: F403
