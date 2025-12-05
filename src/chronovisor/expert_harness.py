"""
Expert Harness: Interface that experts expose to the controller.

Experts become passive sensors. They:
- Read through the shared lens surface
- Expose: local gain, tilt, stability score, out-of-tolerance flags
- Operate under whatever geometry the controller provides
"""
