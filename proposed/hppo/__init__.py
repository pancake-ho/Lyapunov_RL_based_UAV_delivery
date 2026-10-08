"""Hierarchical PPO (frame PPO + slot PPO) on the P3 system model.

The physical model is imported from ``config_p3`` and ``env/p3`` unchanged.
This package only adds: an episodic two-timescale environment wrapper with
action projection/masking, two factorized PPO agents, detailed history
logging, and an offline trace verifier.
"""
