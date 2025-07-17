#!/usr/bin/env python3
"""
Simple entry point for the Audio Evaluation Pipeline.

This is a convenience script that imports and runs the main CLI function.
For development and direct execution without installing the package.
"""

if __name__ == "__main__":
    from audio_evals.cli import main
    main()