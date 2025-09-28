# RaySR (“Racer”): Super-Resolution for Radio Maps

<div align="center">
<img src="https://img.shields.io/badge/python-3.11-blue.svg" />
<img src="https://img.shields.io/badge/build-passing-brightgreen.svg" />
<img src="https://img.shields.io/badge/license-apache--2.0-blue.svg" />
</div>

:signal_strength: This project aims to enhance Radio Maps with Super Resolution and is inspired by [DLSS](https://www.nvidia.com/en-gb/geforce/technologies/dlss/). The project is part of the course [DD2430](https://www.kth.se/student/kurser/kurs/DD2430?l=en) at KTH and is done in collaboration with Ericsson.

<p align="center">
  <img src="assets/etoile_grid.png" width="45%" alt="Radio Map Scene">
  <img src="assets/san_francisco_mesh.png" width="45%" alt="Radio Map Example">
    <br>
    <em>Left:</em> Example of a radio map scene with transmitters (red). <em>Right:</em> Example of a radio map over San Francisco.
    <br>
  <img src="assets/high_res.png" width="45%" alt="High Resolution">
  <img src="assets/low_res.png" width="45%" alt="Low Resolution">
    <br>
    <em>Left:</em> High-resolution radio map. <em>Right:</em> Example of "downsampled" low-resolution radio map.
</p>

## Setup
1. **Install** [uv](https://github.com/astral-sh/uv).
2. **Sync environment**: ``uv sync``
3. **Run commands**: ``uv run <command>``
4. **(Dev)**: Install pre-commit hooks: ``pre-commit install``

## Environment Instructions
- To add dependencies, add them to `pyproject.toml` and run `uv sync`.
- Optionally, you can use `uv add <package>` to add a package and sync the environment, e.g. `uv add 'requests==2.31.0'`.
