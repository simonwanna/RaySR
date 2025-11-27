# RaySR (“Racer”): Super-Resolution for Radio Maps

<div align="center">
<img src="https://img.shields.io/badge/python-3.11-blue.svg" />
<img src="https://img.shields.io/badge/build-passing-brightgreen.svg" />
<img src="https://img.shields.io/badge/license-apache--2.0-blue.svg" />
</div>

:signal_strength: This project aims to enhance Radio Maps with Super Resolution and is inspired by [DLSS](https://www.nvidia.com/en-gb/geforce/technologies/dlss/). The project is part of the course [DD2430](https://www.kth.se/student/kurser/kurs/DD2430?l=en) at KTH and is done in collaboration with Ericsson.

<p align="center">
  <img src="assets/etoile.png" width="45%" alt="Radio Map Scene">
  <img src="assets/san_francisco_mesh.png" width="45%" alt="Radio Map Example">
    <br>
    <em>Left:</em> Example of a radio map data sample with transmitters (blue). <em>Right:</em> Example of a radio map over San Francisco with transmitters (red).
    <br>
</p>

## Setup
1. **Install** [uv](https://github.com/astral-sh/uv) by running ``curl -LsSf https://astral.sh/uv/install.sh | sh``.
2. **Sync environment**: ``uv sync``
3. **Run commands**: ``uv run <command>``
4. **(Dev)**: Install pre-commit hooks: ``pre-commit install``
- Might need `LLVM` installed... 

## Running the POC
See [src/poc/readme.md](src/poc/readme.md) for detailed instructions on how to run the proof of concept code.
Available commands:
- `uv run generate`: Generate synthetic radio map data.
- `uv run train`: Train the super-resolution model.
- `uv run test`: Evaluate the model.