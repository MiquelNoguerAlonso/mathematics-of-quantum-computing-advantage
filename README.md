# The Mathematics of Quantum Computing Advantage

*Foundations, measurement, and operational cost frontiers*  
Miquel Noguer i Alonso · Artificial Intelligence Finance Institute

[Read the manuscript (PDF)](Quantum_Computing_Advantage.pdf) · [LaTeX source](Quantum_Computing_Advantage.tex) · [Verification guide](README.txt)

The paper studies the cost of distinguishing quantum states when measurements, preparation, and circuit depth carry explicit prices. For a family of qubit experiments with two readouts, it gives the minimum worst-case expected cost at each error tolerance. A separate analysis bounds the cost of sequential discrimination for a noisy binary phase oracle.

## Build the manuscript

With pdfLaTeX, BibTeX, and latexmk installed, run the following command from the repository root.

```sh
make
```

## Reproduce the numerical checks

The validation environment used Python 3.12. In a Python virtual environment, install the pinned dependencies and run all fourteen verification scripts.

```sh
python3 -m pip install -r requirements.txt
make verify
```

For the exact readout-price frontier alone, run `python3 verify_sharp_characterization.py`. Its [results](sharp_characterization_results.json) include the rational optimal costs and stopping thresholds. The [validation manifest](validation_manifest.json) lists the checks and file hashes for the manuscript package.
