.PHONY: all verify clean
all:
	latexmk -pdf Quantum_Computing_Advantage.tex

verify:
	python3 verify_examples.py
	python3 research_checks.py
	python3 verify_boost.py
	python3 verify_case_study.py
	python3 verify_measurements.py
	python3 verify_readout_patch.py
	python3 verify_operational_frontier.py
	python3 verify_interval_certificates.py
	python3 verify_foundational_tools.py
	python3 verify_joint_frontier.py
	python3 verify_refinements.py
	python3 verify_table_rates.py
	python3 verify_deep_extensions.py
	python3 verify_sharp_characterization.py

clean:
	latexmk -c Quantum_Computing_Advantage.tex
