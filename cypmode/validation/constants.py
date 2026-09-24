"""Shared constants with no heavy dependencies of their own.

Split out so docs/figures/generate_validation_figure.py (matplotlib only)
can import COORDINATION_DISTANCE_CUTOFF_A without pulling in Biopython,
which cypmode/validation/structures.py needs but this figure script
doesn't.
"""

# Dative Fe-N coordination bonds in solved Type II CYP structures run
# ~2.0-2.3 A (Poulos & Johnson, ch. 3 in Ortiz de Montellano (ed.),
# Cytochrome P450: Structure, Mechanism, and Biochemistry, 3rd ed., 2005).
# A margin above that range distinguishes a coordinating pose from one
# where the ligand is merely nearby in the pocket.
COORDINATION_DISTANCE_CUTOFF_A = 2.6
