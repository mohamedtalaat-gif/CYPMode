# SPDX-FileCopyrightText: 2026 Muhammad Talaat
# SPDX-License-Identifier: AGPL-3.0-or-later
# Additional terms under section 7(b) of the licence: see NOTICE.

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

# Human CYP3A4, UniProt P08684. The default target for
# cypmode.validation.pipeline -- pass a different sequence for a different
# protein/construct, and note that a different construct can shift residue
# numbering (find_coordinating_cysteine() re-derives the axial cysteine's
# position from whatever sequence it's given, rather than assuming this
# one's numbering carries over).
CYP3A4_SEQUENCE = (
    "MALIPDLAMETWLLLAVSLVLLYLYGTHSHGLFKKLGIPGPTPLPFLGNILSYHKGFCMFDMECHKKYGKVWGFYDGQQPVLAITDPDMIKTVLVKECYSVFTNRRPFGPVGFMKSAISIAEDEEW"
    "KRLRSLLSPTFTSGKLKEMVPIIAQYGDVLVRNLRREAETGKPVTLKDVFGAYSMDVITSTSFGVNIDSLNNPQDPFVENTKKLLRFDFLDPFFLSITVFPFLIPILEVLNICVFPREVTNFLRKSVKR"
    "MKESRLEDTQKHRVDFLQLMIDSQNSKETESHKALSDLELVAQSIIFIFAGYETTSSVLSFIMYELATHPDVQQKLQEEIDAVLPNKAPPTYDTVLQMEYLDMVVNETLRLFPIAMRLERVCKKDVEIN"
    "GMFIPKGVVVMIPSYALHRDPKYWTEPEKFLPERFSKKNKDNIDPYIYTPFGSGPRNCIGMRFALMNMKLALIRVLQNFSFKPCKETQIPLKLSLGGLLQPEKPVVLKVESRDGTVSGA"
)

# Human CYP2D6, UniProt P10635 -- fetched directly from UniProt's own REST
# API (rest.uniprot.org/uniprotkb/P10635.fasta), 497 residues, confirmed
# matching the real, published length. find_coordinating_cysteine() on
# this real sequence locates a single, unambiguous match at position 443
# (Cys443) -- the same real, generic conserved P450 heme-binding motif
# search already used for CYP3A4's Cys442, no CYP2D6-specific
# special-casing needed. Use with cypmode.validation.pipeline's
# target="cyp2d6" (see that module's own docstring for why target matters --
# it was a hardcoded "cyp3a4" literal in file/directory naming until this
# session, which would have silently collided a CYP2D6 run's files with
# CYP3A4's own).
CYP2D6_SEQUENCE = (
    "MGLEALVPLAVIVAIFLLLVDLMHRRQRWAARYPPGPLPLPGLGNLLHVDFQNTPYCFDQLRRRFGDVFSLQLAWTPVVVLNGLAAVREALVTHGEDTADRPPVPITQILGFGPRSQGVFLARYGPAWREQ"
    "RRFSVSTLRNLGLGKKSLEQWVTEEAACLCAAFANHSGRPFRPNGLLDKAVSNVIASLTCGRRFEYDDPRFLRLLDLAQEGLKEESGFLREVLNAVPVLLHIPALAGKVLRFQKAFLTQLDELLTEHRMTW"
    "DPAQPPRDLTEAFLAEMEKAKGNPESSFNDENLRIVVADLFSAGMVTTSTTLAWGLLLMILHPDVQRRVQQEIDDVIGQVRRPEMGDQAHMPYTTAVIHEVQRFGDIVPLGVTHMTSRDIEVQGFRIPKG"
    "TTLITNLSSVLKDEAVWEKPFRFHPEHFLDAQGHFVKPEAFLPFSAGRRACLGEPLARMELFLFFTSLLQHFSFSVPTGQPRPSHHGVFAFLVSPSPYELCAVPR"
)
