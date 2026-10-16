"""Force-field relaxation of a predicted ligand pose in its frozen pocket.

The PoseBusters paper reports that a short minimisation with a force field repairs most of the poses that a deep-learning docking model leaves invalid (clashes, strained
bonds and angles), and the challenge scorer gives a pose that fails any of its checks an LDDT-PLI of 0. This module relaxes the ligand of a pose in the submission format
(protein chain A, ligand residue ``LIG``, heme residue ``HEM``) and writes the same file with the new ligand coordinates:

* the protein and the heme are frozen, so no bonded protein or heme parameters are needed; they only supply nonbonded terms (charges and Lennard-Jones) to the ligand;
* protein: Amber ff14SB with hydrogens added by OpenMM's Modeller; the cysteine bound to the iron is the thiolate ``CYM``;
* heme: the nonbonded parameters of the CHARMM36 ``HEME`` template that OpenMM ships (heme b, 73 atoms, net charge -2). The pose holds heavy atoms only, so the charge of every
  heme hydrogen is folded into the heavy atom it is bonded to (the net charge stays -2) and the hydrogens carry no Lennard-Jones term. CHARMM gives the iron no Lennard-Jones
  term (epsilon 0), so a ligand atom is held at the iron by the restraint below;
* ligand: OpenFF 2.2.1 (Sage) with AM1-BCC-like partial charges from the NAGL graph network, hydrogens added from the heavy-atom coordinates;
* ligand-environment interaction: Lennard-Jones (Lorentz-Berthelot) plus Coulomb with a constant dielectric of 4, without cutoff (an interaction group of a ligand and a
  frozen environment of a few thousand atoms);
* restraints: a weak harmonic tether on the ligand's heavy atoms to their predicted positions (the relaxation repairs a pose, it does not re-dock it), and, when a ligand nitrogen
  coordinates the iron in the prediction (closer than 3.0 A), a harmonic restraint that keeps that distance at the predicted value.

The minimiser is OpenMM's L-BFGS on the CPU platform. Run it in the conda environment ``openmm`` (``environment/openmm-environment.yml``):

    PYTHONPATH=. ~/miniforge3/envs/openmm/bin/python -m cypmode.validation.relax --manifest jobs.json

``jobs.json`` is a list of ``{"pdb": ..., "smiles": ..., "out": ...}``; finished outputs are kept, so a second run only does what is missing, and a ``<out>.json`` file next to
every output records the energies and the movement of the ligand.
"""

from __future__ import annotations

import argparse
import json
import sys
import xml.etree.ElementTree as ET
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np

KCAL_PER_A2 = 418.4  # 1 kcal/mol/A^2 in kJ/mol/nm^2
TETHER_K = 2.0 * KCAL_PER_A2  # heavy atoms of the ligand to their predicted positions
FE_N_K = 500.0 * KCAL_PER_A2  # iron-nitrogen coordination at its predicted distance
FE_N_MAX_A = 3.0  # a ligand nitrogen closer to the iron than this counts as coordinating
DIELECTRIC = 4.0
COULOMB = 138.935456  # kJ nm / (mol e^2)
MAX_ITERATIONS = 5000
TOLERANCE = 0.5  # kJ/mol/nm
LIGAND_FORCE_FIELD = "openff-2.2.1.offxml"
CHARGE_MODEL = "openff-gnn-am1bcc-1.0.0.pt"


@dataclass
class RelaxResult:
    heavy: np.ndarray  # relaxed heavy-atom coordinates of the ligand in angstrom, in the order of the file
    energy_before: float  # kJ/mol, ligand + ligand-environment + restraints
    energy_after: float
    interaction_before: float  # ligand-environment energy alone
    interaction_after: float
    rmsd: float  # heavy-atom movement in angstrom
    fe_n_before: float  # angstrom (NaN when no ligand nitrogen coordinates the iron)
    fe_n_after: float

    def summary(self) -> dict:
        record = asdict(self)
        record.pop("heavy")
        return record


def heme_parameters() -> dict[str, tuple[float, float, float]]:
    """Heavy atom name -> (charge with the charges of its hydrogens folded in, sigma in nm, epsilon in kJ/mol) from the CHARMM36 ``HEME`` residue shipped with OpenMM."""
    import openmm.app as app

    root = ET.parse(Path(app.__file__).parent / "data" / "charmm36.xml").getroot()
    template = next(r for r in root.iter("Residue") if r.get("name") == "HEME")
    atoms = {a.get("name"): (a.get("type"), float(a.get("charge"))) for a in template.iter("Atom")}
    lj = {a.get("type"): (float(a.get("sigma")), float(a.get("epsilon"))) for a in root.find("LennardJonesForce").iter("Atom")}
    charge = {name: value[1] for name, value in atoms.items()}
    for bond in template.iter("Bond"):
        a, b = bond.get("atomName1"), bond.get("atomName2")
        for hydrogen, heavy in ((a, b), (b, a)):
            if atoms[hydrogen][0].startswith("H") and not atoms[heavy][0].startswith("H"):
                charge[heavy] += charge[hydrogen]
    total = sum(charge[n] for n, (t, _) in atoms.items() if not t.startswith("H"))
    if abs(total - round(total)) > 1e-6 or round(total) != -2:
        raise RuntimeError(f"the folded heme charge is {total}, not -2")
    return {name: (charge[name], *lj[atoms[name][0]]) for name, (type_, _) in atoms.items() if not type_.startswith("H")}


def protein_environment(pdb: Path, iron_position_nm: np.ndarray):
    """Charges, sigmas, epsilons (OpenMM units) and positions (nm) of the protein with hydrogens (Amber ff14SB), with the cysteine nearest the iron as the thiolate ``CYM``.

    The prediction holds heavy atoms only and no terminal oxygen, so PDBFixer adds the missing heavy atoms of the termini before Modeller adds the hydrogens.
    """
    import openmm as mm
    import openmm.app as app
    from openmm import unit
    from pdbfixer import PDBFixer

    fixer = PDBFixer(filename=str(pdb))
    fixer.removeHeterogens(keepWater=False)
    fixer.findMissingResidues()
    fixer.missingResidues = {}
    fixer.findMissingAtoms()
    fixer.addMissingAtoms()
    topology, positions = fixer.topology, fixer.positions
    modeller = app.Modeller(topology, positions)
    field = app.ForceField("amber14-all.xml")
    variants = [None] * modeller.topology.getNumResidues()
    sulfurs = [(np.linalg.norm(np.array(positions[a.index].value_in_unit(unit.nanometer)) - iron_position_nm), r.index) for r in modeller.topology.residues() if r.name == "CYS" for a in r.atoms() if a.name == "SG"]
    thiolate = None
    if sulfurs:
        distance, index = min(sulfurs)
        if distance < 0.33:
            variants[index] = "CYX"  # no thiol hydrogen is added; the residue is renamed to the Amber thiolate template below
            thiolate = index
    modeller.addHydrogens(field, variants=variants)
    if thiolate is not None:
        list(modeller.topology.residues())[thiolate].name = "CYM"
    system = field.createSystem(modeller.topology, nonbondedMethod=app.NoCutoff, constraints=None)
    nonbonded = next(f for f in system.getForces() if isinstance(f, mm.NonbondedForce))
    values = np.array([[v.value_in_unit_system(unit.md_unit_system) for v in nonbonded.getParticleParameters(i)] for i in range(system.getNumParticles())])
    return values[:, 0], values[:, 1], values[:, 2], np.array(modeller.positions.value_in_unit(unit.nanometer))


def ligand_molecule(pdb: Path, smiles: str):
    """The ligand of the file as an RDKit molecule with bond orders from the SMILES, hydrogens added from the heavy-atom coordinates (heavy atoms first, in file order)."""
    from rdkit import Chem

    from cypmode.validation.official_posebusters import ligand_copies

    copies = ligand_copies(pdb, smiles)
    if len(copies) != 1:
        raise ValueError(f"{len(copies)} ligand copies in {pdb}; exactly one is expected")
    heavy = copies[0]
    return heavy, Chem.AddHs(heavy, addCoords=True)


def ligand_system(with_hydrogens):
    """OpenMM system of the ligand (OpenFF Sage, NAGL charges) in vacuum, without cutoff."""
    import openmm as mm
    from openff.interchange import Interchange
    from openff.toolkit import ForceField, Molecule
    from openff.toolkit.utils.nagl_wrapper import NAGLToolkitWrapper

    molecule = Molecule.from_rdkit(with_hydrogens, allow_undefined_stereo=True, hydrogens_are_explicit=True)
    molecule.assign_partial_charges(CHARGE_MODEL, toolkit_registry=NAGLToolkitWrapper())
    interchange = Interchange.from_smirnoff(ForceField(LIGAND_FORCE_FIELD), [molecule], charge_from_molecules=[molecule])
    system = interchange.to_openmm_system()
    for force in system.getForces():
        if isinstance(force, mm.NonbondedForce):
            force.setNonbondedMethod(mm.NonbondedForce.NoCutoff)
    return system


def relax_pose(pdb: Path, smiles: str, out: Path, threads: int = 2) -> RelaxResult:
    """Relax the ligand of ``pdb`` in its frozen pocket and write ``out`` (the same file with the new ligand coordinates)."""
    import openmm as mm
    import openmm.app as app
    from openmm import unit

    structure = app.PDBFile(str(pdb))
    topology, positions = structure.topology, structure.positions
    hemes = [r for r in topology.residues() if r.name == "HEM"]
    if len(hemes) != 1:
        raise ValueError(f"{len(hemes)} heme residues in {pdb}; exactly one is expected")
    heme_atoms = list(hemes[0].atoms())
    heme_xyz = np.array([positions[a.index].value_in_unit(unit.nanometer) for a in heme_atoms])
    parameters = heme_parameters()
    unknown = [a.name for a in heme_atoms if a.name not in parameters]
    if unknown:
        raise ValueError(f"heme atoms without a CHARMM36 HEME parameter: {unknown}")
    iron = next(i for i, a in enumerate(heme_atoms) if a.name == "FE")

    heavy, full = ligand_molecule(pdb, smiles)
    lig_xyz = np.array(full.GetConformer().GetPositions()) * 0.1  # nm
    n_heavy = heavy.GetNumAtoms()
    system = ligand_system(full)
    n_lig = system.getNumParticles()
    if n_lig != full.GetNumAtoms():
        raise RuntimeError("the ligand system and the molecule have different atom counts")

    q_prot, sigma_prot, eps_prot, xyz_prot = protein_environment(pdb, heme_xyz[iron])
    q_hem = np.array([parameters[a.name][0] for a in heme_atoms])
    sigma_hem = np.array([parameters[a.name][1] for a in heme_atoms])
    eps_hem = np.array([parameters[a.name][2] for a in heme_atoms])
    q_env, sigma_env, eps_env = np.r_[q_prot, q_hem], np.r_[sigma_prot, sigma_hem], np.r_[eps_prot, eps_hem]
    xyz_env = np.vstack([xyz_prot, heme_xyz])
    n_env = len(q_env)
    fe_particle = n_lig + len(q_prot) + iron

    nonbonded = next(f for f in system.getForces() if isinstance(f, mm.NonbondedForce))
    ligand_values = [nonbonded.getParticleParameters(i) for i in range(n_lig)]
    for _ in range(n_env):
        system.addParticle(0.0)  # a particle without mass does not move
        nonbonded.addParticle(0.0, 1.0, 0.0)
    pocket = mm.CustomNonbondedForce(f"4*epsilon*((sigma/r)^12-(sigma/r)^6)+{COULOMB}*q1*q2/({DIELECTRIC}*r); sigma=0.5*(sigma1+sigma2); epsilon=sqrt(epsilon1*epsilon2)")
    for name in ("q", "sigma", "epsilon"):
        pocket.addPerParticleParameter(name)
    for q, sigma, epsilon in ligand_values:
        pocket.addParticle([q.value_in_unit(unit.elementary_charge), sigma.value_in_unit(unit.nanometer), epsilon.value_in_unit(unit.kilojoule_per_mole)])
    for q, sigma, epsilon in zip(q_env, sigma_env, eps_env, strict=True):
        pocket.addParticle([float(q), float(sigma), float(epsilon)])
    for k in range(nonbonded.getNumExceptions()):  # the platform wants the same exclusions in every nonbonded force; they are inside the ligand and never meet the interaction group
        i, j = nonbonded.getExceptionParameters(k)[:2]
        pocket.addExclusion(i, j)
    pocket.setNonbondedMethod(mm.CustomNonbondedForce.NoCutoff)
    pocket.addInteractionGroup(set(range(n_lig)), set(range(n_lig, n_lig + n_env)))
    pocket.setForceGroup(1)
    system.addForce(pocket)

    tether = mm.CustomExternalForce("0.5*k*((x-x0)^2+(y-y0)^2+(z-z0)^2)")
    tether.addGlobalParameter("k", TETHER_K)
    for name in ("x0", "y0", "z0"):
        tether.addPerParticleParameter(name)
    for i in range(n_heavy):
        tether.addParticle(i, [float(v) for v in lig_xyz[i]])
    tether.setForceGroup(2)
    system.addForce(tether)

    nitrogens = [a.GetIdx() for a in heavy.GetAtoms() if a.GetSymbol() == "N"]
    distances = {i: float(np.linalg.norm(lig_xyz[i] - heme_xyz[iron])) * 10.0 for i in nitrogens}
    coordinating = min(distances, key=distances.get) if distances and min(distances.values()) < FE_N_MAX_A else None
    if coordinating is not None:
        bond = mm.CustomBondForce("0.5*k*(r-r0)^2")
        bond.addPerBondParameter("k")
        bond.addPerBondParameter("r0")
        bond.addBond(coordinating, fe_particle, [FE_N_K, distances[coordinating] * 0.1])
        bond.setForceGroup(3)
        system.addForce(bond)

    platform = mm.Platform.getPlatformByName("CPU")
    context = mm.Context(system, mm.VerletIntegrator(0.001), platform, {"Threads": str(threads)})
    start = np.vstack([lig_xyz, xyz_env])
    context.setPositions(start)

    def energy(groups: int | None = None) -> float:
        return context.getState(getEnergy=True, groups=groups if groups is not None else 0xFFFFFFFF).getPotentialEnergy().value_in_unit(unit.kilojoule_per_mole)

    before, interaction_before = energy(), energy(1 << 1)
    mm.LocalEnergyMinimizer.minimize(context, TOLERANCE, MAX_ITERATIONS)
    after, interaction_after = energy(), energy(1 << 1)
    moved = context.getState(getPositions=True).getPositions(asNumpy=True).value_in_unit(unit.nanometer)[:n_lig]
    relaxed = moved[:n_heavy] * 10.0
    fe_after = float(np.linalg.norm(moved[coordinating] - heme_xyz[iron])) * 10.0 if coordinating is not None else float("nan")
    result = RelaxResult(relaxed, before, after, interaction_before, interaction_after, float(np.sqrt(np.mean(np.sum((relaxed - lig_xyz[:n_heavy] * 10.0) ** 2, axis=1)))),
                         distances[coordinating] if coordinating is not None else float("nan"), fe_after)
    write_pdb(pdb, relaxed, out)
    return result


def write_pdb(pdb: Path, heavy_xyz: np.ndarray, out: Path) -> None:
    """``pdb`` with the coordinates of its ``LIG`` atoms replaced by ``heavy_xyz`` (angstrom, file order)."""
    lines, k = [], 0
    for line in Path(pdb).read_text().splitlines():
        if line.startswith(("ATOM", "HETATM")) and line[17:20].strip() == "LIG":
            x, y, z = heavy_xyz[k]
            line = f"{line[:30]}{x:8.3f}{y:8.3f}{z:8.3f}{line[54:]}"
            k += 1
        lines.append(line)
    if k != len(heavy_xyz):
        raise ValueError(f"{k} ligand atom lines in {pdb} for {len(heavy_xyz)} coordinates")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(lines) + "\n")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--manifest", required=True, help="JSON list of {pdb, smiles, out}")
    parser.add_argument("--threads", type=int, default=2)
    args = parser.parse_args()
    jobs = json.loads(Path(args.manifest).read_text())
    failures = 0
    for job in jobs:
        out = Path(job["out"])
        out.parent.mkdir(parents=True, exist_ok=True)
        record = out.with_suffix(out.suffix + ".json")
        if out.exists() and record.exists():
            continue
        try:
            result = relax_pose(Path(job["pdb"]), job["smiles"], out, args.threads)
            record.write_text(json.dumps(result.summary()))
            print(f"{out.name}: energy {result.energy_before:.0f} -> {result.energy_after:.0f} kJ/mol, ligand moved {result.rmsd:.2f} A", flush=True)
        except Exception as exc:  # noqa: BLE001 - one pose that cannot be set up must not stop the batch
            failures += 1
            record.write_text(json.dumps({"error": f"{type(exc).__name__}: {exc}"}))
            print(f"{out.name}: FAILED {type(exc).__name__}: {exc}", flush=True)
    print(f"{len(jobs) - failures} of {len(jobs)} poses relaxed")
    return 0


if __name__ == "__main__":
    sys.exit(main())
