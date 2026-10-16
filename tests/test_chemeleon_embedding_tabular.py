"""TabICL and TabPFN on the frozen CheMeleon embedding are registered like the other cached candidates, wait for their inputs instead of failing, and are waited for by the long Chemprop family."""

from pathlib import Path

from cypmode.challenge.aux_experiments import DIAGNOSTICS, candidates_of
from cypmode.challenge.cached_candidates import GROUPS
from cypmode.challenge.final_tdi_build import GROUPS as FINAL_GROUPS
from cypmode.challenge.nested_runners_torch import CANDIDATES
from cypmode.challenge.nested_runners_torch import GROUPS as RUNNERS

JOBS = Path("queue/jobs")
NAMES = ("tabicl_chemeleon", "tabpfn_chemeleon")


def test_each_group_is_known_to_the_runner_the_cache_reader_the_addition_test_and_the_final_build():
    for name in NAMES:
        assert [c.name for c in GROUPS[name]()] == CANDIDATES[name] == [name]
        assert name in RUNNERS and name in FINAL_GROUPS
        assert [c.name for c in candidates_of(name)] == [name] and DIAGNOSTICS[name] == (name,)


def test_the_fill_jobs_wait_for_the_embedding_cache_and_the_additions_for_both_caches():
    for job, group in (("246_tabicl_chemeleon_CYP3A4", "tabicl_chemeleon"), ("247_tabicl_chemeleon_CYP2D6", "tabicl_chemeleon"), ("249_tabpfn_chemeleon_CYP3A4", "tabpfn_chemeleon"), ("250_tabpfn_chemeleon_CYP2D6", "tabpfn_chemeleon")):
        text = (JOBS / f"{job}.sh").read_text()
        assert "need_file data/tdi_track/outputs/caches/chemeleon_embeddings.npz" in text and f"--groups {group} " in text
        assert f"chunk_status {job} " in text
    for job in ("249_tabpfn_chemeleon_CYP3A4.sh", "250_tabpfn_chemeleon_CYP2D6.sh"):
        assert "need_file data/tdi_track/inputs/tabpfn_licence_accepted.txt" in (JOBS / job).read_text()
    assert "need_done 246_tabicl_chemeleon_CYP3A4 247_tabicl_chemeleon_CYP2D6" in (JOBS / "248_tabicl_chemeleon_addition.sh").read_text()
    assert "need_done 249_tabpfn_chemeleon_CYP3A4 250_tabpfn_chemeleon_CYP2D6" in (JOBS / "251_tabpfn_chemeleon_addition.sh").read_text()


def test_the_long_chemprop_family_and_its_speed_test_wait_for_the_new_jobs():
    text = Path("queue/lib.sh").read_text()
    cheap = text.split("cheap_candidates_settled() {", 1)[1].split("\n}", 1)[0]
    for job in ("246_tabicl_chemeleon_CYP3A4", "247_tabicl_chemeleon_CYP2D6", "248_tabicl_chemeleon_addition", "249_tabpfn_chemeleon_CYP3A4", "250_tabpfn_chemeleon_CYP2D6", "251_tabpfn_chemeleon_addition"):
        assert job in cheap
