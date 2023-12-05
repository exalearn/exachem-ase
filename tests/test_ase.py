"""Test the TAMM interface"""
from pathlib import Path
from math import isclose
import json

from ase.build import molecule
from pytest import fixture, mark

from exachem.calc import ExaChem

_test_files = Path(__file__).parent / 'files'
exachem_dir = Path('/home/lward/Software/exachem/bin/bin')


def make_command(level):
    if not exachem_dir.exists():
        return f"mpirun -n 3 {exachem_dir}/ExaChem exachem.json 1> exachem.out 2> exachem.err"
    else:
        return f'cp -r {_test_files / "outputs" / level / "*"} .'


@fixture()
def atoms():
    return molecule('H2O')


@fixture()
def exachem_template():
    with (_test_files / 'template.json').open() as fp:
        return json.load(fp)


@mark.parametrize('level,target', [('ccsd', -2074.600227), ('scf', -2068.773588), ('mp2', -2074.346445)])
def test_tamm(level, target, exachem_template, atoms, tmpdir):
    command = make_command(level)
    calc = ExaChem(command=command, method=level, basisset='cc-pvdz', template=exachem_template, directory=tmpdir)
    eng = calc.get_potential_energy(atoms)
    assert isclose(eng, target)
