"""Interface to ASE calculator"""
import re
import json
import logging
from collections import defaultdict
from pathlib import Path

from ase.calculators.calculator import FileIOCalculator
from ase import units, Atoms

logger = logging.getLogger(__name__)


class ExaChem(FileIOCalculator):
    """ExaChem implements many quantum chemistry methods as scalable algorithms on GPU

    Set the initial charges and magnetic moments to control the charge and multiplicity of a
    calculation, respectively.
    """

    default_parameters = {
        'basisset': 'cc-pvdz',
        'method': 'scf',
        'template': {}
    }
    implemented_properties = ['energy']

    def write_input(self, atoms: Atoms, properties=None, system_changes=None):
        # Write the atoms to geometry output
        geometry = {
            'coordinates': [],
            'units': 'angstrom'
        }
        for sym, pos in zip(atoms.symbols, atoms.positions):
            coords = "\t".join(f"{x:.8f}" for x in pos)
            geometry['coordinates'].append(
                f'{sym}\t{coords}'
            )

        # Join geometry and basis with the rest of the settings
        output = defaultdict(dict)
        output.update(self.parameters['template'])
        output['geometry'] = geometry
        output['basis'] = {'basisset': self.parameters['basisset']}
        output['TASK'] = {self.parameters['method']: True}

        # Determine the charge and multiplicity of the system
        charge = atoms.get_initial_charges().sum()
        if charge != 0:
            output['SCF']['charge'] = charge

        magmom = int(round(abs(atoms.get_initial_magnetic_moments().sum())))
        if magmom > 0:
            output['SCF']['multiplicity'] = magmom + 1
            output['SCF']['scf_type'] = 'unrestricted'

        # Write to disk in the output directory
        with open(Path(self.directory) / 'exachem.json', 'w') as fp:
            json.dump(output, fp, indent=2)

    def read_results(self):
        # Load the output file
        with open(Path(self.directory) / 'exachem.json') as fp:
            output_file = json.load(fp)
        scf_type = output_file['SCF'].get('scf_type', 'restricted')

        # Find the output directory
        level = self.parameters['method']
        basis = self.parameters['basisset']
        output_path = Path(self.directory) / f'exachem.{basis}_files'

        # Read the energy
        if level == "mp2":
            # If none, assume this is an MP2 computation and read from the stdout
            output = (output_path / '..' / 'exachem.out').resolve().read_text()

            # Find the SCF energy
            scf_energy_match = re.search(r'\*\* Total SCF energy =\s+([-\d\.]+)', output)
            scf_energy = float(scf_energy_match.group(1))

            # Find the MP2 energy
            mp2_energy_match = re.search(rf'{"Closed" if scf_type == "restricted" else "Open"}-Shell MP2 energy / hartree:\s+([-\d\.]+)', output)
            mp2_energy = float(mp2_energy_match.group(1))
            energy = mp2_energy + scf_energy
        else:
            output_json_dir = output_path / scf_type / 'json'
            output_json = output_json_dir / f'exachem.{basis}.{level}.json'

            # Read in the energy
            with output_json.open() as fp:
                output = json.load(fp)["output"]
            if level == 'ccsd_t':
                energy = output["CCSD(T)"]["(T)Energies"]["total"]
            elif level == 'ccsd':
                energy = output["CCSD"]["final_energy"]["total"]
            elif level == "scf":
                energy = output["SCF"]["final_energy"]
            else:
                raise NotImplementedError(f'No support for {level} yet')
        self.results['energy'] = energy * units.Ha
