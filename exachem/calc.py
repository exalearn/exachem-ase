"""Interface to ASE calculator"""
import re
import json
import logging
from pathlib import Path

from ase.calculators.calculator import FileIOCalculator
from ase import units, Atoms

logger = logging.getLogger(__name__)


class ExaChem(FileIOCalculator):
    """ExaChem implements many quantum chemistry methods as scalable algorithms on GPU"""

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
        output = self.parameters['template'].copy()
        output['geometry'] = geometry
        output['basis'] = {'basisset': self.parameters['basisset']}
        output['TASK'] = {self.parameters['method']: True}

        # Write to disk in the output directory
        with open(Path(self.directory) / 'exachem.json', 'w') as fp:
            json.dump(output, fp, indent=2)

    def read_results(self):
        # Find the output directory
        output_paths = list(Path(self.directory).glob('exachem.*_files'))
        if len(output_paths) > 1:  # pragma: no cover
            raise ValueError(f'Found {len(output_paths)} output directories when expecting one')
        output_path = output_paths[0]

        # Read the energy
        level = self.parameters['method']
        basis = self.parameters['basisset']
        if level == "mp2":
            # If none, assume this is an MP2 computation and read from the stdout
            output = (output_path / '..' / 'exachem.out').read_text()

            # Find the SCF energy
            scf_energy_match = re.search(r'\*\* Total SCF energy =\s+([-\d\.]+)', output)
            scf_energy = float(scf_energy_match.group(1))

            # Find the MP2 energy
            mp2_energy_match = re.search(r'Closed-Shell MP2 energy / hartree:\s+([-\d\.]+)', output)
            mp2_energy = float(mp2_energy_match.group(1))
            energy = mp2_energy + scf_energy
        else:
            output_json_dir = output_path / 'restricted' / 'json'
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
