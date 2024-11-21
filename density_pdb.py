import requests
from math import cos, radians, sqrt
from io import StringIO
from Bio.PDB import PDBParser


def calculate_density_from_pdb(pdb_id):
    """
    Calculate the density of a molecule from its PDB ID.
    
    Parameters:
    - pdb_id (str): The PDB ID of the molecule.

    Returns:
    - density (float): The calculated density in g/cm³.
    - vm (float): Matthews coefficient in Å³/Da.
    """
    # Fetch the PDB file
    url = f"https://files.rcsb.org/download/{pdb_id}.pdb"
    response = requests.get(url)
    if response.status_code != 200:
        raise ValueError(f"PDB ID {pdb_id} not found.")
    pdb_content = response.text

    # Parse crystallographic data
    a = b = c = alpha = beta = gamma = z = None
    for line in pdb_content.splitlines():
        if line.startswith("CRYST1"):
            a = float(line[6:15].strip())
            b = float(line[15:24].strip())
            c = float(line[24:33].strip())
            alpha = float(line[33:40].strip())
            beta = float(line[40:47].strip())
            gamma = float(line[47:54].strip())
            z = int(line[62:66].strip())  # Z value
            break

    if None in [a, b, c, alpha, beta, gamma, z]:
        raise ValueError("Crystallographic data not found in the PDB file.")

    # Calculate unit cell volume
    alpha, beta, gamma = map(radians, [alpha, beta, gamma])
    unit_cell_volume = a * b * c * sqrt(
        1 - cos(alpha)**2 - cos(beta)**2 - cos(gamma)**2 + 2 * cos(alpha) * cos(beta) * cos(gamma)
    )  # in Å³

    # Parse molecular weight
    parser = PDBParser(QUIET=True)
    structure = parser.get_structure(pdb_id, StringIO(pdb_content))
    molecular_weight = 0.0
    for atom in structure.get_atoms():
        molecular_weight += atom.mass

    if molecular_weight == 0.0:
        raise ValueError("Failed to calculate molecular weight from the PDB file.")

    # Calculate density
    avogadro_number = 6.022e23  # mol⁻¹
    unit_cell_volume_cm3 = unit_cell_volume * 1e-24  # Convert Å³ to cm³
    density = (z * molecular_weight) / (avogadro_number * unit_cell_volume_cm3)  # g/cm³

    # Calculate Matthews coefficient
    vm = unit_cell_volume / (molecular_weight * z)  # Å³/Da

    return density, vm


def fetch_pdb_details(pdb_id):
    """
    Fetch experimental details from the PDB entry.
    """
    url = f"https://www.rcsb.org/fasta/entry/{pdb_id}/download"
    response = requests.get(url)
    if response.status_code != 200:
        raise ValueError(f"PDB ID {pdb_id} not found.")
    pdb_content = response.text

    # Parse for experimental details (REMARK or EXPDTA sections)
    exp_details = []
    for line in pdb_content.splitlines():
        if line.startswith("REMARK") or line.startswith("EXPDTA"):
            exp_details.append(line.strip())
    return exp_details


def calculate_mother_liquor_density(pdb_id):
    """
    To calculate the density of the mother liquor for a crystal given its PDB ID, you would need to account for the solvent content in the crystal, typically using the Matthews coefficient (VM).

    Parameters:
    - pdb_id (str): The PDB ID of the molecule.

    Returns:
    - liquor_density (float): Mother liquor density in g/cm³.
    - solvent_content (float): Solvent content as a fraction.
    """
    crystal_density, vm = calculate_density_from_pdb(pdb_id)

    # Estimate solvent content
    solvent_content = 1 - (1.23 / vm)

    # Calculate mother liquor density
    liquor_density = crystal_density / (1 - solvent_content)

    return liquor_density, solvent_content
