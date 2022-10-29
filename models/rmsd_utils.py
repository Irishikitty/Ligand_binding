import torch
from sklearn.metrics import pairwise_distances
import pycpd as pypd
from Bio.PDB import NeighborSearch, PDBParser, Selection
from Bio.PDB.vectors import Vector, calc_angle, calc_dihedral
import numpy as np


# Predefined ========================================
# RADIUS = 5  # radius to define sphere containing donors


# ================================================================================================


def rmsd(predict, true, num_atoms):

    assert np.array(predict).shape == np.array(true).shape

    dis = distance_3D(predict[:,1:], np.array(true).T[1:].T)
    dis_total_min = np.sqrt((dis**2) / num_atoms)

    return dis_total_min


# ======================================================================================================================


def extract_dis_vectors(dm, num):
    '''
    :param dm: (predicted) pairwise distance mx
    :param num: num of ligand
    :return: [protein_env, ligand] rectangular mx
    '''
    dis_vectors = [dm[index, num:].tolist() for index in range(num)]

    return dis_vectors


# ================================================================================================


def distance_to_loc(vector_d, verts):
    '''
    build mx A, Y and compute beta

    :param vector_d: [1, ligand] each row of rectangular mx = r_atom1, r_atom2, ..
    :param verts: [protein_env, 3]

    :return: beta hat
    '''
    vector_d = np.asarray(vector_d)
    verts = np.asarray(verts)
    A = 2 * (verts[1:] - verts[0]) # [2(b1-a1)  2(b2-a2) 2(b3-a3)]
    Y = vector_d[0] ** 2 - vector_d[1:] ** 2 + np.sum(verts[1:] ** 2 - verts[0] ** 2, axis=1)
    loc = np.matmul(np.matmul(np.linalg.pinv(np.matmul(A.T, A)), A.T), Y)

    return loc

# ======================================================================================================================


def distance_3D(x, y, axis=None):
    diff = np.array(x) - np.array(y)
    diff = diff ** 2
    return np.sqrt(np.sum(diff, axis=axis))

# ======================================================================================================================

def get_rmsd(y_fake, random_env_atom_data, true_ligands, ligandLength):
    '''
    Recover distance mx to 3D coordinates for ligands

    :param y_fake: predicted pairwise distance
    :param random_env_atom_data: permuted data (No permutation)
    :param true_ligands: True 3d coordinates
    :param ligandLength:

    :return: rmsd between recovered 3d and the truth
    '''
    # [protein_env, ligand_num] rectangular mx
    dis_vectors = extract_dis_vectors(y_fake, ligandLength)
    # [ligand_num, 3] predicted beta hat
    temp_ligand_locs = np.array([distance_to_loc(dis_vectors[j], np.stack(random_env_atom_data).T[1:].T).tolist() for j in range(ligandLength)])
    # [ligand_num, 4]
    estimated_ligands_data = np.vstack([[22]*ligandLength, temp_ligand_locs.T]).T
    # target = [np.stack(true_ligands[i]).reshape(3, ) for i in range(len(true_ligands))]
    target = np.array([[22] + np.squeeze([i.numpy() for i in true_ligands[0]]).tolist()])
    # difference
    pred_rmsd = rmsd(estimated_ligands_data, target, ligandLength)

    return pred_rmsd, estimated_ligands_data

def get_locs(y_fake, random_env_atom_data, ligandLength):

    dis_vectors = extract_dis_vectors(y_fake, ligandLength)

    temp_ligand_locs = np.array([distance_to_loc(dis_vectors[j], random_env_atom_data.T[1:].T).tolist() for j in
                          range(ligandLength)])

    estimated_ligands_data = np.vstack([[22]*ligandLength, temp_ligand_locs.T]).T

    return estimated_ligands_data


# ======================================================================================================================
def data_toTensor(ligand, atoms, moving = True, dim_max=256, channels = 23):
    '''
    3D coordinates of input (ligand + atom)
    Ouptut tensor with shape [23, dim_sum = 250, 250]

    Input:
        :param ligand: list of [22 x y z]
        :param atoms: list of [c x y z]
        :param dim_max: 250 as default
    Output:

    '''
    atoms = [np.stack(atoms[i]).reshape(4,) for i in range(len(atoms))]
    ligand = [np.stack(ligand[i]).reshape(4,) for i in range(len(ligand))]

    numAtoms, _ = np.array(atoms).shape
    numLigandAtoms, _ = np.array(ligand).shape

    A_coordinates = pairwise_distances([[x, y, z] for [_, x, y, z] in ligand] + [[x, y, z] for [i, x, y, z] in atoms])
    types = [int(i) for [i, _, _, _] in atoms]
    container = torch.zeros(channels-1, *A_coordinates.shape)
    for i, c in enumerate(types):
        for j in range(len(types)):
            container[c - 1, i, j] += 1
            container[c - 1, j, i] += 1
    container[container > 0] = 1  # as indicator
    # build tensor
    A_coordinates = torch.from_numpy(A_coordinates.reshape(-1, *A_coordinates.shape))
    A_coordinates = torch.cat([A_coordinates, container], dim=0)
    A_coordinates[22, :numLigandAtoms, :numLigandAtoms] = 1
    # move blocks
    out = torch.zeros(channels, dim_max, dim_max)  # first channel container
    out[:, :numLigandAtoms, :numLigandAtoms] += A_coordinates[:, :numLigandAtoms, :numLigandAtoms]  # upper left
    out[:, 50:50 + numAtoms, 50:50 + numAtoms] += A_coordinates[:, numLigandAtoms:, numLigandAtoms:]  # lower right
    if moving == True:
        out[:, :numLigandAtoms, 50:50 + numAtoms] += A_coordinates[:, :numLigandAtoms, numLigandAtoms:]  # rectangle upper
        out[:, 50:50 + numAtoms, :numLigandAtoms] += A_coordinates[:, numLigandAtoms:, :numLigandAtoms]  # rectangle lower
    return out


# ======================================================================================================================
def data_to_0_1(sample):
    '''
    Normalize to [-1, 1]
    :param sample: input mx/tensor
    :return: normalized input
    '''

    sample = 2 * sample /80 - 1
    return sample


def get_input_dis_matrix(start, env_atoms, target):
    target = np.squeeze([i.numpy() for i in target[0]])
    target = np.array([[22]+target.tolist()])
    tempFakeA = data_toTensor(start, env_atoms, moving = True, dim_max=256, channels=23)
    tempTrueB = data_toTensor(target, env_atoms, moving = True, dim_max=256, channels=23)

    tempFakeA[0] = data_to_0_1(tempFakeA[0])  # [-1,1]
    tempTrueB[0] = data_to_0_1(tempTrueB[0])  # [-1,1]

    starting_rmsd = rmsd(start, [np.stack(target[i]).reshape(4,) for i in range(len(target))], len(target))

    fakeA = tempFakeA.unsqueeze(0).float()
    trueB = tempTrueB.unsqueeze(0).float()
    assert fakeA.shape == (1, 23, 256, 256)
    assert trueB.shape == (1, 23, 256, 256)
    env_atoms = [np.stack(env_atoms[i]).reshape(4,) for i in range(len(env_atoms))]

    return fakeA, trueB, starting_rmsd, env_atoms

def get_neighbor_points(pred_coords, true_ligand, protein_env, distance = 5):
    '''
    Extract points in a sphere with radius = DISTANCE
    Just consider heavy atoms

    :param pred_coords: [ligand_len, 3] usually [1, 3]
    :param true_ligand: [ligand_len, 3] usually [1, 3]
    :param protein_env: [protein_len, 3] usually [200, 3]
    :param distance: default = 3.5

    :return: two sets of points of same size
    '''
    # X
    pt2 = pairwise_distances(true_ligand.reshape(-1,3), protein_env)
    pt2 = protein_env[(pt2<=distance)[0]]

    # Y
    pt1 = pairwise_distances(pred_coords.reshape(-1,3), protein_env)
    pt1.sort()
    pt1 = protein_env[(pt1<=distance)[0]][:len(pt2)]

    return pt1, pt2[:len(pt1)]


def coherent_point_registration(pt1, pt2, min_points = 2):
    '''
    Perform coherent point registration
    If # points < min_points, then ignore

    :param pt1: X: [N, 3]
    :param pt2: Y: [M, 3] -> RY
    :param min_points: default = 3 for a surface in 3D
    :return: RMSD between two polyhedrons
    '''
    if len(pt1) == 1:
        RMSD = np.linalg.norm(pt1 - pt2, 2)
        return RMSD
    elif len(pt1) >= min_points:
        reg = pypd.RigidRegistration(**{'X': pt1, 'Y': pt2})
        data, _ = reg.register() # data: [M, 3]
        RMSD = np.sqrt(np.sum((data-pt1)**2)/len(data))    # sqrt(dist^2/N)
        return RMSD

def intersection(array1, array2):
    lst3 = [value for value in array1 if value in array2]
    return lst3

# Dictionary ======================================
# donor dictionary
donor_dict = {'ARG': ('NE', 'NH1', 'NH2'), 'ASN': ('ND2','OD1'),
              'GLN': ('NE2','OE1'), 'HIS': ('ND1, NE2'), 'LYS': ('NZ'),
              'SER': ('OG'), 'THR': ('OG1'), 'TRP': ('NE1'), 'TYR': ('OH'),
              'ASP': ('OD1','OD2'), 'GLU': ('OE1')}

# neighbor dictionary
neighbor_dict = {'ARG NH1': ('CZ', 'NE'),
                 'ARG NH2': ('CZ', 'NE'),
                 'ARG NE': ('CD', 'CG'),
                 'ASN ND2': ('CG', 'CB'),
                 'ASN OD1': ('CG', 'CB'),
                 'ASP OD1': ('CG', 'CB'),
                 'ASP OD2': ('CG', 'CB'),
                 'GLN NE2': ('CD', 'CG'),
                 'GLN OE1': ('CD', 'CG'),
                 'GLU OE1': ('CD', 'CG'),
                 'GLU OE2': ('CD', 'CG'),
                 'HIS ND1': ('CG', 'CB'),
                 'HIS NE2': ('CD2', 'CG'),
                 'LYS NZ': ('CE', 'CD'),
                 'SER OG': ('CB', 'CA'),
                 'THR OG1': ('CB', 'CA'),
                 'TRP NE1': ('CD1', 'CG'),
                 'TYR OH': ('CZ', 'CE1')}


def cal_angle(v1, v2, v3):
    '''
    Calculate angle method
    representing 3 connected points.

    :param v1, v2, v3: the tree points that define the angle
    :type v1, v2, v3: L{Vector}

    :return: angle
    :rtype float
    '''

    v1 = Vector(v1)
    v2 = Vector(v2)
    v3 = Vector(v3)

    return calc_angle(v1, v2, v3)


def cal_dihedral(v1, v2, v3, v4):
    '''
    Calculate dihedral angle method
    representing 3 connected points.

    :param v1, v2, v3, v4: the tree points that define the angle
    :type v1, v2, v3, v4: L{Vector}

    :return: angle
    :rtype float
    '''

    v1 = Vector(v1)
    v2 = Vector(v2)
    v3 = Vector(v3)
    v4 = Vector(v4)

    return calc_dihedral(v1, v2, v3, v4)


# Donors, Neighbors =================================

def donor_neighbor(QUERY: np.array, QUERY_real: np.array,
                   NAME: str, RADIUS = 4.5, min_points = 3):
    '''
    Input query coordinates and PDB file name
    Output a dict with donors, neighbors and angles

    Parameters
    ----------
    QUERY : pred location np.array ex: np.array([50., -15., 81.])
    QUERY_real : real location
    NAME : str: '5iqd'

    Returns: list of dict {'donor': v,'neighbor1': v, 'neighbor2': v,
                   'angle': v, 'dihedral': v}
    -------
    None.

    '''
    p = PDBParser(QUIET=True)
    structure = p.get_structure(NAME, NAME + '.pdb')
    res_list = Selection.unfold_entities(structure, 'A')
    ns = NeighborSearch(res_list)

    def find_neighbor(location, neighbor_name):
        candidates = ns.search(location, 1.8, 'A')
        candidates = [(i.name, i.get_coord()) for i in candidates if i.name == neighbor_name]
        return candidates[0]

    # all residuals in sphere (QUERT)
    neighbor_lst = ns.search(QUERY, RADIUS, 'R')
    output = []
    for i in neighbor_lst:
        atom_lst = i.get_atoms()
        for j in atom_lst:
            atom_coord = j.get_coord()
            if np.linalg.norm(atom_coord - QUERY) <= RADIUS:
                output.append((i.get_resname(), j.name, j.get_coord()))

    # donors within RADIUS
    donor_lst = []
    for (i, j, k) in output:
        if (donor_dict.get(i) is not None) and (j in donor_dict.get(i)):
            donor_lst.append((i + ' ' + j, k, np.linalg.norm(k - QUERY, 1)))
    # donor_lst.sort(key=lambda x: x[2])

    # neighbor list
    output = []
    for (ij, k, _) in donor_lst:
        neighbor_name = neighbor_dict.get(ij)
        if neighbor_name is not None:
            neighbor1 = find_neighbor(k, neighbor_name[0])
            neighbor2 = find_neighbor(neighbor1[1], neighbor_name[1])

            angle = cal_angle(QUERY, k, neighbor1[1])
            dihedral = cal_dihedral(QUERY, k, neighbor1[1], neighbor2[1])
            angle_real = cal_angle(QUERY_real, k, neighbor1[1])
            dihedral_real = cal_dihedral(QUERY_real, k, neighbor1[1], neighbor2[1])
            dist = np.abs(QUERY-k)
            dist_real = np.abs(QUERY_real-k)

            output.append({'donor': (ij, k), 'neighbor1': neighbor1, 'neighbor2': neighbor2,
                           'angle': angle, 'dihedral': dihedral, 'angle_real': angle_real, 'dihedral_real': dihedral_real,
                           'dist': dist , 'dist_real': dist_real})
        else:
            continue

    # geometry RMSD
    # all residuals in sphere (QUERY_real)
    neighbor_lst = ns.search(QUERY_real, RADIUS, 'R')
    temp = []
    for i in neighbor_lst:
        atom_lst = i.get_atoms()
        for j in atom_lst:
            atom_coord = j.get_coord()
            if np.linalg.norm(atom_coord - QUERY) <= RADIUS:
                temp.append((i.get_resname(), j.name, j.get_coord()))

    # (donor_name, coords, dist_metal)
    donor_real_lst = []
    for (i, j, k) in temp:
        if (donor_dict.get(i) is not None) and (j in donor_dict.get(i)):
            donor_real_lst.append((i + ' ' + j, k, np.linalg.norm(k - QUERY_real, 1)))
    # donor_real_lst.sort(key=lambda x: x[2])
    missing = np.max([len(donor_real_lst) - len(donor_lst),0])

    if len(donor_real_lst) < min_points:
        return output,  -1
    elif (len(donor_real_lst) >= min_points) and (len(donor_lst) < min_points):
        return output, missing * 100
    elif len(donor_lst) != len(donor_real_lst):
        M = min(len(donor_lst), len(donor_real_lst))
        donor_lst = donor_lst[:M]
        donor_real_lst = donor_real_lst[:M]
    pt1 = [donor_real_lst[i][1] for i in range(len(donor_lst))]
    pt2 = [donor_lst[i][1] for i in range(len(donor_lst))]
    coord_rmsd = coherent_point_registration(np.asarray(pt1), np.asarray(pt2))

    # find the 3rd closest distance
    donor_real_lst.sort(key=lambda x: x[2])

    return output, coord_rmsd, donor_real_lst[2][2]

# Compute avg deviations ..
def compute_metrics(QUERY, QUERY_real, metrics: list):
    '''
    Compute coord_fitness

    Parameters
    ----------
    metrics : TYPE
        DESCRIPTION.

    Returns
    -------
    None.

    '''
    N = len(metrics)
    if N == 0:
        return 100/3, 100/3, 100/3
    else:
        angle_dev = 0
        dihedral_dev = 0
        distance_dev = 0
        for i in metrics:
            angle_dev += np.abs(np.sin(i['angle_real'] - i['angle']))
            dihedral_dev += np.abs(np.sin(i['dihedral_real'] - i['dihedral']))
            distance_dev = np.linalg.norm(i['dist']-i['dist_real'], 1)
        return angle_dev / N, dihedral_dev / N, distance_dev / N


def compute_coord_fitness(QUERY, QUERY_real, NAME, RADIUS):
    '''
    Adding all together

    Parameters
    ----------
    QUERY : TYPE
        DESCRIPTION.
    QUERY_real : TYPE
        DESCRIPTION.
    NAME : TYPE
        DESCRIPTION.

    Returns
    -------
    TYPE
        DESCRIPTION.

    '''
    metrics, coord_rmsd, distances = donor_neighbor(QUERY, QUERY_real, NAME, RADIUS)
    if coord_rmsd >= 100.0:
        return coord_rmsd, coord_rmsd
    else:
        coord_directionality = np.sum(compute_metrics(QUERY, QUERY_real, metrics))
        return coord_rmsd, coord_rmsd+coord_directionality, distances


