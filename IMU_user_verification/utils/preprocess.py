import numpy as np
from scipy.integrate import cumtrapz, trapz


def get_rotation_matrix_from_vector(v):
    q1 = v[0]
    q2 = v[1]
    q3 = v[2]
    if len(v) == 4:
        q0 = v[3]
    else:    
        q0 = 1 - q1**2 - q2**2 - q3**2
        q0 = np.sqrt(q0) if q0 > 0 else 0
    sq_q1 = 2 * q1**2
    sq_q2 = 2 * q2**2
    sq_q3 = 2 * q3**2
    q1_q2 = 2 * q1 * q2
    q3_q0 = 2 * q3 * q0
    q1_q3 = 2 * q1 * q3
    q1_q3 = 2 * q1 * q3
    q2_q0 = 2 * q2 * q0
    q2_q3 = 2 * q2 * q3
    q1_q0 = 2 * q1 * q0
    R = np.array([
        [1.0 - sq_q2 - sq_q3, q1_q2 - q3_q0, q1_q3 + q2_q0],
        [q1_q2 + q3_q0, 1.0 - sq_q1 - sq_q3, q2_q3 - q1_q0],
        [q1_q3 - q2_q0, q2_q3 + q1_q0, 1.0 - sq_q1 - sq_q2],
    ])
    if len(v) == 4:
        R = np.hstack((R, np.zeros((3, 1))))
        R = np.vstack((R, np.array([[0.0, 0.0, 0.0, 1.0]])))
    return R

def get_rotation_matrices(x):
    rotation_matrices = np.zeros(x.shape[:-1] + (3, 3))
    xx = x.reshape((-1, 3))
    rmt = rotation_matrices.reshape((-1, 3, 3))
    for i in range(xx.shape[0]):
        rmt[i, :, :] = get_rotation_matrix_from_vector(xx[i])
    rotation_matrices = rmt.reshape(x.shape[:-1] + (3, 3))
    return rotation_matrices


def known_rot_starts(rot):
    nonzero_mask = np.any(rot != 0, axis=-1)
    pred_zero = np.concatenate((np.ones((nonzero_mask.shape[0], 1), dtype=bool), (~nonzero_mask)[:, :-1]), axis=1)
    rot_single_start_mask = np.logical_and(nonzero_mask, pred_zero)
    assertion_idx, col_idx = np.where(np.logical_and(nonzero_mask, pred_zero))
    col_idx_res = np.ones(nonzero_mask.shape[0], dtype=col_idx.dtype) * nonzero_mask.shape[0]
    for ass_id, cid in zip(assertion_idx, col_idx):
        col_idx_res[ass_id] = cid
    return col_idx_res


def rotate_known(x, rot, axis=1):
    start_indices = known_rot_starts(rot)
    rotated = np.zeros_like(x)
    for i in range(x.shape[0]):
        if start_indices[i] == rot.shape[1]:
            continue
        rmt = get_rotation_matrices(rot[i, start_indices[i]:, :])
        xx = np.einsum('dmn,dn->dm', rmt, x[i, start_indices[i]:, :])
        rotated[i, start_indices[i]:, :] = xx
    return rotated


def diff_known(x, rot, axis=1):
    start_indices = known_rot_starts(rot)
    diffed = np.zeros_like(x)
    for i in range(x.shape[0]):
        if start_indices[i] == rot.shape[1]:
            continue
        xx = x[i, start_indices[i]:, :]
        diff_value = np.concatenate((
            np.zeros((1, xx.shape[-1])),
            np.diff(xx, axis=-2)), axis=-2)
        diffed[i, start_indices[i]:, :] = diff_value
    return diffed


def unrotate_known(x, rot, axis=1):
    start_indices = known_rot_starts(rot)
    rotated = np.zeros_like(x)
    for i in range(x.shape[0]):
        if start_indices[i] == rot.shape[1]:
            continue
        rmt = get_rotation_matrices(rot[i, start_indices[i]:, :])
        #print(np.einsum('dmn,dkn->dmk', rmt, rmt))
        #print(rmt[0] @ rmt[0].T)
        #assert np.allclose(np.einsum('dmn,dkn->dmk', rmt, rmt), np.stack([np.eye(3)] * rmt.shape[0], axis=0))
        xx = np.einsum('dnm,dn->dm', rmt, x[i, start_indices[i]:, :])
        rotated[i, start_indices[i]:, :] = xx
    return rotated


def integrate_known(x, rot, axis=1):
    start_indices = known_rot_starts(rot)
    integrated = np.zeros_like(x)
    for i in range(x.shape[0]):
        if start_indices[i] == rot.shape[1]:
            continue
        rmt = get_rotation_matrices(rot[i, start_indices[i]:, :])
        xx = np.einsum('dmn,dn->dm', rmt, x[i, start_indices[i]:, :])
        integrated[i, start_indices[i]:, :] = cumtrapz(xx, dx=0.02, axis=0, initial=0)
    return integrated


def degrav_known(x, rot, grav_vec, axis=1):
    start_indices = known_rot_starts(rot)
    degrav = np.zeros_like(x)
    for i in range(x.shape[0]):
        if start_indices[i] == rot.shape[1]:
            continue
        rmt = get_rotation_matrices(rot[i, start_indices[i]:, :])
        xx = np.einsum('dmn,dn->dm', rmt, x[i, start_indices[i]:, :])
        degrav[i, start_indices[i]:, :] = xx - grav_vec[None, :]
    return degrav


def get_linaccel_features(accel, rot):
    grav_const = np.array([0.0, 0.0, 9.806634201818664])
    rot_accel_known = rotate_known(accel, rot)
    rot_linaccel_known = degrav_known(accel, rot, grav_const)
    unrot_accel_known = unrotate_known(rot_accel_known, rot)
    unrot_linaccel_known = unrotate_known(rot_linaccel_known, rot)
    
    rot_linaccel_diff = diff_known(rot_linaccel_known, rot)
    unrot_linaccel_diff = diff_known(unrot_linaccel_known, rot)
    rot_linaccel_int = integrate_known(rot_linaccel_known, rot)
    unrot_linaccel_int = integrate_known(unrot_accel_known, rot)
    
    return np.concatenate((
        rot_linaccel_known,
        rot_linaccel_diff,
        rot_linaccel_int,
        unrot_linaccel_known,
        unrot_linaccel_diff,
        unrot_linaccel_int
    ), axis=-1)

def generate_dataset_features(accel_dataset, gyro_dataset, rotation_dataset, magnetic_dataset):
    dataset = {
        stage: {
            'data': np.concatenate((get_linaccel_features(accel_dataset[stage]['data'], rotation_dataset[stage]['data']),
                                    np.concatenate((
                                        integrate_known(accel_dataset[stage]['data'], rotation_dataset[stage]['data']),
                                        cumtrapz(accel_dataset[stage]['data'], dx=0.02, axis=1, initial=0.0),
                                        rotate_known(accel_dataset[stage]['data'], rotation_dataset[stage]['data']),
                                        accel_dataset[stage]['data'],
                                        np.concatenate((np.zeros((accel_dataset[stage]['data'].shape[0], 1, accel_dataset[stage]['data'].shape[2])), np.diff(accel_dataset[stage]['data'], axis=1)), axis=1)), axis=-1),
                                    np.concatenate((
                                        integrate_known(gyro_dataset[stage]['data'], rotation_dataset[stage]['data']),
                                        cumtrapz(gyro_dataset[stage]['data'], dx=0.02, axis=1, initial=0.0),
                                        rotate_known(gyro_dataset[stage]['data'], rotation_dataset[stage]['data']),
                                        gyro_dataset[stage]['data'],
                                        np.concatenate((np.zeros((gyro_dataset[stage]['data'].shape[0], 1, gyro_dataset[stage]['data'].shape[2])), np.diff(gyro_dataset[stage]['data'], axis=1)), axis=1)), axis=-1),
                                    np.concatenate((
                                        rotation_dataset[stage]['data'],
                                        np.concatenate((np.zeros((rotation_dataset[stage]['data'].shape[0], 1, rotation_dataset[stage]['data'].shape[2])), np.diff(rotation_dataset[stage]['data'], axis=1)), axis=1)), axis=-1),
                                    np.concatenate((
                                        integrate_known(magnetic_dataset[stage]['data'], magnetic_dataset[stage]['data']),
                                        rotate_known(magnetic_dataset[stage]['data'], rotation_dataset[stage]['data']),
                                        magnetic_dataset[stage]['data'],
                                        np.concatenate((np.zeros((magnetic_dataset[stage]['data'].shape[0], 1, magnetic_dataset[stage]['data'].shape[2])), np.diff(magnetic_dataset[stage]['data'], axis=1)), axis=1)), axis=-1)
                                   ), axis=-1),
            'labels': accel_dataset[stage]['labels']
        }
        for stage in accel_dataset
    }
    return dataset
