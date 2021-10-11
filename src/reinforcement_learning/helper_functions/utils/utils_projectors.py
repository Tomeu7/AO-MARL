import numpy as np

def get_projector_measurements2modes(imat, m2v, nfilt=None):
    """
    :param imat: Interaction Matrix. Shape: measurements x commands.
    :param m2v: Btt modes to Volts matrix. Shape: commands x modes.
    :param nfilt: Number of modes filtered
    :return: Projector from measurements 2 modes
    """

    # 1. filter modes
    m2v_filtered = np.zeros((m2v.shape[0], m2v.shape[1] - nfilt))
    m2v_filtered[:, :m2v_filtered.shape[1] - 2] = m2v[:, :m2v.shape[1] - (nfilt + 2)]
    m2v_filtered[:, m2v_filtered.shape[1] - 2:] = m2v[:, m2v.shape[1] - 2:]

    # 2. measurements x modes filtered
    imat_in_modes = imat.dot(m2v_filtered)
    # 3. modes filtered x measurements ??
    projector = np.linalg.inv(imat_in_modes.T.dot(imat_in_modes)).dot(imat_in_modes.T)

    return projector


def get_projector_wfsphase2modes(supervisor, m2v):
    """
    Note we consider pzt dm as the first DM and tt DM as the second DM
    :param supervisor: Supervisor Object
    :param m2v: Btt modes to Volts matrix. Shape: commands x modes.
    :return: Projector from wfs phase 2 modes
    """
    supervisor.reset()
    pupil = supervisor.get_m_pupil()
    modes_to_wfs = np.zeros((m2v.shape[1] - 2, np.where(pupil)[0].shape[0]))
    tt_to_wfs = np.zeros((2, np.where(pupil)[0].shape[0]))

    for mode in range(m2v.shape[1]):
        send_to_dm = m2v[:, mode]  # This has shape volts x modes
        if mode < (m2v.shape[1] - 2):
            supervisor.dms._dms.d_dms[0].set_com(send_to_dm[:-2])
            supervisor.dms._dms.d_dms[0].comp_shape()
            supervisor.wfs.raytrace(index=0, dms=supervisor.dms, ncpa=False, reset=True)
            phase = supervisor.wfs.get_wfs_phase(0)
            pupil = supervisor.get_m_pupil()
            phase = phase[np.where(pupil)]
            modes_to_wfs[mode, :] = phase.copy()
            supervisor.reset()
        else:
            supervisor.dms._dms.d_dms[1].set_com(send_to_dm[-2:])
            supervisor.dms._dms.d_dms[1].comp_shape()
            supervisor.wfs.raytrace(index=0, dms=supervisor.dms, ncpa=False, reset=True)
            phase = supervisor.wfs.get_wfs_phase(0)
            pupil = supervisor.get_m_pupil()
            phase = phase[np.where(pupil)]
            tt_to_wfs[mode - (m2v.shape[1] - 2), :] = phase.copy()
            supervisor.reset()

    projector = np.linalg.inv(modes_to_wfs.dot(modes_to_wfs.transpose())).dot(modes_to_wfs)
    projector_tt = np.linalg.inv(tt_to_wfs.dot(tt_to_wfs.transpose())).dot(tt_to_wfs)

    projector = np.concatenate([projector, projector_tt])

    return projector


def get_projector_dm2modes(supervisor, v2m):
    """
    With this function you can plot each mode how it would look in the DM.
    TT is not included but it can be added easely.
    Note we consider pzt dm as the first DM and tt DM as the second DM
    :param supervisor: Supervisor Object
    :param v2m: Volts 2 Btt modes matrix. Shape: modes x commands.
    :return: Projector from dm shape 2 modes
    """
    dm = supervisor.dms.get_dm_shape(0)
    modes_to_dm = np.zeros((v2m.shape[0], dm.shape[0], dm.shape[1]))
    for mode in range(v2m.shape[0]):
        send_to_dm = v2m[mode, :]
        supervisor.dms._dms.d_dms[0].set_com(send_to_dm, send_to_dm.size)
        supervisor.dms._dms.d_dms[0].comp_shape()
        dm = supervisor.dms.get_dm_shape(0)
        modes_to_dm[mode, :, :] = dm.copy()

    modes_to_dm_reshaped = modes_to_dm.reshape(v2m.shape[0], dm.shape[0] * dm.shape[1])
    projector = \
        np.linalg.inv(modes_to_dm_reshaped.dot(modes_to_dm_reshaped.transpose())).dot(modes_to_dm_reshaped)

    return projector