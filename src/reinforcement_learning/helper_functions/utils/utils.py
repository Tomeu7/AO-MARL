import numpy as np

def get_projector_measurements2modes(imat, v2m, m2v, nfilt=None):
    """
    :imat measurements x commands
    """
    print("Imat", imat.shape)
    print("v2m", v2m.shape)
    print("m2v", m2v.shape)
    # 1. measurements x modes
    method = 3
    print("method", method)
    #  D = np.array(rtc.d_control[0].d_imat)
    #     #D = ao.imat_geom(wfs,config.p_wfss,config.p_controllers[0],dms,config.p_dms,meth=0)
    #     # Filtering on Btt modes
    #     Btt_filt = np.zeros((Btt.shape[0], Btt.shape[1] - nfilt))
    #     Btt_filt[:, :Btt_filt.shape[1] - 2] = Btt[:, :Btt.shape[1] - (nfilt + 2)]
    #     Btt_filt[:, Btt_filt.shape[1] - 2:] = Btt[:, Btt.shape[1] - 2:]
    #
    #     # Modal interaction basis
    #     Dm = D.dot(Btt_filt)
    #     # Direct inversion
    #     Dmp = np.linalg.inv(Dm.T.dot(Dm)).dot(Dm.T)
    #     # Command matrix
    #     cmat = Btt_filt.dot(Dmp)
    #     rtc.d_control[0].set_cmat(cmat.astype(np.float32))
    if method == 1:
        # Imat (592, 338)
        # v2m (335, 338)
        # m2v (338, 335)
        # Action range [ 0  1  2  3  4  5  6  7  8  9 10 11 12 13 14 15 16 17 18 19 20 21 22 23
        #  24 25 26 27 28 29 30 31 32 33 34 35 36 37 38 39]
        # method 1
        # Imat in modes (592, 335)
        # Imat in modes after filtering (592, 40)
        # Projector (592, 40)
        # r= -23350219000000.0
        # r= -22585933000000.0
        # r= -4205974000000.0
        imat_in_modes = imat.dot(v2m.T)
        print("Imat in modes", imat_in_modes.shape)
        # 2. filter, measurements x modes filtered
        if action_range is not None:
            imat_in_modes = imat_in_modes[:, action_range]
            print("Imat in modes after filtering", imat_in_modes.shape)
        # 3. modes filtered x measurements ??
        projector = np.linalg.inv(imat_in_modes.dot(imat_in_modes.transpose())).dot(imat_in_modes)
        print("Projector", projector.shape)
    elif method == 2:
        imat_in_modes = v2m.dot(imat.T)
        print("Imat in modes", imat_in_modes.shape)
        # 2. filter, measurements x modes filtered
        if action_range is not None:
            imat_in_modes = imat_in_modes[action_range, :]
            print("Imat in modes after filtering", imat_in_modes.shape)
        # 3. modes filtered x measurements ??
        projector = np.linalg.inv(imat_in_modes.dot(imat_in_modes.transpose())).dot(imat_in_modes)
        print("Projector", projector.shape)
        # Imat (592, 338)
        # v2m (335, 338)
        # m2v (338, 335)
        # Action range [ 0  1  2  3  4  5  6  7  8  9 10 11 12 13 14 15 16 17 18 19 20 21 22 23
        #  24 25 26 27 28 29 30 31 32 33 34 35 36 37 38 39]
        # method 2
        # Imat in modes (335, 592)
        # Imat in modes after filtering (40, 592)
        # Projector (40, 592)
        # r= -74332540000.0
        # r= -75610210000.0
        # r= -24050147000.0
    else:

        # 1. filter modes
        m2v_filtered = np.zeros((m2v.shape[0], m2v.shape[1] - nfilt))
        m2v_filtered[:, :m2v_filtered.shape[1] - 2] = m2v[:, :m2v.shape[1] - (nfilt + 2)]
        m2v_filtered[:, m2v_filtered.shape[1] - 2:] = m2v[:, m2v.shape[1] - 2:]

        print("m2v-filtered", m2v_filtered.shape)
        # 2. measurements x modes filtered
        imat_in_modes = imat.dot(m2v_filtered)
        print("Imat in modes after filtering", imat_in_modes.shape)
        # 3. modes filtered x measurements ??
        projector = np.linalg.inv(imat_in_modes.T.dot(imat_in_modes)).dot(imat_in_modes.T)
        print("Projector", projector.shape)

    return projector

def get_projector_wfsphase2modes(supervisor, v2m, m2v):
    """

    :param supervisor:
    :param v2m:
    :return:
    """
    mode = 2
    supervisor.reset()
    if mode == 1:
        actubas = supervisor.basis.compute_influ_basis(0).toarray()
        # actubas /= actubas.shape[1]
        row_sums = actubas.sum(axis=1)
        actubas = actubas / row_sums[:, np.newaxis]
        actubas_tt = supervisor.basis.compute_influ_basis(2).toarray()
        # actubas_tt /= actubas_tt.shape[1]
        row_sums = actubas_tt.sum(axis=1)
        actubas_tt = actubas_tt / row_sums[:, np.newaxis]
        projector = np.concatenate([actubas, actubas_tt])
        # projector = m2v.T.dot(actubas)
    else:
        # apply the pupil instead of full
        pupil = supervisor.get_m_pupil()
        # v2m: modes x volts, v2m.shape[0] modes shape
        modes_to_wfs = np.zeros((v2m.shape[0] - 2, np.where(pupil)[0].shape[0]))
        tt_to_wfs = np.zeros((2, np.where(pupil)[0].shape[0]))

        for mode in range(v2m.shape[0]):
            # send_to_dm = v2m[mode, :]  # This has shape modes x volts
            send_to_dm = m2v[:, mode]  # This has shape volts x modes
            if mode < (v2m.shape[0] - 2):
                supervisor.dms._dms.d_dms[0].set_com(send_to_dm[:-2])
                supervisor.dms._dms.d_dms[0].comp_shape()
                supervisor.wfs.raytrace(index=0, dms=supervisor.dms, ncpa=False, reset=True)
                phase = supervisor.wfs.get_wfs_phase(0)
                pupil = supervisor.get_m_pupil()
                phase = phase[np.where(pupil)]
                modes_to_wfs[mode, :] = phase.copy()
                supervisor.reset()
            else:
                try:
                    supervisor.dms._dms.d_dms[2].set_com(send_to_dm[-2:])
                    supervisor.dms._dms.d_dms[2].comp_shape()
                    supervisor.wfs.raytrace(index=0, dms=supervisor.dms, ncpa=False, reset=True)
                    phase = supervisor.wfs.get_wfs_phase(0)
                    pupil = supervisor.get_m_pupil()
                    phase = phase[np.where(pupil)]
                    tt_to_wfs[mode - (v2m.shape[0] - 2), :] = phase.copy()
                    supervisor.reset()
                except:
                    supervisor.dms._dms.d_dms[1].set_com(send_to_dm[-2:])
                    supervisor.dms._dms.d_dms[1].comp_shape()
                    supervisor.wfs.raytrace(index=0, dms=supervisor.dms, ncpa=False, reset=True)
                    phase = supervisor.wfs.get_wfs_phase(0)
                    pupil = supervisor.get_m_pupil()
                    phase = phase[np.where(pupil)]
                    tt_to_wfs[mode - (v2m.shape[0] - 2), :] = phase.copy()
                    supervisor.reset()

        # modes_to_wfs_reshaped = modes_to_wfs.reshape(v2m.shape[0] - 2, phase.shape[0] * phase.shape[1])
        projector = \
          np.linalg.inv(modes_to_wfs.dot(modes_to_wfs.transpose())).dot(modes_to_wfs)
        # projector = modes_to_wfs.T.dot(np.linalg.inv(modes_to_wfs.dot(modes_to_wfs.T)))

        projector_tt = \
           np.linalg.inv(tt_to_wfs.dot(tt_to_wfs.transpose())).dot(tt_to_wfs)
        # projector_tt = tt_to_wfs.T.dot(np.linalg.inv(tt_to_wfs.dot(tt_to_wfs.T)))


        # identity = projector.dot(modes_to_wfs_reshaped)

        # not_identity = modes_to_wfs_reshaped.dot(projector)
        try:
            projector = np.concatenate([projector, projector_tt])
        except:
            projector = np.concatenate([projector.T, projector_tt.T])
    return projector

def get_projector_dm2modes(supervisor, v2m, plot_modes=False):
    """

    :param supervisor:
    :param v2m:
    :param plot_modes:
    :return:
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