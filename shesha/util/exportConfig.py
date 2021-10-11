from collections import OrderedDict
import numpy as np
from shesha.constants import CentroiderType, WFSType

def export_config(supervisor):
    """
    Extract and convert compass supervisor configuration parameters 
    into 2 dictionnaries containing relevant AO parameters

    Args:
        root: (object), COMPASS supervisor object to be parsed

    Returns : 2 dictionnaries
    """
    aodict = OrderedDict()
    dataDict = {}

    if (supervisor.config.p_tel is not None):
        aodict.update({"teldiam": supervisor.config.p_tel.diam})
        aodict.update({"telobs": supervisor.config.p_tel.cobs})
        aodict.update({"pixsize": supervisor.config.p_geom._pixsize})
        # TURBU
        aodict.update({"r0": supervisor.config.p_atmos.r0})
        aodict.update({"Fe": 1 / supervisor.config.p_loop.ittime})
        aodict.update({"nbTargets": len(supervisor.config.p_targets)})
    else:
        aodict.update({"nbTargets": 1})
 
    # WFS
    aodict.update({"nbWfs": len(supervisor.config.p_wfss)})
    aodict.update({"nbCam": aodict["nbWfs"]})
    aodict.update({"nbOffaxis": 0})
    aodict.update({"nbNgsWFS": 1})
    aodict.update({"nbLgsWFS": 0})
    aodict.update({"nbFigSensor": 0})
    aodict.update({"nbSkyWfs": aodict["nbWfs"]})
    aodict.update({"nbOffNgs": 0})

    # DMS
    aodict.update({"nbDms": len(supervisor.config.p_dms)})
    aodict.update({"Nactu": supervisor.config.p_controllers[0].nactu})
    # List of things
    aodict.update({"list_NgsOffAxis": []})
    aodict.update({"list_Fig": []})
    aodict.update({"list_Cam": [0]})
    aodict.update({"list_SkyWfs": [0]})
    aodict.update({"list_ITS": []})
    aodict.update({"list_Woofer": []})
    aodict.update({"list_Tweeter": []})
    aodict.update({"list_Steering": []})

    listOfNstatesPerController = []
    listOfcontrolLawTypePerController = []
    for control in supervisor.config.p_controllers:
        listOfNstatesPerController.append(control.nstates)
        listOfcontrolLawTypePerController.append(control.type)
    aodict.update({"list_nstatesPerController": listOfNstatesPerController})
    aodict.update({"list_controllerType": listOfcontrolLawTypePerController})

    # fct of Nb of wfss
    NslopesList = []
    NsubapList = []
    listWfsType = []
    listCentroType = []

    pyrModulationList = []
    pyr_npts = []
    pyr_pupsep = []
    pixsize = []
    xPosList = []
    yPosList = []
    fstopsize = []
    fstoptype = []
    npixPerSub = []
    nxsubList = []
    nysubList = []
    lambdaList = []
    dms_seen = []
    colTmpList = []
    noise = []
    #new_hduwfsl = pfits.HDUList()
    #new_hduwfsSubapXY = pfits.HDUList()
    for i in range(aodict["nbWfs"]):
        #new_hduwfsl.append(pfits.ImageHDU(supervisor.config.p_wfss[i]._isvalid))  # Valid subap array
        #new_hduwfsl[i].header["DATATYPE"] = "valid_wfs%d" % i
        dataDict["wfsValid_" + str(i)] = supervisor.config.p_wfss[i]._isvalid

        xytab = np.zeros((2, supervisor.config.p_wfss[i]._validsubsx.shape[0]))
        xytab[0, :] = supervisor.config.p_wfss[i]._validsubsx
        xytab[1, :] = supervisor.config.p_wfss[i]._validsubsy
        dataDict["wfsValidXY_" + str(i)] = xytab

        #new_hduwfsSubapXY.append(pfits.ImageHDU(xytab))  # Valid subap array inXx Y on the detector
        #new_hduwfsSubapXY[i].header["DATATYPE"] = "validXY_wfs%d" % i
        pixsize.append(supervisor.config.p_wfss[i].pixsize)
        """
        if (supervisor.config.p_centroiders[i].type == "maskedpix"):
            factor = 4
        else:
            factor = 2
        NslopesList.append(
                supervisor.config.p_wfss[i]._nvalid * factor)  # slopes per wfs
        """
        listCentroType.append(
                supervisor.config.p_centroiders[i].
                type)  # assumes that there is the same number of centroiders and wfs
        NsubapList.append(supervisor.config.p_wfss[i]._nvalid)  # subap per wfs
        listWfsType.append(supervisor.config.p_wfss[i].type)
        xPosList.append(supervisor.config.p_wfss[i].xpos)
        yPosList.append(supervisor.config.p_wfss[i].ypos)
        fstopsize.append(supervisor.config.p_wfss[i].fssize)
        fstoptype.append(supervisor.config.p_wfss[i].fstop)
        nxsubList.append(supervisor.config.p_wfss[i].nxsub)
        nysubList.append(supervisor.config.p_wfss[i].nxsub)
        lambdaList.append(supervisor.config.p_wfss[i].Lambda)
        if (supervisor.config.p_wfss[i].dms_seen is not None):
            dms_seen.append(list(supervisor.config.p_wfss[i].dms_seen))
            noise.append(supervisor.config.p_wfss[i].noise)

        if (supervisor.config.p_centroiders[i].type == CentroiderType.MASKEDPIX):
            NslopesList.append(supervisor.config.p_wfss[i]._nvalid * 4)  # slopes per wfs
        else:
            NslopesList.append(supervisor.config.p_wfss[i]._nvalid * 2)  # slopes per wfs

        if (supervisor.config.p_wfss[i].type == "pyrhr"):
            pyrModulationList.append(supervisor.config.p_wfss[i].pyr_ampl)
            pyr_npts.append(supervisor.config.p_wfss[i].pyr_npts)
            pyr_pupsep.append(supervisor.config.p_wfss[i].pyr_pup_sep)
            npixPerSub.append(1)
        else:
            pyrModulationList.append(0)
            pyr_npts.append(0)
            pyr_pupsep.append(0)
            npixPerSub.append(supervisor.config.p_wfss[i].npix)
    """
    confname = filepath.split("/")[-1].split('.conf')[0]
    print(filepath.split(".conf")[0] + '_wfsConfig.fits')
    new_hduwfsl.writeto(
            filepath.split(".conf")[0] + '_wfsConfig.fits', overwrite=True)
    new_hduwfsSubapXY.writeto(
            filepath.split(".conf")[0] + '_wfsValidXYConfig.fits', overwrite=True)
    """
    if (len(dms_seen) != 0):
        aodict.update({"listWFS_dms_seen": dms_seen})

    aodict.update({"listWFS_NslopesList": NslopesList})
    aodict.update({"listWFS_NsubapList": NsubapList})
    aodict.update({"listWFS_CentroType": listCentroType})
    aodict.update({"listWFS_WfsType": listWfsType})
    aodict.update({"listWFS_pixarc": pixsize})
    aodict.update({"listWFS_pyrModRadius": pyrModulationList})
    aodict.update({"listWFS_pyrModNPts": pyr_npts})
    aodict.update({"listWFS_pyrPupSep": pyr_pupsep})
    aodict.update({"listWFS_fstopsize": fstopsize})
    aodict.update({"listWFS_fstoptype": fstoptype})
    aodict.update({"listWFS_NsubX": nxsubList})
    aodict.update({"listWFS_NsubY": nysubList})
    aodict.update({"listWFS_Nsub": nysubList})
    aodict.update({"listWFS_NpixPerSub": npixPerSub})
    aodict.update({"listWFS_Lambda": lambdaList})
    if (len(noise) != 0):
        aodict.update({"listWFS_noise": noise})

    listDmsType = []
    NactuX = []
    Nactu = []
    unitPerVolt = []
    push4imat = []
    coupling = []
    push4iMatArcSec = []
    #new_hdudmsl = pfits.HDUList()

    for j in range(aodict["nbDms"]):
        listDmsType.append(supervisor.config.p_dms[j].type)
        NactuX.append(
                supervisor.config.p_dms[j].nact)  # nb of actuators across the diameter !!
        Nactu.append(supervisor.config.p_dms[j]._ntotact)  # nb of actuators in total
        unitPerVolt.append(supervisor.config.p_dms[j].unitpervolt)
        push4imat.append(supervisor.config.p_dms[j].push4imat)
        coupling.append(supervisor.config.p_dms[j].coupling)
        tmp = []
        if (supervisor.config.p_dms[j]._i1 is
                    not None):  # Simu Case where i1 j1 is known (simulated)
            if (supervisor.config.p_dms[j].type != 'tt'):
                tmpdata = np.zeros((4, len(supervisor.config.p_dms[j]._i1)))
                tmpdata[0, :] = supervisor.config.p_dms[j]._j1
                tmpdata[1, :] = supervisor.config.p_dms[j]._i1
                tmpdata[2, :] = supervisor.config.p_dms[j]._xpos
                tmpdata[3, :] = supervisor.config.p_dms[j]._ypos
            else:
                tmpdata = np.zeros((4, 2))

            dataDict["dmData" + str(j)] = tmpdata
            """
            new_hdudmsl.append(pfits.ImageHDU(tmpdata))  # Valid subap array
            new_hdudmsl[j].header["DATATYPE"] = "valid_dm%d" % j
            """
            #for k in range(aodict["nbWfs"]):
            #    tmp.append(supervisor.computeDMrange(j, k))

            push4iMatArcSec.append(tmp)

    # new_hdudmsl.writeto(filepath.split(".conf")[0] + '_dmsConfig.fits', overwrite=True)
    if (len(push4iMatArcSec) != 0):
        aodict.update({"listDMS_push4iMat": push4imat})
        aodict.update({"listDMS_unitPerVolt": unitPerVolt})
    aodict.update({"listDMS_Nxactu": NactuX})
    aodict.update({"listDMS_Nyactu": NactuX})
    aodict.update({"listDMS_Nactu": Nactu})

    aodict.update({"listDMS_type": listDmsType})
    aodict.update({"listDMS_coupling": coupling})

    if (supervisor.config.p_targets is not None):  # simu case
        listTargetsLambda = []
        listTargetsXpos = []
        listTargetsYpos = []
        listTargetsDmsSeen = []
        listTargetsMag = []
        listTARGETS_pixsize = []
        for k in range(aodict["nbTargets"]):
            listTargetsLambda.append(supervisor.config.p_targets[k].Lambda)
            listTargetsXpos.append(supervisor.config.p_targets[k].xpos)
            listTargetsYpos.append(supervisor.config.p_targets[k].ypos)
            listTargetsMag.append(supervisor.config.p_targets[k].mag)
            listTargetsDmsSeen.append(list(supervisor.config.p_targets[k].dms_seen))
            PSFPixsize = (supervisor.config.p_targets[k].Lambda * 1e-6) / (
                    supervisor.config.p_geom._pixsize *
                    supervisor.config.p_geom.get_ipupil().shape[0]) * 206265.
            listTARGETS_pixsize.append(PSFPixsize)

        aodict.update({"listTARGETS_Lambda": listTargetsLambda})
        aodict.update({"listTARGETS_Xpos": listTargetsXpos})
        aodict.update({"listTARGETS_Ypos": listTargetsYpos})
        aodict.update({"listTARGETS_Mag": listTargetsMag})
        aodict.update({"listTARGETS_DmsSeen": listTargetsDmsSeen})
        aodict.update({"listTARGETS_pixsize": listTARGETS_pixsize})

    listDmsType = []
    Nslopes = sum(NslopesList)
    Nsubap = sum(NsubapList)
    aodict.update({"Nslopes": Nslopes})
    aodict.update({"Nsubap": Nsubap})
    return aodict, dataDict

