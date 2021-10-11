import matplotlib.pyplot as plt
import numpy as np
from guardians.drax import get_tar_image, plotCovCor, get_err
from guardians.gamora import psf_rec_Vii
import h5py

plt.style.use("ggplot")
OUT_FOLDER = "output/error_budget/h5_files/roket_output/"


def cutsPSF(filename, psf, psfs):
    """
    Plots cuts of two PSF along X and Y axis for comparison
    Args:
        filename: (str): path to the ROKET file
        psf: (np.ndarray[ndim=2,dtype=np.float32]): first PSF
        psfs: (np.ndarray[ndim=2,dtype=np.float32]): second PSF
    """
    f = h5py.File(filename, 'r')
    Lambda_tar = f.attrs["_Param_target__Lambda"][0]
    RASC = 180 / np.pi * 3600.
    pixsize = Lambda_tar * 1e-6 / (psf.shape[0] * f.attrs["_Param_tel__diam"] / f.attrs[
            "_Param_geom__pupdiam"]) * RASC
    x = (np.arange(psf.shape[0]) - psf.shape[0] / 2) * pixsize / (
            Lambda_tar * 1e-6 / f.attrs["_Param_tel__diam"] * RASC)

    plt.figure(num=1, figsize=(8, 8))

    # Plot 1
    plt.subplot(2, 1, 1)
    plt.plot(x, psf[psf.shape[0] // 2, :], color="blue")
    plt.plot(x, psfs[psf.shape[0] // 2, :], color="red")
    plt.plot(x,
             np.abs(psf[psf.shape[0] // 2, :] - psfs[psf.shape[0] // 2, :]),
             color="green")
    plt.yscale('log')
    plt.xlabel("X-axis angular distance [units of lambda/D]")
    plt.ylabel("Normalized intensity")
    plt.legend(["PSF Roket", "PSF Compass", "Diff"])
    plt.xlim(-20, 20)
    plt.ylim(1e-7, 1)

    # Plot 2
    plt.subplot(2, 1, 2)
    plt.plot(x, psf[:, psf.shape[0] // 2], color="blue")
    plt.plot(x, psfs[:, psf.shape[0] // 2], color="red")
    plt.plot(x,
             np.abs(psf[:, psf.shape[0] // 2] - psfs[:, psf.shape[0] // 2]),
             color="green")
    plt.yscale('log')
    plt.xlabel("Y-axis angular distance [units of lambda/D]")
    plt.ylabel("Normalized intensity")
    plt.legend(["PSF Roket", "PSF Compass", "Diff"])
    plt.xlim(-20, 20)
    plt.ylim(1e-7, 1)

    par_real = filename.split("/")[-1]
    plt.savefig(OUT_FOLDER + "psf_comp" + par_real[:-3] + '.png')
    f.close()
    plt.close("all")


def check_strehl_roket_vs_compass(parameter_file, RL):
    _, _, psf, _ = psf_rec_Vii(parameter_file, RL=RL)
    psf_compass = get_tar_image(parameter_file)

    print("1. L.E. SR from Roket reconstruction:", psf.max())
    print("2. L.E. SR from Compass:", psf_compass.max())

    cutsPSF(parameter_file, psf, psf_compass)


# 2. Errors computation

def errors_computation(parameter_file, num_agents):
    f = h5py.File(parameter_file, 'r')

    P = f["P"][:]

    err_rl = get_err(parameter_file, RL=True)
    err_rl = P.dot(err_rl).var(axis=1)

    err_linear = get_err(parameter_file, RL=False)
    err_linear = P.dot(err_linear).var(axis=1)

    if err_rl.shape[0] > 600:
        get_error_per_agent_8m(err_linear, err_rl, parameter_file, num_agents, P)
    else:
        get_error_per_agent_2m(err_linear, err_rl, parameter_file, P)
    f.close()


def get_error_per_agent_8m(original_result, final_result, parameter_file, num_agents, P):
    list_original_results = []
    list_final_results = []
    pad = int(1260/42)
    for modes in range(0, 1260, pad):
        list_original_results.append(np.sum(original_result[modes:(modes + pad)]))
        list_final_results.append(np.sum(final_result[modes:(modes + pad)]))

    list_original_results.append(np.sum(original_result[-2:]))
    list_final_results.append(np.sum(final_result[-2:]))

    plt.figure(figsize=(9, 6))
    plt.bar(np.arange(len(list_original_results)), list_original_results, label="Integrator", color="#348ABD")
    plt.bar(np.arange(len(list_final_results)), list_final_results, label="MARL", color="#E24A33")
    plt.xlabel("Agent number")
    plt.ylabel(r'Variance per agent $(\mu m^2)$')
    plt.legend()
    par_real = parameter_file.split("/")[-1]
    out_folder = OUT_FOLDER + "error_per_rl_" + par_real[:-3] + ".pdf"
    plt.savefig(out_folder)

    plt.figure(figsize=(9, 6))
    plt.plot(original_result, label="Integrator")
    plt.plot(final_result, label="Integrator + RL")
    plt.xlabel("Mode")
    plt.ylabel(r'Variance per mode $(\mu m^2)$')
    plt.legend()
    par_real = parameter_file.split("/")[-1]
    out_folder = OUT_FOLDER + "error_per_mode_" + par_real[:-3] + ".png"
    plt.savefig(out_folder)

    # Ratios
    ratios = np.array(list_final_results)/np.array(list_original_results)
    print("Ratios", ratios)
    plt.figure(figsize=(9, 6))
    plt.bar(np.arange(len(list_original_results)), ratios)
    plt.xlabel("Agent number")
    plt.ylabel('Ratios variance per agent (MARL/Int)')
    plt.legend()
    par_real = parameter_file.split("/")[-1]
    out_folder = OUT_FOLDER + "ratios_" + par_real[:-3] + ".pdf"
    plt.savefig(out_folder)


def get_error_per_agent_2m(original_result, final_result, parameter_file, P):
    plt.bar(np.arange(P.shape[0]), original_result, label="linear")
    plt.bar(np.arange(P.shape[0]), final_result, label="linear + rl")
    plt.xlabel("Mode number")
    plt.ylabel(r'Variance per mode $(\mu m^2)$')
    plt.legend()
    par_real = parameter_file.split("/")[-1]
    out_folder = OUT_FOLDER + "error_per_mode_" + par_real[:-3] + ".png"
    plt.savefig(out_folder)

# 3. Covariance Correlation


def get_result_breakdown(parameter_file,
                         num_agents=43,
                         RL=True):

    parameter_file = "output/error_budget/h5_files/" + parameter_file

    # ----------------------------------------  1. Check it works (RL as a mark) ---------------------------------------
    check_strehl_roket_vs_compass(parameter_file, RL)

    # ----------------------------------------  2. Errors computation ----------------------------------------
    errors_computation(parameter_file, num_agents)

    # ----------------------------------------  3. Covariance Correlation ----------------------------------------

    plotCovCor(parameter_file, plot=True, include_anisoplanatism=False, include_noise=True)