import pycbc.psd
import numpy as np
from time import gmtime, time, strftime, localtime

# import gwsurrogate
import os
import h5py
import functools
from pathlib import Path
import itertools

# from argparse import ArgumentParser
from tqdm import tqdm

# from einops import rearrange, reduce, repeat
os.environ["CUDA_VISIBLE_DEVICES"] = "1"

EPS = 1e-5

# #######################################################################################################################
# #######################################################################################################################


class GW_SE_Dataset(object):
    def __init__(
        self,
        length=16000,
        sample_rate=0.1,
        use_AAK=False,
        use_NRSur=False,
        use_PyCBC=False,
        BWD=False,
        SGWB=False,
    ):
        # parameter range
        if use_AAK:
            param_idx = dict(M=0, a=1, e0=2, Y0=3)
            nparams = 4
            self.param_idx = param_idx
            self.nparams = nparams
            self.AAK_par_range = dict(
                # M=[5, 7],
                M=[1e5, 1e7],
                a=[1e-3, 0.99],
                e0=[1e-3, 0.5],
                Y0=[-0.98, 0.99],
            )

        if use_PyCBC or use_NRSur:
            param_idx = dict(M=0, q=1, spin1z=2, spin2z=3)
            nparams = 4
            self.param_idx = param_idx
            self.nparams = nparams
            self.bbh_par_range = dict(
                M=[6, 8], q=[0.01, 1], spin1z=[-0.99, 0.99], spin2z=[-0.99, 0.99]
            )

        if BWD:
            param_idx = dict(f1=0, f_dot1=1, f2=2, f_dot2=3)
            nparams = 4
            self.param_idx = param_idx
            self.nparams = nparams
            self.BWD_par_range = dict(
                f1=[1e-3, 4e-3],
                f_dot1=[-4e-17, 7e-16],
                f2=[4e-3, 15e-3],
                f_dot2=[-3e-15, 4e-14],
            )

        self.sampling_rate = sample_rate
        self.buffer = 1000
        self.time_duration = (
            length + 2 * self.buffer
        ) / self.sampling_rate  # [s]   #3600.0*24*3
        self.f_min = max(3.0e-5, EPS + 1.0 / self.time_duration)  # Hertz

        self.SNR = [30.0]
        self.psd = self.get_psd(fname="LISA_PSD.txt")

        self.waveform_dataset = {
            "train": {"noisy": [], "clean": []},
            "test": {"noisy": [], "clean": []},
        }
        self.waveform_par = {"train": [], "test": []}
        if use_AAK:
            self.init_AAK_waveform_generator()
        if use_NRSur:
            self.init_NRSur_waveform_generator()
        if use_PyCBC:
            from pycbc.waveform import get_td_waveform

            self.get_td_waveform = get_td_waveform
        if SGWB:
            alpha = -11.352
            f_star = 1e-3
            n_t = 2 / 3
            self.sgwb_psd = self.get_sgwb_psd(alpha=alpha, n_t=n_t, f_star=f_star)
        return

    def init_NRSur_waveform_generator(self, model="NRHybSur3dq8"):
        import gwsurrogate

        p = Path(gwsurrogate.catalog.download_path())
        fname = p / "{}.h5".format(model)
        if not fname.exists():
            print("Downloading {}.h5".format(model))
            gwsurrogate.catalog.pull(model)
        self.NRSur = gwsurrogate.LoadSurrogate(model)

    def init_AAK_waveform_generator(self, use_gpu=True):
        from few.waveform import Pn5AAKWaveform

        # keyword arguments for inspiral generator (RunKerrGenericPn5Inspiral)
        inspiral_kwargs = {
            "DENSE_STEPPING": 0,  # we want a sparsely sampled trajectory
            "max_init_len": int(
                1e4
            ),  # all of the trajectories will be well under len = 1000
        }

        # keyword arguments for summation generator (AAKSummation)
        sum_kwargs = {
            "use_gpu": use_gpu,  # GPU is availabel for this type of summation
            "pad_output": False,
        }

        self.AAK = Pn5AAKWaveform(
            inspiral_kwargs=inspiral_kwargs, sum_kwargs=sum_kwargs, use_gpu=use_gpu
        )

    def tukey(self, M, alpha=0.5):
        """
        Tukey window code copied from scipy
        """
        n = np.arange(0, M)
        width = int(np.floor(alpha * (M - 1) / 2.0))
        n1 = n[0 : width + 1]
        n2 = n[width + 1 : M - width - 1]
        n3 = n[M - width - 1 :]

        w1 = 0.5 * (1 + np.cos(np.pi * (-1 + 2.0 * n1 / alpha / (M - 1))))
        w2 = np.ones(n2.shape)
        w3 = 0.5 * (1 + np.cos(np.pi * (-2.0 / alpha + 1 + 2.0 * n3 / alpha / (M - 1))))
        w = np.concatenate((w1, w2, w3))

        return np.array(w[:M])

    @property
    def f_max(self):
        """Set the maximum frequency to half the sampling rate."""
        return self.sampling_rate / 2.0

    @f_max.setter
    def f_max(self, f_max):
        self.sampling_rate = 2.0 * f_max

    @property
    def delta_t(self):
        return 1.0 / self.sampling_rate

    @delta_t.setter
    def delta_t(self, delta_t):
        self.sampling_rate = 1.0 / delta_t

    @property
    def delta_f(self):
        return 1.0 / self.time_duration

    @delta_f.setter
    def delta_f(self, delta_f):
        self.time_duration = 1.0 / delta_f

    @property
    def Nt(self):
        return int(self.time_duration * self.sampling_rate)

    @property
    def Nf(self):
        return int(self.f_max / self.delta_f + 0.5) + 1

    @property
    def sample_times(self):
        """Array of times at which waveforms are sampled."""
        return np.linspace(
            0.0, self.time_duration, num=self.Nt, endpoint=False, dtype=np.float32
        )

    @property
    @functools.lru_cache()
    def sample_frequencies(self):
        return np.linspace(
            0.0, self.f_max, num=self.Nf, endpoint=True, dtype=np.float32
        )

    def m1_m2_from_M_q(self, M, q):
        """Compute individual masses from total mass and mass ratio.

        Choose m1 >= m2.

        Arguments:
            M {float} -- total mass
            q {mass ratio} -- mass ratio, 0.0< q <= 1.0

        Returns:
            (float, float) -- (mass_1, mass_2)
        """

        m1 = M / (1.0 + q)
        m2 = q * m1

        return m1, m2

    def m1_m2_from_M_Chirp_q(self, M_Chirp, q):
        q = 1 / q
        eta = q / (1 + q) ** 2
        M = M_Chirp * eta ** (-3 / 5)
        return self.m1_m2_from_M_q(M, 1 / q)

    def get_psd(self, fname="LISA_PSD_2.txt"):
        psd = pycbc.psd.read.from_txt(
            fname, self.Nf, self.delta_f, self.f_min, is_asd_file=False
        )
        return psd.data

    def LISA_fpfc(self, theta_S_bar, phi_S_bar, phi_0_bar=0):
        yr = 3600 * 24 * 365
        phi_t_bar = phi_0_bar + self.sample_times * (2 * np.pi / yr)
        theta_t_bar = 0.5 * np.pi
        cos_theta_S_bar = np.cos(theta_S_bar)
        sin_theta_S_bar = np.sin(theta_S_bar)
        cos_phi_S_bar = np.cos(phi_t_bar - phi_S_bar)
        sin_phi_S_bar = np.sin(phi_t_bar - phi_S_bar)

        cos_theta_S = (
            0.5 * cos_theta_S_bar - 0.5 * np.sqrt(3) * sin_theta_S_bar * cos_phi_S_bar
        )
        # sin_theta_S = np.sqrt(1- cos_theta_S * cos_theta_S)
        phi_S = np.pi / 12 + np.arctan2(
            (np.sqrt(3) * cos_theta_S_bar + sin_theta_S_bar * cos_phi_S_bar),
            (2 * sin_theta_S_bar * cos_phi_S_bar),
        )
        cos_phi_S = np.cos(phi_S)
        sin_phi_S = np.sin(phi_S)
        cos_2phi_S = np.cos(2 * phi_S)
        sin_2phi_S = np.sin(2 * phi_S)

        psi_S = np.arctan2(
            (
                np.sqrt(3) * np.cos(phi_t_bar)
                + 2 * sin_theta_S_bar * cos_theta_S_bar * cos_phi_S_bar
            ),
            (2 * sin_theta_S_bar * sin_phi_S_bar),
        )
        cos_2psi_S = np.cos(2 * psi_S)
        sin_2psi_S = np.sin(2 * psi_S)

        tmp1 = 1 / 2 * (1 + cos_theta_S * cos_theta_S)
        # fp
        tmp2 = cos_2phi_S * cos_2psi_S
        tmp3 = cos_theta_S * sin_2phi_S * sin_2psi_S
        # fc
        tmp4 = cos_2phi_S * sin_2psi_S
        tmp5 = cos_theta_S * sin_2phi_S * cos_2psi_S
        fp = tmp1 * (tmp2 - tmp3)
        fc = tmp1 * (tmp4 + tmp5)
        return fp, fc

    def proj(self, hp, hc, fp, fc):
        return np.sqrt(3) / 2 * (fp * hp + fc * hc)

    def gen_bbh_signal_nrsur(self, m1, m2, f_low=1e-5, chiA=[0, 0, 0], chiB=[0, 0, 0]):
        q = m1 / m2  # m1>m2
        # chiA = [0, 0, 0.5]
        # chiB = [0, 0, -0.7]
        M = m1 + m2  # Total masss in solar masses
        dist_mpc = 100  # distance in megaparsecs
        # dt = 1./8192       # step size in seconds
        # f_low = 20         # initial frequency in Hz
        t, h, dyn = self.NRSur(
            q,
            chiA,
            chiB,
            dt=self.delta_t,
            f_low=f_low,
            mode_list=[(2, 2)],
            M=M,
            dist_mpc=dist_mpc,
            units="mks",
        )
        n = self.sampling_rate * self.time_duration
        return h[(2, 2)].real[-int(n) :]

    def gen_bbh_signal_pycbc(self, par, apx="SEOBNRv4_opt"):
        M = 10 ** par[self.param_idx["M"]]
        # M = par[self.param_idx['M']]
        q = par[self.param_idx["q"]]
        ratio = M / 100
        m1, m2 = self.m1_m2_from_M_q(100, q)
        # print(ratio*self.f_min)
        f_lower = np.max([ratio * self.f_min, 3])

        hp, hc = self.get_td_waveform(
            approximant=apx,
            mass1=m1,
            mass2=m2,
            spin1z=par[self.param_idx["spin1z"]],
            spin2z=par[self.param_idx["spin2z"]],
            delta_t=self.delta_t / ratio,
            f_lower=f_lower,
        )

        a = np.where(hp.sample_times < self.delta_t / ratio)
        merge_idx = a[0][-1]
        # length = int(self.time_duration*self.sampling_rate)
        idx1 = merge_idx - int((self.Nt - 2 * self.buffer) * 3 / 4) - self.buffer
        idx2 = idx1 + self.Nt - self.buffer

        fp, fc = self.LISA_fpfc(0, 0, 0)
        if idx1 > 0:
            try:
                tmp = hp.data[idx2]

                hp1 = hp.data[idx1:idx2]
                hc1 = hc.data[idx1:idx2]
                return self.proj(hp1, hc1, fp, fc)
            except:
                hp1 = hp.data[-self.Nt :]
                hc1 = hc.data[-self.Nt :]
                return self.proj(hp1, hc1, fp, fc)
        else:
            try:
                tmp = hp.data[self.Nt]

                hp1 = hp.data[: self.Nt]
                hc1 = hc.data[: self.Nt]
                return self.proj(hp1, hc1, fp, fc)
            except:
                hp1 = np.zeros(self.Nt)
                hc1 = np.zeros(self.Nt)
                hp1[: len(hp)] = hp
                hc1[: len(hc)] = hc
                return self.proj(hp1, hc1, fp, fc)

    def gen_emri_signal(self, par):
        # set initial parameters
        # M = 1e5
        mu = 10
        # M = 10 ** par[self.param_idx['M']]
        M = par[self.param_idx["M"]]
        # SMBH spin
        a = par[self.param_idx["a"]]
        # a = 0.8
        # semi-latus rectum ---'separation'
        p0 = 20.0
        # initial eccentricity
        # e0 = 0.1
        e0 = par[self.param_idx["e0"]]
        # PN orbital parameter
        # iota0 = 0.7
        # iota0 = par[self.param_idx['iota0']]
        # Y0 = np.cos(iota0)
        Y0 = par[self.param_idx["Y0"]]

        # 3 direction mode initial phase
        Phi_phi0 = 0.0
        Phi_theta0 = 0.0
        Phi_r0 = 0.0

        # sky location
        qS = 1e-6
        phiS = 1e-6

        # SMBH spin direction
        qK = 1e-6
        phiK = 1e-6

        # D_L
        dist = 1.0
        # ...............
        mich = False
        # ...............
        dt = self.delta_t
        T = 0.01
        waveform = self.AAK(
            M,
            mu,
            a,
            p0,
            e0,
            Y0,
            qS,
            phiS,
            qK,
            phiK,
            dist,
            Phi_phi0=Phi_phi0,
            Phi_theta0=Phi_theta0,
            Phi_r0=Phi_r0,
            mich=mich,
            dt=dt,
            T=T,
        )
        # n = self.sampling_rate * self.time_duration
        hp = waveform.get().real[-self.Nt :]
        hc = waveform.get().imag[-self.Nt :]
        fp, fc = self.LISA_fpfc(0, 0, 0)
        return self.proj(hp, hc, fp, fc)

    def gen_bwd_signal(self, par):
        f = par[self.param_idx["f1"]]
        f_dot = par[self.param_idx["f_dot1"]]
        phi = 2 * np.pi * f * self.sample_times + np.pi * f_dot * self.sample_times**2
        hp = np.cos(phi)
        hc = np.sin(phi)
        fp, fc = self.LISA_fpfc(0, 0, 0)
        return self.proj(hp, hc, fp, fc)

    def gen_noise(self):
        """
        Generates noise from a psd
        """
        T_obs = self.time_duration
        fs = self.sampling_rate
        psd = self.psd

        N = T_obs * fs  # the total number of time samples
        # Nf = N // 2 + 1
        dt = 1 / fs  # the sampling time (sec)
        df = 1 / T_obs

        amp = np.sqrt(0.25 * T_obs * psd)
        idx = np.argwhere(psd == 0.0)
        amp[idx] = 0.0
        re = amp * np.random.normal(0, 1, self.Nf)
        im = amp * np.random.normal(0, 1, self.Nf)
        re[0] = 0.0
        im[0] = 0.0
        x = N * np.fft.irfft(re + 1j * im) * df

        return x

    def get_sgwb_psd(self, alpha=-11.352, n_t=2 / 3, f_star=1e-3):
        psd = np.zeros(self.sample_frequencies.shape, dtype=np.float64)

        h2_Omega_gw = 10 ** (alpha) * (self.sample_frequencies / f_star) ** (n_t)
        a = (3 / (2 * np.pi**2)) * ((3.24e-18) ** 2)
        h = 0.67

        fidx1 = int(self.f_min / self.delta_f)
        fidx2 = int(self.f_max / self.delta_f)
        psd[fidx1:fidx2] = (h2_Omega_gw[fidx1:fidx2]) / (
            self.sample_frequencies[fidx1:fidx2] ** 3
        )
        psd_tmp = a * h**2 * psd
        return psd_tmp

    def gen_sgwb_signal(self):
        """
        Generates noise from a psd
        """
        T_obs = self.time_duration
        fs = self.sampling_rate
        psd = self.sgwb_psd

        N = T_obs * fs  # the total number of time samples
        # Nf = N // 2 + 1
        dt = 1 / fs  # the sampling time (sec)
        df = 1 / T_obs

        amp = np.sqrt(0.25 * T_obs * psd)
        idx = np.argwhere(psd == 0.0)
        amp[idx] = 0.0
        re = amp * np.random.normal(0, 1, self.Nf)
        im = amp * np.random.normal(0, 1, self.Nf)
        re[0] = 0.0
        im[0] = 0.0
        x = N * np.fft.irfft(re + 1j * im) * df

        return x

    def whiten_data(self, data, flag="td"):
        """
        Takes an input timeseries and whitens it according to a psd
        """
        duration = self.time_duration
        sample_rate = self.sampling_rate
        psd = self.psd

        if flag == "td":
            # FT the input timeseries - window first
            win = self.tukey(int(duration * sample_rate), alpha=1.0 / 8.0)
            xf = np.fft.rfft(win * data)
            # xf = np.fft.rfft(data)
        else:
            xf = data

        # deal with undefined PDS bins and normalise
        idx = np.argwhere(psd > 0.0)
        invpsd = np.zeros(psd.size)
        invpsd[idx] = 1.0 / psd[idx]
        xf *= np.sqrt(2.0 * invpsd / sample_rate)

        # Detrend the data: no DC component.
        xf[0] = 0.0

        if flag == "td":
            # Return to time domain.
            x = np.fft.irfft(xf)
            return x
        else:
            return xf

    def get_snr(self, data):
        """
        computes the snr of a signal given a PSD starting from a particular frequency index
        """
        T_obs = self.time_duration
        fs = self.sampling_rate
        psd = self.psd
        fmin = self.f_min

        N = int(T_obs * fs)
        df = 1.0 / T_obs
        dt = 1.0 / fs
        fidx = int(fmin / df)

        win = self.tukey(N, alpha=1.0 / 8.0)
        idx = np.argwhere(psd > 0.0)
        invpsd = np.zeros(psd.size)
        invpsd[idx] = 1.0 / psd[idx]

        xf = np.fft.rfft(data * win) * dt

        SNRsq = 4.0 * np.sum((np.abs(xf[fidx:]) ** 2) * invpsd[fidx:]) * df
        return np.sqrt(SNRsq)

    def get_inner_product(self, data1, data2):
        """
        computes the snr of a signal given a PSD starting from a particular frequency index
        """
        T_obs = self.time_duration
        fs = self.sampling_rate
        psd = self.psd
        fmin = self.f_min

        N = int(T_obs * fs)
        df = 1.0 / T_obs
        dt = 1.0 / fs
        fidx = int(fmin / df)

        win = self.tukey(N, alpha=1.0 / 8.0)
        idx = np.argwhere(psd > 0.0)
        invpsd = np.zeros(psd.size)
        invpsd[idx] = 1.0 / psd[idx]

        xf1 = np.fft.rfft(data1 * win) * dt
        xf2 = np.fft.rfft(data2 * win) * dt
        SNRsq = (
            2.0
            * np.sum(
                (
                    xf1[fidx:] * np.conjugate(xf2[fidx:])
                    + np.conjugate(xf1[fidx:]) * xf2[fidx:]
                )
                * invpsd[fidx:]
            )
            * df
        )
        return SNRsq.real

    def generate_emri_dataset(self, n_grid=[4, 3]):
        """
        Args:
            grid_per_par : grid point of each parameter
        """
        # training set
        snr = self.SNR[0]
        grid_per_par = n_grid[0]
        par = np.zeros([self.nparams, grid_per_par])
        for pkey, idx in self.param_idx.items():
            par[idx, :] = np.linspace(
                self.AAK_par_range[pkey][0],
                self.AAK_par_range[pkey][1],
                num=grid_per_par,
                endpoint=True,
            )

        self.AAK_train_par = par

        waveform_num = grid_per_par**self.nparams
        self.waveform_dataset["train"]["clean"] = np.zeros(
            [int(waveform_num), self.Nt - 2 * self.buffer]
        )
        self.waveform_dataset["train"]["noisy"] = np.zeros(
            [int(waveform_num), self.Nt - 2 * self.buffer]
        )
        self.waveform_par["train"] = np.zeros([int(waveform_num), self.nparams])
        for i, par_idx in tqdm(
            enumerate(itertools.product(range(grid_per_par), repeat=self.nparams))
        ):
            p = [par[j, par_idx[j]] for j in range(self.nparams)]
            hp = self.gen_emri_signal(p)
            # print(hp.shape)
            noise = self.gen_noise()
            data = snr * hp / self.get_snr(hp) + noise
            hp = self.whiten_data(hp)
            data = self.whiten_data(data)
            # cut
            if self.buffer > 0:
                hp_cut = hp[self.buffer : -self.buffer]
                data_cut = data[self.buffer : -self.buffer]
            else:
                hp_cut = hp
                data_cut = data
            a1 = np.max(np.abs(hp_cut))
            a2 = np.max(np.abs(data_cut))

            self.waveform_dataset["train"]["clean"][i] = hp_cut / a1
            self.waveform_dataset["train"]["noisy"][i] = data_cut / a2
            self.waveform_par["train"][i] = np.array(p)

        # ________________________________
        del par
        del waveform_num
        # _________________________________
        grid_per_par = n_grid[1]

        # test set
        par = np.zeros([self.nparams, grid_per_par])
        for pkey, idx in self.param_idx.items():
            dpd2 = (
                (self.AAK_par_range[pkey][1] - self.AAK_par_range[pkey][0])
                / (par.shape[1])
                / 2
            )
            par[idx, :] = np.linspace(
                self.AAK_par_range[pkey][0] + dpd2,
                self.AAK_par_range[pkey][1] - dpd2,
                num=grid_per_par,
                endpoint=True,
            )

        self.AAK_test_par = par

        waveform_num = grid_per_par**self.nparams
        self.waveform_dataset["test"]["clean"] = np.zeros(
            [int(waveform_num), self.Nt - 2 * self.buffer]
        )
        self.waveform_dataset["test"]["noisy"] = np.zeros(
            [int(waveform_num), self.Nt - 2 * self.buffer]
        )
        self.waveform_par["test"] = np.zeros([int(waveform_num), self.nparams])

        for i, par_idx in tqdm(
            enumerate(itertools.product(range(grid_per_par), repeat=self.nparams))
        ):
            p = [par[j, par_idx[j]] for j in range(self.nparams)]
            hp = self.gen_emri_signal(p)
            noise = self.gen_noise()
            data = snr * hp / self.get_snr(hp) + noise
            hp = self.whiten_data(hp)
            data = self.whiten_data(data)
            # cut
            if self.buffer > 0:
                hp_cut = hp[self.buffer : -self.buffer]
                data_cut = data[self.buffer : -self.buffer]
            else:
                hp_cut = hp
                data_cut = data
            a1 = np.max(np.abs(hp_cut))
            a2 = np.max(np.abs(data_cut))

            self.waveform_dataset["test"]["clean"][i] = hp_cut / a1
            self.waveform_dataset["test"]["noisy"][i] = data_cut / a2
            self.waveform_par["test"][i] = np.array(p)

        for i in self.waveform_dataset.keys():
            for j in self.waveform_dataset[i].keys():
                self.waveform_dataset[i][j] = np.array(self.waveform_dataset[i][j])

        # self.normalize_data()
        return

    def generate_bwd_dataset(self, n_grid=[4, 3, 11, 7]):
        """
        Args:
            grid_per_par : grid point of each parameter
        """
        # training set
        snr = self.SNR[0]
        # grid_per_par = n_grid[0]
        # par = np.zeros([self.nparams,grid_per_par])
        # for pkey,idx in self.param_idx.items():
        #     par[idx,:] = np.linspace(self.AAK_par_range[pkey][0],self.AAK_par_range[pkey][1],num=grid_per_par,endpoint=True)
        nparams = int(self.nparams / 2)
        par1 = np.zeros([nparams, n_grid[0]])
        par2 = np.zeros([nparams, n_grid[2]])
        par1[0, :] = np.linspace(
            self.BWD_par_range["f1"][0],
            self.BWD_par_range["f1"][1],
            num=n_grid[0],
            endpoint=True,
        )
        par1[1, :] = np.linspace(
            self.BWD_par_range["f_dot1"][0],
            self.BWD_par_range["f_dot1"][1],
            num=n_grid[0],
            endpoint=True,
        )
        par2[0, :] = np.linspace(
            self.BWD_par_range["f2"][0],
            self.BWD_par_range["f2"][1],
            num=n_grid[2],
            endpoint=True,
        )
        par2[1, :] = np.linspace(
            self.BWD_par_range["f_dot2"][0],
            self.BWD_par_range["f_dot2"][1],
            num=n_grid[2],
            endpoint=True,
        )

        par = np.hstack([par1, par2])
        self.BWD_train_par = par
        grid_per_par = n_grid[0] + n_grid[2]
        waveform_num = (grid_per_par) ** (nparams)
        self.waveform_dataset["train"]["clean"] = np.zeros(
            [int(waveform_num), self.Nt - 2 * self.buffer]
        )
        self.waveform_dataset["train"]["noisy"] = np.zeros(
            [int(waveform_num), self.Nt - 2 * self.buffer]
        )
        self.waveform_par["train"] = np.zeros([int(waveform_num), nparams])
        for i, par_idx in tqdm(
            enumerate(itertools.product(range(grid_per_par), repeat=nparams))
        ):
            p = [par[j, par_idx[j]] for j in range(nparams)]
            hp = self.gen_bwd_signal(p)
            # print(hp.shape)
            noise = self.gen_noise()
            data = snr * hp / self.get_snr(hp) + noise
            hp = self.whiten_data(hp)
            data = self.whiten_data(data)
            # cut
            if self.buffer > 0:
                hp_cut = hp[self.buffer : -self.buffer]
                data_cut = data[self.buffer : -self.buffer]
            else:
                hp_cut = hp
                data_cut = data
            a1 = np.max(np.abs(hp_cut))
            a2 = np.max(np.abs(data_cut))

            self.waveform_dataset["train"]["clean"][i] = hp_cut / a1
            self.waveform_dataset["train"]["noisy"][i] = data_cut / a2
            self.waveform_par["train"][i] = np.array(p)

        # ________________________________
        del par, par1, par2
        del waveform_num
        # _________________________________
        # grid_per_par = n_grid[1]

        # test set
        # par = np.zeros([self.nparams,grid_per_par])
        # for pkey,idx in self.param_idx.items():
        #     dpd2 = (self.AAK_par_range[pkey][1]-self.AAK_par_range[pkey][0])/(par.shape[1])/2
        #     par[idx,:] = np.linspace(self.AAK_par_range[pkey][0]+dpd2,
        #                              self.AAK_par_range[pkey][1]-dpd2,
        #                              num=grid_per_par,endpoint=True)
        par1 = np.zeros([nparams, n_grid[1]])
        par2 = np.zeros([nparams, n_grid[3]])
        dpd2 = (
            (self.BWD_par_range["f1"][1] - self.BWD_par_range["f1"][0])
            / (par1.shape[1])
            / 2
        )
        par1[0, :] = np.linspace(
            self.BWD_par_range["f1"][0] + dpd2,
            self.BWD_par_range["f1"][1] - dpd2,
            num=n_grid[1],
            endpoint=True,
        )

        dpd2 = (
            (self.BWD_par_range["f_dot1"][1] - self.BWD_par_range["f_dot1"][0])
            / (par1.shape[1])
            / 2
        )
        par1[1, :] = np.linspace(
            self.BWD_par_range["f_dot1"][0] + dpd2,
            self.BWD_par_range["f_dot1"][1] - dpd2,
            num=n_grid[1],
            endpoint=True,
        )

        dpd2 = (
            (self.BWD_par_range["f2"][1] - self.BWD_par_range["f2"][0])
            / (par1.shape[1])
            / 2
        )
        par2[0, :] = np.linspace(
            self.BWD_par_range["f2"][0] + dpd2,
            self.BWD_par_range["f2"][1] - dpd2,
            num=n_grid[3],
            endpoint=True,
        )

        dpd2 = (
            (self.BWD_par_range["f_dot2"][1] - self.BWD_par_range["f_dot2"][0])
            / (par1.shape[1])
            / 2
        )
        par2[1, :] = np.linspace(
            self.BWD_par_range["f_dot2"][0] + dpd2,
            self.BWD_par_range["f_dot2"][1] - dpd2,
            num=n_grid[3],
            endpoint=True,
        )

        par = np.hstack([par1, par2])
        self.BWD_test_par = par
        grid_per_par = n_grid[1] + n_grid[3]
        waveform_num = (grid_per_par) ** (nparams)

        self.BWD_test_par = par

        # waveform_num = grid_per_par**nparams
        self.waveform_dataset["test"]["clean"] = np.zeros(
            [int(waveform_num), self.Nt - 2 * self.buffer]
        )
        self.waveform_dataset["test"]["noisy"] = np.zeros(
            [int(waveform_num), self.Nt - 2 * self.buffer]
        )
        self.waveform_par["test"] = np.zeros([int(waveform_num), nparams])

        for i, par_idx in tqdm(
            enumerate(itertools.product(range(grid_per_par), repeat=nparams))
        ):
            p = [par[j, par_idx[j]] for j in range(nparams)]
            hp = self.gen_bwd_signal(p)
            noise = self.gen_noise()
            data = snr * hp / self.get_snr(hp) + noise
            hp = self.whiten_data(hp)
            data = self.whiten_data(data)
            # cut
            if self.buffer > 0:
                hp_cut = hp[self.buffer : -self.buffer]
                data_cut = data[self.buffer : -self.buffer]
            else:
                hp_cut = hp
                data_cut = data
            a1 = np.max(np.abs(hp_cut))
            a2 = np.max(np.abs(data_cut))

            self.waveform_dataset["test"]["clean"][i] = hp_cut / a1
            self.waveform_dataset["test"]["noisy"][i] = data_cut / a2
            self.waveform_par["test"][i] = np.array(p)

        for i in self.waveform_dataset.keys():
            for j in self.waveform_dataset[i].keys():
                self.waveform_dataset[i][j] = np.array(self.waveform_dataset[i][j])

        # self.normalize_data()
        return

    def generate_sgwb_dataset(self, n_train=5e4, n_test=1e4):
        """
        Args:
            grid_per_par : grid point of each parameter
        """
        # training set
        self.waveform_dataset["train"]["clean"] = np.zeros([int(n_train), self.Nt])
        self.waveform_dataset["train"]["noisy"] = np.zeros([int(n_train), self.Nt])
        self.waveform_par["train"] = np.zeros([int(n_train), 4])
        for i in tqdm(range(int(n_train + 0.5))):
            hp = self.gen_sgwb_signal()
            # print(hp.shape)
            noise = self.gen_noise()
            data = hp + noise
            hp = self.whiten_data(hp)
            data = self.whiten_data(data)
            a1 = np.max(np.abs(hp))
            a2 = np.max(np.abs(data))
            self.waveform_dataset["train"]["clean"][i] = hp / a1
            self.waveform_dataset["train"]["noisy"][i] = data / a2
            self.waveform_par["train"][i] = np.array([0, 0, 0, 0])

        # _____________________________
        # test set
        self.waveform_dataset["test"]["clean"] = np.zeros([int(n_test), self.Nt])
        self.waveform_dataset["test"]["noisy"] = np.zeros([int(n_test), self.Nt])
        self.waveform_par["test"] = np.zeros([int(n_test), 4])

        for i in tqdm(range(int(n_test + 0.5))):
            hp = self.gen_sgwb_signal()
            noise = self.gen_noise()
            data = hp + noise
            hp = self.whiten_data(hp)
            data = self.whiten_data(data)
            a1 = np.max(np.abs(hp))
            a2 = np.max(np.abs(data))
            self.waveform_dataset["test"]["clean"][i] = hp / a1
            self.waveform_dataset["test"]["noisy"][i] = data / a2
            self.waveform_par["test"][i] = np.array([0, 0, 0, 0])

        for i in self.waveform_dataset.keys():
            for j in self.waveform_dataset[i].keys():
                self.waveform_dataset[i][j] = np.array(self.waveform_dataset[i][j])

        # self.normalize_data()
        return

    def generate_noise_dataset(self, n_train=5e4, n_test=1e4):
        """
        Args:
            grid_per_par : grid point of each parameter
        """
        # training set
        self.waveform_dataset["train"]["clean"] = np.zeros(
            [int(n_train), self.Nt - 2 * self.buffer]
        )
        self.waveform_dataset["train"]["noisy"] = np.zeros(
            [int(n_train), self.Nt - 2 * self.buffer]
        )
        self.waveform_par["train"] = np.zeros([int(n_train), 4])
        for i in tqdm(range(int(n_train + 0.5))):
            noise = self.gen_noise()
            noise = self.whiten_data(noise)
            # cut
            noise_cut = noise[self.buffer : -self.buffer]
            a2 = np.max(np.abs(noise_cut))

            self.waveform_dataset["train"]["noisy"][i] = noise_cut / a2
            self.waveform_par["train"][i] = np.array([0, 0, 0, 0])

        # _____________________________
        # test set
        self.waveform_dataset["test"]["clean"] = np.zeros(
            [int(n_test), self.Nt - 2 * self.buffer]
        )
        self.waveform_dataset["test"]["noisy"] = np.zeros(
            [int(n_test), self.Nt - 2 * self.buffer]
        )
        self.waveform_par["test"] = np.zeros([int(n_test), 4])

        for i in tqdm(range(int(n_test + 0.5))):
            noise = self.gen_noise()
            noise = self.whiten_data(noise)
            # cut
            noise_cut = noise[self.buffer : -self.buffer]
            a2 = np.max(np.abs(noise_cut))

            self.waveform_dataset["test"]["noisy"][i] = noise_cut / a2
            self.waveform_par["test"][i] = np.array([0, 0, 0, 0])

        for i in self.waveform_dataset.keys():
            for j in self.waveform_dataset[i].keys():
                self.waveform_dataset[i][j] = np.array(self.waveform_dataset[i][j])
        return

    def generate_smbhb_dataset(self, n_grid=[4, 3]):
        """
        Args:
            n_grid : grid point of each parameter
        """
        # training set
        snr = self.SNR[0]
        grid_per_par = n_grid[0]
        par = np.zeros([self.nparams, grid_per_par])
        for pkey, idx in self.param_idx.items():
            par[idx, :] = np.linspace(
                self.bbh_par_range[pkey][0],
                self.bbh_par_range[pkey][1],
                num=grid_per_par,
                endpoint=True,
            )

        self.bbh_train_par = par

        waveform_num = grid_per_par**self.nparams
        self.waveform_dataset["train"]["clean"] = np.zeros(
            [int(waveform_num), self.Nt - 2 * self.buffer]
        )
        self.waveform_dataset["train"]["noisy"] = np.zeros(
            [int(waveform_num), self.Nt - 2 * self.buffer]
        )
        self.waveform_par["train"] = np.zeros([int(waveform_num), self.nparams])
        for i, par_idx in tqdm(
            enumerate(itertools.product(range(grid_per_par), repeat=self.nparams))
        ):
            p = [par[j, par_idx[j]] for j in range(self.nparams)]
            s = self.gen_bbh_signal_pycbc(p)
            # print(hp.shape)
            # s = self.proj(hp,hc,fp,fc)
            noise = self.gen_noise()
            data = snr * s / self.get_snr(s) + noise
            s = self.whiten_data(s)
            data = self.whiten_data(data)
            # cut
            if self.buffer > 0:
                hp_cut = s[self.buffer : -self.buffer]
                data_cut = data[self.buffer : -self.buffer]
            else:
                hp_cut = s
                data_cut = data
            a1 = np.max(np.abs(hp_cut))
            a2 = np.max(np.abs(data_cut))

            self.waveform_dataset["train"]["clean"][i] = hp_cut / a1
            self.waveform_dataset["train"]["noisy"][i] = data_cut / a2
            self.waveform_par["train"][i] = np.array(p)

        # ________________________________
        del par
        del waveform_num
        # _________________________________
        grid_per_par = n_grid[1]

        # test set
        par = np.zeros([self.nparams, grid_per_par])
        for pkey, idx in self.param_idx.items():
            dpd2 = (
                (self.bbh_par_range[pkey][1] - self.bbh_par_range[pkey][0])
                / (par.shape[1])
                / 2
            )
            par[idx, :] = np.linspace(
                self.bbh_par_range[pkey][0] + dpd2,
                self.bbh_par_range[pkey][1] - dpd2,
                num=grid_per_par,
                endpoint=True,
            )

        self.bbh_test_par = par

        waveform_num = grid_per_par**self.nparams
        self.waveform_dataset["test"]["clean"] = np.zeros(
            [int(waveform_num), self.Nt - 2 * self.buffer]
        )
        self.waveform_dataset["test"]["noisy"] = np.zeros(
            [int(waveform_num), self.Nt - 2 * self.buffer]
        )
        self.waveform_par["test"] = np.zeros([int(waveform_num), self.nparams])

        for i, par_idx in tqdm(
            enumerate(itertools.product(range(grid_per_par), repeat=self.nparams))
        ):
            p = [par[j, par_idx[j]] for j in range(self.nparams)]
            s = self.gen_bbh_signal_pycbc(p)

            noise = self.gen_noise()
            data = snr * s / self.get_snr(s) + noise
            s = self.whiten_data(s)
            data = self.whiten_data(data)
            # cut
            if self.buffer > 0:
                hp_cut = s[self.buffer : -self.buffer]
                data_cut = data[self.buffer : -self.buffer]
            else:
                hp_cut = s
                data_cut = data
            a1 = np.max(np.abs(hp_cut))
            a2 = np.max(np.abs(data_cut))

            self.waveform_dataset["test"]["clean"][i] = hp_cut / a1
            self.waveform_dataset["test"]["noisy"][i] = data_cut / a2
            self.waveform_par["test"][i] = np.array(p)

        for i in self.waveform_dataset.keys():
            for j in self.waveform_dataset[i].keys():
                self.waveform_dataset[i][j] = np.array(self.waveform_dataset[i][j])

        # self.normalize_data()
        return

    def normalize_data(
        self,
    ):
        for i in self.waveform_dataset.keys():
            for j in self.waveform_dataset[i].keys():
                max_array = np.max(np.abs(self.waveform_dataset[i][j]), axis=1)
                for k in range(self.waveform_dataset[i][j].shape[0]):
                    self.waveform_dataset[i][j][k] /= max_array[k]
        return

    def calc_matches(self, d1, d2):
        if len(d1) < len(d2):
            d2 = d2[: len(d1)]
        elif len(d1) > len(d2):
            d1 = d1[: len(d2)]
        fft1 = np.fft.fft(d1)
        fft2 = np.fft.fft(d2)
        norm1 = np.mean(np.abs(fft1) ** 2)
        norm2 = np.mean(np.abs(fft2) ** 2)
        inner = np.mean(fft1.conj() * fft2).real
        return inner / np.sqrt(norm1 * norm2)

    def save_waveform(self, DIR=".", data_fn="waveform_dataset.hdf5"):
        p = Path(DIR)
        p.mkdir(parents=True, exist_ok=True)

        f_data = h5py.File(p / data_fn, "w")

        data_name = "0"
        for i in self.waveform_dataset.keys():
            for j in self.waveform_dataset[i].keys():
                data_name = i + "_" + j
                f_data.create_dataset(
                    data_name,
                    data=self.waveform_dataset[i][j],
                    compression="gzip",
                    compression_opts=9,
                )

        for i in self.waveform_par.keys():
            data_name = i + "_" "par"
            f_data.create_dataset(
                data_name,
                data=self.waveform_par[i],
                compression="gzip",
                compression_opts=9,
            )
        f_data.close()

    def load_waveform(self, DIR=".", data_fn="waveform_dataset.hdf5"):
        p = Path(DIR)

        f_data = h5py.File(p / data_fn, "r")
        data_name = "0"
        for i in self.waveform_par.keys():
            data_name = i + "_" "par"
            self.waveform_par[i] = f_data[data_name][:, :]

        for i in self.waveform_dataset.keys():
            for j in self.waveform_dataset[i].keys():
                data_name = i + "_" + j
                self.waveform_dataset[i][j] = f_data[data_name][:, :]

        f_data.close()


def main():
    GWD = GW_SE_Dataset(sample_rate=1.0)
    return 0


if __name__ == "__main__":
    main()
