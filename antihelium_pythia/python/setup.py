"""
This file factors out a bunch of code that's cluttering up the analysis notebook.
"""
import numpy as np
from scipy.optimize import brentq
from scipy.interpolate import interp1d
from scipy.integrate import quad
import itertools
from math import factorial
import matplotlib.pyplot as plt

dataDir = "../data/"
masses = [100, 1000]
colors = {100: "goldenrod", 1000: "steelblue"}
channels = ["bbbar", "W+W-"]
finalStates = ["pbar", "Dbar", "3Hebar"]
lines = {"pbar": "-", "Dbar": ":", "3Hebar": "--"}

### Physical constants
mp = 0.93827 # proton mass in GeV
mn = 0.93957 # neutron mass in GeV
mD = 1.8756 # deuterium mass in GeV
m3He = 2.8094 # helium mass in GeV
m3H = 2.8094 # tritium mass in GeV

### Utility functions

def plotBinErrs(binl, binr, val, err, fmt):
    """
    Plots binned data with error bars in the form of little boxes!
    """
    for bl, br, l, u in zip(binl, binr, val-err, val+err):
        plt.fill_between([bl, br], [l, l], [u, u], color=fmt, alpha=1.0, lw=0.75)

def rigToT(rig, A, absZ, mN):
    """
    Convert rigidity (GV) to kinetic energy per nucleon (GeV/n)
    """
    return (np.sqrt(absZ**2 * rig**2 + mN**2) - mN) / float(A)

def rigDataToTData(data, A, absZ, mN):
    """
    Given an array with columns rigidity (GV), bin left (GV), bin right (GV), flux ((m^2 sr s GV)^-1),
    constructs corresponding bins as a function of kinetic energy per nucleon (GeV/n) and flux in units of
    (m^2 sr s GeV/n)^-1. Also returns rescaled total error.
    """
    deltaR = data[:, 2] - data[:, 1]
    binr = rigToT(data[:, 2], A, absZ, mN)
    binl = rigToT(data[:, 1], A, absZ, mN)
    deltaT = binr - binl

    # Convert bin centers, rescale flux, sum and rescale errors
    return rigToT(data[:, 0], A, absZ, mN), data[:, 3] * np.divide(deltaR, deltaT), np.sqrt(data[:, 4]**2 \
            + data[:, 5]**2) * np.divide(deltaR, deltaT), binl, binr

def nucParams(species):
    """
    Returns mass, mass number (A) and proton number (Z) for the provided species.
    
    Arguments
    -species: "pbar", "Dbar", "3Hebar", "3Hbar"
    
    Returns: float, int, int
        m_species, A (positive), Z (can be negative)
    """
    if species == "pbar":
        return mp, 1, -1
    elif species == "Dbar":
        return mD, 2, -1
    elif species == "3Hebar":
        return m3He, 3, -2
    elif species == "3Hbar":
        return m3H, 3, -1

def toLabelStr(s):
    """
    Given a channel or final state, produces a LaTeX label for plotting
    """
    if s == "W+W-":
        return r"W^+W^-"
    elif s == "bbbar":
        return r"b\bar{b}"
    elif s == "Dbar":
        return r"\bar{D}"
    elif s == "3Hebar":
        return r"\overline{^3He}"
    elif s == "pbar":
        return r"\bar{p}"
    else:
        return ""

def loadSpectra():
    """
    Loads pbar, Dbar and anti-3Hebar spectra into a dictionary.
    """
    spectra = dict()

    for m in masses:
        for c in channels:
            for fs in finalStates:
                if fs == "pbar":
                    # No uncertainty from p_coal
                    spectra[(m, c, fs)] = np.loadtxt(dataDir + "mDM_%iGeV_%s_%s.csv"%(m, c, fs), delimiter=" ")
                else:
                    spectra[(m, c, fs)] = {"mean": np.loadtxt(dataDir + "mDM_%iGeV_%s_%s.csv"%(m, c, fs),
                        delimiter=" "),
                        "lower": np.loadtxt(dataDir + "mDM_%iGeV_%s_%s_lower.csv"%(m, c, fs), delimiter=" "),
                        "upper": np.loadtxt(dataDir + "mDM_%iGeV_%s_%s_upper.csv"%(m, c, fs), delimiter=" ")}

    return spectra

### Propagation functions
def get3HebarPropFn(propMethod, intMethod):
    """
    Returns an antihelium propagation function P(T), defined by
        dPhi/dT|_T = P(T) dN/dT|_T,
    where dPhi/dT is the differential flux and T is kinetic energy per nucleon.
    P(T) has units of m^-2 s^-1 sr^-1 and is defined as
        P(T) = (rhoLoc/0.39 GeV/cm^3)^2 (100 GeV/mDM)^2 (sigmaV/3e-26 cm^3/s) P_num(T),
    where P_num(T) is a function fit to the output of a numerical code.

    Arguments
    -propMethod: "MIN", "MED" or "MAX"
        Specifies propagation setup. TODO: see notes for details
    -intMethod: "MethodAnn" or "MethodInel"
        Specifies how antihelium interacts with cosmic rays. TODO: see notes

    Returns: lambda T, mDM, sigmaV=3.0e-26, rhoLoc=0.39.
        P(T). Gives 0 if unless 0.1 GeV/n < T < 50 GeV/n.
    """
    params = np.genfromtxt(dataDir + "3Hebar_prop_fits.csv", dtype=None)

    for p in params:
        if p[0] == propMethod and p[1] == intMethod:
            # Extract the parameters
            a0, a1, a2, a3, a4 = list(p)[2:]
            # Numerical part of the propagation function.
            pNum = lambda T: np.exp(a0+a1*np.log(T)+a2*np.log(T)**2+a3*np.log(T)**3+a4*np.log(T)**4) \
                    * (T < 50.0) * (T > 0.1)

            return lambda T, mDM, sigmaV=3.0e-26, rhoLoc=0.39: (rhoLoc/0.39)**2 * (100.0/mDM)**2 \
                    * (sigmaV/3.0e-26) * pNum(T)

def getPropFn(species, propMethod):
    """
    Returns a propagation function P(T), defined by
        dPhi/dT|_T = P(T) dN/dT|_T,
    where dPhi/dT is the differential flux and T is kinetic energy per nucleon.
    P(T) has units of m^-2 s^-1 sr^-1.

    Arguments
    -species: "pbar" or "Dbar"
    -propMethod: "MIN", "MED" or "MAX"
        Specifies propagation setup. See notes/PPPC4DMID paper for details.

    Returns: T, mDM, sigmaV=3.0e-26 cm^3/s, rhoLoc=0.39 GeV/cm^-3 -> float
        P(T). Gives 0 unless 100 MeV/n < T < 100 TeV/n (for pbar), 50 MeV/n < T < 50 TeV/n (for Dbar).
    """
    if species not in ["pbar", "Dbar"]:
        print "Error: invalid particle species"
        return None

    mN, A, _ = nucParams(species)

    # Range restrictions differ for pbar and Dbar
    rangeFn = None
    if species == "pbar":
        rangeFn = lambda T: (T < 1.0e5) * (T > 0.1)
    elif species == "Dbar":
        rangeFn = lambda T: (T > 0.05) * (T < 5.0e4)

    for p in np.genfromtxt(dataDir + species + "_prop_fits.csv", dtype=None):
        if p[0] == propMethod:
            # Extract the parameters
            a0, a1, a2, a3, a4, a5 = list(p)[1:]

            # Numerical part of the propagation function.
            R = lambda T: 10.0**(a0 + a1*np.log10(T) + a2*np.log10(T)**2 + a3*np.log10(T)**3
                                    + a4*np.log10(T)**4 + a5*np.log10(T)) * rangeFn(T)

            # 1/A is present because PPPC4DMID write injection spectrum as function of total
            # kinetic energy rather than kinetic energy per nucleon. 10^4 m^2 / cm^2 and
            # c = 9.454*10^23 cm / Myr are needed to give the correct units.
            return lambda T, mDM, sigmaV=3.0e-26, rhoLoc=0.39: 1.0e4 * 9.454e23 \
                    * np.sqrt(1.0-mN**2/(A*T+mN)**2)*rhoLoc**2*sigmaV*R(T) / (8.0*np.pi*A*mDM**2)

def solarMod(species, TIS, phiIS, phiF):
    """
    Computes flux at top of atmosphere as function of kinetic energy per
    nucleon at the top of the atmosphere.
    
    Arguments
    -TIS: float
        Energy per nucleon at boundary of solar system
    -phiIS: float
        Nucleus flux at TIS near boundary of solar system (GeV^-1)
    -species: "pbar", "Dbar", "3Hebar"
        Nucleus species
    -phiF: float
        Fisk potential (GV)
        
    Returns: float, float
        Modulated kinetic energy per nucleon TTOA (GeV/n), modulated flux 
        phiTOA (GeV^-1)
    """
    mN, A, Z = nucParams(species)
        
    # Force field approximation. Factor of e converts phiF to be in MeV
    TTOA = TIS - phiF * np.abs(Z) / float(A)
    
    return TTOA, (2*mN*A*TTOA + A**2*TTOA**2)/(2*mN*A*TIS + A**2*TIS**2) * phiIS

def propagate(species, mDM, Ts, dNdTs, phiF, propMethod="MED", sigmaV=3.0e-26, intMethod3Hebar=None):
    """
    Propagates injection spectrum

    Arguments
    -species: "pbar", "Dbar", "3Hebar"
    -mDM: float
        Dark matter mass
    -Ts: [float]
        List of kinetic energies per nucleon at production (GeV/n)
    -dNdTs: [float]
        Injection spectrum (GeV^-1)
    -sigmaV: float
        Annihilation cross section in cm^3/s
    -propMethod: "MIN", "MED", "MAX"
        Galactic propagation setting
    -phiF: float
        Fisk potential (GV)
    -intMethod3Hebar: string
        If propagating antihelium, this argument must be provided to specify how the antihelium interacts
        with the ISM

    Returns: [float], [float]
        List of TOA kinetic energies per nucleon (GeV/n), list of TOA
        flux (m^-2 s^-1 sr^-1 GeV^-1)
    """
    propFn = None
    if species == "3Hebar":
        propFn = get3HebarPropFn(propMethod=propMethod, intMethod=intMethod3Hebar)
    else:
        propFn = getPropFn(propMethod=propMethod, species=species)

    # Apply galactic modulation
    dPhidTIS = propFn(Ts, mDM, sigmaV) * dNdTs

    # Apply solar propagation
    return solarMod(species, Ts, dPhidTIS, phiF)

### Upper limits and observations

# AMS-02 antiproton flux
amspbarFluxData = np.loadtxt(dataDir + "ams_pbar_flux.csv")
amspbarBins, amspbarFluxes, amspbarFluxErrs, amspbarBinl, amspbarBinr = rigDataToTData(amspbarFluxData, 1, 1, mp)

### Dbar upper limits
# BESS
bessDbarLim = 1.9e-4 # (m^2 s sr Gev/n)^-1
bessDbarTs = [0.17, 1.15]
# AMS-02 estimated sensitivity
amsDbarEstLowLim = 2.06e-6 # (m^2 s sr GeV/n)^-1
amsDbarTsLow = [0.179, 0.73] # low region, GeV/n
amsDbarEstHigh1Lim = 1.028e-6
amsDbarTsHigh1 = [2.43, 3.71] # high region 1, GeV/n
amsDbarEstHigh2Lim = 1.94e-6
amsDbarTsHigh2 = [3.71, 4.65] # high region, GeV/n
# GAPS sensitivities
gapsDbarEstLim = 2.0e-6 # need to divide to convert to (GeV/n)^-1!
gapsDbarTs = [0.1, 0.259]
gaps3HebarEstLim = 1.5e-9 / 3.0 # need to divide to convert to (GeV/n)^-1!
gaps3HebarTs = [0.1, 0.4]

### Acceptances

### AMS-02 pbar acceptance
# AMS-02 (very rough) estimate of $\bar{p}$ acceptance (arXiv:hep-ph/990448)
accpbarEst = lambda T: 2.2e7/(1e8*(2.84-0.1)) * (T >= 0.1) * (T <= 2.84)

# Another estimate of the antiproton acceptance: divide event counts from pbar paper by flux, observation time and bin width
deltaTpbar = amspbarBinr - amspbarBinl # bin widths in kinetic energy per nucleon
accpbar = np.divide(amspbarFluxData[:, 6], 3.15e7*4.0 * deltaTpbar * amspbarFluxes) # 4 YEARS OF DATA!!!

### AMS-02 Dbar acceptance
# Load AMS-02's Dbar acceptance curve
accDbarData = np.loadtxt(dataDir + "ams_Dbar_acceptance.csv")
accDbarData[:, 0] = rigToT(accDbarData[:, 0], 2, 1, mD)
# Acceptance in m^2 sr as a function of kinetic energy per nucleon in GeV/n
accDbar = interp1d(accDbarData[:, 0], accDbarData[:, 1], kind="linear", bounds_error=False, fill_value=0.0)
# Alternative estimate from arXiv:hep-ph/990448
accDbarEst = lambda T: 5.5e7/(1e8*(2.84-0.1)) * (T >= 0.1) * (T <= 2.84)

### AMS-02 anti-3Hebar acceptance
# I'll use this, which is just the geometric acceptance. Not sure how to get a better estimate. The lower
# bound comes from the lowest rigidity bin in He search.
accHe = lambda T: 0.5 * (T >= rigToT(2.0, 3, 2, m3He)) # m^2 sr

# Try converting the flux sensitivity. Note that this gives an acceptance that's larger than AMS-02's
# geometrical acceptance, and therefore makes no sense!
amsHeFluxData = np.loadtxt(dataDir + "ams_He_flux.csv") # only load rigidity, flux
amsHeBins, amsHeFluxes, amsHeFluxErrs, _, _ = rigDataToTData(amsHeFluxData, 3, 2, m3He)
amsHeInterp = interp1d(amsHeBins, amsHeFluxes, kind="linear", bounds_error=False, fill_value=0.0)
# Load 18-year sensitivity
amsHebarHeSensData = np.loadtxt(dataDir + "ams_antiHe_He_18yr_sensitivity.csv")
# Convert rigidity to kinetic energy per nucleon
amsHebarHeSensData[:, 0] = rigToT(amsHebarHeSensData[:, 0], 3, 2, m3He)
# Interpolate the sensitivity
amsHebarHeSens = interp1d(amsHebarHeSensData[:, 0], amsHebarHeSensData[:, 1], kind="linear",
        bounds_error=False, fill_value=0.0)
# Now convert 18-year antiHe / He ratio to 5-year antiHe flux sensitivity
amsHebarSens = lambda T: amsHeInterp(T) * amsHebarHeSens(T) * 5.0 / 18.0

### Optimistic values for AMS-02's geomagnetic cutoff efficiency
amsGeoEffData = np.loadtxt(dataDir + "ams_geomagnetic_cutoff_eff.csv")
amsGeoEffpbar = interp1d(rigToT(amsGeoEffData[:, 0], 1, 1, mp), amsGeoEffData[:, 1] * 0.01,
        bounds_error=False, fill_value=0.0)
amsGeoEffDbar = interp1d(rigToT(amsGeoEffData[:, 0], 2, 1, mD), amsGeoEffData[:, 1] * 0.01,
        bounds_error=False, fill_value=0.0)
amsGeoEff3Hebar = interp1d(rigToT(amsGeoEffData[:, 0], 3, 2, m3He), amsGeoEffData[:, 1] * 0.01,
        bounds_error=False, fill_value=0.0)

### Anti-3Hebar background estimates
# Estimate from arxiv:1401.4017 (Cirelli's antihelium paper)
bg3HebarEstData = np.loadtxt(dataDir + "3Hebar_background_1401.4017.csv")
bg3HebarBins, bg3HebarFluxes = bg3HebarEstData[:, 0], bg3HebarEstData[:, 1]

# Estimate from arxiv:1704.05431 (Blum's paper)
bg3HebarBlumData = np.loadtxt(dataDir + "blum_3Hebar_background.csv", delimiter=",")
bg3HebarBlumBins, bg3HebarBlumFluxesUpper = bg3HebarBlumData[:, 0], bg3HebarBlumData[:, 1]
# Convert rigidity to kinetic energy per nucleon
bg3HebarBlumBins = rigToT(bg3HebarBlumBins, 3, 2, m3He)
# Convert flux from (m^2 s sr GV)^-1 (ie, dN/dR) to (m^2 s sr GeV/n)^-1 (dN/dT) using
#   dR/dT = (A T + m) / (Z sqrt(A T (A T + 2 m)))
bg3HebarBlumFluxesUpper = np.asarray([(3.0*t+m3He) / (2.0 * np.sqrt(3.0*t*(3.0*t+2*m3He))) * flux
    for t, flux in zip(bg3HebarBlumBins, bg3HebarBlumFluxesUpper)])
# Lower part of their result is computed by divide upper flux by 10:
bg3HebarBlumFluxesLower = bg3HebarBlumFluxesUpper / 10.0


### Functions to compute yield and constrain <sigma v>
def numNucObs(spectra, mDM, species, channel, phiF, propMethod="MED", sigmaV=3.0e-26, tObs=5.0, loc="mean",
        Tmin=None, Tmax=None, intMethod3Hebar=None):
    """
    Arguments
    -tObs: float
        Observation time (years)

    Returns: float
        Expected number of Dbar observations at AMS
    """
    # Load spectrum, propagate to TOA
    spectrum = spectra[(mDM, channel, species)]
    TTOAs, dPhidTTOAs = propagate(species, mDM, spectrum[loc][:, 0], spectrum[loc][:, 1], phiF, propMethod,
            sigmaV, intMethod3Hebar)

    # Make sure there's overlap between the energy range and spectrum range
    if (Tmin != None and Tmin > max(TTOAs)) or (Tmax != None and Tmax < min(TTOAs)):
        return 0.0

    # Create interpolating function
    interpFlux = interp1d(TTOAs, dPhidTTOAs, kind="linear")

    # Multiply by acceptance and observation time and integrate
    acc = None
    amsGeoEff = None
    if species == "Dbar":
        acc = accDbar
        amsGeoEff = amsGeoEffDbar
    elif species == "3Hebar":
        acc = accHe
        amsGeoEff = amsGeoEff3Hebar
    else:
        print "Species not recognized"
        return 0.0

    # If no bounds are set, integrate over all available energies
    if Tmin == None:
        Tmin = min(TTOAs)
    if Tmax == None:
        Tmax = max(TTOAs)

    return quad(lambda T: tObs*3.154e7 * interpFlux(T) * acc(T) * amsGeoEff(T), Tmin, Tmax, limit=1000)[0]

def match3HebarYield(spectra, mDM, channel, phiF, propMethod, intMethod, loc, tObs=1.0, Tmin=None, Tmax=None):
    """
    Finds <sigma v> such that AMS-02 observes 1 anti-helium3 nucleus per tObs with kinetic energy per nucleon
    between Tmin and Tmax.
    """
    # Since number of nuclei \propto flux \propto <sigma v>, this is actually trivial:
    nNucRef = numNucObs(spectra, mDM, "3Hebar", channel, phiF, propMethod, sigmaV=3.0e-26, tObs=tObs, loc=loc,
            Tmin=Tmin, Tmax=Tmax, intMethod3Hebar=intMethod)

    if nNucRef == 0.0:
        return np.inf
    else:
        return 3.0e-26 / nNucRef

### Plotting functions
def plotAntinuc(spectra, mDMs, ch, phiF, pm, svDict, intMethod3Hebar):
    """
    Plots fluxes for the three antinucleus species.

    Arguments
    -mDMs: list of DM masses
    -ch: "bbbar" or "W+W-"
    -pm: propagation model
    -svDict: a dictionary with keys of the form (mDM, channel, propagationMethod, loc) that returns a value of
        <sigma v>
    """
    for mDM in mDMs:
        # Get sigmav upper and lower for these parameters
        svU = svDict[(mDM, ch, pm, phiF, "upper")]
        svL = svDict[(mDM, ch, pm, phiF, "lower")]

        # Plot pbar spectrum
        phipbar = spectra[(mDM, ch, "pbar")]
        TTOAspbarL, dPhidTTOAspbarL = propagate("pbar", mDM, phipbar[:, 0], phipbar[:, 1], phiF, pm, svL)
        TTOAspbarU, dPhidTTOAspbarU = propagate("pbar", mDM, phipbar[:, 0], phipbar[:, 1], phiF, pm, svU)

        plt.fill(np.append(TTOAspbarL, TTOAspbarU[::-1]), np.append(dPhidTTOAspbarL, dPhidTTOAspbarU[::-1]),
                color=colors[mDM], alpha=0.15)

        # Plot upper and lower Dbar spectrum
        phiDbarL = spectra[(mDM, ch, "Dbar")]["upper"]
        TTOAsDbarL, dPhidTTOAsDbarL = propagate("Dbar", mDM, phiDbarL[:, 0], phiDbarL[:, 1], phiF, pm, svL)

        phiDbarU = spectra[(mDM, ch, "Dbar")]["lower"]
        TTOAsDbarU, dPhidTTOAsDbarU = propagate("Dbar", mDM, phiDbarU[:, 0], phiDbarU[:, 1], phiF, pm, svU)

        plt.fill(np.append(TTOAsDbarL, TTOAsDbarU[::-1]), np.append(dPhidTTOAsDbarL, dPhidTTOAsDbarU[::-1]),
                color=colors[mDM], alpha=0.4)#, hatch="\\\\//")

        # Plot 3Hebar spectrum. Clip off last point for 1000 GeV curves since they look weird.
        phi3HebarL = spectra[(mDM, ch, "3Hebar")]["lower"]
        last_idx = -1 if mDM == 1000 else phi3HebarL.shape[0]
        TTOAs3HebarL, dPhidTTOAs3HebarL = propagate("3Hebar", mDM, phi3HebarL[:last_idx, 0],
                phi3HebarL[:last_idx, 1], phiF, pm, svL, intMethod3Hebar)

        phi3HebarU = spectra[(mDM, ch, "3Hebar")]["upper"]
        last_idx = -1 if mDM == 1000 else phi3HebarU.shape[0]
        TTOAs3HebarU, dPhidTTOAs3HebarU = propagate("3Hebar", mDM, phi3HebarU[:last_idx, 0],
                phi3HebarU[:last_idx, 1], phiF, pm, svU, intMethod3Hebar)

        plt.fill(np.append(TTOAs3HebarL, TTOAs3HebarU[::-1]), np.append(dPhidTTOAs3HebarL,
            dPhidTTOAs3HebarU[::-1]), color=colors[mDM], alpha=1, label=r"$m_\chi = %i\ {\rm GeV}$"%mDM)

def commonFormatting(ch, labels=True):
    # Useful for plotting other functions
    TMin = 0.1
    TMax = 500.0
    Ts = np.logspace(np.log10(TMin), np.log10(TMax), 200)

    # AMS-02 3Hebar sensitivity (extrapolated from 18-year sensitivity to antiHe/He flux)
    #plt.plot(Ts, amsHebarSens(Ts), 'r--')

    gapsColor = "darkViolet"
    amsColor = "fuchsia"
    bessColor = "m"

    # Dbar upper limit
    plt.plot(bessDbarTs, 2*[bessDbarLim], color=bessColor, linewidth=2)

    # AMS-02 expected Dbar sensitivty
    plt.plot(amsDbarTsLow, 2*[amsDbarEstLowLim], color=amsColor, linewidth=2)
    plt.plot(amsDbarTsHigh1, 2*[amsDbarEstHigh1Lim], color=amsColor, linewidth=2)
    plt.plot(2*[amsDbarTsHigh1[1]], [amsDbarEstHigh1Lim, amsDbarEstHigh2Lim], color=amsColor, linewidth=2)
    plt.plot(amsDbarTsHigh2, 2*[amsDbarEstHigh2Lim], color=amsColor, linewidth=2)

    # GAPS estimated bounds
    plt.plot(gapsDbarTs, 2*[gapsDbarEstLim], color=gapsColor, linewidth=2)
    #plt.plot(gaps3HebarTs, 2*[gaps3HebarEstLim], color=gapsColor, linewidth=2)

    # AMS-02 antiproton flux
    plotBinErrs(amspbarBinl, amspbarBinr, amspbarFluxes, amspbarFluxErrs, 'r')

    # 3Hebar background estimate from arxiv:1401.4017
    plt.plot(bg3HebarBins, bg3HebarFluxes, color='green')
    # 3Hebar background from arxiv:1704.05431 (Blum et al)
    plt.fill_between(bg3HebarBlumBins, bg3HebarBlumFluxesLower, bg3HebarBlumFluxesUpper, color="darkgreen",
            alpha=0.3)

    if labels:
        #plt.text(4.8e-1, 3e-10, r"$\overline{^3{\rm He}}$ (GAPS)", fontsize=8, color=gapsColor)
        plt.text(1.2e-1, 8e-7, r"$\bar{D}$ (GAPS)", fontsize=8, color=gapsColor)
        plt.text(7e-1, 3e-6, r"$\bar{D}$ (AMS-02)", fontsize=8, color=amsColor)
        plt.text(2.4e-1, 7e-5, r"$\bar{D}$ (BESS)", fontsize=8, color=bessColor)
        plt.text(2e-1, 2.9e-2, r"$\bar{p}$ (AMS-02)", fontsize=8, color='r')
        plt.text(1e0, 7e-12, r"$\overline{^3{\rm He}}$ bg", fontsize=10, color="green")

    # Label plot
    plt.title(r"$\chi\chi\to %s$"%toLabelStr(ch))
    plt.xlabel(r"Kinetic energy per nucleon, T (GeV/n)")
    plt.ylabel(r"$\Phi\ ({\rm m}^2\ {\rm s}\ {\rm sr}\ {\rm GeV}/n)^{-1} $")

    # Fix axes
    plt.xscale("log")
    plt.yscale("log")
    plt.xlim(TMin, TMax)
    plt.ylim(1e-12, 1)
    plt.yticks(np.logspace(-12, 0, 13))


