import numpy as np
from scipy.interpolate import UnivariateSpline


def get_St0(hr, sigmad, amax, sigmagGI):
    """
    hr:  h/r (aspect ratio) of the disk at the planet's radius
    sigmad: disk dust surface density in g/cm^2
    
    assume amax = 100 um and sigmag < sigmagGI
    amax: in unit of micron.
    
    """
    sigmags = [0.1, 0.3, 1, 3, 10, 30, 100]
    amaxs = [100, 1000, 10000]
    idx_amax = np.argmin([abs(amax-amaxs[i]) for i in range(3)])
    if idx_amax == 0:
        idx_sigg = get_sigmag_DSD1(hr, sigmad) 
    elif idx_amax == 1:
        idx_sigg = get_sigmag_1mm(hr, sigmad)
    elif idx_amax == 2:
        idx_sigg = get_sigmag_DSD2(hr, sigmad)
    
    if (sigmags[idx_sigg] > sigmagGI):
        print ("The gas density is larger than that triggers GI. Use the latter.")
        idx_sigg = np.argmin([abs(sigmagGI-sigmags[i]) for i in range(7)])
        if (sigmags[idx_sigg] > sigmagGI):
            idx_sigg = idx_sigg - 1
        
    if idx_amax == 0:
        return idx_sigg
    elif idx_amax == 1:
        return idx_sigg - 2  # since St propto a/Sigma
    elif idx_amax == 2:
        return idx_sigg - 4

def get_sigmag_DSD1(hr, sigmad):
    """
    hr:  h/r (aspect ratio) of the disk at the planet's radius
    sigmad: disk dust surface density in g/cm^2
    
    assume amax = 100 um
    return 0 to 6. They represent
    0.1, 0.3, 1, 3, 10, 30, 100
    
    This function is a simplified version of table 18.
    
    """
    
    hrmodels = np.array([0.05, 0.07, 0.1])
    distohrm = [abs(hr - hrmodel) for hrmodel in hrmodels]
    idxmin = np.argmin(distohrm)
    #print ("using h/r={:} in Fig. 18".format(hrmodels[idxmin]))
           
    if idxmin == 0: 
        #h/r=0.05
        if sigmad < 0.0035:
            return 0
        elif sigmad < 0.006:
            return 1
        elif sigmad < 0.015:
            return 2
        elif sigmad < 0.04:
            return 3
        elif sigmad < 0.15:
            return 4
        elif sigmad < 0.4:
            return 5
        else: return 6
    
    if idxmin == 1:
        #h/r=0.07
        if sigmad < 0.006:
            return 0
        elif sigmad < 0.1:
            return 1
        elif sigmad < 0.015:
            return 2
        elif sigmad < 0.045:
            return 3
        elif sigmad < 0.15:
            return 4
        elif sigmad < 0.4:
            return 5
        else: return 6
        
    if idxmin == 2:
        #h/r=0.1
        if sigmad < 0.007:
            return 0
        elif sigmad < 0.015:
            return 1
        elif sigmad < 0.025:
            return 2
        elif sigmad < 0.05:
            return 3
        elif sigmad < 0.15:
            return 4
        elif sigmad < 0.4:
            return 5
        else: return 6
        
        
def get_sigmag_DSD2(hr, sigmad):
    """
    hr:  h/r (aspect ratio) of the disk at the planet's radius
    sigmad: disk dust surface density in g/cm^2
    
    get the surface density
    if amax = 1cm
    
    # (0.1, 0.3), 1, 3, 10, 30, 100
    This function is a simplified version of table 18.
    
    """
    
    hrmodels = np.array([0.05, 0.07, 0.1])
    distohrm = [abs(hr - hrmodel) for hrmodel in hrmodels]
    idxmin = np.argmin(distohrm)
    #print ("using h/r={:} in Fig. 18 DSD2".format(hrmodels[idxmin]))
           
    if idxmin == 0: 
        #h/r=0.05
        
        if sigmad < 0.06:
            return 2
        elif sigmad < 0.3:
            return 3
        elif sigmad < 0.55:
            return 4
        elif sigmad < 0.9:
            return 5
        else: return 6
    
    if idxmin == 1:
        #h/r=0.07
        if sigmad < 0.085:
            return 2
        elif sigmad < 0.3:
            return 3
        elif sigmad < 0.8:
            return 4
        elif sigmad < 1.05:
            return 5
        else: return 6
        
    if idxmin == 2:
        #h/r=0.1
        if sigmad < 0.06:
            return 2
        elif sigmad < 0.3:
            return 3
        elif sigmad < 0.9:
            return 4
        elif sigmad < 1.05:
            return 5
        else: return 6
        
def get_sigmag_1mm(hr, sigmad):
    """
    get the gas surface density if amax=1mm
    """
    idx_DSD1 = get_sigmag_DSD1(hr, sigmad)
    idx_DSD2 = get_sigmag_DSD2(hr, sigmad)
    return (idx_DSD1+idx_DSD2)//2

def get_Mp(Delta, idxSt, Mstar, hr, alpha):
    """
    Delta:  gap width (rout-rin)/rout
            rout is the gap's outer edge
            rin  is the gap's inner edge
            defined in Zhang et al. 2018
            around equation 21
            edge is where 
            Sig = [Sig(rpeak)+Sig(rgap)]/2
            
    Stidx:  0 to 6 correspond to
            0.157, 0.0523, 0.0157,
            0.00523, 0.00157,
            0.000523, 0.000157
            
    Mstar:  stellar mass in solar mass
    hr   :  h/r at the planet's position
    alpha:  alpha viscosity
            since
            
            Kprime = q*(h/r)^-0.18*alpha^-0.31
            (equation 22)
            
            a decade decrease of alpha means 
            a factor of 
            10**-0.31 = 0.49 
            decrease in planet mass
    Note: the mass might be different from Zhang et al.
          try idxSt+1 or idxSt-1 should get the exact solution
          This is because Fig. 18 also has Mp dependence, but
          it's simplified in get_sigmag* functions.
    """
    Mj_Msun = 0.0009543
    Me_Msun = 3.00273e-6
    
    wAB3p5 = np.array([[1.09, 1.73, 2.00, 1.25, 1.18, 0.98, 1.11],
        [0.07, 0.24, 0.36, 0.27, 0.29, 0.25, 0.29]])
    # Constants of A and B in Table 1, Zhang et al. 2018
    
    if (idxSt < 0) or (idxSt > 6):
        print ("the Stokes number is out of range.")
        return None
    if (idxSt <= 1):
        print ("the fitting at such a high St has a large scattering.")
    A, B = wAB3p5[:, idxSt]
    
    Kprime = (Delta/A)**(1./B)
    q = Kprime / (hr**(-0.18) * alpha**(-0.31))
    Mp = q / Mj_Msun * Mstar # jupiter mass
    
    print ("q = {:1.1e} Mj/M*".format(q / Mj_Msun))
    print ("q = {:1.1e} Me/M*".format(q / Me_Msun))    
    print ("Mp = {:1.1e} Mj".format(Mp))
    return Mp


def measure_widths(gaprad, ringrad, radbins, radprofile, innerlimit, outerlimit):
    """
    gaprad:  gap's radius 
    ringrad: ring's radius 
    radbins: radial grids 
    radprofile: radial profile (e.g., intensity)
    innerlimit:  the inner radius that is analyzed
    outerlimit: the outer radius that is analyzed
    adapted from Jane Huang, DSHARP II, 2018
    
    return rout-rin, rout
    """
    radbins2 = np.arange(0.5, np.max(radbins), 0.1)
    interpprofile = np.interp(radbins2, radbins, radprofile)
    gapindex = np.argmin(np.abs(radbins2-gaprad))
    ringindex = np.argmin(np.abs(radbins2-ringrad))
    gapintensity = interpprofile[gapindex]
    ringintensity = interpprofile[ringindex]
    meanintensity = 0.5*(gapintensity+ringintensity)
    rmean = radbins2[np.argmin(np.abs(interpprofile[gapindex:ringindex]-meanintensity))+gapindex]
    gapwidth = rmean-radbins2[10*innerlimit:gapindex][np.argmin(np.abs(interpprofile[10*innerlimit:gapindex]-meanintensity))]
    ringwidth = radbins2[ringindex:outerlimit*10][np.argmin(np.abs(interpprofile[ringindex:10*outerlimit]-meanintensity))]-rmean
    return gapwidth, rmean


def Tmidr(rAU, Lstar_Lsun=1., phi=0.02):
    """
    calculate midplane temperature
    """
    au = 1.496e+11
    Lsun = 3.828e26
    sigmaSB = 5.670367e-8
    Lstar = Lstar_Lsun * Lsun
    r = rAU * au
    Tmid = (phi*Lstar/(8.*np.pi*r**2*sigmaSB))**(0.25)
    return Tmid

def calculate_hr(rgap, Lstar, Mstar, phi=0.02):
    """
    calculate h/r
    rgap: gap's radius in au
    Lstar: stellar luminosity in Lsun
    Mstar: stellar mass in Msun
    phi:  the flaring angle
    """
    mmw = 2.35
    m_H = 1.6733e-27 # in kg
    Msun = 1.989e30 # kg
    k   = 1.380649e-23 # SI
    G   = 6.6743e-11   # SI
    au  = 149597870700.0 # m
    
    T = Tmidr(rgap, Lstar, phi)
    Cs  = np.sqrt(k*T/ (mmw*m_H))
    v_k = np.sqrt(G*(Mstar*Msun)/ (rgap*au))
    Omega_k = v_k / (rgap*au)
    hr  = Cs/v_k
    return hr

def calculate_sigmagGI(rgap, Lstar, Mstar, phi=0.02):
    """
    calculate SigmagGI
    rgap: gap's radius in au
    Lstar: stellar luminosity in Lsun
    Mstar: stellar mass in Msun
    phi:  the flaring angle
    """
    mmw = 2.35
    m_H = 1.6733e-27 # in kg
    Msun = 1.989e30 # kg
    k   = 1.380649e-23 # SI
    G   = 6.6743e-11   # SI
    au  = 149597870700.0 # m
    
    T = Tmidr(rgap, Lstar, phi)
    Cs  = np.sqrt(k*T/ (mmw*m_H))
    v_k = np.sqrt(G*(Mstar*Msun)/ (rgap*au))
    Omega_k = v_k / (rgap*au)
    sigmagGI = Cs*Omega_k/ (np.pi*G) / 10 # 1/10 to convert to cgs
    return sigmagGI

def Bnu(T, nu=240e9):
    from scipy.constants.constants import c, h, k
    # in unit W*sr^-1*m^-2*Hz^-1
    return 2*h*nu**3/c**2 * 1./(np.exp(h*nu/(k*T)) - 1)

def BnuJyperSr(T, nu=240e9):
    # in unit Jy/Sr
    return Bnu(T, nu) * 1e26

def getSigma_A(theta_maj, theta_min):
    "Major and minor beam in arcsec"
    theta_maj = theta_maj / 3600. /(180./np.pi) 
    theta_min = theta_min / 3600. /(180./np.pi) 
    Sigma_A = np.pi*theta_maj*theta_min / (4. * np.log(2.))
    return Sigma_A
    
def BnuJyperBeam(T, Sigma_A, nu=240e9):
        # in unit Jy/beam
    return BnuJyperSr(T) * Sigma_A

def tau_r(rAU, intens, params):
    """
    intens: intensity in mJy/beam
    params: [Lstar/Lsun: central star luminosity, 
             phi: the factor accounting for flaring angle, 
             theta_maj: beam major axis FWHM arcsec, 
             theta_min: beam minor axis FWHM arcsec, 
             nu:  observation frequency (Hz)]
    """
    Lstar_Lsun, phi, theta_maj, theta_min, nu = params
    T = Tmidr(rAU, Lstar_Lsun, phi)
    Sigma_A = getSigma_A(theta_maj, theta_min)
    JyperBeam = BnuJyperBeam(T, Sigma_A, nu)
    oneMinusExpMinusTau = intens / JyperBeam
    tau = -np.log(1. - oneMinusExpMinusTau)
    return tau

def sigmadgap(r, sigma_d, rgap, left_end=1.1, right_end=2.0):
    """
    calculate the average dust mass around the gap
    r:      radial grids
    sigmad: dust surface density profile
    rgap :  position of the gap
    left_end:  the lower bound for the integration in [r_p]
    right_end: the upper end for the intergration  in [r_p]
    The right_end needs to tune to a smaller value if
    it's larger than the disk's outer edge. Emission might
    be negtive there.
    """
    sigma_dr2pir = UnivariateSpline(r, 2.*np.pi*r*sigma_d, s=0)
    ave_sigma = sigma_dr2pir.integral(left_end*rgap, right_end*rgap) \
    / (np.pi* ((right_end*rgap)**2 - (left_end*rgap)**2))
    return ave_sigma



