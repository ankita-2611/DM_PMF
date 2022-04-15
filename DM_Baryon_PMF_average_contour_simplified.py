

import matplotlib.pylab as plt
import numpy as np
import scipy.integrate as sp
import scipy.special as sp1

'---------------------------'
'--COSMOLOGICAL PARAMETERS--'
'-------(Planck 2015)-------'
'---------------------------'

Omega_b_h2 = 0.0223         # Physical baryon density parameter
Omega_c_h2 = 0.1188         # Physical dark matter density parameter
W  = -1                     # Equation state of dark energy 
Omega_b = 0.0486            # Baryon density parameter
Omega_c = 0.2589            # Dark Matter density parameter
Omega_m = 0.3            # Matter density parameter
Omega_lambda = 0.6911       # Dark Energy density parameter
h = 0.6774
H0 = (100*h)*(1e3/3.086e22) # Hubble constant in SI units

"NOTE:- Omega_k = 0 for LCDM model"
"NOTE:- Omega_rad = 0 if we consider the above values"

'---------------------------'
'------OTHER CONSTANTS------'
'---------------------------'

hp = 6.626e-34              # Planck constant
c = 3.0e8                   # Speed of light
me = 9.1e-31                # Mass of electron
kB = 1.38e-23               # Boltzmann constant
sigma_T = 6.6524e-29        # Thomson scattering cross-section 
sigma_SB = 5.67e-8          # Stefan-Boltzmann constant
T0 = 2.725                  # CMB Temperature at z = 0
del_E = 1.632e-18           # Ly_alpha transition energy
E2 = 5.44e-19               # Energy of 2s orbital of hydrogen
E1 = 2.176e-18              # Energy of 1s orbital of hydrogen
G = 6.67e-11                # Gravitational constant


G_CGS = 6.67e-8                     #in cm3 gm-1 sec-2
Mpc = 3.085677581e24                #in centimetres
gauss = 1.0e-4                      #in tesla
H0_CGS = (100*h)*(1e3/3.086e22)     #100*h*10**5.*Mpc**-1., same in SI and CGS unit
kB_CGS = 1.38e-16                   #cm2 g s-2 K-1 or egr/K

B0 = 0.05e-9                         # magnetic field gauss
AD_ON = 1.                          # write 1 to enable ambipolar diffusion heating
DT_ON = 1.                          # write 1 to enable decaying turbulence heating
E1s = 2.176e-11                     #in erg

'----------------------------'
'''HELIUM FRACTION'''
'----------------------------'

Y = 0.24
f_He = 0.079

'-------------------------------------------------------------'
'-------CONSTANTS FOR COLLISIONAL COUPLING COEFFEICIENTS------'
'-------------------------------------------------------------'

T_star = 0.068             
A_10 = 2.85e-15          # Einstein coefficient for spontaneous emission

'-----------------------------------------'
'''FUNCTION TO DETERMINE CMB TEMPERATURE'''
'-----------------------------------------'

def T_CMB(red):
    T_gamma = T0*(1.0 + red)
    return T_gamma

'------------------------------------------'
'''FUNCTION TO DETERMINE HUBBLE PARAMETER'''
'------------------------------------------'
'We have assumed a matter dominated universe'

def H(red):
    H_z= H0*(Omega_m*(1.0 + red)**3.0)**0.5
    return H_z

def H_CGS(red):
    return H0_CGS*(Omega_m*(1.0 + red)**3.0)**0.5

'---------------------------------------------------------'
'''FUNCTION TO DETERMINE NEUTRAL HYDROGEN NUMBER DENSITY'''
'---------------------------------------------------------'

def nH(red):
    NH = 8.403*Omega_b_h2*(1.0 + red)**3.0
    return NH

def nH_CGS(red):
    return 8.403e-6*Omega_b_h2*(1.0 + red)**3.0          #cm-3

'-----------------------------------------------------------'
'''FUNCTION TO DETERMINE RECOMBINATION COEFFICIENT (alpha)'''
'-----------------------------------------------------------'

def alpha_e(Tg):
    a = 4.309
    b = -0.6166
    cp = 0.6703
    d = 0.5300
    F = 1.14             # Fudge factor
    t = Tg/(1.0e4)
    
    alpha = F*1.0e-19*((a*t**b)/(1.0 + cp*t**d))
    return alpha

'------------------------------------------------------------'
'''FUNCTION TO DETERMINE PHOTOIONIZATION COEFFICIENT (beta)'''  
'------------------------------------------------------------'
 
def beta_e(T_gamma): #SI unit, note that T_gamma has been used to calculate beta as suggested in chluba, 2015, mnras
    beta = alpha_e(T_gamma)*2.4093e21*T_gamma**(1.5)*np.exp(-39420.289/T_gamma)
    return beta

'---------------------------'
'''FUNCTION TO DETERMINE C'''
'---------------------------'

def C1(red,x, Tg):
    K = 7.1898e-23/(H(red))
    Lambda = 8.22458 
    
    Cr = (1.0 + K*Lambda*(1.0 - x)*nH(red))/(1.0 + K*(Lambda + beta_e(Tg))*(1.0 - x)*nH(red))
    return Cr

'---------------------------------------------'
'''FUNCTION TO DETERMINE DARK MATTER DENSITY'''
'---------------------------------------------'

def rho_DM(red):
    rho_DM_eng =  (Omega_m-Omega_b)*((3*H0**2*c**2)/(8*np.pi*G))*(1.0 + red)**3
    return rho_DM_eng

'----------------------------------------'
'''FUNCTION TO DETERMINE MATTER DENSITY'''
'----------------------------------------'

def rho_M(red):
    rho_M_eng = (Omega_m)*((3*H0**2*c**2)/(8*np.pi*G))*(1.0 + red)**3
    return rho_M_eng    

'----------------------------------------'
'''FUNCTION TO DETERMINE BARYON DENSITY'''
'----------------------------------------'

def rho_B(red):
    rho_B_eng = (Omega_b)*((3*H0**2*c**2)/(8*np.pi*G))*(1.0 + red)**3
    return rho_B_eng

'---------------------------------------------'
'''FUNCTION TO DETERMINE BARYON MASS DENSITY'''
'---------------------------------------------'

def rho_b_CGS(red): #CGS unit
    return (Omega_b)*((3*H0_CGS**2)/(8*np.pi*G_CGS))*(1.0 + red)**3

'------------------------------'
'''FUNCTION TO DETERMINE u_th'''
'------------------------------'

def u_th(Tg, Tx, dm):
    mx = dm
    return c*2.936835e-7*(((Tg/mb) + (Tx/mx))**(0.5))

'------------------------------'
'''FUNCTION TO DETERMINE F(r)'''
'------------------------------'
  
def F_r(vel_xb, Tg, Tx, dm):
    u_therm = u_th(Tg, Tx, dm)
    rv = vel_xb/u_therm
    F = sp1.erf(rv/np.sqrt(2.0)) - np.sqrt(2.0/np.pi)*np.exp((-rv**2.0)/2.0)*rv
    return F

'----------------------------------------'
'''FUNCTION TO DETERMINE F(r)/Vel_xb^2'''
'----------------------------------------'
def Fr_by_velxb2(vel_xb, Tg, Tx, dm):
    u_therm = u_th(Tg, Tx, dm)
    rv = vel_xb/u_therm
    if rv >= 0.1:
        F = sp1.erf(rv/np.sqrt(2.0)) - np.sqrt(2.0/np.pi)*np.exp((-rv**2.0)/2.0)*rv
        F = F/vel_xb**2
        return F
    else:
        F = np.sqrt(2.0/np.pi)*(rv/3.0 - rv**3.0/10.0 + rv**5.0/56.0)
        F = F/u_therm**2
        return F
        
'----------------------------------------'
'''FUNCTION TO DETERMINE F(r)/Vel_xb'''
'----------------------------------------'
def Fr_by_velxb(vel_xb, Tg, Tx, dm):
    u_therm = u_th(Tg, Tx, dm)
    rv = vel_xb/u_therm
    if rv >= 0.1:
        F = sp1.erf(rv/np.sqrt(2.0)) - np.sqrt(2.0/np.pi)*np.exp((-rv**2.0)/2.0)*rv
        F = F/vel_xb
        return F
    else:
        F = np.sqrt(2.0/np.pi)*(rv**2.0/3.0 - rv**4.0/10.0 + rv**6.0/56.0)
        F = F/u_therm
        return F


'---------------------------------------'
'''FUNCTION TO DETERMINE THE DRAG TERM'''
'---------------------------------------'

def Drag_vxb(vel_xb, red, Tg, Tx, sigma_45, dm):
    mx = dm
    D_Vxb2 = 1.*2.63e+7*h*(Omega_m**0.5)*((1.0 + red)**0.5)*sigma_45*Fr_by_velxb2(vel_xb, Tg, Tx, dm)/(mb + mx)
    return D_Vxb2

'---------------------------------------'
'''FUNCTION TO DETERMINE Q_b_coupling'''
'---------------------------------------'

def Q_b_coupling(Tx, Tg, red, vel_xb, sigma_45, dm):
    mx = dm
    u_therm = u_th(Tg, Tx, dm)
    rv = vel_xb/u_therm
    Q_b1 = (2.0/3.0)*2.10e+7*(Omega_m - Omega_b)*(h**2.0)*((1.0 + red)**0.5)*sigma_45*np.exp(-(rv**2)/2.0)*(mb/((mb + mx)**2.0))*(Tx - Tg)/(h*(Omega_m**0.5)*u_therm**3)/1.
    return Q_b1


'----------------------------------'
'''FUNCTION TO DETERMINE Q_b_drag'''
'----------------------------------'

def Q_b_drag(Tx, Tg, red, vel_xb, sigma_45, dm):
    mx = dm
    Q_b_d = 1.*2.26e+3*(Omega_m - Omega_b)*h**2*(1.0+red)**0.5*sigma_45*(mb*mx/((mb + mx)**2.0))*Fr_by_velxb(vel_xb, Tg, Tx, dm)/(h*Omega_m**0.5) 
    return Q_b_d

'--------------------------------------'
'''FUNCTION TO DETERMINE Q_x_coupling'''
'--------------------------------------'

def Q_x_coupling(Tx, Tg, red, vel_xb, sigma_45, dm):
    mx = dm
    u_therm = u_th(Tg, Tx, dm)
    rv = vel_xb/u_therm
    Q_x1 = (2.0/3.0)*2.10e+7*Omega_b*(h**2.0)*((1.0 + red)**0.5)*sigma_45*np.exp(-(rv**2)/2.0)*(mx/((mb + mx)**2.0))*(Tg - Tx)/(h*(Omega_m**0.5)*u_therm**3)/1.
    return Q_x1


'-----------------------------------'
'''FUNCTION TO DETERMINE Q_x_drag'''
'-----------------------------------'

def Q_x_drag(Tx, Tg, red, vel_xb, sigma_45, dm):
    mx = dm
    Q_x_d = 1.*2.26e+3*Omega_b*h**2*(1.0+red)**0.5*sigma_45*(mb*mx/((mb + mx)**2.0))*Fr_by_velxb(vel_xb, Tg, Tx, dm)/(h*Omega_m**0.5)  
    return Q_x_d


'----------------------------------'
'''FUNCTION TO DETERMINE MAG FIELD AND RELEVANT'''
'----------------------------------'

def B(red):
    return B0*(1. + red)**2.

def rho_B_pmf(red):
    return B0**2.*(1. + red)**4./(8.0*np.pi) # CGS unit erg/cm^3

k_max = 286.91*(1.0e-9/B0)  # 1/Mpc unit
#L = 1.0/k_max    # Mpc unit

def l_d(red):
    return (1.075e22)*(B0/1.0e-9)/(1+red)  # cm unit
    
'----------------------------------------'
'''FUNCTION TO DETERMINE SPECTRAL INDEX'''
'----------------------------------------'

n = -2.9
m = 2.0*(n+3.0)/(n+5.0)

'-------------------------------------------------------'
'''FUNCTION TO DETERMINE COLLISIONAL IONIZATION COEFF.'''
'-------------------------------------------------------'

def k_ion(Tg): #CGS unit,  cm 3 s −1, denoted as gamma in sethi, subramanian paper, 2005
    U = E1s/(kB_CGS*Tg)
    return 0.291e-7*U**0.39*np.exp(-U)/(0.232+U)    #cm3s-1 adopted from Section B, Minoda et. al. 2017 paper
    #TH=1.58e5
    #return 5.85e-9*(Tg/1.0e4)**0.5*np.exp(-TH/Tg) #cm^3/s adopted from subsection 2.1, Chluba et al.2015, MNRAS, 451, 2244, 
    #T4 = Tg/1.0e4
    #return 8.2e-16*np.exp(-15.78/T4)   #from http://www.astro.caltech.edu/~srk/Ay126/Lectures/Lecture18/CIE_Notes.pdf
    #return 2.7e-8*(Tg**(-1.5))*np.exp(-43000.0/Tg)  #from https://iopscience.iop.org/article/10.1086/313388/pdf

'----------------------------------'
'''FUNCTION TO DETERMINE TIME'''
'----------------------------------'

tibytd = 14.8*(1.0e-9/B0)*(1.0/k_max)  #t = t_i/t_d, k_max should be in unit of Mpc^-1

'--------------------------------------------------------------------------'
'''FUNCTION TO DETERMINE RATE OF ENERGY INPUT DUE TO AMBIPOLAR DIFFUSION'''
'--------------------------------------------------------------------------'

#def L_square(red):
#    return (rho_B_pmf(red)/l_d(red))**2.

def f_L(n):
    return 0.8313*(1 - 1.02e-2*n)*(n)**1.105

def curl_B(red, n, rho_B_red): # in CGS unit
    return ((rho_B_red/l_d(red))**2.)*f_L(n+3.0)

def gamma_AD(red, Tg, x, n, rho_B_red): # in CGS unit #((1./8.*np.pi)**2.)*
    return (0.126*((1.-x)*(rho_B_red**2.)*((1.0e-9/B0)**2.)*f_L(n+3.0))/(x*(Tg**0.375)*(h**4.)*(Omega_b**2.)*(1.+red)**4.))         #in cm-1 g s-3

def pmf_AD(red, Tg, x, n, rho_B_red):
    return ((2.236e37/(1.+red)**9.5)*((1.-x)*(rho_B_red**2.)*((1.0e-9/B0)**2.)*f_L(n+3.0))/(x*(Tg**0.375)*(h**7.)*(Omega_b**3.)*(Omega_m**0.5)*(1.+f_He+x)))

def rho_AD(red, Tg, x, n, rho_B_red):
    return ((3.93e16*(1.0-x)*((1.0e-9/B0)**2.0)*f_L(n+3.0)*rho_B_red**2.)/(x*((1.0+red)**6.5)*(h**5)*np.sqrt(Omega_m)*(Omega_b**2.0)*(Tg**0.375)))
    
'------------------------------------------------------------------------------'
'''FUNCTION TO DETERMINE RATE OF ENERGY INPUT DUE TO DECAYING TURBULENCE (DT)'''
'------------------------------------------------------------------------------'

def gamma_DT(red, rho_B_red):
    return (4.86e-18*m*(np.log(1+tibytd))**m*rho_B_red*h*Omega_m**0.5*(1.+red)**1.5)/((np.log(1+tibytd)+1.5*np.log((1+zi)/(1+red)))**(m+1))

def pmf_DT(red, x, rho_B_red):
    return (8.623e20/(1.+red)**4.)*(m*(np.log(1+tibytd))**m*rho_B_red)/(Omega_b*h**2.*(1.+f_He+x)*(np.log(1+tibytd)+1.5*np.log((1+zi)/(1+red)))**(m+1))

def rho_DT(red, rho_B_red):
    return ((1.5*m*rho_B_red*(np.log(1.+tibytd))**m)/((1.+red)*(np.log(1+tibytd)+1.5*np.log((1+zi)/(1+red)))**(m+1)))


'----------------------------------------------------------------------------------'
'''FUNCTION TO DETERMINE GAS KINETIC TEMPERATURE (Tg) AND IONIZATION FRACTION (x)'''
'----------------------------------------------------------------------------------'

def func(r, red, cross, dm):
    Tg = r[0]
    x = r[1]
    Tx = r[2]
    V_xb = r[3]
    rho_B_red = r[4]
    
    f_Tg = ((2.0*Tg)/(1.0 + red)) - ((2.70877e-20*(T_CMB(red) - Tg)*(1.0 + red)**(1.5)*(x/(1.0 + f_He + x)))/(H0*np.sqrt(Omega_m))) - Q_b_coupling(Tx, Tg, red, V_xb, cross, dm) - Q_b_drag(Tx, Tg, red, V_xb, cross, dm)/1. - AD_ON*pmf_AD(red, Tg, x, n, rho_B_red) - DT_ON*pmf_DT(red, x, rho_B_red) 
    f_x =  (C1(red,x,Tg)*(alpha_e(Tg)*x**2*nH(red) - beta_e(T_CMB(red))*(1.0 - x)*np.exp(-118260.87/T_CMB(red))))/(H(red)*(1.0 + red))  - (k_ion(Tg)*nH(red)*(1.-x)*x)/(H(red)*(1.0 + red))
    #f_x =  (C1(red,x,Tg)*(alpha_e(Tg)*x**2*nH(red) - beta_e(T_CMB(red))*(1.0 - x)*np.exp(-118260.87/Tg)))/(H(red)*(1.0 + red)) 
    f_Tx = ((2.0*Tx)/(1.0 + red)) - Q_x_coupling(Tx, Tg, red, V_xb, cross, dm) - Q_x_drag(Tx, Tg, red, V_xb, cross, dm)/1.
    f_Vxb = (V_xb/(1.0 + red)) + Drag_vxb(V_xb, red, Tg, Tx, cross, dm)
    f_rho_B = (4.*rho_B_red)/(1.0 + red) + AD_ON*rho_AD(red, Tg, x, n, rho_B_red) + DT_ON*rho_DT(red, rho_B_red) #+ 1.*(1.*gamma_DT(red,rho_B_red)/(H_CGS(red)*(1.+red)) + 0.*gamma_AD(red,Tg,x,n,rho_B_red)/(H_CGS(red)*(1.+red)))
    
    return np.array([f_Tg, f_x, f_Tx, f_Vxb, f_rho_B], float)





zf = 9.0                        # Final redshift
zi = 1010.0                     # Initial redshift
del_z = -0.01                   # Step-size

z = np.arange(zi, zf, del_z)


'----------CMB TEMPERATURE---------'

T_gamma = T0*(1.0 + z)

'-------------------------------------------------------------------------------------------------'
'''FUNCTION TO DETERMINE THE VALUE OF PROBABILITY DISTRIBUTION FOR EACH INTIAL RELATIVE VELOCITY'''
'-------------------------------------------------------------------------------------------------'

def prob_func(v_xb):    
    return 4*np.pi*v_xb**2*(np.exp((-3*v_xb**2)/(2*Vrms**2)))*(1.0/((2.0/3)*np.pi*Vrms**2)**(1.5))


Vxb = np.arange(2.0e-6*c, 5.0e-4*c, 2.0e-5*c)      # Range of initial relative velocties

Vrms = 1e-4*c                          # RMS velocity


'----------------------------------------------------'
'''---------------INITIAL CONDITIONS---------------'''
'---- (Tg = T_CMB at z = 1010 and x = 0.05497)-----'''
'-----(Tx = 0.0 at z = 1010 and V_xb = 1e-4*c)-----'''
'----------------------------------------------------'

n1 = 100         #cross-section grid 
n2 = 100         #mass grid 
cross_sec_45 = 10**(np.linspace(-2, 3, n1))    # Cross-section range in terms of sigma_45
dark_mass = 10**(np.linspace(-4, 2, n2))       # DM mass range in GeV

sigma_300 = []                       # Array to store allowed cross-section
mass_dark_300 = []                   # Array to store allowed mass


mb = 0.938 
xfrac_standard_17=0.0001958782928613447

f= open("mass-sigma-temp_B0.05_100grid.txt","w+")
f1= open("mass-sigma-temp-constrained_B0.05_100grid.txt", "w+")

for i in range(0,n1,1):
    for j in range(0,n2,1):
        T_21_xb = []  # Array to store T_21 data for each initial conditions (array within and array)
        T_gas_vxb = []                        
        P_Vxb = []
        x_frac_vxb = []
        Vxb = np.arange(2.0e-6*c, 5.0e-4*c, 2.0e-5*c)      # Range of initial relative velocties                           # the value of the probabilty distribution for each initial velocity        
        for vxb in Vxb:
            '----------------------------------------------------'
            '''---------------INITIAL CONDITIONS---------------'''
            '---- (Tg = T_CMB at z = 1010 and x = 0.05497) ----'''
            '-----(Tx = 0.0 at z = 1010 and V_xb_0 = 1e-4*c)---'''
            '----------------------------------------------------'
            
            r0 = np.array([T_CMB(zi), 0.05497, 0.0, vxb, rho_B_pmf(zi)], float)    # Initial conditions for Tg, x, Tx and V_xb

        
            '''------SOLVING THE EQUATIONS--------'''
 
                       
            r = sp.odeint(func, r0, z, args = (cross_sec_45[i], dark_mass[j]))            # Solving the coupled differential equation
            T_gas = r[:,0]                        # Stores Tg as an array
            x_points = r[:,1]                     # Stores x as an array
            T_dark = r[:,2]                       # Stores Tx as an array
            Vel_xb = r[:,3]                       # Stores V_xb as an array
            B_energy_density = r[:,4]
            B_z = np.sqrt(8.*np.pi*B_energy_density)
            
            '----------CMB TEMPERATURE---------'
            
            T_gamma = T0*(1.0 + z)


            '''
            '-------CALCULATION OF COLLISIONAL COUPLING COEFFICIENT and spin temperature------'
            
            K_HH = 3.1e-17*T_gas**(0.357)*np.exp(-32.0/T_gas)     # Using the fitting formula
            nHI = 8.403*Omega_b_h2*(1.0 + z)**3.0
            
            C_10 = K_HH*nHI
            
            x_c = (T_star*C_10)/(A_10*T_gamma)     # Collisional coupling coefficient 
            
            
            '------CALCULATION OF SPIN TEMPERATURE------'
            
            T_spin = ((1.0 +  x_c)*T_gas*T_gamma)/(x_c*T_gamma + T_gas)'''
        
            T_21 = 23.0*((0.15/Omega_m)*((1.0 + z)/10))**(0.5)*(Omega_b*h/0.02)*(1.0 - (T_gamma/T_gas))
            P_Vxb.append(prob_func(vxb))
            T_gas_vxb.append(T_gas)
            T_21_xb.append(T_21)
            x_frac_vxb.append(x_points)
            '''------The following steps does a statistical average of the T_21 signal over the velocity distribution---'''
        T_gas_avg = []
        T_b_avg = []
        x_frac_avg = []
        for k in range(len(P_Vxb)):
            T_gas_avg.append(T_gas_vxb[k]*P_Vxb[k])
            T_b_avg.append(T_21_xb[k]*P_Vxb[k])
            x_frac_avg.append(x_frac_vxb[k]*P_Vxb[k])
        T_gas_avg = sum(T_gas_avg)/(1.0*sum(P_Vxb))
        T_21_avg = sum(T_b_avg)/(sum(P_Vxb))
        x_frac_avg = sum(x_frac_avg)/(sum(P_Vxb))
        Percentage_diff_xfrac=(xfrac_standard_17-x_frac_avg[99280])*100.0/xfrac_standard_17
        #print (len(P_Vxb))
        #f.write("%e\t%e\t%e\t%e\n", dark_mass[j], cross_sec_45[i], T_21_avg[99280], x_frac_avg[99280], Percentage_diff_xfrac)
        f.write(str(dark_mass[j]) + "\t" + str(cross_sec_45[i]) + "\t" +str(T_gas_avg[99280]) + "\t" +str(-1*T_21_avg[99280])+ "\t" + str(Percentage_diff_xfrac) +"\n")


            
        if(-1*T_21_avg[99280] >= 300.0 and -1*T_21_avg[99280] <= 1000.0 ):
            sigma_300.append(cross_sec_45[i])
            mass_dark_300.append(dark_mass[j])
            f1.write(str(dark_mass[j]) + "\t" + str(cross_sec_45[i]) + "\t" +str(T_gas_avg[99280]) + "\t" + str(-1*T_21_avg[99280])+ "\t" + str(Percentage_diff_xfrac) +"\n")
            #np.savetxt('temp-xfracs.txt',dark_mass[j],cross_sec_45[i], -1*T_21_avg[99280], Percentage_diff_xfrac)
            #Percentage_diff_xfrac=(xfrac_standard_17-x_points[99280])*100.0/xfrac_standard_17
            #print(dark_mass[j],cross_sec_45[i], T_gas[99280],T_21[99280], x_points[99280], Percentage_diff_xfrac)
            #print(dark_mass[j],cross_sec_45[i], Percentage_diff_xfrac)  
f.close()
f1.close()            
            
plt.loglog(mass_dark_300, sigma_300, '.')
plt.xlim(1e-4, 1e2)
plt.ylim(1e-2, 1e3)
plt.xlabel(r'$m_{\chi}$')
plt.ylabel(r'$\sigma_{45}$')
plt.title('Bounds on cross-section and dark matter mass')
plt.show()


 
