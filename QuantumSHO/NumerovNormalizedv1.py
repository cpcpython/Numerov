#-----------------------------------------------------------------------------
# Evaluation of Quantum Simple Harmonic Oscillator : H2 molecule in atomic unit 
# H2 k = 0.37 au (AKChandra, CHEMICAL PHYSICS LETTERS 1972); Req = 0.7 au
# xp = 1.0 au, xn = -1.0 au,  xp = xpositive limit
# Use the command: eog *.png  To view image files
# version 1. GPKM 14.02.2025
#------------------------------------------------------------------------------


import numpy as np
import matplotlib.pyplot as plt
import math

#-------------------------------------------------------------
# main function to get value of wavefunction on the array/grid
#-------------------------------------------------------------
def callNumerov(E,xp,n,ynm1,yn):
    K=0.33           # force constant
    h= (2*xp)/n      # grid size from -xp to xp 
    m=1837*1/2       # reduced mass of Hydrogen molecule
# starting points
    y_n_minus_1= ynm1 # first y
    y_n        = yn   # second y
    y          =[]    # third y, the calculated point
    xarray     =[]    # x values in f(x
    x=-xp   # x is a starting point, say from the far left point
    while x < xp:     # Numerov method loop
        g_n_minus_1    =  2*m*(E-K*(x)*(x)/2)
        g_n            =  2*m*(E-K*(x+h)*(x+h)/2)
        g_n_plus_1     =  2*m*(E-K*(x+2*h)*(x+2*h)/2)
        y_n_plus_1 = (2*y_n*(1 - (5*h**2/12)*g_n) - y_n_minus_1*(1 + (h**2/12)*g_n_minus_1)) / (1 + (h**2/12)*g_n_plus_1)
        y.append(y_n_plus_1)
        xarray.append(x+2*h)
        x=x+h
        y_n_minus_1=y_n
        y_n        =y_n_plus_1
    return xarray,y
#=======================================================================
# IMPORTANT PARAMETERS  
#=======================================================================
# delE a small start E
grid=3200        # N division from -xp to xp ie. grid number
ynm1=1e-15       #0.0001     # far left point value
yn  =2e-15       #0.0005     # ynm1+1th point value
xp  =2.0         # range of oscillator as [-xp...0...xp] in atomic unit
node=0
Itermax=500000   # Iteration limit
print("Step size:h = ",2*xp/grid)
#======================================================================


#---------------------------------------------------------------------
#  Simpson 1/3 rule for Normalizing Eigenfunctions
#---------------------------------------------------------------------
def Simpson(func,a,b,n):
# Simple Simpson Integrations over func array 
    n = n 
    h = (b - a) / (n - 1)
    I_simp = (h/3) * (func[0] + 2*sum(func[:n-2:2]) + 4*sum(func[1:n-1:2]) + func[n-1])
    return(I_simp)


#-----------------------------------------------------------------------
# INITIAL BRACKETING OF ENERGY [Emin < delE < Emax] where delE is the Energy and we find Wavefunction of this delE
#-----------------------------------------------------------------------
def findHermiteFunctions(Emin, Emax):
    InEmin=Emin; InEmax=Emax # Just storing the original values

    delE=Emin; xo,yo=callNumerov(delE,xp,grid,ynm1,yn)
    Eendymin=yo[grid-1]

    delE=Emax; xo,yo=callNumerov(delE,xp,grid,ynm1,yn)
    Eendymax=yo[grid-1]

    Emid=((Emin+Emax)/2.);delE=Emid ; xo,yo=callNumerov(delE,xp,grid,ynm1,yn)
    Eendymid=yo[grid-1]

    # BISECTION TO FIND APPROX. LOCATION OF delE in which y(+xp) ~ 0
    bisecN=100
    print("=== === <<  FIRST BISECTION BEGIN >> === ===")
    for egy in range(1,bisecN,1):
        print("=== First Bisection Cycle: === : \t",egy)
        delEmin=Emin; xo,yo=callNumerov(delEmin,xp,grid,ynm1,yn); Eendymin=yo[grid-1]
        delEmax=Emax; xo,yo=callNumerov(delEmax,xp,grid,ynm1,yn); Eendymax=yo[grid-1]
        Emid=((Emin+Emax)/2.);
        delEmid=Emid ; xo,yo=callNumerov(delEmid,xp,grid,ynm1,yn); Eendymid=yo[grid-1]
        print("First bisection - Loop Values of delEmin,delEmid,delEmax::\t", delEmin,delEmid,delEmax)

        # if it finds ideal biseaction region which contains a definte Root, it will break the loop over here
        if(Eendymid*Eendymin < 0 ):
            print("Exit in First bisection - Values of delEmin,delEmid,delEmax::\t", delEmin,delEmid,delEmax)
            print("Ideal bisective region found ...")
            break;
        # Bisection method - Two possibilities [1] monotonically decreasing but all positive; [2] reagion with real root
        # For Possibility [1] Lowest decreasing part 
        # it find minimum bracketed values for root search, if f(x) are having same sign
        if(abs(Eendymid) < abs(Eendymin)):
            Emin=delEmid;

    print("=== === === <<< FIRST BISECTION END >>> === === ===")

    # ------------------------------------------------------------------------------------
    # Exiting if Emin and Emax didnt change at all 
    # sometime it wont give sufficient, so it can exited from the below for loop
    if((Emin==InEmin) and (Emax==InEmax) and egy != 1): # egy != 1 need since sometime a single bisection loop find optimum bisection bracket
        print("No solutions in this Energy Interval,[",Emin,",",Emax, "]. Exiting ...")
        return 0.0,yo,xo
    # -------------------------------------------------------------------------------------
    # important: If first section gives Emin~Emid~Emax we dont go furthur and should be returned Null
    tolSec=1e-12
    if(abs(Emid-Emin)<tolSec and abs(Emid-Emax)< tolSec and abs(Emin-Emax)<tolSec):
        print("No solutions in this Energy Interval where Emin~Emid~Emax: Exiting from the Second Bisection ...")
        return 0,xo,yo; # exiting ...

    bs1=Emin;bs2=Emid;bs3=Emax
    print("*** *** *** <<< SECOND BISECTION BEGIN >>> *** *** *** ",Emin,Emid,Emax)
    secondbs=1; tolSec=0.0000001    # Second bisection for finding approximated Eigenfunction
    while(secondbs < 50):          # Hopefully 50 bisection is enough !

        delEmid=(delEmax+delEmin)/2
        if((Eendymin)*(Eendymid) < 0):
            Emax=delEmid; print("Root=================1 Emin Emax N:", Emin,Emax,secondbs)
        if ((Eendymax)*(Eendymid) <0 ):
            Emin=delEmid; print("Root=================2 Emin Emax N:",Emin,Emax,secondbs)

        delEmin=Emin; xo,yo=callNumerov(delEmin,xp,grid,ynm1,yn); Eendymin=yo[grid-1]
        delEmax=Emax; xo,yo=callNumerov(delEmax,xp,grid,ynm1,yn); Eendymax=yo[grid-1]
        Emid=((Emin+Emax)/2.);
        delEmid=Emid ; xo,yo=callNumerov(delEmid,xp,grid,ynm1,yn); Eendymid=yo[grid-1]

        # the below means points doesnt changeing att all in this loop
        if(secondbs>10000):
            if(abs(bs1-Emin)<tolSec and abs(bs2-Emid)< tolSec and abs(bs3-Emax)<tolSec):
                print("No solutions in this Energy Interval,[",Emin,",",Emax, "] Exiting from the Second Bisection ...")
                return 0,xo,yo; # exiting ...

        secondbs=secondbs+1
        if(abs(Emid-InEmax) <=1e-15):# sometime Emid tends to the limit of InEmax, that give error, so it should be avoided
            print("Emid-InEmax are very small, Exiting....")    
            return 0,xo,yo

        if(abs(Emid-Emin)< 1e-15): # Crucial step ~ Machine Precision
            print("*** Break in Second Bisection ***")
            # End Points contains noises so it should be removed like xo[100:1500] yo[100:1500]
            plt.plot(xo[200:3000],yo[200:3000], linestyle='--', marker='o', color='g')
            plt.savefig('LARGE_Emidsho_'+str(Emid)+'H2.png')
            plt.close()
            print("************* *** *** *** *** Convergence Achieved *** *** *** *** *************")
            # Main Return 
            return Emid,xo[200:3000],yo[200:3000]#break
        
    print("******************************************* SECOND BISECTION END  **********************************************")

    # Main Results
    return Emid,xo,yo
#-------------------------------------------------------------------------------------
# function to plot SHO+Energies+Wfn etc.
#-------------------------------------------------------------------------------------
# Function to plot Potential Parabola and Classical Turn points etc.
def PlotParabolaPlus(Energies,K,XAll,PsiAll):
    #---------------------------------------
    # Plot (1/2)Kx**2 Parabola
    #---------------------------------------
    y=[]
    xarray=[]
    y=[ (1/2)*K*x**2 for x in XAll]
    xarray=XAll
    #-----------------------------------------
    En=Energies[0]

    # insert Lines of Energy into Parabola
    fig, ax = plt.subplots()
    ax.plot(xarray, y) # plot parabola
    for i in range(len(Energies)):
        print("Energy Level,n=",i," E = ", Energies[i])
        En=Energies[i]
        ClassTP= math.sqrt(2*En/K) # Classical Turning Point Marking
        ax.hlines(y=En, xmin=ClassTP, xmax=-ClassTP, linewidth=1, color='y')
        ClassTPplt1 = plt.Circle((  -ClassTP,En ), 0.002 ); ClassTPplt2 = plt.Circle((  ClassTP,En ), 0.002 )
        ax.add_artist( ClassTPplt1)
        ax.add_artist( ClassTPplt2)
#        ax.plot(xarray,PsiAll[i])  # here plots WFN_n
        #squares=list(map(lambda x:x+1000,PsiAll[0]))
        EadjustedWfn=list(map(lambda x:En+(x/1),PsiAll[i])) # note x/1
        ax.plot(xarray,EadjustedWfn)

    plt.show()
    plt.savefig('shopotential.png')
    plt.close()
# main inputs : in this verrsion no Tolerance is used in findHermiteFunctions()

InEmin=0.00 ; InEmax = 0.01
#Converged Energies
Econverged=[]
PsiAll=[]
XAll=[]
for i in range (32): # upto range_32 is checked [ n=16 ] i upto 75 is checked.
    print(InEmin,InEmax,"///////////////////////////////////////////////////////////////////////////")
    ## FIND pSI BETWEEN[INEMIN TO INEMAX] INTERVAL ##################################################
    ee,xx,yy = findHermiteFunctions(InEmin,InEmax)   # Program Call Begins ...
    #################################################################################################
    if(ee != 0):
        Econverged.append(ee)
        # Normalized Psi should be used from here onwards: yy = unNormalized Psi
        # little bit Normalization work; N = Sqrt[ Int[ f*f]dx] so,----------------------
        l = [x * x for x in yy] # l=Psi^2 here
        # len(l) = 2800
        NormalizedC=math.sqrt(Simpson(l,xx[0],xx[2800-1],len(l)))        # note we eliminated far right, far left Psi values for noise reduction
        print(NormalizedC) 
        yyNormed = [x/NormalizedC for x in yy]
        #--------------------------------------------------------------------------------
        # Choice - 1
#       PsiAll.append(yy)       # for UnNormalized Psi
        # Choice - 2
        PsiAll.append(yyNormed) # if we want Normalized Psi
        #--------------------------------------------------------------------------------
        # Checking whether Normalization correct or not:
        Psi2 = [x*x for x in yyNormed] # Psi^2
        IntPsiNorm=Simpson(Psi2,xx[0],xx[2800-1],grid-400)
        print("Normalization check: ∫(Psi_Normalized)^2 dx ==1 ? If  n= \t",i,"NumericalValue: \t",IntPsiNorm,"+++++++++++++++++++++++++++")

        XAll=xx 
    InEmin=InEmax
    InEmax=InEmax+0.01



K=0.33
#xLim=2.
Epot=Econverged
PlotParabolaPlus(Epot,K,XAll,PsiAll)

