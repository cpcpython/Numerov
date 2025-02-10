#-----------------------------------------------------------------------------
# Evaluation of Quantum Simple Harmonic Oscillator : H2 molecule in atomic unit 
# H2 k = 0.33 au   Req = 0.7 au
# xp = 1.0 au, xn = -2.0 au,  xp = xpositive limit
# Use the command: eog *.png  To view image files
#  
#------------------------------------------------------------------------------


import numpy as np
import matplotlib.pyplot as plt

#-------------------------------------------------------------
# main function to get value of wavefunction on the array/grid
#-------------------------------------------------------------
def callNumerov(E,xp,n,ynm1,yn):
    K=0.33           # force constant
    h= (2*xp)/n      # grid size from -xp to xp 
    m=1837*1/2       # reduced mass of Hydrogen molecule
#   print("Step size:h = ",h)
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
#   print("== End Numerov Subroutine  ==")
    return xarray,y
#=======================================================================
# IMPORTANT PARAMETERS  
#=======================================================================
# delE a small start E
grid=1600        # N division from -xp to xp ie. grid number
ynm1=1e-15       #0.0001     # far left point value
yn  =2e-15       #0.0005     # ynm1+1th point value
xp  =2.0         # range of oscillator as [-xp...0...xp] in atomic unit
node=0
#Tol =0.5         # Need some Trial and Error Inspection of Graph of the y(xp_end)
Itermax=500000   # Iteration limit
print("Step size:h = ",2*xp/grid)
#======================================================================

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
        print("=== FOR LOOP 1 Numerov Cycle Begin === : ",egy)
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

        print("=== === << FOR LOOP 1 Numerov Cycle END >> === === : ",egy)


    print("=== === === <<< FIRST BISECTION END >>> === === ===")

    # ------------------------------------------------------------------------------------
    # Exiting if Emin and Emax didnt change at all 
    # sometime it wont give sufficient, so it can exited from the below for loop
    if((Emin==InEmin) and (Emax==InEmax) and egy != 1): # egy != 1 need since sometime a single bisection loop find optimum bisection bracket
        print("No solutions in this Energy Interval,[",Emin,",",Emax, "]. Exiting ...")
        return 0.0,yo,xo
    # -------------------------------------------------------------------------------------

    bs1=Emin;bs2=Emid;bs3=Emax
    print("*** *** *** <<< SECOND BISECTION BEGIN >>> *** *** *** ")
    secondbs=1; tolSec=0.0000001    # Second bisection for finding approximated Eigenfunction
    while(secondbs < 500):          # Hopefully 50 bisection is enough !

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
            plt.plot(xo[100:1500],yo[100:1500], linestyle='--', marker='o', color='g')
            plt.savefig('LARGE_Emidsho_'+str(Emid)+'H2.png')
            plt.close()
            print("************* *** *** *** *** Convergence Achieved *** *** *** *** *************")

            break
        
    print("******************************************* SECOND BISECTION END  **********************************************")

    # Main Results
    return Emid,xo,yo

# main inputs : in this verrsion no Tolerance is used in findHermiteFunctions()

InEmin=0.00 ; InEmax = 0.01
for i in range (32): # upto range_32 is checked! n=16 is ok
    print(InEmin,InEmax,"///////////////////////////////////////////////////////////////////////////")
    ee,xx,yy = findHermiteFunctions(InEmin,InEmax)
    InEmin=InEmax
    InEmax=InEmax+0.01

