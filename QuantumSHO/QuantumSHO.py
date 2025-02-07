gpkmohan_mbcet@general1gpk:~/Numerov/Numerov1/v2/grid/noParity/correct/fine/Bisection/finegrid$ cat   Numerov-bisection_0-0.1BohrOK.py 
#-----------------------------------------------------------------------------
# Evaluation of Quantum Simple Harmonic Oscillator : H2 molecule in atomic unit 
# H2 k = 0.37 au (AKChandra, CHEMICAL PHYSICS LETTERS 1972); Req = 0.7 au
# xp = 1.0 au, xn = -1.0 au,  xp = xpositive limit
# Use the command: eog *.png  To view image files
# version 1. GPKM 30.01.2025
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
    print("Step size:h = ",h)
# starting points
    y_n_minus_1= ynm1 # first y
    y_n        = yn   # second y
    y          =[]    # third y, the calculated point
    xarray     =[]
    x=-xp   # x is a starting point
    while x < xp:
        g_n_minus_1    =  2*m*(E-K*(x)*(x)/2)
        g_n            =  2*m*(E-K*(x+h)*(x+h)/2)
        g_n_plus_1     =  2*m*(E-K*(x+2*h)*(x+2*h)/2)
        y_n_plus_1 = (2*y_n*(1 - (5*h**2/12)*g_n) - y_n_minus_1*(1 + (h**2/12)*g_n_minus_1)) / (1 + (h**2/12)*g_n_plus_1)
        y.append(y_n_plus_1)
        xarray.append(x+2*h)
        x=x+h
        y_n_minus_1=y_n
        y_n        =y_n_plus_1
    print("== End Numerov ==")
    return xarray,y
#=======================================================================
# IMPORTANT PARAMETERS  
#=======================================================================
# delE a small start E
grid=400        # N division from -xp to xp ie. grid number
ynm1=0.0001     # far left point value
yn  =0.0005     # ynm1+1th point value
xp  =1.0        # range of oscillator as [-xp...0...xp] in atomic unit
node=0
Tol =0.5        # Need some Trial and Error Inspection of Graph of the y(xp_end)
Itermax=500000   # Iteration limit
#======================================================================

#-----------------------------------------------------------------------
# INITIAL BRACKETING OF ENERGY [Emin < delE < Emax] where delE is the Energy and we find Wavefunction of this delE
#-----------------------------------------------------------------------
def findHermiteFunctions(Emin, Emax,TolE):
    InEmin=Emin; InEmax=Emax # Just storing the original values

    delE=Emin; xo,yo=callNumerov(delE,xp,grid,ynm1,yn)
    Eendymin=yo[grid-1]

    delE=Emax; xo,yo=callNumerov(delE,xp,grid,ynm1,yn)
    Eendymax=yo[grid-1]

    Emid=((Emin+Emax)/2.);delE=Emid ; xo,yo=callNumerov(delE,xp,grid,ynm1,yn)
    Eendymid=yo[grid-1]

    # BISECTION TO FIND APPROX. LOCATION OF delE in which y(+xp) ~ 0
    bisecN=10
    print("=== FIRST BISECTION BEGIN ===")
    for egy in range(1,bisecN,1):
        print("=== FOR LOOP 1 Numerov Cycle Begin === : ",egy)
        delEmin=Emin; xo,yo=callNumerov(delEmin,xp,grid,ynm1,yn); Eendymin=yo[grid-1]
        delEmax=Emax; xo,yo=callNumerov(delEmax,xp,grid,ynm1,yn); Eendymax=yo[grid-1]
        Emid=((Emin+Emax)/2.);
        delEmid=Emid ; xo,yo=callNumerov(delEmid,xp,grid,ynm1,yn); Eendymid=yo[grid-1]

        # if it finds ideal biseaction region which contains a definte Root, it will break the loop over here
        if(Eendymid*Eendymin < 0 ):
            print("Ideal bisective region found ...")
            break;
        # Bisection method - Two possibilities [1] monotonically decreasing but all positive; [2] reagion with real root
        # [1] Lowest decreasing part 
        # it find minimum bracketed values for root search, if f(x) are having same sign
        if(abs(Eendymid) < abs(Eendymin)):
            Emin=delEmid;

        print("=== FOR LOOP 1 Numerov Cycle END === : ",egy)


    print("=== === === FIRST BISECTION END === === ===")
    # exiting if Emin and Emax didnt change at all ----------------------------------------
    # sometime it wont give sufficient, so it can exited from the below for loop
    if((Emin==InEmin) and (Emax==InEmax)):
        print("No solutions in this Energy Interval,[",Emin,",",Emax, "]. Exiting ...")
        return 0.0,yo,xo
    # -------------------------------------------------------------------------------------

    bs1=Emin;bs2=Emid;bs3=Emax
    print("*** *** *** SECOND BISECTION BEGIN *** *** *** ")

    secondbs=1; tolSec=0.000001
    while(secondbs < 25000):
#        # Sometime there will be NO eigen functions in some intervales after bisection as above...
#        if(InEmin==Emin and InEmax==Emax):
#            break

        delEmid=(delEmax+delEmin)/2
        if((Eendymin)*(Eendymid) < 0):
            Emax=delEmid; print("Root=================1", Emin,Emax)
        if ((Eendymax)*(Eendymid) <0 ):
           Emin=delEmid; print("Root=================2",Emin,Emax)

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

        if(abs(Emid-Emin)<0.00001):
            print("*** Break in Second Bisection ***")
            break
        
    print("******************************************* SECOND BISECTION END  **********************************************")

    # sometime this delE can go beyond the range [Emin ... Emax], if so quit
    ShiftE=0.000
    print(delE,delE-ShiftE,Emin,Emax,50000000,TolE)
    if(delE-ShiftE< InEmin): # delE-0.002 is needed for iteratively increasing the accuracy of delE
        print("No suitable solution in this region**********************************************")
        return 0.,xo,yo  # 0 means failed in search

    # Otherwise we can Rectify the wavefunction with lesser Tolerance Value
    delE=Emin-ShiftE #  we shifted Energy a bit down for iterative procedure to rectify delE
    
    print("Refine the Eigenvalue, E for Eigenfunction ... ")
    for egy1 in range(1,Itermax,1):
        xo,yo=callNumerov(delE,xp,grid,ynm1,yn)
        print(delE,yo[grid-1])
        if(delE>InEmax): # we dont let delE to cross the max.E in the bracket [Emin,Emax]
            print("delE crossed Emax !, exiting ...")
            return 0.,xo,yo 
        if(abs(yo[grid-1]) < TolE):
            plt.plot(xo,yo, linestyle='--', marker='o', color='g')
            plt.savefig('FinalEsho_'+str(delE)+'shoH2.png')
            plt.close()
            print("*** *** *** *** *** Convergence Achieved *** *** *** *** ***")
            break
        delE=delE+0.0000001
        if(egy==Itermax-1):
            print("*** *** *** *** *** Convergence Failed *** *** *** *** ***")

        # Main Results
    return delE,xo,yo

# main inputs

InEmin=0.00 ; InEmax = 0.01
ee,xx,yy = findHermiteFunctions(InEmin,InEmax,0.1)

InEmin=0.01 ; InEmax = 0.02
ee,xx,yy = findHermiteFunctions(InEmin,InEmax,0.1)

InEmin=0.02 ; InEmax = 0.03
ee,xx,yy = findHermiteFunctions(InEmin,InEmax,0.05)

InEmin=0.03 ; InEmax = 0.04
ee,xx,yy = findHermiteFunctions(InEmin,InEmax,0.05)

InEmin=0.04 ; InEmax = 0.05
ee,xx,yy = findHermiteFunctions(InEmin,InEmax,0.05)

InEmin=0.05 ; InEmax = 0.06
ee,xx,yy = findHermiteFunctions(InEmin,InEmax,0.01)

InEmin=0.06 ; InEmax = 0.07
ee,xx,yy = findHermiteFunctions(InEmin,InEmax,0.01)

InEmin=0.07 ; InEmax = 0.08
ee,xx,yy = findHermiteFunctions(InEmin,InEmax,0.01)

InEmin=0.08 ; InEmax = 0.09
ee,xx,yy = findHermiteFunctions(InEmin,InEmax,0.01)

