#!/usr/bin/env python

# give the right permissions (for a Mac or Linux): chmod +x {scriptname}


#==============================================================================
# WELCOME
#==============================================================================
#    Welcome to RainyDay, a framework for coupling remote sensing precipitation
#    fields with Stochastic Storm Transposition for assessment of rainfall-driven hazards.
#    Copyright (C) 2015  Daniel Benjamin Wright (danielb.wright@gwisc.edu)
#
#    This program is free software: you can redistribute it and/or modify
#    it under the terms of the GNU General Public License as published by
#    the Free Software Foundation, either version 3 of the License, or
#    (at your option) any later version.
#
#    This program is distributed in the hope that it will be useful,
#    but WITHOUT ANY WARRANTY; without even the implied warranty of
#    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#    GNU General Public License for more details.
#
#    You should have received a copy of the GNU General Public License
#    along with this program.  If not, see <http://www.gnu.org/licenses/>.
#
#==============================================================================


#==============================================================================
# IMPORT STUFF!
#==============================================================================
from StringIO import StringIO
import os
import sys
import numpy as np
import scipy as sp
from scipy.interpolate import griddata
import glob
import math
from timeit import default_timer as timer  
from datetime import datetime, date, time, timedelta      
import time
from copy import deepcopy

from scipy import ndimage
from pytz import timezone
import pytz
utc = pytz.utc  
import pickle
from mpl_toolkits.basemap import Basemap, addcyclic  
from scipy import stats
from scipy import misc
from netCDF4 import Dataset, num2date, date2num
import warnings

# plotting stuff, really only needed for diagnostic plots
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm

# import RainyDay functions
import RainyDay_utilities.RainyDay_functions as RainyDay



#==============================================================================
# DIFFERENT PROJECTIONS
#==============================================================================
HRAP="+proj=stere +lat_0=90 +lat_ts=60 +lon_0=-105 +k=1 +x_0=0 +y_0=0 +ellps=WGS84 +units=m +no_defs"
GEOG="+proj=longlat +ellps=WGS84 +datum=WGS84 +no_defs"


#==============================================================================
# RAINFALL CLASS
# THIS CONTAINS INFORMATION ABOUT THE SPECIFIED INPUT RAINFALL DATASET
#==============================================================================
class GriddedRainProperties(object):
    def __init__(self,dataset,bndbox,subind,subextent,dimensions,subdimensions,spatialres,projection,timeres,timeunits,spatialunits,rainunits,nodata,notes):
        self.dataset=dataset                #  INPUT RAINFALL DATA SOURCE (TMPA, STAGE IV, ETC.)
        self.bndbox=bndbox                  # COORDINATES OF FULL DATASET BOUNDING BOX (THIS SHOULD BE THE IN THE ORDER [WEST LON,EAST LON,SOUTH LAT,NOTH LAT])
        self.subind=subind                  # MATRIX INDICIES OF BOUNDING BOX (XMIN,XMAX,YMIN,YMAX)
        self.subextent=subextent            # COORDINATES OF USER-DEFINED BOUNDING BOX (NOTE: THIS WILL BE AUTOMATICALLY CALCULATED)
        self.dimensions=dimensions          # WHAT ARE THE SPATIAL DIMENSIONS OF THE INPUT DATASET (NOTE: EVEN IF THE INPUT DATA YOU ARE USING IS FOR A SUBDOMAIN, THIS SHOULD BE THE DIMENSIONS OF THE FULL DATASET)
        self.subdimensions=subdimensions    # WHAT ARE THE DIMENSIONS OF THE SUBDOMAIN (IN THIS SCRIPT, THIS WILL BE RESET TO THE ACTUAL DIMENSION OF THE INPUT DATA, IF NEEDED)
        self.spatialres=spatialres          # WHAT IS THE SPATIAL RESOLUTION OF THE INPUT DATA [dx,dy]?  CURRENTLY THIS SCRIPT WILL ONLY HANDLE RECTANGULAR GRIDS IN DEGREES (BUT DX AND DY DON'T HAVE TO BE EQUAL)
        self.projection=projection          # WHAT IS THE PROJECTION?  CURRENTLY ONLY HANDLES RECTANGULAR LAT/LON ('GEOG')
        self.timeres=timeres                # TEMPORAL RESOLUTION OF INPUT DATA IN MINUTES
        self.timeunits=timeunits            # TEMPORAL UNITS.  CURRENTLY MUST BE MINUTES
        self.spatialunits=spatialunits      # SPATIAL UNITS (CURRENTLY MUST BE DEGREES) [Xres,Yres]
        self.rainunits=rainunits            # RAINFALL UNITS (CURRENTLY MUST BE MM/HR)
        self.nodata=nodata                  # MISSING DATA FLAG
        self.notes=notes                    # ANY SPECIAL NOTES?


#==============================================================================
# RAINFALL INFO
# NOTE: "BOUNDING BOXES"-bndbox, subbox, CONSIDER THE COORDINATES OF THE EDGE OF THE BOUNDING BOX
#==============================================================================
    
emptyprop=GriddedRainProperties('emptyprop',
                            [-999.,-999.-999.-999.],
                            [999, 999, 999, 999],
                            [-999.,-999.-999.-999.],
                            [999, 999],
                            [999, 999],
                            [99.,99.],
                            GEOG,
                            99.,
                            "minutes",
                            "degrees",
                            "mm/hr",
                            -9999.,
                            "none")
   
                   
################################################################################
# "MAIN"
################################################################################

print '''Welcome to RainyDay, a framework for coupling remote sensing precipitation
fields with Stochastic Storm Transposition for assessment of rainfall-driven hazards.
    Copyright (C) 2015  Daniel Benjamin Wright (danielb.wright@gmail.com)

    This program is free software: you can redistribute it and/or modify
    it under the terms of the GNU General Public License as published by
    the Free Software Foundation, either version 3 of the License, or
    (at your option) any later version.

    This program is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU General Public License for more details.

    You should have received a copy of the GNU General Public License
    along with this program.  If not, see <http://www.gnu.org/licenses/>.'''


#==============================================================================
# USER DEFINED INFO
#==============================================================================
start = time.time()
parameterfile='ttt'

#parameterfile=np.str(sys.argv[1])
parameterfile='/Users/daniel/Google_Drive/RainyDay2/RainyDayGit/Example/RainyDayExample.sst'

if os.path.isfile(parameterfile)==False:
    sys.exit("You either didn't specify a parameter file, or it doesn't exist.")
else:
    print "reading in the parameter file..."
    cardinfo=np.loadtxt(parameterfile, comments="#",dtype="str", unpack=False)


#==============================================================================
# USER DEFINED VARIABLES
#==============================================================================
wd=str(cardinfo[cardinfo[:,0]=="MAINPATH",1][0])
scenarioname=str(cardinfo[cardinfo[:,0]=="SCENARIO",1][0])
fullpath=wd+'/'+scenarioname
if os.path.isdir(fullpath)==False:
    os.system('mkdir -p -v %s' %(fullpath))
os.chdir(wd)


# PROPERTIES RELATED TO SST PROCEDURE:
catalogname=str(cardinfo[cardinfo[:,0]=="CATALOGNAME",1][0])
catalogname=wd+'/'+catalogname

CreateCatalog=cardinfo[cardinfo[:,0]=="CREATECATALOG",1][0]
if CreateCatalog.lower()=='true':
    CreateCatalog=True
else:
    CreateCatalog=False
    if os.path.isfile(catalogname)==False:
        sys.exit("You need to create a storm catalog first.")
        
acceltype=cardinfo[cardinfo[:,0]=="ACCELERATOR",1][0]
if acceltype.lower()=='weave':
    print "Weave selected!"
    sys.exit("weave isn't supported anymore!")
    weavecheck=True
    numbacheck=False
    #from scipy import weave
    #from scipy.weave import converters 
elif acceltype.lower()=='numba':
    print "Numba selected!"
    numbacheck=True
    weavecheck=False
    #import cython
    #import pyximport; pyximport.install()
    #pyximport.install(setup_args={'include_dirs':[np.get_include()]})
    #import CatalogCython
    #from numba import jit

else:
    print "No acceleration selected!"
    weavecheck=False
    numbacheck=False


nstorms=np.int(cardinfo[cardinfo[:,0]=="NSTORMS",1][0])
nsimulations=np.int(cardinfo[cardinfo[:,0]=="NYEARS",1][0])
nrealizations=np.int(cardinfo[cardinfo[:,0]=="NREALIZATIONS",1][0])
timeseparation=np.int(cardinfo[cardinfo[:,0]=="TIMESEPARATION",1][0]) 
duration=np.int(cardinfo[cardinfo[:,0]=="DURATION",1][0])
if isinstance(duration,(int,long))==False or duration<0:
    sys.exit("Specified duration is not a positive integer value!")
    
if timeseparation<duration:
    timeseparation=duration
    
samplingtype=cardinfo[cardinfo[:,0]=="COUNTSAMPLE",1][0]
if samplingtype.lower()!='poisson' and samplingtype.lower()!='empirical':
    sys.exit("unrecognized storm count resampling type, options are: 'poisson' or 'empirical'")

areatype=str(cardinfo[cardinfo[:,0]=="POINTAREA",1][0])
if areatype.lower()=="basin":
    wsmaskshp=str(cardinfo[cardinfo[:,0]=="WATERSHEDSHP",1][0])

    shpproj=cardinfo[cardinfo[:,0]=="SHAPEPROJECTION",1][0]
    if shpproj=='geographic':
        shpproj=GEOG
    else:
        sys.exit("The shapefile needs to be in Lat/Lon")

elif areatype.lower()=="point":
    ptlat=np.float32(cardinfo[cardinfo[:,0]=="POINTLAT",1][0]) 
    ptlon=np.float32(cardinfo[cardinfo[:,0]=="POINTLON",1][0]) 
elif areatype.lower()=="box":
    box1=np.float32(cardinfo[cardinfo[:,0]=="BOX_YMIN",1][0])
    box2=np.float32(cardinfo[cardinfo[:,0]=="BOX_YMAX",1][0])
    box3=np.float32(cardinfo[cardinfo[:,0]=="BOX_XMIN",1][0])
    box4=np.float32(cardinfo[cardinfo[:,0]=="BOX_XMAX",1][0])
    boxarea=np.array([box3,box4,box1,box2])

else:
    sys.exit("unrecognized area type")
    
# RESAMPLING TYPE-THIS CODE IS UGLY...
resampletype=cardinfo[cardinfo[:,0]=="RESAMPLER",1][0]
if resampletype.lower()=='kernel':
    resampletype='kernel'
elif resampletype.lower()=='user':
    resampletype='user'
    sys.exit("sadly we aren't set up for manual resampling yet")
elif resampletype.lower()=='uniform':
    resampletype='uniform'
elif resampletype.lower()=='intensity':
    sys.exit('The "intensity-based" resampling scheme is not finished yet')
    resampletype='intensity'
    stackbins=cardinfo[cardinfo[:,0]=="NBINS",1][0]
    try:
        stackbins=np.int32(stackbins)
    except:
        sys.exit('NBINS is not an integer!')
else:
    sys.exit("You specified an unknown resampling method")
    
do_deterministic=cardinfo[cardinfo[:,0]=="DETERMINISTIC",1][0]
if do_deterministic.lower():
    deterministic=True
else:
    deterministic=False

    
lim1=np.float32(cardinfo[cardinfo[:,0]=="LATITUDE_MIN",1][0])
lim2=np.float32(cardinfo[cardinfo[:,0]=="LATITUDE_MAX",1][0])
lim3=np.float32(cardinfo[cardinfo[:,0]=="LONGITUDE_MIN",1][0]) 
lim4=np.float32(cardinfo[cardinfo[:,0]=="LONGITUDE_MAX",1][0]) 
inarea=np.array([lim3,lim4,lim1,lim2])
    
DoDiagnostics=cardinfo[cardinfo[:,0]=="DIAGNOSTICPLOTS",1][0]
if DoDiagnostics.lower()=='true':
    DoDiagnostics=True
else:
    DoDiagnostics=False
    
diagpath=fullpath+'/Diagnostics/'
if os.path.isdir(wd+scenarioname+'/Diagnostics')==False:
    os.system('mkdir %s' %(diagpath))
#diagpath=diagpath+'/'+scenarioname

FreqAnalysis=cardinfo[cardinfo[:,0]=="FREQANALYSIS",1][0]
if FreqAnalysis.lower()=='true':
    FreqAnalysis=True
else:
    FreqAnalysis=False
    
DoDiagMovies=cardinfo[cardinfo[:,0]=="DIAGNOSTICMOVIES",1][0]
if DoDiagMovies.lower()=='true':
    DoDiagMovies=True
    #matplotlib.use("Agg")
    import matplotlib.animation as manimation
    #FFMpegWriter = manimation.writers['ffmpeg']
else:
    DoDiagMovies=False
        
FreqFile=fullpath+'/'+scenarioname+'.FreqAnalysis'

Scenarios=cardinfo[cardinfo[:,0]=="SCENARIOS",1][0]
if Scenarios.lower()=='true':
    Scenarios=True
    WriteName=wd+'/'+scenarioname+'/Realizations'
    os.system('mkdir %s' %(WriteName))
    WriteName=WriteName+'/'+scenarioname
else:
    Scenarios=False

# EXLCUDE CERTAIN MONTHS
exclude=cardinfo[cardinfo[:,0]=="EXCLUDEMONTHS",1][0]
if exclude.lower()!="none":
    exclude=exclude.split(',')
    excludemonths=np.empty(len(exclude),dtype="int32")
    for i in range(0,len(exclude)):
        excludemonths[i]=np.int(exclude[i])
else:
    excludemonths=False
    
# INCLUDE ONLY CERTAIN YEARS
includeyr=cardinfo[cardinfo[:,0]=="INCLUDEYEARS",1][0] 
if includeyr.lower()!="all":
    if ',' in includeyr:
        includeyr=includeyr.split(',')
        includeyears=np.empty(len(includeyr),dtype="int32")
        for i in range(0,len(includeyr)):
            includeyears[i]=np.int(includeyr[i])
    elif '-' in includeyr:
        includeyr=includeyr.split('-')
        includeyears=np.arange(np.int(includeyr[0]),np.int(includeyr[1])+1,dtype='int32')
    else:
        includeyears=np.empty((1),dtype="int32")
        includeyears[0]=np.int(includeyr)
else:
    includeyears=False

# MINIMUM RETURN PERIOD THRESHOLD FOR WRITING RAINFALL SCENARIO (THIS WILL HAVE A HUGE IMPACT ON THE SIZE OF THE OUTPUT FILES!)  IN PRINCIPLE IT IS PERHAPS BETTER TO USE AN INTENSITY BUT THIS IS CLEANER!
RainfallThreshYear=np.int(cardinfo[cardinfo[:,0]=="RETURNTHRESHOLD",1][0])

# DIRECTORY INFO-WHERE DOES THE INPUT DATASET RESIDE?
inpath=str(cardinfo[cardinfo[:,0]=="RAINPATH",1][0])

      
    
BaseMap=cardinfo[cardinfo[:,0]=="BASEMAP",1][0]
if BaseMap.lower()!='none':
    BaseMap=BaseMap.split('.')[0]
    BaseField=cardinfo[cardinfo[:,0]=="BASEFIELD",1][0]


# SENSITIVITY ANALYSIS OPTIONS:
IntensitySens=cardinfo[cardinfo[:,0]=="SENS_INTENSITY",1][0]
if IntensitySens.lower()!='false':
    IntensitySens=1.+np.float(IntensitySens)/100.
else:
    IntensitySens=1.

FrequencySens=cardinfo[cardinfo[:,0]=="SENS_FREQUENCY",1][0]     
if FrequencySens.lower()!='false':
    if samplingtype.lower()!='poisson':
        sys.exit("Sorry, you can't modify the resampling frequency unless you use the Poisson resampler.")
    else:
        FrequencySens=1.0+np.float(FrequencySens)/100.
else:
    FrequencySens=1.

    
userdistr=cardinfo[cardinfo[:,0]=="INTENSDISTR",1][0]
if userdistr.lower()=='false':
    userdistr=np.zeros((1),dtype='bool')
elif len(userdistr.split(','))==3:
    print "reading user-defined rainfall intensity distribution..."
    userdistr=np.array(userdistr.split(','),dtype='float32')
else:
    sys.exit("There is a problem with the INTENSDISTR entry!")
 
if Scenarios:
    if np.any(np.core.defchararray.find(cardinfo[:,0],"SPINPERIOD")>-1):
        pretime=cardinfo[cardinfo[:,0]=="SPINPERIOD",1][0]   
         
        if pretime.lower()=='none':
            prependrain=False
            print 'spinup rainfall will be not included in rainfall scenarios...'
        else:
            try:
                pretime=np.int(pretime)
                prependrain=True
                if pretime==0.:
                    prependrain=False
                    print 'the spinup time you provided was 0, no spinup rainfall will be included...'
                elif pretime<0.:
                    print "the spin-up time value is negative."
                    sys.exit(0)
                elif pretime<1.:
                    print 'the spinup time you provided was short, rounding up to 1 day...'
                    pretime=1.0
                else:
                    print np.str(pretime)+' days of spinup rainfall will be included in rainfall scenarios...'
            except:
                sys.exit('unrecognized value provided for SPINPERIOD.')
           
    else:
        prependrain=False
        print 'no SPINPERIOD field was provided.  Spin-up rainfall will be not be included...'

if np.any(np.core.defchararray.find(cardinfo[:,0],"SPREADTYPE")>-1):       
    spread=cardinfo[cardinfo[:,0]=="SPREADTYPE",1][0]
    if spread.lower()=='ensemble':
        spreadtype='ensemble'
        print '"ensemble spread" will be calculated for rainfall frequency analysis...'
    else:
        try:
            int(spread)
        except:
            print 'unrecognized value for SPREADTYPE.'
            
        if int(spread)>=0 and int(spread)<=100:
            spreadtype='quantile'
            quantilecalc=int(spread)
            print spread+'th quantiles will be calculated for rainfall frequency analysis...'
        else: 
            print 'invalid quantile range for frequency analysis...'
else:
    spreadtype='ensemble'
                
    
if np.any(np.core.defchararray.find(cardinfo[:,0],"RETURNLEVELS")>-1):
    speclevels=cardinfo[cardinfo[:,0]=="RETURNLEVELS",1][0]
    if speclevels.lower()=='all':
        print 'using all return levels...'
        alllevels=True
    else:
        alllevels=False
        if ',' in speclevels:
            speclevels=speclevels.split(',')
            try:
                speclevels=np.float32(speclevels,dtype="float32")
            except:
                print "Non-numeric value provided to RETURNLEVELS."
                
            speclevels=speclevels[speclevels<=nsimulations+0.00001]
            speclevels=speclevels[speclevels>=0.99999]
        else:
            sys.exit("The format of RETURNLEVELS isn't recognized.  It should be 'all' or a comma separated list.")
else:
    alllevels=True
      
      
rotation=False
if np.any(np.core.defchararray.find(cardinfo[:,0],"ROTATIONANGLE")>-1):
    rotangle=cardinfo[cardinfo[:,0]=="ROTATIONANGLE",1][0]
    if rotangle.lower()=='none' or areatype.lower()=="point":
        rotation=False            
    else:
        rotation=True
        if len(rotangle.lower().split(','))!=3:
            sys.exit('Unrecognized entry provided for ROTATIONANGLE.  Should be "none" or "-X,+Y,Nangles".')
        else:
            minangle,maxangle,nanglebins=rotangle.split(',')
            try:
                minangle=np.float32(minangle)
            except:
                print 'Unrecognized entry provided for ROTATIONANGLE minimum angle.'
            try:
                maxangle=np.float32(maxangle)
            except:
                print 'Unrecognized entry provided for ROTATIONANGLE maximum angle.'
            try:
                nanglebins=np.int32(nanglebins,dtype="float32")
            except:
                print 'Unrecognized entry provided for ROTATIONANGLE number of bins.'
                
            if minangle>0. or maxangle<0.:
                sys.exit('The minimum angle should be negative and the maximum angle should be positive.')
    
if rotation:
    print "storm rotation will be used..."
    delarray=[]

    spversion=sp.__version__.split('.')
    if int(spversion[0])<1:
        if int(spversion[1])<9:
            sys.exit('Your version of Scipy is too old to handle the rotation scheme.  Either do not use rotation or update Scipy.')
  
# should add a "cell-centering" option!

        
#==============================================================================
# THIS BLOCK CONFIGURES SEVERAL THINGS
#============================================================================== 
    
initseed=0
np.random.seed(initseed)
global rainprop   
rainprop=deepcopy(emptyprop) 
   
    
#==============================================================================
# FIND THE MULTIPLES OF 24 HOURS TO DO 24-HOUR BIAS CORRECTION
# this section is going to be used later
#==============================================================================
# IT MIGHT BE WISE TO MAKE THIS BIT OPTIONAL...
#if duration%24!=0:
#    sys.exit("Duration is not a multiple of 24 hours!  This version isn't set up for that!")
#else:
tempdur=duration   
    
print "SHOULD CHECK FOR ROBUSTNESS OF NODATA LATER ON"
   
 
#==============================================================================
# CREATE NEW STORM CATALOG, IF DESIRED
#==============================================================================  
if CreateCatalog:
    print "creating a new storm catalog..."
    
    flist=RainyDay.createfilelist(inpath,includeyears,excludemonths)
  
    
    # GET SUBDIMENSIONS, ETC. FROM THE NETCDF FILE RATHER THAN FROM RAINPROPERTIES  
    rainprop.spatialres,rainprop.dimensions,rainprop.bndbox,rainprop.timeres,rainprop.nodata=RainyDay.rainprop_setup(flist[0])
    

    
    #==============================================================================
    # SET UP THE SUBGRID INFO
    #==============================================================================
    # 'subgrid' defines the transposition domain
 
    rainprop.subextent,rainprop.subind,rainprop.subdimensions=RainyDay.findsubbox(inarea,rainprop)
    
    #subgrid,gx,gy=RainyDay.creategrids(rainprop)
    ingridx,ingridy=np.meshgrid(np.arange(rainprop.subextent[0],rainprop.subextent[1]-rainprop.spatialres[0]/1000,rainprop.spatialres[0]),np.arange(rainprop.subextent[3],rainprop.subextent[2]+rainprop.spatialres[1]/1000,-rainprop.spatialres[1]))        

    lonrange=ingridx[0,:]
    latrange=ingridy[:,0]


#==============================================================================
# IF A STORM CATALOG ALREADY EXISTS, USE IT
#==============================================================================  
    
else:
    print "reading an existing storm catalog..."
    if os.path.isfile(catalogname)==False:
        sys.exit("can't find the storm catalog")
    else:
        #catrain,cattime,latrange,lonrange,catx,caty,catmax,_=RainyDay.readcatalog(catalogname)
        rainprop.spatialres,rainprop.dimensions,rainprop.bndbox,rainprop.timeres,rainprop.nodata,catrain,cattime,latrange,lonrange,catx,caty,catmax=RainyDay.rainprop_setup(catalogname,catalog=True)
        if nstorms>cattime.shape[0]:
            print "WARNING: The storm catalog has fewer storms than the specified nstorms"
            nstorms=cattime.shape[0] 

    rainprop.bndbox=np.array([lonrange[0],lonrange[-1]+rainprop.spatialres[0],latrange[-1]-rainprop.spatialres[1],latrange[0]],dtype='float32')
    rainprop.subextent=np.array([lonrange[0],lonrange[-1]+rainprop.spatialres[0],latrange[-1]-rainprop.spatialres[1],latrange[0]],dtype='float32')
        
    rainprop.subind=np.array([0,len(lonrange)-1,len(latrange)-1,0],dtype='int32')
    rainprop.dimensions=np.array([len(latrange),len(lonrange)],dtype='int32')  
    rainprop.subdimensions=rainprop.dimensions

    ingridx,ingridy=np.meshgrid(np.arange(rainprop.subextent[0],rainprop.subextent[1]-rainprop.spatialres[0]/1000,rainprop.spatialres[0]),np.arange(rainprop.subextent[3],rainprop.subextent[2]+rainprop.spatialres[1]/1000,-rainprop.spatialres[1]))        

         
#==============================================================================
# SET UP GRID MASK
#==============================================================================

print "setting up the grid information and masks..."
if areatype.lower()=="basin":
    if os.path.isfile(wsmaskshp)==False:
        sys.exit("can't find the basin shapefile!")
    else:
        catmask=RainyDay.rastermaskGDAL(wsmaskshp,shpproj,rainprop,'fraction',fullpath)
        catmask=catmask.reshape(ingridx.shape,order='F')

elif areatype.lower()=="point":
    catmask=np.zeros((rainprop.subdimensions))
    yind=np.where((latrange-ptlat)>0)[0][-1]
    xind=np.where((ptlon-lonrange)>0)[0][-1]  
    if xind==0 or yind==0:
        sys.exit('the point you defined is too close to the edge of the box you defined!')
    else:
        catmask[yind,xind]=1.0
elif areatype.lower()=="box":
    finelat=np.arange(latrange[0],latrange[-1]-rainprop.spatialres[1]+rainprop.spatialres[0]/1000,-rainprop.spatialres[1]/25)
    finelon=np.arange(lonrange[0],lonrange[-1]+rainprop.spatialres[0]-rainprop.spatialres[0]/1000,rainprop.spatialres[0]/25)

    subindy=np.logical_and(finelat>boxarea[2]+rainprop.spatialres[1]/1000,finelat<boxarea[3]+rainprop.spatialres[1]/1000)
    subindx=np.logical_and(finelon>boxarea[0]-rainprop.spatialres[0]/1000,finelon<boxarea[1]-rainprop.spatialres[0]/1000)
    
    tx,ty=np.meshgrid(subindx,subindy)
    catmask=np.array(np.logical_and(tx==True,ty==True),dtype='float32')
    
    if len(finelat[subindy])<25 and len(finelon[subindx])<25:    
        print 'WARNING: you set POINTAREA to "box", but the box is smaller than a single pixel.  This is not advised.  Either set POINTAREA to "point" or increase the size of the box.'   
    
    if len(finelat[subindy])==1 and len(finelon[subindx])==1:
        catmask=np.zeros((rainprop.subdimensions))
        yind=np.where((latrange-ptlat)>0)[0][-1]
        xind=np.where((ptlon-lonrange)>0)[0][-1]  
        if xind==0 or yind==0:
            sys.exit('the point you defined is too close to the edge of the box you defined!')
        else:
            catmask[yind,xind]=1.0
    else:
        from scipy import ndimage
        def block_mean(ar, fact):
            assert isinstance(fact, int), type(fact)
            sx, sy = ar.shape
            X, Y = np.ogrid[0:sx, 0:sy]
            regions = sy/fact * (X/fact) + Y/fact
            res = ndimage.mean(ar, labels=regions, index=np.arange(regions.max() + 1))
            res.shape = (sx/fact, sy/fact)
            return res

        catmask=block_mean(catmask,25)          # this scheme is a bit of a numerical approximation but I doubt it makes much practical difference  
else:
    sys.exit("unrecognized area type!")
    
        
# TRIM THE GRID DOWN
csum=np.where(np.sum(catmask,axis=0)==0)
rsum=np.where(np.sum(catmask,axis=1)==0)

xmin=np.min(np.where(np.sum(catmask,axis=0)!=0))
xmax=np.max(np.where(np.sum(catmask,axis=0)!=0))
ymin=np.min(np.where(np.sum(catmask,axis=1)!=0))
ymax=np.max(np.where(np.sum(catmask,axis=1)!=0))

trimmask=np.delete(catmask,csum,axis=1)
trimmask=np.delete(trimmask,rsum,axis=0)
maskwidth=trimmask.shape[1]
maskheight=trimmask.shape[0]
trimmask=np.array(trimmask,dtype='float32')
catmask=np.array(catmask,dtype='float32')

timeseparation=np.timedelta64(timeseparation,'h')
timestep=np.timedelta64(rainprop.timeres,'m')

mnorm=np.sum(trimmask)

xlen=rainprop.subdimensions[1]-maskwidth+1
ylen=rainprop.subdimensions[0]-maskheight+1


#################################################################################
# STEP 1: CREATE STORM CATALOG
#################################################################################

if CreateCatalog: 
    print "reading rainfall files..."
    
    #==============================================================================
    # SET UP OUTPUT VARIABLE
    #==============================================================================

    rainarray=np.zeros((tempdur*60/rainprop.timeres,rainprop.subdimensions[0],rainprop.subdimensions[1]),dtype='float32')  
    
    rainsum=np.zeros((ylen,xlen),dtype='float32')
    #tsum=np.zeros((ylen*xlen),dtype='float32')
    rainarray[:]=np.nan
    raintime=np.empty((tempdur*60/rainprop.timeres),dtype='datetime64[m]')
    raintime[:]=np.datetime64(datetime(1900,1,1,0,0,0))
    
    catmax=np.zeros((nstorms),dtype='float32')

    catrain=np.zeros((nstorms,tempdur*60/rainprop.timeres,rainprop.subdimensions[0],rainprop.subdimensions[1]),dtype='float32')
    cattime=np.empty((nstorms,tempdur*60/rainprop.timeres),dtype='datetime64[m]')
    cattime[:]=np.datetime64(datetime(1900,1,1,0,0,0))
    catloc=np.empty((nstorms),dtype='float32')
    
    catx=np.zeros((nstorms),dtype='int32')
    caty=np.zeros((nstorms),dtype='int32')
    
#    stackmask=np.zeros(((ylen)*(xlen),rainprop.subdimensions[0],rainprop.subdimensions[1]),dtype='float32')
#    if weavecheck==False and cythoncheck==False:
#        for k in range(0,(ylen)*(xlen)):
#            for i in range(0,(ylen)*(xlen)):
#                y=i/xlen
#                x=i-y*xlen
#                stackmask[k,(y):(y+maskheight),(x):(x+maskwidth)]=trimmask
#    plt.imshow(stackmask[-1,:])       
#    
    #==============================================================================
    # READ IN RAINFALL
    #==============================================================================
    filerange=range(0,len(flist))
    start = time.time()
    for i in filerange: 
        infile=flist[i]
        inrain,intime,inlatitude,inlongitude=RainyDay.readnetcdf(infile,inbounds=rainprop.subind)
        inrain[inrain<0.]=np.nan
        print 'Processing file '+str(i+1)+' out of '+str(len(flist))+'('+str(100*(i+1)/len(flist))+'%): '+infile

        # THIS FIRST PART BUILDS THE STORM CATALOG
        for k in np.arange(0,24*60/rainprop.timeres,1):     
            starttime=intime[k]-np.timedelta64(tempdur,'h')
            raintime[-1]=intime[k]
            #rainarray[-1,:]=np.reshape(inrain[k,subgrid],(rainprop.subdimensions[0],rainprop.subdimensions[1]))
            rainarray[-1,:]=np.reshape(inrain[k,:],(rainprop.subdimensions[0],rainprop.subdimensions[1]))
            #rainarray[-1,:]=inrain[k,:]            
            subtimeind=np.where(np.logical_and(raintime>starttime,raintime<=raintime[-1]))
            subtime=np.arange(raintime[-1],starttime,-timestep)[::-1]
            temparray=np.squeeze(np.nansum(rainarray[subtimeind,:],axis=1))
            
            if numbacheck:
                rainmax,ycat,xcat=RainyDay.catalogNumba(temparray,trimmask,xlen,ylen,maskheight,maskwidth,rainsum)
            else:
                rainmax,ycat,xcat=RainyDay.catalogAlt(temparray,trimmask,xlen,ylen,maskheight,maskwidth,rainsum)
                                

            tempmin=np.min(catmax)
            if rainmax>tempmin:
                minind=np.argmin(catmax)
                checksep=intime[k]-cattime[:,-1]
                if (checksep<timeseparation).any():
                    checkind=np.where(checksep<timeseparation)
                    if rainmax>=catmax[checkind]:
                        catmax[checkind]=rainmax
                        cattime[checkind,:]=subtime
                        catx[checkind]=xcat
                        caty[checkind]=ycat
                        catrain[checkind,:]=rainarray
                else:  
                    catmax[minind]=rainmax
                    cattime[minind,:]=subtime
                    catx[minind]=xcat
                    caty[minind]=ycat
                    catrain[minind,:]=rainarray
   
            rainarray[0:-1,:]=rainarray[1:tempdur*60/rainprop.timeres,:]
            raintime[0:-1]=raintime[1:tempdur*60/rainprop.timeres]

    sind=np.argsort(catmax)
    cattime=cattime[sind,:]
    catx=catx[sind]
    caty=caty[sind]    
    catrain=catrain[sind,:]  
    catmax=catmax[sind]/mnorm*rainprop.timeres/60.
    end = time.time()   
    print "catalog timer: "+str((end - start)/60.)+" minutes"
    
    # WRITE CATALOG
    print "writing storm catalog..."
    RainyDay.writecatalog(catrain,catmax,catx,caty,cattime,latrange,lonrange,catalogname,nstorms,catmask,parameterfile)   


#################################################################################
# STEP 2: RESAMPLING
#################################################################################
print "trimming storm catalog..."

origstormsno=np.arange(0,nstorms)

if CreateCatalog==False:
    if duration>catrain.shape[1]*rainprop.timeres/60.:
        sys.exit("The specified duration is longer than the length of the storm catalog")

                
# EXLCUDE "BAD" STORMS OR FOR DOING SENSITIVITY ANALYSIS TO PARTICULAR STORMS.  THIS IS PARTICULARLY NEEDED FOR ANY RADAR RAINFALL PRODUCT WITH SERIOUS ARTIFACTS
includestorms=np.ones((len(catmax)),dtype="bool")
exclude=cardinfo[cardinfo[:,0]=="EXCLUDESTORMS",1][0] 
if exclude.lower()!="none":
    exclude=exclude.split(',')
    for i in range(0,len(exclude)):
        includestorms[np.int(exclude[i])-1]=False
        
catrain=catrain[includestorms,:]
catmax=catmax[includestorms]
catx=catx[includestorms]
caty=caty[includestorms]
cattime=cattime[includestorms,:]
modstormsno=origstormsno[includestorms]  

if nstorms<np.sum(includestorms==True):
    catrain=catrain[-nstorms:,:]
    catmax=catmax[-nstorms:]
    catx=catx[-nstorms:]
    caty=caty[-nstorms:]
    cattime=cattime[-nstorms:,:]
    nstorms=np.shape(catx)
else:    
    nstorms=np.sum(includestorms==True)

# EXCLUDE STORMS BY MONTH AND YEAR
# THIS SECTION EXISTS IN CASE YOU WANT TO USE A PRE-EXISTING STORM CATALOG THAT HASN'T CONSIDERED ANY MONTH-BASED OR YEAR-BASED EXCLUSION
if CreateCatalog==False:        
    catyears=cattime[:,0].astype('datetime64[Y]').astype(int)+1970
    if cattime.shape[1]>1:
        catmonths=cattime[:,-2].astype('datetime64[M]').astype(int)-(catyears-1970)*12+1   # set to "-2" instead of "-1" to catch a very specific situation
    else:
        catmonths=cattime[:,-1].astype('datetime64[M]').astype(int)-(catyears-1970)*12+1
    catinclude=np.ones(cattime.shape[0],dtype="bool")
    if isinstance(includeyears, (bool))==False:
        excludeyears=set(catyears)^set(includeyears)
        for j in excludeyears:
            catinclude[catyears==j]=False
    if isinstance(excludemonths,(bool))==False:
        for j in excludemonths:
            catinclude[catmonths==j]=False
    nstorms=np.sum(catinclude==True)
    catrain=catrain[catinclude,:]
    catmax=catmax[catinclude]
    catx=catx[catinclude]
    caty=caty[catinclude]
    cattime=cattime[catinclude,:]
    modstormsno=modstormsno[catinclude]
    
catmax=catmax*IntensitySens
catrain=catrain*IntensitySens


#==============================================================================
# If the storm catalog has a different duration than the specified duration, fix it!
# find the max rainfall for the N-hour duration, not the M-day duration
#==============================================================================
if CreateCatalog==False:   
    print "checking storm catalog duration, and adjusting if needed..."
    if 60./rainprop.timeres*duration!=np.float(catrain.shape[1]):
        print "Storm catalog duration is longer than the specified duration, finding max rainfall periods for specified duration..."
        dur_maxind=np.array((nstorms),dtype='int32')
        dur_x=0
        dur_y=0
        dur_j=0
        temprain=np.zeros((nstorms,duration*60/rainprop.timeres,rainprop.subdimensions[0],rainprop.subdimensions[1]),dtype='float32')
        rainsum=np.zeros((rainprop.subdimensions[0]-maskheight+1,rainprop.subdimensions[1]-maskwidth+1),dtype='float32')
    
        temptime=np.empty((nstorms,duration*60/rainprop.timeres),dtype='datetime64[m]')
        for i in range(0,nstorms):
            if (100*((i+1)%(nstorms/10)))==0:
                print 'adjusting duration of storms, '+str(100*(i+1)/nstorms)+'% complete...'
            dur_max=0.
            for j in range(0,catrain.shape[1]-duration*60/rainprop.timeres):
                maxpass=np.nansum(catrain[i,j:j+int(duration*60./rainprop.timeres),:],axis=0)
                if weavecheck:
                    maxtemp,tempy,tempx=RainyDay.catalogweave(maxpass,trimmask,np.int(xlen),np.int(ylen),np.int(maskheight),np.int(maskwidth),rainsum)    
                elif numbacheck:
                    maxtemp,tempy,tempx=RainyDay.catalogNumba(maxpass,trimmask,xlen,ylen,maskheight,maskwidth,rainsum)                       
                else:
                    maxtemp,tempy,tempx=RainyDay.catalogAlt(maxpass,trimmask,xlen,ylen,maskheight,maskwidth,rainsum)    
                    
                if maxtemp>dur_max:
                    dur_max=maxtemp
                    dur_x=tempx
                    dur_y=tempy
                    dur_j=j
            catmax[i]=dur_max
            catx[i]=dur_x
            caty[i]=dur_y
            temprain[i,:]=catrain[i,dur_j:dur_j+int(duration*60./rainprop.timeres),:]
            temptime[i,:]=cattime[i,dur_j:dur_j+int(duration*60./rainprop.timeres)]
        catrain=temprain
        cattime=temptime
    
        sind=np.argsort(catmax)
        cattime=cattime[sind,:]
        catx=catx[sind]
        caty=caty[sind]
        catrain=catrain[sind,:]
        catmax=catmax[sind]*rainprop.timeres/60./mnorm

 
    
#==============================================================================
# IF THE USER IS SUPPLYING A DISTRIBUTION FOR THE INTENSITY, NORMALIZE THE FIELDS
# SO THAT THE INTENSITY CAN BE APPLIED PROPERLY
#==============================================================================
if userdistr.all()!=False:   
    print "normalizing rainfall fields..."
    for i in range(0,nstorms):
        tempmax=np.nanmax(np.nansum(catrain[i,:],axis=0))
        catrain[i,:]=catrain[i,:]/tempmax

    
#==============================================================================
# Create kernel density smoother, even if you don't use it for resampling
#==============================================================================
print "calculating kernel density smoother..."

kx,ky=np.meshgrid(range(0,rainprop.subdimensions[1]-maskwidth),range(0,rainprop.subdimensions[0]-maskheight))
kpositions=np.vstack([ky.ravel(),kx.ravel()])

checkind=np.where(np.logical_and(np.logical_and(caty!=0,catx!=0),np.logical_and(caty!=ylen-1,catx!=xlen-1)))

invalues=np.vstack([caty[checkind], catx[checkind]])
stmkernel=stats.gaussian_kde(invalues)
pltkernel=np.reshape(stmkernel(kpositions), kx.shape)

pltkernel=pltkernel/np.sum(pltkernel)
cumkernel=np.reshape(np.cumsum(pltkernel),(kx.shape))


#==============================================================================
# If you're using intensity-dependent resampling, calculate the kernel "stack"
#==============================================================================
if resampletype=='intensity':

    print 'calculating kernel density "stack" for intensity-dependent resampling...'
    

    stackkernel=np.empty((stackbins,cumkernel.shape[0],cumkernel.shape[1]),dtype='float64')
    stackcat=np.empty((stackbins),dtype='float32')
    
    stackstep=len(checkind[0])/stackbins
    ctr=0
    for i in range(0,len(checkind[0])-stackstep,stackstep):
        invalues=np.vstack([caty[checkind][i:i+stackstep], catx[checkind][i:i+stackstep]])
        stackcat[ctr]=catmax[checkind][i]
        stmkernel=stats.gaussian_kde(invalues)
        tmpkernel=np.reshape(stmkernel(kpositions), kx.shape)
        
        tmpkernel=tmpkernel/np.sum(tmpkernel)

        stackkernel[ctr]=np.reshape(np.cumsum(tmpkernel),(kx.shape))
        ctr=ctr+1



#==============================================================================
# DO YOU WANT TO CREATE DIAGNOSTIC PLOTS?
#==============================================================================

if DoDiagnostics:
    
    # this is for a very specific situation....
    if np.any(np.isnan(catmax)):
        catmax[:]=100.

    if areatype.lower()=="box":
        from matplotlib.patches import Polygon
        def plot_rectangle(bmap, lonmin,lonmax,latmin,latmax):
            xs = [lonmin,lonmax,lonmax,lonmin,lonmin]
            ys = [latmin,latmin,latmax,latmax,latmin]

            p = Polygon([(lonmin,latmin),(lonmin,latmax),(lonmax,latmax),(lonmax,latmin)],facecolor='grey',edgecolor='black',alpha=0.5,linewidth=2)
            plt.gca().add_patch(p)

    print "preparing diagnostic plots..."
    
    if rainprop.subdimensions[0]>rainprop.subdimensions[1]:
        figsizex=5
        figsizey=5+0.25*5*np.float(rainprop.subdimensions[0])/rainprop.subdimensions[1]
    elif rainprop.subdimensions[0]<rainprop.subdimensions[1]: 
        figsizey=5
        figsizex=5+0.25*5*np.float(rainprop.subdimensions[0])/rainprop.subdimensions[1]
    else:
        figsizey=5   
        figsizex=5
       
    bmap=Basemap(llcrnrlon=rainprop.subextent[0],llcrnrlat=rainprop.subextent[2],urcrnrlon=rainprop.subextent[1],urcrnrlat=rainprop.subextent[3],projection='cyl',resolution='l')    
    outerextent=np.array(rainprop.subextent,dtype='float32')   
    for i in range(0,nstorms):    
        temprain=np.nansum(catrain[i,:],axis=0)*rainprop.timeres/60.
        if userdistr.all()==False:     
            temprain[temprain<0.05*catmax[i]]=np.nan
        fig = plt.figure()
        fig.set_size_inches(figsizex,figsizey)
        f1=plt.imshow(temprain, interpolation='none',extent=outerextent,cmap='Blues')
        plt.title('Storm '+str(i+1)+': '+str(cattime[i,-1])+'\nMax Rainfall:'+str(round(catmax[i]))+' mm @ Lat/Lon:'+"{:6.1f}".format(latrange[caty[i]]-(maskheight/2+maskheight%2)*rainprop.spatialres[0])+u'\N{DEGREE SIGN}'+','+"{:6.1f}".format(lonrange[catx[i]]+(maskwidth/2+maskwidth%2)*rainprop.spatialres[0])+u'\N{DEGREE SIGN}')
        #bmap=Basemap(llcrnrlon=rainprop.subextent[0],llcrnrlat=rainprop.subextent[2],urcrnrlon=rainprop.subextent[1],urcrnrlat=rainprop.subextent[3],projection='cyl',resolution='l')
        bmap.drawcoastlines(linewidth=1.25)
        bmap.drawparallels(np.linspace(rainprop.subextent[2],rainprop.subextent[3],2),labels=[1,0,0,0],fmt='%6.1f')
        bmap.drawmeridians(np.linspace(rainprop.subextent[0],rainprop.subextent[1],2),labels=[1,0,0,1],fmt='%6.1f')
        if BaseMap.lower()!='none':
            bmap.readshapefile(BaseMap,BaseField,color="grey")
        if areatype.lower()=="basin" and os.path.isfile(wsmaskshp):
            wmap=Basemap(llcrnrlon=outerextent[0],llcrnrlat=outerextent[2],urcrnrlon=outerextent[1],urcrnrlat=outerextent[3],projection='cyl')
            try:            
                wmap.readshapefile(wsmaskshp.split('.')[0],str(0),color="black")
            except ValueError:
                if i==0:
                    print "problem plotting the watershed map; skipping..."
        elif areatype.lower()=="box":
            plot_rectangle(bmap,boxarea[0],boxarea[1],boxarea[2],boxarea[3])
        elif areatype.lower()=="point":
            plt.scatter(ptlon,ptlat,color="b")
        if rainprop.subdimensions[1]>rainprop.subdimensions[0]:
            cb=plt.colorbar(f1,orientation='horizontal')
        else:
            cb=plt.colorbar(f1)
        cb.set_label('Total Rainfall [mm]')
        plt.scatter(lonrange[catx[i]]+(maskwidth/2+maskwidth%2)*rainprop.spatialres[0],latrange[caty[i]]-(maskheight/2+maskheight%2)*rainprop.spatialres[1],s=10,facecolors='none',edgecolors='r',alpha=0.5)
        plt.savefig(diagpath+'Storm'+str(i+1)+'_'+str(cattime[i,-1]).split('T')[0]+'.png',dpi=250)
        plt.close()       
 
         
    # PLOT STORM OCCURRENCE PROBABILITIES
    fig = plt.figure()
    fig.set_size_inches(figsizex,figsizey)
    f1=plt.imshow(pltkernel,interpolation="none",extent=rainprop.subextent,cmap='Reds')
    plt.title("Probability of storm occurrence")
    #bmap=Basemap(llcrnrlon=rainprop.subextent[0],llcrnrlat=rainprop.subextent[2],urcrnrlon=rainprop.subextent[1],urcrnrlat=rainprop.subextent[3],projection='cyl',resolution='l')
    bmap.drawcoastlines(linewidth=1.25)
    bmap.drawparallels(np.linspace(rainprop.subextent[2],rainprop.subextent[3],2),labels=[1,0,0,0],fmt='%6.1f')
    bmap.drawmeridians(np.linspace(rainprop.subextent[0],rainprop.subextent[1],2),labels=[1,0,0,1],fmt='%6.1f')
    if BaseMap.lower()!='none':
        bmap.readshapefile(BaseMap,BaseField,color="grey")
    if areatype.lower()=="basin" and os.path.isfile(wsmaskshp):
        wmap=Basemap(llcrnrlon=rainprop.subextent[0],llcrnrlat=rainprop.subextent[2],urcrnrlon=rainprop.subextent[1],urcrnrlat=rainprop.subextent[3],projection='cyl')
        try:        
            wmap.readshapefile(wsmaskshp.split('.')[0],str(0),color="black")
        except ValueError:
            print "problem plotting the watershed map; skipping..."
            
    elif areatype.lower()=="box":
        plot_rectangle(bmap,boxarea[0],boxarea[1],boxarea[2],boxarea[3])
    elif areatype.lower()=="point":
        plt.scatter(ptlon,ptlat,color="b")
    #if rainprop.subdimensions[1]>rainprop.subdimensions[0]:
    #    cb=plt.colorbar(orientation='horizontal',ticks=[np.nanmin(pltkernel),np.nanmax(pltkernel)])
    #else:
    #    cb=plt.colorbar(ticks=[np.nanmin(pltkernel),np.nanmax(pltkernel)])
    if rainprop.subdimensions[1]>rainprop.subdimensions[0]:
        cb=plt.colorbar(f1,orientation='horizontal')
    else:
        cb=plt.colorbar(f1)
    
    plt.scatter(lonrange[catx]+(maskwidth/2+maskwidth%2)*rainprop.spatialres[0],latrange[caty]-(maskheight/2+maskheight%2)*rainprop.spatialres[1],s=catmax/2,facecolors='k',edgecolors='none',alpha=0.75)
    plt.savefig(diagpath+'KernelDensity.png',dpi=250)
    #plt.savefig(diagpath+'KernelDensity.pdf')
    plt.close('all')
    
    # PLOT AVERAGE STORM RAINFALL
    avgrain=np.nansum(catrain,axis=(0,1))/nstorms
    fig = plt.figure()
    fig.set_size_inches(figsizex,figsizey)
    f1=plt.imshow(avgrain,interpolation="none",extent=rainprop.subextent,cmap="Blues")
    plt.title("Average storm rainfall")
    #bmap=Basemap(llcrnrlon=rainprop.subextent[0],llcrnrlat=rainprop.subextent[2],urcrnrlon=rainprop.subextent[1],urcrnrlat=rainprop.subextent[3],projection='cyl',resolution='l')
    bmap.drawcoastlines(linewidth=1.25)
    bmap.drawparallels(np.linspace(rainprop.subextent[2],rainprop.subextent[3],2),labels=[1,0,0,0],fmt='%6.1f')
    bmap.drawmeridians(np.linspace(rainprop.subextent[0],rainprop.subextent[1],2),labels=[1,0,0,1],fmt='%6.1f')
    if BaseMap.lower()!='none':
        bmap.readshapefile(BaseMap,BaseField,color="grey")
    if areatype.lower()=="basin" and os.path.isfile(wsmaskshp):
        wmap=Basemap(llcrnrlon=rainprop.subextent[0],llcrnrlat=rainprop.subextent[2],urcrnrlon=rainprop.subextent[1],urcrnrlat=rainprop.subextent[3],projection='cyl')
        try:
            wmap.readshapefile(wsmaskshp.split('.')[0],str(0),color="black")
        except ValueError:
            print "problem plotting the watershed map; skipping..."
    elif areatype.lower()=="box":
        plot_rectangle(bmap,boxarea[0],boxarea[1],boxarea[2],boxarea[3])
    elif areatype.lower()=="point":
        plt.scatter(ptlon,ptlat,color="b")
    if rainprop.subdimensions[1]>rainprop.subdimensions[0]:
        cb=plt.colorbar(f1,orientation='horizontal')
    else:
        cb=plt.colorbar(f1)
    cb.set_label('rainfall [mm/storm]')
    plt.scatter(lonrange[catx]+(maskwidth/2+maskwidth%2)*rainprop.spatialres[0],latrange[caty]-(maskheight/2+maskheight%2)*rainprop.spatialres[1],s=catmax/2,facecolors='k',edgecolors='none',alpha=0.75)
    plt.savefig(diagpath+'AvgStormRain.png',dpi=250)
    #plt.savefig(diagpath+'AvgStormRain.pdf')
    plt.close()
        
         
if DoDiagMovies:
    "preparing diagnostic movies..."
    
    if rainprop.subdimensions[0]>rainprop.subdimensions[1]:
        figsizex=5
        figsizey=5+0.25*5*np.float(rainprop.subdimensions[0])/rainprop.subdimensions[1]
    elif rainprop.subdimensions[0]<rainprop.subdimensions[1]: 
        figsizey=5
        figsizex=5+0.25*5*np.float(rainprop.subdimensions[0])/rainprop.subdimensions[1]
    else:
        figsizey=5   
        figsizex=5
        
    outerextent=np.array([rainprop.subextent[0],rainprop.subextent[1],rainprop.subextent[2],rainprop.subextent[3]],dtype='float32')      

    def init():
        f1.set_array(temprain[0,:])
        titl=plt.title(tstr+'\n'+str(temptime[0]))
        return [f1,titl]

    def animate(i):
        f1.set_array(temprain[i,:])
        titl=plt.title(tstr+'\n'+str(temptime[i]))
        return [f1,titl]

    for i in range(0,nstorms):
        temprain=catrain[i,:]
        #temprain[temprain<0.5]=np.nan
        temptime=cattime[i,:]
        fig = plt.figure()
        fig.set_size_inches(figsizex,figsizey)
        tstr='Storm '+str(i+1)
        titl=plt.title(tstr+'\n'+str(temptime[0,]))
        #pylab.hold(True)
        f1=plt.imshow(temprain[0,:],extent=rainprop.subextent,interpolation="none",cmap='Blues',norm=LogNorm(vmin=0.1, vmax=np.nanmax(temprain)))
        #bmap=Basemap(llcrnrlon=rainprop.subextent[0],llcrnrlat=rainprop.subextent[2],urcrnrlon=rainprop.subextent[1],urcrnrlat=rainprop.subextent[3],projection='cyl',resolution='l')
        bmap.drawcoastlines(linewidth=1.25)
        bmap.drawparallels(np.linspace(rainprop.subextent[2],rainprop.subextent[3],2),labels=[1,0,0,0],fmt='%6.1f')
        bmap.drawmeridians(np.linspace(rainprop.subextent[0],rainprop.subextent[1],2),labels=[1,0,0,1],fmt='%6.1f')

        if BaseMap.lower()!='none':
            bmap.readshapefile(BaseMap,BaseField,color="grey")
        if areatype.lower()=="basin" and os.path.isfile(wsmaskshp):
            wmap=Basemap(llcrnrlon=rainprop.subextent[0],llcrnrlat=rainprop.subextent[2],urcrnrlon=rainprop.subextent[1],urcrnrlat=rainprop.subextent[3],projection='cyl')
            try:
                if i==0:
                    wmap.readshapefile(wsmaskshp.split('.')[0],str(0),color="black")
            except ValueError:
                print "problem plotting the watershed map; skipping..."
        elif areatype.lower()=="box":
            plot_rectangle(bmap,boxarea[0],boxarea[1],boxarea[2],boxarea[3])
        elif areatype.lower()=="point":
            plt.scatter(ptlon,ptlat,color="b")
        if rainprop.subdimensions[1]>rainprop.subdimensions[0]:
            cb=plt.colorbar(f1,orientation='horizontal')
        else:
            cb=plt.colorbar(f1)
        cb.set_label('Rain Rate (mm/hr)')
        anim=manimation.FuncAnimation(fig,animate,init_func=init,frames=temprain.shape[0],interval=25,blit=True)
        anim.save(diagpath+'StormMovie_'+str(i+1)+'_'+str(cattime[i,-1]).split('T')[0]+'.mp4',fps=3,dpi=250)   # animation seems buggy, I can only create movies for odd frames-per-second rates       
        plt.close()  


     # this is for a very specific situation....
    if np.any(catmax==100.):
        catmax[:]=np.nan
        
        
#==============================================================================
# DO THE RESAMPLING!
#==============================================================================
print "resampling and transposing..."

if np.all(includeyears==False):
    nyears=len(range(min(cattime[:,-1].astype('datetime64[Y]').astype(int)),max(cattime[:,-1].astype('datetime64[Y]').astype(int))+1))
else:
    nyears=len(includeyears)
    
    
# resampling counts options:
if samplingtype.lower()=='poisson':
    lrate=len(catmax)/nyears*FrequencySens                  
    ncounts=np.random.poisson(lrate,(nsimulations,nrealizations))
    cntr=0
    ncounts[ncounts==0]=1
else:
    _,yrscount=np.unique(cattime[:,-1].astype('datetime64[Y]').astype(int)+1970,return_counts=True)
    if len(yrscount)<nyears:
        yrscount=np.append(yrscount,np.ones(nyears-len(yrscount),dtype='int32'))
    ncounts=np.random.choice(yrscount,(nsimulations,nrealizations),replace=True)   
    
    
whichstorms=np.empty((np.nanmax(ncounts),ncounts.shape[0],ncounts.shape[1]),dtype='int32')
whichstorms[:]=-999

if rotation==True:
    randangle=(maxangle-minangle)*np.random.random_sample(((np.nanmax(ncounts),ncounts.shape[0],ncounts.shape[1])))+minangle
    
    angbins=np.linspace(minangle,maxangle,nanglebins)
    angs=math.pi/180*angbins
    anglebins=np.digitize(randangle.ravel(),angbins).reshape(np.nanmax(ncounts),ncounts.shape[0],ncounts.shape[1])


# DOES THIS PROPERLY HANDLE STORM EXCLUSIONS???  I think so...
for i in range(0,np.nanmax(ncounts)):
    whichstorms[i,ncounts>=i+1]=np.random.randint(0,nstorms-1,(len(ncounts[ncounts>=i+1])))

whichrain=np.zeros((whichstorms.shape),dtype='float32')
testrain=np.zeros((whichstorms.shape),dtype='float32')
whichx=np.zeros((whichstorms.shape),dtype='int32')
whichy=np.zeros((whichstorms.shape),dtype='int32')


for i in range(0,nstorms):
    # UNIFORM RESAMPLING
    if resampletype=='uniform':
        whichx[whichstorms==i]=np.random.randint(0,np.int(rainprop.subdimensions[1])-maskwidth-1,len(whichx[whichstorms==i]))
        whichy[whichstorms==i]=np.random.randint(0,np.int(rainprop.subdimensions[0])-maskheight-1,len(whichy[whichstorms==i]))
 
    # KERNEL-BASED RESAMPLING
    elif resampletype=='kernel':
        rndloc=np.random.random_sample(len(whichx[whichstorms==i]))
        if weavecheck:
            whichx[whichstorms==i],whichy[whichstorms==i]=RainyDay.weavekernel(rndloc,cumkernel)
        elif numbacheck:
            whichx[whichstorms==i],whichy[whichstorms==i]=RainyDay.numbakernel(rndloc,cumkernel)
        else:
            whichx[whichstorms==i],whichy[whichstorms==i]=RainyDay.pykernel(rndloc,cumkernel)
        
    
    # SET UP MANUAL PDF RESAMPLING
    elif resampletype=='manual':  
        sys.exit("not configured for manually supplied pdf yet!")
        
    # "STACK" FOR INTENSITY-CONDITIONED RESAMPLING
    elif resampletype=='intensity':
        rndloc=np.random.random_sample(len(whichx[whichstorms==i]))
        tempkernel=stackkernel[np.where((catmax[i]> (stackcat-0.001) )==True)[0][-1],:]
        if weavecheck:
            whichx[whichstorms==i],whichy[whichstorms==i]=RainyDay.weavekernel(rndloc,tempkernel)
        elif numbacheck:
            whichx[whichstorms==i],whichy[whichstorms==i]=RainyDay.numbakernel(rndloc,tempkernel)
        else:
            whichx[whichstorms==i],whichy[whichstorms==i]=RainyDay.pykernel(rndloc,tempkernel)


    passrain=np.nansum(catrain[i,:],axis=0)

    if rotation: 
        print 'rotating storms for transposition, '+str(100*(i+1)/nstorms)+'% complete...'
        delarray.append([])
         
        xctr=catx[i]+maskwidth/2.
        yctr=caty[i]+maskheight/2.
        xlinsp=np.linspace(-xctr,rainprop.subdimensions[1]-xctr,rainprop.subdimensions[1])
        ylinsp=np.linspace(-yctr,rainprop.subdimensions[0]-yctr,rainprop.subdimensions[0])
        ingridx,ingridy=np.meshgrid(xlinsp,ylinsp)
        ingridx=ingridx.flatten()
        ingridy=ingridy.flatten()
        outgrid=np.column_stack((ingridx,ingridy))       
        

        binctr=0
        for cbin in np.unique(anglebins):
            #print "really should fix the center of rotation! to be the storm center"
            rotx=ingridx*np.cos(angs[binctr])+ingridy*np.sin(angs[binctr])
            roty=-ingridx*np.sin(angs[binctr])+ingridy*np.cos(angs[binctr])
            rotgrid=np.column_stack((rotx,roty))
            delaunay=sp.spatial.qhull.Delaunay(rotgrid)
            delarray[i].append(delaunay)
            interp=sp.interpolate.LinearNDInterpolator(delaunay,passrain.flatten(),fill_value=0.)
            tpass=np.reshape(interp(outgrid),rainprop.subdimensions)
            whichrain[np.logical_and(whichstorms==i,anglebins==cbin)]=RainyDay.SSTalt(tpass,whichx[np.logical_and(whichstorms==i,anglebins==cbin)],whichy[np.logical_and(whichstorms==i,anglebins==cbin)],trimmask,xmin,xmax,ymin,ymax,maskheight,maskwidth)*rainprop.timeres/60./mnorm
            binctr=binctr+1
    else:
        whichrain[whichstorms==i]=RainyDay.SSTalt(passrain,whichx[whichstorms==i],whichy[whichstorms==i],trimmask,xmin,xmax,ymin,ymax,maskheight,maskwidth)*rainprop.timeres/60./mnorm
    


# HERE ARE THE ANNUAL MAXIMA!!!
maxrain=np.nanmax(whichrain,axis=0)

# HERE THE OPTIONAL USER SPECIFIED INTENSITY DISTRIBUTION IS APPLIED    
if userdistr.all()!=False:
    rvs=sp.stats.genextreme.rvs(userdistr[2],loc=userdistr[0],scale=userdistr[1],size=maxrain.shape).astype('float32')
    maxrain=maxrain*rvs
    
# PULL OUT THE CORRESPONDING TRANSPOSITION INFORMATION
maxind=np.nanargmax(whichrain,axis=0)

# THIS ISN'T VERY ELEGANT
maxx=np.empty((maxind.shape),dtype="int32")
maxy=np.empty((maxind.shape),dtype="int32")
maxstorm=np.empty((maxind.shape),dtype="int32")
if rotation:
    maxangles=np.empty((maxind.shape),dtype="float32")
for i in range(0,np.max(ncounts)):
    maxx[maxind==i]=whichx[i,maxind==i]
    maxy[maxind==i]=whichy[i,maxind==i]
    maxstorm[maxind==i]=whichstorms[i,maxind==i]
    if rotation:
        maxangles[maxind==i]=randangle[i,maxind==i]

# RANK THE STORMS BY INTENSITY AND ASSIGN RETURN PERIODS
exceedp=np.linspace(1,1./nsimulations,nsimulations)
returnperiod=1/exceedp
sortind=np.argsort(maxrain,axis=0)
sortrain=np.sort(maxrain,axis=0)
sortx=np.empty((maxind.shape),dtype="int32")
sorty=np.empty((maxind.shape),dtype="int32")
sortstorms=np.empty((maxind.shape),dtype="int32")
sortangle=np.empty((maxind.shape),dtype="float32")

for i in range(0,nrealizations):
    sortx[:,i]=maxx[sortind[:,i],i]
    sorty[:,i]=maxy[sortind[:,i],i]
    sortstorms[:,i]=maxstorm[sortind[:,i],i]
    if rotation:
        sortangle[:,i]=maxangles[sortind[:,i],i]
        
    
# FIND THE TIMES:
sorttimes=np.zeros((maxind.shape[0],maxind.shape[1],cattime.shape[1]),dtype="datetime64[m]")
whichorigstorm=np.zeros((maxind.shape[0],maxind.shape[1]),dtype='int32')
for i in range(0,nstorms):
    sorttimes[sortstorms==i,:]=cattime[i,:]
    whichorigstorm[sortstorms==i]=modstormsno[i]+1
    
    
if alllevels==False:
    reducedlevind=[]
    def find_nearest(array,value):
        idx = (np.abs(array-value)).argmin()
        return idx
    
    for i in range(0,len(speclevels)):
        reducedlevind.append(find_nearest(returnperiod,speclevels[i]))  
    
    returnperiod=returnperiod[reducedlevind]
    sortrain=sortrain[reducedlevind,:]
    sortstorms=sortstorms[reducedlevind,:]
    sorttimes=sorttimes[reducedlevind,:]
    exceedp=exceedp[reducedlevind]
    sortx=sortx[reducedlevind,:]
    sorty=sorty[reducedlevind,:]
    whichorigstorm=whichorigstorm[reducedlevind,:]
    
    
#################################################################################
# STEP 2a (OPTIONAL): Find the single storm maximized storm rainfall-added DBW 7/19/2017
#################################################################################    

if deterministic:
    print "finding maximizing rainfall..."
    
    nanmask=deepcopy(trimmask)
    nanmask[np.isclose(nanmask,0.)]=np.nan
    nanmask[np.isclose(nanmask,0.)==False]=1.0
    max_trnsx=catx[-1]
    max_trnsy=caty[-1]
    if rotation==False:
        # there is some small bug that I don't understand either here or in the storm catalog creation, in which maxstm_avgrain will not exactly match catmax[-1] unless areatype is a point
        maxstm_rain=np.multiply(catrain[-1,:,max_trnsy:(max_trnsy+maskheight),max_trnsx:(max_trnsx+maskwidth)],nanmask)
        maxstm_avgrain=np.nansum(np.multiply(catrain[-1,:,max_trnsy:(max_trnsy+maskheight),max_trnsx:(max_trnsx+maskwidth)],trimmask))/mnorm
        maxstm_ts=np.nansum(np.multiply(maxstm_rain,trimmask)/mnorm,axis=(1,2))
        maxstm_time=cattime[-1,:]
    else:  
        prevmxstm=0.
        maxstm_rain=np.empty((catrain.shape[1],nanmask.shape[0],nanmask.shape[1]),dtype='float32')
        for i in range(0,nstorms):
            passrain=np.nansum(catrain[i,:],axis=0)
            xctr=catx[i]+maskwidth/2.
            yctr=caty[i]+maskheight/2.
            xlinsp=np.linspace(-xctr,rainprop.subdimensions[1]-xctr,rainprop.subdimensions[1])
            ylinsp=np.linspace(-yctr,rainprop.subdimensions[0]-yctr,rainprop.subdimensions[0])
            ingridx,ingridy=np.meshgrid(xlinsp,ylinsp)
            ingridx=ingridx.flatten()
            ingridy=ingridy.flatten()
            outgrid=np.column_stack((ingridx,ingridy))       
        
            for tempang in angbins:
                #print "really should fix the center of rotation! to be the storm center"
                rotx=ingridx*np.cos(tempang)+ingridy*np.sin(tempang)
                roty=-ingridx*np.sin(tempang)+ingridy*np.cos(tempang)
                rotgrid=np.column_stack((rotx,roty))
                delaunay=sp.spatial.qhull.Delaunay(rotgrid)
                interp=sp.interpolate.LinearNDInterpolator(delaunay,passrain.flatten(),fill_value=0.)
                train=np.reshape(interp(outgrid),rainprop.subdimensions)
                temp_maxstm_avgrain=np.nansum(np.multiply(train[max_trnsy:(max_trnsy+maskheight),max_trnsx:(max_trnsx+maskwidth)],trimmask))/mnorm
                if temp_maxstm_avgrain>prevmxstm:
                    maxstm_avgrain=temp_maxstm_avgrain
                    prevmxstm=maxstm_avgrain
                    maxstm_time=cattime[-i,:]
                    
                    for k in range(0,len(maxstm_time)):
                        interp=sp.interpolate.LinearNDInterpolator(delaunay,catrain[i,k,:].flatten(),fill_value=0.)
                        maxstm_rain[k,:]=np.reshape(interp(outgrid),rainprop.subdimensions)[max_trnsy:(max_trnsy+maskheight),max_trnsx:(max_trnsx+maskwidth)]
                    maxstm_rain=np.multiply(maxstm_rain,nanmask)
                    maxstm_ts=np.nansum(np.multiply(maxstm_rain,trimmask)/mnorm,axis=(1,2))


#################################################################################
# STEP 3 (OPTIONAL): RAINFALL FREQUENCY ANALYSIS
#################################################################################

if FreqAnalysis:
    print "preparing frequency analysis..."
    
    if spreadtype=='ensemble':
        spreadmin=np.nanmin(sortrain,axis=1)
        spreadmax=np.nanmax(sortrain,axis=1)    
    else:
        spreadmin=np.percentile(sortrain,(100-quantilecalc)/2,axis=1)
        spreadmax=np.percentile(sortrain,quantilecalc+(100-quantilecalc)/2,axis=1)

    freqanal=np.column_stack((exceedp,returnperiod,spreadmin,RainyDay.mynanmean(sortrain,1),spreadmax))
    
    np.savetxt(FreqFile,freqanal,delimiter=',',header='prob.exceed,returnperiod,minrain,meanrain,maxrain',fmt='%6.3f',comments='')
    
  
    import matplotlib.patches as mpatches
    from matplotlib.font_manager import FontProperties
    warnings.filterwarnings('ignore')
    fontP = FontProperties()
    fontP.set_size('xx-small')
    fig, ax = plt.subplots(1)
    line1, = plt.plot(exceedp, RainyDay.mynanmean(sortrain,1), lw=1, label='Average', color='blue')

    ax.fill_between(exceedp, spreadmin, spreadmax, facecolor='dodgerblue', alpha=0.5,label='Ensemble Variability')
    red_patch = mpatches.Patch(color='dodgerblue', label='Spread')
    plt.legend(handles=[line1,red_patch],loc='lower right',prop = fontP)

    plt.ylim(1,1000)
    ax.set_xlabel('Exceedance Probability [-]\n1/(Return Period) [year]')
    ax.set_ylabel('Rainfall [mm]')
    ax.set_yscale('log')
    ax.set_xscale('log')
    plt.gca().invert_xaxis()
    ax.grid()
    plt.tight_layout()
    plt.savefig(diagpath+'FrequencyAnalysis.png',dpi=250)
    plt.close('all')
       
       
    
#################################################################################
# STEP 4 (OPTIONAL): WRITE RAINFALL SCENARIOS
#################################################################################



if Scenarios:
    
    print "writing spacetime rainfall scenarios..."
    
    # if desired, this will "pad" the beginning of the scenarios to add a spin-up period.  Not recommended to use long spin-ups, due to memory constraints
    # this scheme basically creates two storm catalogs and mixes them up!
    if prependrain==True:
        print "prepending rainfall for model spin-up..."
        outerextent=np.array(rainprop.subextent,dtype='float32') 
        if CreateCatalog==False:
            # need to generate a file list since it hasn't already been done
            flist=RainyDay.createfilelist(inpath,includeyears,excludemonths)
            if len(flist)==0:
                sys.exit("couldn't prepend the rainfall files for spinup period because the input rainfall files weren't available.") 

        _,_,inlatitude,inlongitude=RainyDay.readnetcdf(flist[0])
        rainprop.dimensions=[len(inlatitude),len(inlongitude)]
        rainprop.bndbox=[np.min(inlongitude),np.max(inlongitude)+rainprop.spatialres[0],np.min(inlatitude)-rainprop.spatialres[1],np.max(inlatitude)]          
        
        rainprop.subextent,rainprop.subind,rainprop.subdimensions=RainyDay.findsubbox(inarea,rainprop)
        innerextent=deepcopy(rainprop.subextent)
        innerind=deepcopy(rainprop.subind)
             
        #subgrid,gx,gy=RainyDay.creategrids(rainprop)
    
        tlen=1440*pretime/rainprop.timeres
        
        precat=np.zeros((catrain.shape[0],tlen,catrain.shape[2],catrain.shape[3]),dtype='float32')

        starttime=np.empty((cattime.shape[0],tlen),dtype='datetime64[m]')
        starttime[:,0]=np.subtract(cattime[:,0],np.timedelta64(np.int32(pretime)*1440,'m'))
        for i in range(0,nstorms):
            starttime[i,:]=np.arange(starttime[i,0],cattime[i,0],np.timedelta64(rainprop.timeres,'m'))   
        cattime=np.concatenate((starttime,cattime),axis=1)
        
        
        # roll 'em back!
        for i in np.arange(0,nstorms):
            print "Creating pre-pending catalog entry for storm "+str(i+1)
            starttime=cattime[i,tlen]-np.timedelta64(np.int32(pretime)*1440,'m')
            startstr=np.str(starttime).replace('-','').split('T')[0]
            prelist=np.core.defchararray.find(flist,startstr)
            preind=np.where(prelist==np.max(prelist))[0][0]
            tlist=flist[preind:preind+np.int(np.round(pretime))] 

            for j in range(0,len(tlist)):
                print 'Pre-pending rainfall with file '+tlist[j]
                inrain,intime,_,_=RainyDay.readnetcdf(tlist[j],inbounds=rainprop.subind)
                inrain[inrain<0.]=np.nan
                
                for k in range(0,24*60/rainprop.timeres):
                    if np.in1d(intime[k],cattime[i,:])[0]: 
                        cattime[i,j*24*60/rainprop.timeres:j*24*60/rainprop.timeres+24*60/rainprop.timeres]=intime
                        precat[i,j*24*60/rainprop.timeres:j*24*60/rainprop.timeres+24*60/rainprop.timeres,:]=np.reshape(inrain,(24*60/rainprop.timeres,rainprop.subdimensions[0],rainprop.subdimensions[1]))
    else:
        precat=np.zeros((catrain.shape[0],0,catrain.shape[2],catrain.shape[3]),dtype='float32')

    if alllevels:
        def find_nearest(array,value):
            idx = (np.abs(array-value)).argmin()
            return idx
            
        minind=find_nearest(returnperiod,RainfallThreshYear)
        writemax=sortrain[minind:,:]
        writex=sortx[minind:,:]
        writey=sorty[minind:,:]
        writestorm=sortstorms[minind:,:]
        writeperiod=returnperiod[minind:]
        writeexceed=exceedp[minind:]
        writetimes=sorttimes[minind:,:]
        whichorigstorm=whichorigstorm[minind:,:]
        if rotation:
            writeangle=sortangle[minind:,:]
            binwriteang=np.digitize(writeangle.ravel(),angbins).reshape(writeangle.shape)
    else:
        writemax=sortrain
        writex=sortx
        writey=sorty
        writestorm=sortstorms
        writeperiod=returnperiod
        writeexceed=exceedp
        writetimes=sorttimes
        if rotation:
            writeangle=sortangle
            binwriteang=np.digitize(writeangle.ravel(),angbins).reshape(writeangle.shape)

    for rlz in range(0,nrealizations):
        print "writing spacetime rainfall scenarios for realization "+str(rlz+1)+"/"+str(nrealizations)
        
        # this statement is only really needed if you are prepending rainfall, but it is fast so who cares?
        outtime=np.empty((writetimes.shape[0],cattime.shape[1]),dtype='datetime64[m]')
        unqstm=np.unique(writestorm[:,rlz])
        for i in range(0,len(unqstm)):
            outtime[writestorm[:,rlz]==unqstm[i],:]=cattime[unqstm[i],:] 
        
        if rotation:
            outrain=RainyDay.SSTspin_write_v2(catrain,writex[:,rlz],writey[:,rlz],writestorm[:,rlz],nanmask,xmin,xmax,ymin,ymax,maskheight,maskwidth,precat,cattime[:,-1],rainprop,rlzanglebin=binwriteang[:,rlz],delarray=delarray,spin=prependrain,flexspin=False,samptype=resampletype,cumkernel=cumkernel,rotation=rotation)
        else:
            outrain=RainyDay.SSTspin_write_v2(catrain,writex[:,rlz],writey[:,rlz],writestorm[:,rlz],nanmask,xmin,xmax,ymin,ymax,maskheight,maskwidth,precat,cattime[:,-1],rainprop,spin=prependrain,flexspin=False,samptype=resampletype,cumkernel=cumkernel,rotation=rotation)

        #outrain[:,:,trimmask==0]=-9999.               # this line produced problems in CUENCAS CONVERSIONS :(
        writename=WriteName+'_SSTrlz'+str(rlz+1)+'.nc'
        subrangelat=latrange[ymin:ymax+1]
        subrangelon=lonrange[xmin:xmax+1]
        print "need to write angles to the realization files"
        RainyDay.writerealization(rlz,nrealizations,writename,outrain,writemax[:,rlz],writestorm[:,rlz],writeperiod,writex[:,rlz],writey[:,rlz],outtime,subrangelat,subrangelon,whichorigstorm[:,rlz])
    
    if deterministic:
        RainyDay.writemaximized(wd+'/'+scenarioname+'/'+scenarioname+'_maximizedstorm.nc',maxstm_rain,maxstm_avgrain,maxstm_ts,max_trnsx,max_trnsy,maxstm_time,subrangelat,subrangelon)
             
end = time.time()   
print "RainyDay has successfully finished!\n"
print "Elapsed time: "+str((end - start)/60.)+" minutes"
   
#################################################################################
# THE END
#################################################################################

