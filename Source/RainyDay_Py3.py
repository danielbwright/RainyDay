#!/usr/bin/env python

# give the right permissions (for a Mac or Linux): chmod +x {scriptname}


#==============================================================================
# WELCOME
#==============================================================================
#    Welcome to RainyDay, a framework for coupling remote sensing precipitation
#    fields with Stochastic Storm Transposition for assessment of rainfall-driven hazards.
#    Copyright (C) 2017  Daniel Benjamin Wright (danielb.wright@gmail.com)
#

#Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

#The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

#THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.#

#==============================================================================


#==============================================================================
# IMPORT STUFF!
#==============================================================================

import os
import sys
import numpy as np
import scipy as sp
import shapefile
import math 
from datetime import datetime, date, time, timedelta      
import time
from copy import deepcopy
from scipy import ndimage, stats
import pandas as pd
from mpl_toolkits.basemap import Basemap

from numba import jit
numbacheck=True

# plotting stuff, really only needed for diagnostic plots
import matplotlib
#matplotlib.use('Agg')
import matplotlib.pyplot as plt

# import RainyDay functions
import RainyDay_utilities_Py3.RainyDay_functions as RainyDay


import warnings
warnings.filterwarnings("ignore")


#==============================================================================
# DIFFERENT PROJECTIONS
#==============================================================================
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

print('''Welcome to RainyDay, a framework for coupling remote sensing precipitation
        fields with Stochastic Storm Transposition for assessment of rainfall-driven hazards.
        Copyright (C) 2017  Daniel Benjamin Wright (danielb.wright@gmail.com)
        Distributed under the MIT Open-Source License: https://opensource.org/licenses/MIT
    
    
    Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:
    
    The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.
    
    THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
''')


#==============================================================================
# USER DEFINED INFO
#==============================================================================
start = time.time()
parameterfile='ttt'
try:
    parameterfile=np.str(sys.argv[1])
except:
    sys.exit("You didn't specify an input ('.sst') file!")
#parameterfile='/Users/daniel/Google_Drive/RainyDay2/IrregularDomainTesting/defaulttesting.sst'
#parameterfile='/Users/daniel/Google_Drive/PapersandReports/MyPapers/FloodStructure/Analyses/StageIV_Turkey72hour.sst'

if os.path.isfile(parameterfile)==False:
    sys.exit("You either didn't specify a parameter file, or it doesn't exist.")
else:
    print("reading in the parameter file...")
    cardinfo=np.loadtxt(parameterfile, comments="#",dtype="str", unpack=False)


#==============================================================================
# USER DEFINED VARIABLES
#==============================================================================

try:
    wd=str(cardinfo[cardinfo[:,0]=="MAINPATH",1][0])
    try:
        scenarioname=str(cardinfo[cardinfo[:,0]=="SCENARIONAME",1][0])
    except Exception:
        print("You did't specify SCENARIONAME, defaulting to 'RainyDayAnalysis'!")
        scenarioname='RainyDayAnalysis'
    fullpath=wd+'/'+scenarioname
    if os.path.isdir(fullpath)==False:
        os.system('mkdir -p -v %s' %(fullpath))
    os.chdir(wd)
except ImportError:
    sys.exit("You did't specify MAINPATH, which is a required field!")


# PROPERTIES RELATED TO SST PROCEDURE:
try:
    catalogname=str(cardinfo[cardinfo[:,0]=="CATALOGNAME",1][0])
except Exception:
    catalogname='SSTstormcatalog.nc'
catalogname=wd+'/'+catalogname

try:
    CreateCatalog=cardinfo[cardinfo[:,0]=="CREATECATALOG",1][0]
except Exception:
    CreateCatalog='true'
    
if CreateCatalog.lower()=='true':
    CreateCatalog=True
else:
    CreateCatalog=False
    if os.path.isfile(catalogname)==False:
        sys.exit("You need to create a storm catalog first.")
 
#try:       
#    acceltype=cardinfo[cardinfo[:,0]=="ACCELERATOR",1][0]
#    if acceltype.lower()=='weave':
#        print("Weave selected!")
#        sys.exit("weave isn't supported anymore!")
#        weavecheck=True
#        numbacheck=False
#        #from scipy import weave
#        #from scipy.weave import converters 
#    elif acceltype.lower()=='numba':
#        print("Numba selected!")
#        numbacheck=True
#        weavecheck=False
#        #import cython
#        #import pyximport; pyximport.install()
#        #pyximport.install(setup_args={'include_dirs':[np.get_include()]})
#        #import CatalogCython
#        #from numba import jit
#    
#    else:
#        print("No acceleration selected. This is gonna be slow!")
#        weavecheck=False
#        numbacheck=False
#except Exception:
#    print("No acceleration selected! This is not recommended if Numba is available on your Python build")
#    weavecheck=False
#    numbacheck=False


try:
    nstorms=np.int(cardinfo[cardinfo[:,0]=="NSTORMS",1][0])
    defaultstorms=False
    print(str(nstorms)+" storms will be identified for storm catalog!")
except Exception:
    #nstorms=100
    print("you didn't specify NSTORMS, defaulting to 20 per year!")
    defaultstorms=True

try:
    nsimulations=np.int(cardinfo[cardinfo[:,0]=="NYEARS",1][0])
    print(str(nsimulations)+" years of synthetic storms will be generated, for a max recurrence interval of "+str(nsimulations)+" years!")
except Exception:
    nsimulations=100
    print("you didn't specify NYEARS, defaulting to 100, for a max recurrence interval of 100 years!")

try:
    nrealizations=np.int(cardinfo[cardinfo[:,0]=="NREALIZATIONS",1][0])
    print(str(nrealizations)+" realizations will be performed!")
except Exception:
    nrealizations=1


try:
    duration=np.float32(cardinfo[cardinfo[:,0]=="DURATION",1][0])
    if duration<=0.:
        sys.exit("Duration is zero or negative!")
except ImportError:
    sys.exit("You didn't specify 'DURATION', which is a required field!")


try:
    samplingtype=cardinfo[cardinfo[:,0]=="RESAMPLING",1][0]
    if samplingtype.lower()!='poisson' and samplingtype.lower()!='empirical':
        sys.exit("unrecognized storm count resampling type, options are: 'poisson' or 'empirical'")
except Exception:
    samplingtype='poisson'

if samplingtype=='poisson':
    print("Poisson-based temporal resampling scheme will be used!")
elif samplingtype=='empirical':
    print("Empirically-based temporal resampling scheme will be used!")

try:
    areatype=str(cardinfo[cardinfo[:,0]=="POINTAREA",1][0])
    if areatype.lower()=="basin":
        try:
            wsmaskshp=str(cardinfo[cardinfo[:,0]=="WATERSHEDSHP",1][0])
            print("You selected 'basin' for 'POINTAREA', please note that if the watershed is not in a regular lat/lon projection such as EPSG4326/WGS 84, the results will likely be incorrect!")
        except ImportError:
            sys.exit("You specified 'basin' for 'POINTAREA' but didn't specify 'WATERSHEDSHP'")            
    elif areatype.lower()=="point":
        try:
            ptlat=np.float32(cardinfo[cardinfo[:,0]=="POINTLAT",1][0]) 
            ptlon=np.float32(cardinfo[cardinfo[:,0]=="POINTLON",1][0]) 
            ctrlat=ptlat
            ctrlon=ptlon
        except ImportError:
            sys.exit("You specified 'point' for 'POINTAREA' but didn't properly specify 'POINTLAT' and 'POINTLON'")
    elif areatype.lower()=="box":
        try:
            box1=np.float32(cardinfo[cardinfo[:,0]=="BOX_YMIN",1][0])
            box2=np.float32(cardinfo[cardinfo[:,0]=="BOX_YMAX",1][0])
            box3=np.float32(cardinfo[cardinfo[:,0]=="BOX_XMIN",1][0])
            box4=np.float32(cardinfo[cardinfo[:,0]=="BOX_XMAX",1][0])
            boxarea=np.array([box3,box4,box1,box2])
            ctrlat=(box1+box2)/2.
            ctrlon=(box3+box4)/2.
        except ImportError:
            sys.exit("You specified 'box' for 'POINTAREA' but didn't properly specify 'BOX_YMIN', 'BOX_YMAX', 'BOX_XMIN', and 'BOX_XMAX'")
    else:
        sys.exit("unrecognized area type")
except ImportError:
    sys.exit("You didn't specify 'POINTAREA', which needs to either be set to 'basin', 'point', or 'box'")
    

# transposition
try:
    transpotype=cardinfo[cardinfo[:,0]=="TRANSPOSITION",1][0]
    if transpotype.lower()=='kernel' or transpotype.lower()=='nonuniform':
        transpotype='kernel'
        print("You selected the kernel density-based non-uniform storm transposition scheme!")
    elif transpotype.lower()=='user':
        transpotype='user'
        sys.exit("sadly we aren't set up for the user-supplied transposition scheme yet")
    elif transpotype.lower()=='uniform':
        transpotype='uniform'
        print("You selected the spatially uniform storm transposition scheme!")
    else:
        sys.exit("You specified an unknown resampling method")
except Exception:
    transpotype='uniform'
    print("You didn't specify 'TRANSPOSITION', defaulting to spatially uniform scheme!")
 
# Use stochastic ratio multiplier?
try:
    rescalingtype=cardinfo[cardinfo[:,0]=="STOCHASTICRESCALING",1][0]
    if rescalingtype.lower()=='true':
        stochrescale=True
        print("You selected the 'Stochastic Ratio Rescaling'!")
        try:
            rescalingfile=cardinfo[cardinfo[:,0]=="RAINDISTRIBUTIONFILE",1][0]
            if os.path.isfile(rescalingfile)==False:
                sys.exit("The rainfall file specified in 'RAINDISTRIBUTIONFILE' cannot be found!")
            #else:
                #intenserain,intensetime,intenselat,intenselon=RainyDay.readintensityfile(rescalingfile)
        except ImportError:
            sys.exit("Even though you 'Stochastic Ratio Rescaling', you didn't specify the file of rainfall distributions!")   
    else:
        print("No rescaling will be performed!")
        stochrescale=False
except Exception:
    print("No rescaling will be performed!")
    stochrescale=False
    
    
    
try:   
    do_deterministic=cardinfo[cardinfo[:,0]=="DETERMINISTIC",1][0]
    if do_deterministic.lower():
        deterministic=True
    else:
        deterministic=False
except Exception:
    deterministic=False

try:
    domain_type=cardinfo[cardinfo[:,0]=="DOMAINTYPE",1][0] 
except Exception:
    domain_type='rectangular'
    
if domain_type.lower()=='rectangular':
    shpdom=False
    ncfdom=False
   
if domain_type.lower()=='irregular':
    print("Irregular domain type selected!")
    domain_type='irregular'
    ncfdom=False
    shpdom=False
    try:
        domainshp=str(cardinfo[cardinfo[:,0]=="DOMAINSHP",1][0])
        if os.path.isfile(domainshp)==False:
            sys.exit("can't find the transposition domain shapefile!")
        else:
            print("You selected 'irregular' for 'DOMAINTYPE', please note that if the domain shapefile is not in a regular lat/lon projection such as EPSG4326/WGS 84, the results will likely be incorrect!")
            shpdom=True
            ds = shapefile.Reader(domainshp,'rb')
            tempbox= ds.bbox
            inarea=np.array([tempbox[0],tempbox[2],tempbox[1],tempbox[3]],dtype='float32')
            
    except Exception:
        pass
    try:
        domainncf=str(cardinfo[cardinfo[:,0]=="DOMAINFILE",1][0])
        if os.path.isfile(domainncf)==False:
            sys.exit("Can't find the transposition domain NetCDF file!")  
        else:
            print("Domain NetCDF file found!")
            ncfdom=True  
            domainmask,domainlat,domainlon=RainyDay.readdomainfile(domainncf)
            res=np.abs(np.mean(domainlat[1:]-domainlat[0:-1]))
            xmin=np.min(np.where(np.sum(domainmask,axis=0)!=0))
            xmax=np.max(np.where(np.sum(domainmask,axis=0)!=0))
            ymin=np.min(np.where(np.sum(domainmask,axis=1)!=0))
            ymax=np.max(np.where(np.sum(domainmask,axis=1)!=0))
            domainmask=domainmask[ymin:(ymax+1),xmin:(xmax+1)]
            inarea=np.array([domainlon[xmin],domainlon[xmax]+res,domainlat[ymax]-res,domainlat[ymin]])
            domainlat=domainlat[ymin:(ymax+1)]
            domainlon=domainlon[xmin:(xmax+1)]
    except Exception:
        pass
    
    if ncfdom==False and shpdom==False:
        sys.exit("You selected 'irregular' for 'DOMAINTYPE', but didn't provide a shapefile or NetCDF file for the domain!")
else:  
    print("rectangular domain type selected!")
    domain_type='rectangular'   
    try:
        lim1=np.float32(cardinfo[cardinfo[:,0]=="LATITUDE_MIN",1][0])
        lim2=np.float32(cardinfo[cardinfo[:,0]=="LATITUDE_MAX",1][0])
        lim3=np.float32(cardinfo[cardinfo[:,0]=="LONGITUDE_MIN",1][0]) 
        lim4=np.float32(cardinfo[cardinfo[:,0]=="LONGITUDE_MAX",1][0]) 
        inarea=np.array([lim3,lim4,lim1,lim2])
    except ImportError:
        sys.exit("need to specify 'LATITUDE_MIN', 'LATITUDE_MAX', 'LONGITUDE_MIN', 'LONGITUDE_MAX'")

if areatype.lower()=="basin" or shpdom:
    try:
        gdalpath=str(cardinfo[cardinfo[:,0]=="GDALPATH",1][0])
    except IndexError:
        print("You wanted to use a shapefile for either the transposition domain or the watershed, but didn't specify GDALPATH. We'll try, but it might not work!")    
        gdalpath=False


try:  
    DoDiagnostics=cardinfo[cardinfo[:,0]=="DIAGNOSTICPLOTS",1][0]
    if DoDiagnostics.lower()=='true':
        DoDiagnostics=True
    else:
        DoDiagnostics=False
except Exception:
    DoDiagnostics=True  

diagpath=fullpath+'/Diagnostics/'
if DoDiagnostics==True:     
    if os.path.isdir(wd+scenarioname+'/Diagnostics')==False:
        os.system('mkdir %s' %(diagpath))

#if DoDiagnostics and shpdom:
#                import geopandas as gpd
#                dshp = gpd.read_file(domainshp)
#                #print stateshp1.crs
#                dshp.crs={}
#                dshp.to_file(domainshp, driver='ESRI Shapefile')
    
try:
    FreqAnalysis=cardinfo[cardinfo[:,0]=="FREQANALYSIS",1][0]
    if FreqAnalysis.lower()=='true':
        FreqAnalysis=True
    else:
        FreqAnalysis=False
except Exception:
    FreqAnalysis=True
    
    
FreqFile=fullpath+'/'+scenarioname+'.FreqAnalysis'

try:
    Scenarios=cardinfo[cardinfo[:,0]=="SCENARIOS",1][0]
    if Scenarios.lower()=='true':
        Scenarios=True
        FreqAnalysis=True
        WriteName=wd+'/'+scenarioname+'/Realizations'
        os.system('mkdir %s' %(WriteName))
        WriteName=WriteName+'/'+scenarioname
    else:
        Scenarios=False
except Exception:
    Scenarios=False
    print("You didn't specify 'SCENARIOS', defaulting to 'false', no scenarios will be written!")


# EXLCUDE CERTAIN MONTHS
try:
    exclude=cardinfo[cardinfo[:,0]=="EXCLUDEMONTHS",1][0]
    if exclude.lower()!="none":
        exclude=exclude.split(',')
        excludemonths=np.empty(len(exclude),dtype="int32")
        for i in range(0,len(exclude)):
            excludemonths[i]=np.int(exclude[i])
    else:
        excludemonths=False
except Exception:
    excludemonths=False
    
# INCLUDE ONLY CERTAIN YEARS
try:
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
except Exception:
    includeyears=False

# MINIMUM RETURN PERIOD THRESHOLD FOR WRITING RAINFALL SCENARIO (THIS WILL HAVE A HUGE IMPACT ON THE SIZE OF THE OUTPUT FILES!)  IN PRINCIPLE IT IS PERHAPS BETTER TO USE AN INTENSITY BUT THIS IS CLEANER!
try:
    RainfallThreshYear=np.int(cardinfo[cardinfo[:,0]=="RETURNTHRESHOLD",1][0])
except Exception:
    RainfallThreshYear=1

# DIRECTORY INFO-WHERE DOES THE INPUT DATASET RESIDE?
try:
    inpath=str(cardinfo[cardinfo[:,0]=="RAINPATH",1][0])
except ImportError:
    sys.exit("You didn't specify 'RAINPATH', which is a required field!")

      
try:   
    BaseMap=cardinfo[cardinfo[:,0]=="BASEMAP",1][0]
    if BaseMap.lower()!='none':
        BaseMap=BaseMap.split('.')[0]
        BaseField=cardinfo[cardinfo[:,0]=="BASEFIELD",1][0]
except Exception:
    BaseMap='none'


# SENSITIVITY ANALYSIS OPTIONS:
try:
    IntensitySens=cardinfo[cardinfo[:,0]=="SENS_INTENSITY",1][0]
    if IntensitySens.lower()!='false':
        IntensitySens=1.+np.float(IntensitySens)/100.
    else:
        IntensitySens=1.
except Exception:
    IntensitySens=1.

try:
    FrequencySens=cardinfo[cardinfo[:,0]=="SENS_FREQUENCY",1][0]     
    if FrequencySens.lower()!='false':
        if samplingtype.lower()!='poisson':
            sys.exit("Sorry, you can't modify the resampling frequency unless you use the Poisson resampler.")
        else:
            FrequencySens=1.0+np.float(FrequencySens)/100.
    else:
        FrequencySens=1.
except Exception:
    FrequencySens=1.    
    pass

if areatype=='point':
    try:
        ARFval=cardinfo[cardinfo[:,0]=="ARFVAL",1][0]     
        if ARFval.lower()!='false':
            arfval=np.float(ARFval)
            
        else:
            FrequencySens=1.
    except Exception:
        arfval=1.    
        pass
else:
    arfval=1.

try:    
    userdistr=cardinfo[cardinfo[:,0]=="INTENSDISTR",1][0]
    if userdistr.lower()=='false':
        userdistr=np.zeros((1),dtype='bool')
    elif len(userdistr.split(','))==3:
        print("reading user-defined rainfall intensity distribution...")
        userdistr=np.array(userdistr.split(','),dtype='float32')
    else:
        sys.exit("There is a problem with the INTENSDISTR entry!")
except Exception:
    userdistr=np.zeros((1),dtype='bool')
    pass
    
 
if Scenarios:
    try:
        if np.any(np.core.defchararray.find(cardinfo[:,0],"SPINPERIOD")>-1):
            pretime=cardinfo[cardinfo[:,0]=="SPINPERIOD",1][0]  
             
            if pretime.lower()=='none' or pretime.lower()=='false':
                prependrain=False
                print('spinup rainfall will be not included in rainfall scenarios...')
            else:
                try:
                    pretime=np.int(pretime)
                    prependrain=True
                    if pretime==0.:
                        prependrain=False
                        print('the spinup time you provided was 0, no spinup rainfall will be included...')
                    elif pretime<0.:
                        print("the spin-up time value is negative.")
                        sys.exit(0)
                    elif pretime<1.:
                        print('the spinup time you provided was short, rounding up to 1 day...')
                        pretime=1.0
                    else:
                        print( np.str(pretime)+' days of spinup rainfall will be included in rainfall scenarios...')
                except:
                    sys.exit('unrecognized value provided for SPINPERIOD.')
               
        else:
            prependrain=False
            print('no SPINPERIOD field was provided.  Spin-up rainfall will be not be included...')
    except Exception:
        print('no SPINPERIOD field was provided.  Spin-up rainfall will be not be included...')

try:
    if np.any(np.core.defchararray.find(cardinfo[:,0],"UNCERTAINTY")>-1):       
        spread=cardinfo[cardinfo[:,0]=="UNCERTAINTY",1][0]
        if spread.lower()=='ensemble':
            spreadtype='ensemble'
            print('"ensemble spread" will be calculated for rainfall frequency analysis...')
        else:
            try:
                int(spread)
            except:
                print('unrecognized value for UNCERTAINTY.')
                
            if int(spread)>=0 and int(spread)<=100:
                spreadtype='quantile'
                quantilecalc=int(spread)
                print(spread+'th quantiles will be calculated for rainfall frequency analysis...')
            else: 
                sys.exit('invalid quantile range for frequency analysis...')
    else:
        spreadtype='ensemble'
except Exception:
    spreadtype='ensemble'
                
try:    
    if np.any(np.core.defchararray.find(cardinfo[:,0],"RETURNLEVELS")>-1):
        speclevels=cardinfo[cardinfo[:,0]=="RETURNLEVELS",1][0]
        if speclevels.lower()=='all':
            print('using all return levels...')
            alllevels=True
        else:
            alllevels=False
            if ',' in speclevels:
                speclevels=speclevels.split(',')
                try:
                    speclevels=np.float32(speclevels,dtype="float32")
                except:
                    print("Non-numeric value provided to RETURNLEVELS.")
                    
                speclevels=speclevels[speclevels<=nsimulations+0.00001]
                speclevels=speclevels[speclevels>=0.99999]
            else:
                sys.exit("The format of RETURNLEVELS isn't recognized.  It should be 'all' or a comma separated list.")
    else:
        alllevels=True
except Exception:
    alllevels=True    
      
     
try:
    rotation=False
    if np.any(np.core.defchararray.find(cardinfo[:,0],"ROTATIONANGLE")>-1):
        try:
            rotangle=cardinfo[cardinfo[:,0]=="ROTATIONANGLE",1][0]
        except ImportError:
            "You're trying to use storm rotation, but didn't specify 'ROTATIONANGLE'"
        if rotangle.lower()=='none' or rotangle.lower()=='false' or areatype.lower()=="point":
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
                    print('Unrecognized entry provided for ROTATIONANGLE minimum angle.')
                try:
                    maxangle=np.float32(maxangle)
                except:
                    print('Unrecognized entry provided for ROTATIONANGLE maximum angle.')
                try:
                    nanglebins=np.int32(nanglebins,dtype="float32")
                except:
                    print('Unrecognized entry provided for ROTATIONANGLE number of bins.')
                    
                if minangle>0. or maxangle<0.:
                    sys.exit('The minimum angle should be negative and the maximum angle should be positive.')
except Exception:
    rotation=False   
    
if rotation:
    print("storm rotation will be used...")
    delarray=[]

spversion=sp.__version__.split('.')
if int(spversion[0])<1:
    if int(spversion[1])<9:
        sys.exit('Your version of Scipy is too old to handle the rotation scheme.  Either do not use rotation or update Scipy.')
  
# should add a "cell-centering" option!

# read in the start and end hour stuff-for doing IDFs for specific times of day-IS THIS ACTUALLY FUNCTIONAL???:
try:
    starthour=np.float32(cardinfo[cardinfo[:,0]=="STARTHOUR",1][0])
    subday=True
except:
    starthour=0.

try: 
    endhour=np.float32(cardinfo[cardinfo[:,0]=="ENDHOUR",1][0])
    subday=True
except:
    endhour=24.
    subday=False
    
if starthour!=0. and endhour!=24.:
    if (endhour-starthour)<duration:
        duration=endhour-starthour 

try:    
    if np.any(np.core.defchararray.find(cardinfo[:,0],"DURATIONCORRECTION")>-1):
        durcorr=cardinfo[cardinfo[:,0]=="DURATIONCORRECTION",1][0]
        if durcorr.lower()=='false':
            durcorrection=False
            print('DURATIONCORRECTION set to "false". No correction will be used.')
        elif durcorr.lower()=='true':
            durcorrection=True
        else:
            print('Invalid option provided for DURATIONCORRECTION (should be "true" or "false"). Defaulting to "false"!')
            durcorrection=False
    else:
        print('No value was given for DURATIONCORRECTION. If applicable, it will default to "false"!')
        durcorrection=False
except Exception:
    durcorrection=False

if CreateCatalog and durcorrection:
    catduration=max([48.,2.*duration])    # this will be slow!
    print('Since DURCORRECTION will be used and a storm catalog will be created, the duration of the catalog will be '+"{0:0.2f}".format(catduration)+' hours')
elif CreateCatalog:
    catduration=duration
    
try:
    timeseparation=np.float32(cardinfo[cardinfo[:,0]=="TIMESEPARATION",1][0])     
except Exception:
    timeseparation=0. 
    

      

#==============================================================================
# THIS BLOCK CONFIGURES SEVERAL THINGS
#============================================================================== 
    
initseed=0
np.random.seed(initseed)
global rainprop   
rainprop=deepcopy(emptyprop) 
    

#==============================================================================
# CREATE NEW STORM CATALOG, IF DESIRED
#==============================================================================  
if CreateCatalog:
    print("creating a new storm catalog...")
    
    flist,nyears=RainyDay.createfilelist(inpath,includeyears,excludemonths)
    if defaultstorms:
        nstorms=nyears*20
  
    
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
    
    if ncfdom and (np.any(domainmask.shape!=rainprop.subdimensions)):
        sys.exit("Something went terribly wrong :(")        # this shouldn't happen



#==============================================================================
# IF A STORM CATALOG ALREADY EXISTS, USE IT
#==============================================================================  
    
else:
    print("reading an existing storm catalog...")
    if os.path.isfile(catalogname)==False:
        sys.exit("can't find the storm catalog")
    else:
        catrain,cattime,latrange,lonrange,catx,caty,catmax,_,_=RainyDay.readcatalog(catalogname)
        rainprop.spatialres,rainprop.dimensions,rainprop.bndbox,rainprop.timeres,rainprop.nodata,catrain,cattime,latrange,lonrange,catx,caty,catmax,domainmask=RainyDay.rainprop_setup(catalogname,catalog=True)
        delt=np.timedelta64(cattime[0,-1]-(cattime[0,0]))
        catduration=(delt.astype('int')+rainprop.timeres)/60.
        if defaultstorms==False:
            if nstorms>cattime.shape[0]:
                print("WARNING: The storm catalog has fewer storms than the specified nstorms")
                nstorms=cattime.shape[0] 
        else:
            darr=np.array(pd.to_datetime(cattime.ravel()).year)
            nstorms_default=(np.max(darr)-np.min(darr)+1)*25
            if nstorms_default>cattime.shape[0]:
                print("WARNING: The storm catalog has fewer storms than the default number of storms")
                nstorms=cattime.shape[0] 
            else:
                nstorms=nstorms_default
            
            
        if ncfdom and (np.any(domainmask.shape!=catrain.shape[2:])):
            sys.exit("The domain mask and the storm catalog are different sizes. Do you know what you're doing?")

    rainprop.bndbox=np.array([lonrange[0],lonrange[-1]+rainprop.spatialres[0],latrange[-1]-rainprop.spatialres[1],latrange[0]],dtype='float32')
    rainprop.subextent=np.array([lonrange[0],lonrange[-1]+rainprop.spatialres[0],latrange[-1]-rainprop.spatialres[1],latrange[0]],dtype='float32')
        
    rainprop.subind=np.array([0,len(lonrange)-1,len(latrange)-1,0],dtype='int32')
    rainprop.dimensions=np.array([len(latrange),len(lonrange)],dtype='int32')  
    rainprop.subdimensions=rainprop.dimensions

    ingridx,ingridy=np.meshgrid(np.arange(rainprop.subextent[0],rainprop.subextent[1]-rainprop.spatialres[0]/1000,rainprop.spatialres[0]),np.arange(rainprop.subextent[3],rainprop.subextent[2]+rainprop.spatialres[1]/1000,-rainprop.spatialres[1]))        


if int(duration*60/rainprop.timeres)<=0:
    sys.exit("it appears that you specified a duration shorter than the temporal resolution of the input data!")


if timeseparation<=0.:
    timeseparation=duration
elif timeseparation>0. and durcorrection==False:
    timeseparation=timeseparation+duration
elif durcorrection:
    timeseparation=timeseparation+catduration 


#============================================================================
# Do the setup to run for specific times of day!
#=============================================================================

if CreateCatalog==False:
    flist,nyears=RainyDay.createfilelist(inpath,includeyears,excludemonths)

if starthour==0 and endhour==24:
    hourinclude=np.ones((int(24*60/rainprop.timeres)),dtype='int32')
else:
    try:
        _,temptime,_,_=RainyDay.readnetcdf(flist[0],inbounds=rainprop.subind)
    except Exception:
        sys.exit("Can't find the input files necessary for setup to calculate time-of-day-specific IDF curves")

    starthour=starthour+np.float(rainprop.timeres/60)   # because the netcdf file timestamps correspond to the end of the accumulation period
    hourinclude=np.zeros((24*60/rainprop.timeres),dtype='int32')
    temphour=np.zeros((24*60/rainprop.timeres),dtype='float32')

    for i in range(0,len(temptime)):
        temphour[i]=temptime[i].astype(object).hour
    
    # the following line will need to be adapted, due to UTC vs. local issues
    hourinclude[np.logical_and(np.greater_equal(temphour,starthour),np.less_equal(temphour,endhour))]=1
    if len(hourinclude)!=len(temptime):
        sys.exit("Something is wrong in the hour exclusion calculation!")

hourinclude=hourinclude.astype('bool')

#temptime[hourinclude] # checked, seems to be working right
 
        
#==============================================================================
# SET UP GRID MASK
#==============================================================================

print("setting up the grid information and masks...")
if areatype.lower()=="basin":
    if os.path.isfile(wsmaskshp)==False:
        sys.exit("can't find the basin shapefile!")
    else:
        catmask=RainyDay.rastermaskGDAL(wsmaskshp,GEOG,rainprop,'fraction',fullpath,gdalpath=gdalpath)
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
        print('WARNING: you set POINTAREA to "box", but the box is smaller than a single pixel.  This is not advised.  Either set POINTAREA to "point" or increase the size of the box.'   )
    
    if len(finelat[subindy])==1 and len(finelon[subindx])==1:
        catmask=np.zeros((rainprop.subdimensions))
        yind=np.where((latrange-ptlat)>0)[0][-1]
        xind=np.where((ptlon-lonrange)>0)[0][-1]  
        if xind==0 or yind==0:
            sys.exit('the point you defined is too close to the edge of the box you defined!')
        else:
            catmask[yind,xind]=1.0
    else:
        def block_mean(ar, fact):
            assert isinstance(fact, int), type(fact)
            sx, sy = ar.shape
            X, Y = np.ogrid[0:sx, 0:sy]
            regions = sy//fact * (X//fact) + Y//fact
            res = ndimage.mean(ar, labels=regions, index=np.arange(regions.max() + 1))
            res.shape = (sx//fact, sy//fact)
            return res

        catmask=block_mean(catmask,25)          # this scheme is a bit of a numerical approximation but I doubt it makes much practical difference  
else:
    sys.exit("unrecognized area type!")
    
 
if domain_type.lower()=='irregular' and shpdom:
    domainmask=RainyDay.rastermaskGDAL(domainshp,GEOG,rainprop,'simple',fullpath,gdalpath=gdalpath).astype('float32')
    #domainmask=catmask.reshape(ingridx.shape,order='F')
elif domain_type.lower()=='irregular' and ncfdom:
    pass
else:
    domainmask=np.ones((catmask.shape),dtype='float32')    


       
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

timeseparation=np.timedelta64(np.int(timeseparation*60.),'m')
timestep=np.timedelta64(rainprop.timeres,'m')

mnorm=np.sum(trimmask)

xlen=rainprop.subdimensions[1]-maskwidth+1
ylen=rainprop.subdimensions[0]-maskheight+1


if ncfdom:
    ingridx_dom,ingridy_dom=np.meshgrid(domainlon,domainlat)        

    ingrid_domain=np.column_stack((ingridx_dom.flatten(),ingridy_dom.flatten())) 
    #ingridx,ingridy=np.meshgrid(np.arange(rainprop.subextent[0],rainprop.subextent[1]-rainprop.spatialres[0]/1000,rainprop.spatialres[0]),np.arange(rainprop.subextent[3],rainprop.subextent[2]+rainprop.spatialres[1]/1000,-rainprop.spatialres[1]))        
    grid_rainfall=np.column_stack((ingridx.flatten(),ingridy.flatten())) 
    
    delaunay=sp.spatial.qhull.Delaunay(ingrid_domain)
    interp=sp.interpolate.NearestNDInterpolator(delaunay,domainmask.flatten())

    # in case you want it back in a rectangular grid:
    domainmask=np.reshape(interp(grid_rainfall),ingridx.shape)    


#################################################################################
# STEP 1: CREATE STORM CATALOG
#################################################################################

if CreateCatalog: 
    print("reading rainfall files...")
    
    #==============================================================================
    # SET UP OUTPUT VARIABLE
    #==============================================================================

    rainarray=np.zeros((int(catduration*60/rainprop.timeres),rainprop.subdimensions[0],rainprop.subdimensions[1]),dtype='float32')  
    
    rainsum=np.zeros((ylen,xlen),dtype='float32')
    rainarray[:]=np.nan
    raintime=np.empty((int(catduration*60/rainprop.timeres)),dtype='datetime64[m]')
    raintime[:]=np.datetime64(datetime(1900,1,1,0,0,0))
    
    catmax=np.zeros((nstorms),dtype='float32')

    catrain=np.zeros((nstorms,int(catduration*60/rainprop.timeres),rainprop.subdimensions[0],rainprop.subdimensions[1]),dtype='float32')
    cattime=np.empty((nstorms,int(catduration*60/rainprop.timeres)),dtype='datetime64[m]')
    cattime[:]=np.datetime64(datetime(1900,1,1,0,0,0))
    catloc=np.empty((nstorms),dtype='float32')
    
    catx=np.zeros((nstorms),dtype='int32')
    caty=np.zeros((nstorms),dtype='int32')
    

    #==============================================================================
    # READ IN RAINFALL
    #==============================================================================
    filerange=range(0,len(flist))
    #filerange=range(0,18)
    start = time.time()
    for i in filerange: 
        infile=flist[i]
        inrain,intime,_,_=RainyDay.readnetcdf(infile,inbounds=rainprop.subind)
        inrain=inrain[hourinclude,:]
        intime=intime[hourinclude]
        
        inrain[inrain<0.]=np.nan
        
        print('Processing file '+str(i+1)+' out of '+str(len(flist))+' ('+"{0:0.0f}".format(100*(i+1)/len(flist))+'%): '+infile.split('/')[-1])

        # THIS FIRST PART BUILDS THE STORM CATALOG
        for k in np.arange(0,len(intime)):     
            starttime=intime[k]-np.timedelta64(int(catduration*60.),'m')
            raintime[-1]=intime[k]
            rainarray[-1,:]=inrain[k,:]
            #rainarray[-1,:]=np.reshape(inrain[k,:],(rainprop.subdimensions[0],rainprop.subdimensions[1]))
            subtimeind=np.where(np.logical_and(raintime>starttime,raintime<=raintime[-1]))
            subtime=np.arange(raintime[-1],starttime,-timestep)[::-1]
            temparray=np.squeeze(np.nansum(rainarray[subtimeind,:],axis=1))
            
            if numbacheck:
                if domain_type=='irregular':
                    rainmax,ycat,xcat=RainyDay.catalogNumba_irregular(temparray,trimmask,xlen,ylen,maskheight,maskwidth,rainsum,domainmask)
                else:
                    rainmax,ycat,xcat=RainyDay.catalogNumba(temparray,trimmask,xlen,ylen,maskheight,maskwidth,rainsum)
            else:
                if domain_type=='irregular':
                    rainmax,ycat,xcat=RainyDay.catalogAlt_irregular(temparray,trimmask,xlen,ylen,maskheight,maskwidth,rainsum,domainmask)
                else:
                    rainmax,ycat,xcat=RainyDay.catalogAlt(temparray,trimmask,xlen,ylen,maskheight,maskwidth,rainsum,domainmask)

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
            
            rainarray[0:-1,:]=rainarray[1:int(catduration*60/rainprop.timeres),:]
            raintime[0:-1]=raintime[1:int(catduration*60/rainprop.timeres)]

    sind=np.argsort(catmax)
    cattime=cattime[sind,:]
    catx=catx[sind]
    caty=caty[sind]    
    catrain=catrain[sind,:]  
    catmax=catmax[sind]/mnorm*rainprop.timeres/60. 
        
    end = time.time()   
    print("catalog timer: "+"{0:0.2f}".format((end - start)/60.)+" minutes")
    
    # WRITE CATALOG
    print("writing storm catalog...")
    RainyDay.writecatalog(catrain,catmax,catx,caty,cattime,latrange,lonrange,catalogname,nstorms,catmask,parameterfile,domainmask)   
        

#################################################################################
# STEP 2: RESAMPLING
#################################################################################
    
print("trimming storm catalog...")


#origstormsno=np.arange(0,nstorms)
origstormsno=np.arange(0,len(catmax),dtype='int32')

if CreateCatalog==False:
    if duration>catrain.shape[1]*rainprop.timeres/60.:
        sys.exit("The specified duration is longer than the length of the storm catalog")

                
# EXLCUDE "BAD" STORMS OR FOR DOING SENSITIVITY ANALYSIS TO PARTICULAR STORMS.  THIS IS PARTICULARLY NEEDED FOR ANY RADAR RAINFALL PRODUCT WITH SERIOUS ARTIFACTS
includestorms=np.ones((len(catmax)),dtype="bool")

try:
    exclude=cardinfo[cardinfo[:,0]=="EXCLUDESTORMS",1][0] 
except Exception:
    exclude='none'
    
if exclude.lower()!="none":
    exclude=exclude.split(',')
    for i in range(0,len(exclude)):
        includestorms[np.int(exclude[i])-1]=False
        
includestorms[np.isclose(catmax,0.)]=False
        
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
    nstorms=np.shape(catx)[0]
    modstormsno=modstormsno[-nstorms:]
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
    
    
    if defaultstorms:
        if nstorms_default<modstormsno.shape[0]:
            modstormsno=modstormsno[-nstorms_default:]
    else:
        modstormsno=modstormsno[catinclude]
    
catmax=catmax*IntensitySens
catrain=catrain*IntensitySens


#==============================================================================
# If the storm catalog has a different duration than the specified duration, fix it!
# find the max rainfall for the N-hour duration, not the M-day duration
#==============================================================================
#sys.exit()

if CreateCatalog==False or (CreateCatalog and durcorrection):   
    print("checking storm catalog duration, and adjusting if needed...")
    
    # if you are using a catalog that is longer in duration than your desired analysis, this happens:
    if 60./rainprop.timeres*duration!=np.float(catrain.shape[1]):
        print("Storm catalog duration is longer than the specified duration, finding max rainfall periods for specified duration...")
    
        # the first run through trims the storm catalog duration to the desired duration, for diagnostic purposes if needed...       
        dur_maxind=np.array((nstorms),dtype='int32')
        dur_x=0
        dur_y=0
        dur_j=0
        temprain=np.zeros((nstorms,int(duration*60/rainprop.timeres),rainprop.subdimensions[0],rainprop.subdimensions[1]),dtype='float32')
        rainsum=np.zeros((rainprop.subdimensions[0]-maskheight+1,rainprop.subdimensions[1]-maskwidth+1),dtype='float32')
        if durcorrection:
            catmax_subdur=np.zeros_like(catmax)
            catx_subdur=np.zeros_like(catx)
            caty_subdur=np.zeros_like(caty)
            #catrain_subdur=catrain
            #cattime_subdur=cattime
    
        temptime=np.empty((nstorms,int(duration*60/rainprop.timeres)),dtype='datetime64[m]')
        for i in range(0,nstorms):
            if (100*((i+1)%(nstorms/10)))==0:
                print('adjusting duration of storms, '+str(100*(i+1)/nstorms)+'% complete...')
            dur_max=0.
            for j in range(0,catrain.shape[1]-int(duration*60/rainprop.timeres)):
                maxpass=np.nansum(catrain[i,j:j+int(duration*60./rainprop.timeres),:],axis=0)
                if numbacheck:
                    if domain_type.lower()=='irregular':
                        maxtemp,tempy,tempx=RainyDay.catalogNumba_irregular(maxpass,trimmask,xlen,ylen,maskheight,maskwidth,rainsum,domainmask)   
                    else:
                        maxtemp,tempy,tempx=RainyDay.catalogNumba(maxpass,trimmask,xlen,ylen,maskheight,maskwidth,rainsum)                       
                else:
                    if domain_type.lower()=='irregular':
                        maxtemp,tempy,tempx=RainyDay.catalogAlt_irregular(maxpass,trimmask,xlen,ylen,maskheight,maskwidth,rainsum,domainmask)    
                    else:
                        maxtemp,tempy,tempx=RainyDay.catalogAlt(maxpass,trimmask,xlen,ylen,maskheight,maskwidth,rainsum)    
                if maxtemp>dur_max:
                    dur_max=maxtemp
                    dur_x=tempx
                    dur_y=tempy
                    dur_j=j

            catmax_subdur[i]=dur_max
            catx_subdur[i]=dur_x
            caty_subdur[i]=dur_y  

            temprain[i,:]=catrain[i,dur_j:dur_j+int(duration*60./rainprop.timeres),:]
            temptime[i,:]=cattime[i,dur_j:dur_j+int(duration*60./rainprop.timeres)]
        
        catrain_subdur=temprain
        cattime_subdur=temptime
    
        sind=np.argsort(catmax_subdur)
        cattime_subdur=cattime_subdur[sind,:]
        catx_subdur=catx_subdur[sind]
        caty_subdur=caty_subdur[sind]
        catrain_subdur=catrain_subdur[sind,:]
        catmax_subdur=catmax_subdur[sind]/mnorm*rainprop.timeres/60.
        if durcorrection==False:
            cattime=cattime_subdur
            catx=catx_subdur
            caty=caty_subdur
            catrain=catrain_subdur
            catmax=catmax_subdur


#==============================================================================
# IF THE USER IS SUPPLYING A DISTRIBUTION FOR THE INTENSITY, NORMALIZE THE FIELDS
# SO THAT THE INTENSITY CAN BE APPLIED PROPERLY
#==============================================================================
if userdistr.all()!=False:   
    print("normalizing rainfall fields...")
    for i in range(0,nstorms):
        tempmax=np.nanmax(np.nansum(catrain[i,:],axis=0))
        catrain[i,:]=catrain[i,:]/tempmax

    
#==============================================================================
# Create kernel density smoother, even if you don't use it for resampling
#==============================================================================
print("calculating kernel density smoother...")

kx,ky=np.meshgrid(np.arange(0,rainprop.subdimensions[1]-maskwidth+1),np.arange(0,rainprop.subdimensions[0]-maskheight+1))

kpositions=np.vstack([ky.ravel(),kx.ravel()])

if domain_type!='irregular':
    # this "checkind" line seems to cause some unrealistic issues with irregular domains, which makes sense
    checkind=np.where(np.logical_and(np.logical_and(caty!=0,catx!=0),np.logical_and(caty!=ylen-1,catx!=xlen-1)))
    invalues=np.vstack([caty[checkind], catx[checkind]])
else:
    invalues=np.vstack([caty, catx])
    
def my_kde_bandwidth(obj, fac=1):     # this 1.5 choice is completely subjective :(
    """We use Scott's Rule, multiplied by a constant factor."""
    return np.power(obj.n, -1./(obj.d+4)) * fac

stmkernel=stats.gaussian_kde(invalues,bw_method=my_kde_bandwidth)
pltkernel=np.multiply(np.reshape(stmkernel(kpositions), kx.shape),domainmask[0:rainprop.subdimensions[0]-maskheight+1,0:rainprop.subdimensions[1]-maskwidth+1])
pltkernel=pltkernel/np.nansum(pltkernel)

tempmask=deepcopy(domainmask[0:rainprop.subdimensions[0]-maskheight+1,0:rainprop.subdimensions[1]-maskwidth+1])

if transpotype=='uniform':
    tempsum=np.nansum(domainmask[0:rainprop.subdimensions[0]-maskheight+1,0:rainprop.subdimensions[1]-maskwidth+1])
    temparr=np.arange(1.,tempsum+1.)/tempsum
    tempmask[np.equal(tempmask,0.)]=np.nan
    tempmask[~np.isnan(tempmask)]=temparr
    tempmask[np.isnan(tempmask)]=100.
    cumkernel=tempmask
elif transpotype=='kernel' or stochrescale:
    # the following is an important fix. All prior kernel-based results are conceptually incorrect-DBW 1/24/2018. But this hasn't been thoroughly vetted!
    transpokernel=np.empty(pltkernel.shape,dtype='float64')
    phome=pltkernel[ymin,xmin]          # I hope this is right... I think so.
    if np.isclose(phome,0.):
        sys.exit("There is a zero probability of transposition to your location of interest. RainyDay's kernel-based non-uniform transposition scheme doesn't know how to cope with this...")
    transpokernel[np.less(pltkernel,phome)]=pltkernel[np.less(pltkernel,phome)]/phome
    transpokernel[np.greater(pltkernel,phome)]=phome/pltkernel[np.greater(pltkernel,phome)]
    transpokernel[np.equal(pltkernel,phome)]=1.0
    transpokernel=sp.ndimage.filters.gaussian_filter(transpokernel, [3,3], mode='nearest')
    transpokernel[np.equal(pltkernel,0.)]=0.
    rescaler=np.nansum(transpokernel)
    transpokernel=transpokernel/rescaler  
    
    cumkernel=np.array(np.reshape(np.cumsum(transpokernel),(kx.shape)),dtype=np.float64)
    tempmask[np.equal(tempmask,0.)]=100.
    cumkernel[np.equal(tempmask,100.)]=100.
 

#==============================================================================
# DO YOU WANT TO CREATE DIAGNOSTIC PLOTS?
#==============================================================================

if DoDiagnostics:       
    if areatype.lower()=="box":
        from matplotlib.patches import Polygon
        def plot_rectangle(bmap, lonmin,lonmax,latmin,latmax):
            p = Polygon([(lonmin,latmin),(lonmin,latmax),(lonmax,latmax),(lonmax,latmin)],facecolor='grey',edgecolor='black',alpha=0.5,linewidth=2)
            plt.gca().add_patch(p)

    print("preparing diagnostic plots...")
    
    if rainprop.subdimensions[0]>rainprop.subdimensions[1]:
        figsizex=5
        figsizey=5+0.25*5*np.float(rainprop.subdimensions[0])/rainprop.subdimensions[1]
    elif rainprop.subdimensions[0]<rainprop.subdimensions[1]: 
        figsizey=5
        figsizex=5+0.25*5*np.float(rainprop.subdimensions[0])/rainprop.subdimensions[1]
    else:
        figsizey=5   
        figsizex=5
      
    if areatype.lower()=="basin" and os.path.isfile(wsmaskshp):
        wmap=Basemap(llcrnrlon=rainprop.subextent[0],llcrnrlat=rainprop.subextent[2],urcrnrlon=rainprop.subextent[1],urcrnrlat=rainprop.subextent[3],projection='cyl')
    if domain_type.lower()=="irregular":
        dmap=Basemap(llcrnrlon=rainprop.subextent[0],llcrnrlat=rainprop.subextent[2],urcrnrlon=rainprop.subextent[1],urcrnrlat=rainprop.subextent[3],projection='cyl')


    bmap=Basemap(llcrnrlon=rainprop.subextent[0],llcrnrlat=rainprop.subextent[2],urcrnrlon=rainprop.subextent[1],urcrnrlat=rainprop.subextent[3],projection='cyl',resolution='l')    
    bmap.drawcoastlines(linewidth=1.25)
    bmap.drawparallels(np.linspace(rainprop.subextent[2],rainprop.subextent[3],2),labels=[1,0,0,0],fmt='%6.1f')
    bmap.drawmeridians(np.linspace(rainprop.subextent[0],rainprop.subextent[1],2),labels=[1,0,0,1],fmt='%6.1f')
    outerextent=np.array(rainprop.subextent,dtype='float32')  
    if areatype.lower()=="basin" and os.path.isfile(wsmaskshp):
        try:            
            wmap.readshapefile(wsmaskshp.split('.')[0],str(0),color="black")
        except ValueError:
            if i==0:
                print("problem plotting the watershed map; skipping...")
    if BaseMap.lower()!='none':
        try: 
            bmap.readshapefile(BaseMap,BaseField,color="grey")
        except ValueError:
            if i==0:
                print("problem plotting the basemap; skipping...")       

    
     
    # PLOT STORM OCCURRENCE PROBABILITIES-there is a problem with the "alignment of the raster and the storm locations
    print("     Creating storm probability map...")
    plot_kernel=np.column_stack([pltkernel,np.zeros((pltkernel.shape[0],catmask.shape[1]-pltkernel.shape[1]))])
    plot_kernel=np.row_stack([plot_kernel,np.zeros((catmask.shape[0]-pltkernel.shape[0],plot_kernel.shape[1]))])

    fig = plt.figure()
    fig.set_size_inches(figsizex,figsizey)
    f1=plt.imshow(plot_kernel,interpolation="none",extent=rainprop.subextent,cmap='Reds')
    #plt.title("Probability of storm occurrence")
    #bmap=Basemap(llcrnrlon=rainprop.subextent[0],llcrnrlat=rainprop.subextent[2],urcrnrlon=rainprop.subextent[1],urcrnrlat=rainprop.subextent[3],projection='cyl',resolution='l')
    bmap.drawcoastlines(linewidth=1.25)
    bmap.drawparallels(np.linspace(rainprop.subextent[2],rainprop.subextent[3],2),labels=[1,0,0,0],fmt='%6.1f')
    bmap.drawmeridians(np.linspace(rainprop.subextent[0],rainprop.subextent[1],2),labels=[1,0,0,1],fmt='%6.1f')
    if BaseMap.lower()!='none':
        bmap.readshapefile(BaseMap,BaseField,color="grey")
    if areatype.lower()=="basin" and os.path.isfile(wsmaskshp):
        try: 
            wmap.readshapefile(wsmaskshp.split('.')[0],str(0),color="black")
        except ValueError:
            if i==0:
                print("problem plotting the watershed map; skipping...")
    elif areatype.lower()=="box":
        plot_rectangle(bmap,boxarea[0],boxarea[1],boxarea[2],boxarea[3])
    elif areatype.lower()=="point":
        plt.scatter(ptlon,ptlat,color="b")
    if domain_type.lower()=="irregular" and shpdom:
        dmap.readshapefile(domainshp.split('.')[0],str(0),color="black")
    if rainprop.subdimensions[1]>rainprop.subdimensions[0]:
        cb=plt.colorbar(f1,orientation='horizontal')
    else:
        cb=plt.colorbar(f1)
    cb.set_label("Probability of storm occurrence")
    #plt.text(probextent[0],probextent[3]-(maskheight/2)*rainprop.spatialres[0],"*Probability map may not extend to the edge of map.\nThat isn't a mistake!",size=6)
    plt.scatter(lonrange[catx]+(maskwidth/2+maskwidth%2)*rainprop.spatialres[0],latrange[caty]-(maskheight/2+maskheight%2)*rainprop.spatialres[1],s=catmax/2,facecolors='k',edgecolors='none',alpha=0.75)
    plt.savefig(diagpath+'ProbabilityOfStorms.png',dpi=250)
    plt.close()
    
    
    # PLOT AVERAGE STORM RAINFALL
    print ("     Creating Rainfall mean map...")
    avgrain=np.nansum(catrain,axis=(0,1))/nstorms*rainprop.timeres/60.
    fig = plt.figure()
    fig.set_size_inches(figsizex,figsizey)
    f1=plt.imshow(avgrain,interpolation="none",extent=rainprop.subextent,cmap="Blues")
    #plt.title("Average storm rainfall")
    bmap.drawcoastlines(linewidth=1.25)
    bmap.drawparallels(np.linspace(rainprop.subextent[2],rainprop.subextent[3],2),labels=[1,0,0,0],fmt='%6.1f')
    bmap.drawmeridians(np.linspace(rainprop.subextent[0],rainprop.subextent[1],2),labels=[1,0,0,1],fmt='%6.1f')
    if BaseMap.lower()!='none':
        bmap.readshapefile(BaseMap,BaseField,color="grey")
    if areatype.lower()=="basin" and os.path.isfile(wsmaskshp):
        try:            
            wmap.readshapefile(wsmaskshp.split('.')[0],str(0),color="black")
        except ValueError:
            if i==0:
                print("problem plotting the watershed map; skipping...")
    elif areatype.lower()=="box":
        plot_rectangle(bmap,boxarea[0],boxarea[1],boxarea[2],boxarea[3])
    elif areatype.lower()=="point":
        plt.scatter(ptlon,ptlat,color="b")
    if domain_type.lower()=="irregular" and shpdom:
        dmap.readshapefile(domainshp.split('.')[0],str(0),color="black")
    if rainprop.subdimensions[1]>rainprop.subdimensions[0]:
        cb=plt.colorbar(f1,orientation='horizontal')
    else:
        cb=plt.colorbar(f1)
    cb.set_label('Mean Storm Total Rainfall [mm]')
    plt.scatter(lonrange[catx]+(maskwidth/2+maskwidth%2)*rainprop.spatialres[0],latrange[caty]-(maskheight/2+maskheight%2)*rainprop.spatialres[1],s=catmax/2,facecolors='k',edgecolors='none',alpha=0.75)
    plt.savefig(diagpath+'MeanStormRain.png',dpi=250)
    plt.close()
    
    # PLOT STORM RAINFALL Standard deviation
    print("     Creating Rainfall standard deviation map...")
    stdrain=np.nanstd(np.nansum(catrain,axis=1)*rainprop.timeres/60.,axis=0)
    fig = plt.figure()
    fig.set_size_inches(figsizex,figsizey)
    f1=plt.imshow(stdrain,interpolation="none",extent=rainprop.subextent,cmap="Greens")
    #plt.title("Average storm rainfall")
    bmap.drawcoastlines(linewidth=1.25)
    bmap.drawparallels(np.linspace(rainprop.subextent[2],rainprop.subextent[3],2),labels=[1,0,0,0],fmt='%6.1f')
    bmap.drawmeridians(np.linspace(rainprop.subextent[0],rainprop.subextent[1],2),labels=[1,0,0,1],fmt='%6.1f')
    if BaseMap.lower()!='none':
        bmap.readshapefile(BaseMap,BaseField,color="grey")
    if areatype.lower()=="basin" and os.path.isfile(wsmaskshp):
        try:            
            wmap.readshapefile(wsmaskshp.split('.')[0],str(0),color="black")
        except ValueError:
            if i==0:
                print("problem plotting the watershed map; skipping...")
    elif areatype.lower()=="box":
        plot_rectangle(bmap,boxarea[0],boxarea[1],boxarea[2],boxarea[3])
    elif areatype.lower()=="point":
        plt.scatter(ptlon,ptlat,color="b")
    if domain_type.lower()=="irregular" and shpdom:
        dmap.readshapefile(domainshp.split('.')[0],str(0),color="black")
    if rainprop.subdimensions[1]>rainprop.subdimensions[0]:
        cb=plt.colorbar(f1,orientation='horizontal')
    else:
        cb=plt.colorbar(f1)
    cb.set_label('Standard Deviation of Storm Total Rainfall [mm]')
    plt.scatter(lonrange[catx]+(maskwidth/2+maskwidth%2)*rainprop.spatialres[0],latrange[caty]-(maskheight/2+maskheight%2)*rainprop.spatialres[1],s=catmax/2,facecolors='k',edgecolors='none',alpha=0.75)
    plt.savefig(diagpath+'StdDevStormRain.png',dpi=250)
    plt.close()

    
    # PLOT EACH STORM
    print ("     Creating storm rainfall maps (this could take a while)...")
    fig = plt.figure(1)
    ax  = fig.add_subplot(111)
    fig.set_size_inches(figsizex,figsizey)

    bmap=Basemap(llcrnrlon=rainprop.subextent[0],llcrnrlat=rainprop.subextent[2],urcrnrlon=rainprop.subextent[1],urcrnrlat=rainprop.subextent[3],projection='cyl',resolution='l')    
    bmap.drawcoastlines(linewidth=1.25)
    bmap.drawparallels(np.linspace(rainprop.subextent[2],rainprop.subextent[3],2),labels=[1,0,0,0],fmt='%6.1f')
    bmap.drawmeridians(np.linspace(rainprop.subextent[0],rainprop.subextent[1],2),labels=[1,0,0,1],fmt='%6.1f')
    #f1=plt.imshow(catrain[0,0,:], interpolation='none',extent=outerextent,cmap='Blues',vmin=0,vmax=np.nanmax(catmax))
    
    if BaseMap.lower()!='none':
        bmap.readshapefile(BaseMap,BaseField,color="grey")
    if areatype.lower()=="basin" and os.path.isfile(wsmaskshp):
        try:            
            wmap.readshapefile(wsmaskshp.split('.')[0],str(0),color="black")
        except ValueError:
            if i==0:
                print("problem plotting the watershed map; skipping...")
    elif areatype.lower()=="box":
        plot_rectangle(bmap,boxarea[0],boxarea[1],boxarea[2],boxarea[3])
    elif areatype.lower()=="point":
        plt.scatter(ptlon,ptlat,color="b")
    if domain_type.lower()=="irregular" and shpdom:
        dmap.readshapefile(domainshp.split('.')[0],str(0),color="black")


    try:
        os.system('rm '+diagpath+'Storm*.png')
    except Exception:
        pass
            
    for i in range(0,nstorms):    
        temprain=np.nansum(catrain[i,:],axis=0)*rainprop.timeres/60.
        if userdistr.all()==False:     
            temprain[temprain<0.05*catmax[i]]=np.nan
        if domain_type.lower()=="irregular":
            temprain=np.multiply(temprain,domainmask)
        ims=bmap.imshow(np.flipud(temprain), interpolation='none',extent=outerextent,cmap='Blues',vmin=0)
        if rainprop.subdimensions[1]>rainprop.subdimensions[0]:
            cb=fig.colorbar(ims,orientation='horizontal',pad=0.05,shrink=.8)
        else:
            cb=plt.colorbar()
        cb.set_label('Total Rainfall [mm]')
        sct=bmap.scatter(lonrange[catx[i]]+(maskwidth/2+maskwidth%2)*rainprop.spatialres[0],latrange[caty[i]]-(maskheight/2+maskheight%2)*rainprop.spatialres[1],s=10,facecolors='none',edgecolors='r',alpha=0.5)
        if durcorrection:
            ttl=plt.title('Storm '+str(i+1)+': '+str(cattime_subdur[i,-1])+'\nMax Rainfall:'+str(round(catmax_subdur[i]))+' mm @ Lat/Lon:'+"{:6.1f}".format(latrange[caty_subdur[i]]-(maskheight/2+maskheight%2)*rainprop.spatialres[0])+u'\N{DEGREE SIGN}'+','+"{:6.1f}".format(lonrange[catx_subdur[i]]+(maskwidth/2+maskwidth%2)*rainprop.spatialres[0])+u'\N{DEGREE SIGN}\nNote that the map is based on '+str(catduration)+'-hour rainfall')
        else:
            ttl=plt.title('Storm '+str(i+1)+': '+str(cattime[i,-1])+'\nMax Rainfall:'+str(round(catmax[i]))+' mm @ Lat/Lon:'+"{:6.1f}".format(latrange[caty[i]]-(maskheight/2+maskheight%2)*rainprop.spatialres[0])+u'\N{DEGREE SIGN}'+','+"{:6.1f}".format(lonrange[catx[i]]+(maskwidth/2+maskwidth%2)*rainprop.spatialres[0])+u'\N{DEGREE SIGN}')
        plt.savefig(diagpath+'Storm'+str(i+1)+'_'+str(cattime[i,-1]).split('T')[0]+'.png',dpi=250) 
        cb.remove()
        ims.remove()
        sct.remove()
    plt.close()  
    
 
    if areatype.lower()=="point":
        aoi_ctr_lat=ptlat
        aoi_ctr_lon=ptlon
    elif areatype.lower()=="box":
        aoi_ctr_lat=0.5*(boxarea[2]+boxarea[3])
        aoi_ctr_lon=0.5*(boxarea[0]+boxarea[1])
    elif areatype.lower()=="basin":
        aoi_ctr_lat=0.5*(latrange[ymin]+latrange[ymax])
        aoi_ctr_lon=0.5*(lonrange[xmin]+lonrange[xmax])
        
    distmat=RainyDay.latlondistance(aoi_ctr_lat,aoi_ctr_lon,ingridy,ingridx)/1000.
    distmat[np.isclose(domainmask,0.)]=np.nan
    
    bmap=Basemap(llcrnrlon=rainprop.subextent[0],llcrnrlat=rainprop.subextent[2],urcrnrlon=rainprop.subextent[1],urcrnrlat=rainprop.subextent[3],projection='cyl',resolution='l')    
    bmap.drawcoastlines(linewidth=1.25)
    bmap.drawparallels(np.linspace(rainprop.subextent[2],rainprop.subextent[3],2),labels=[1,0,0,0],fmt='%6.1f')
    bmap.drawmeridians(np.linspace(rainprop.subextent[0],rainprop.subextent[1],2),labels=[1,0,0,1],fmt='%6.1f')
    ims=bmap.imshow(np.flipud(distmat),extent=rainprop.subextent, interpolation='none')
    if rainprop.subdimensions[1]>rainprop.subdimensions[0]:
        cb=fig.colorbar(ims,orientation='horizontal',pad=0.05,shrink=.8)
    else:
        cb=plt.colorbar()
        cb.set_label('Distance from Area of Interest [km]')
    plt.scatter(lonrange[catx]+(maskwidth/2+maskwidth%2)*rainprop.spatialres[0],latrange[caty]-(maskheight/2+maskheight%2)*rainprop.spatialres[1],s=catmax/2,facecolors='k',edgecolors='none',alpha=0.75)
    plt.scatter(aoi_ctr_lon,aoi_ctr_lat,color="red")
    ttl=plt.title('Storm Locations and sizes (black dots)\nand distance from Area of Interest')
    plt.savefig(diagpath+'DistancesFromAOI.png',dpi=250) 
       
    plt.scatter(RainyDay.latlondistance(aoi_ctr_lat,aoi_ctr_lon,latrange[caty]-(maskheight/2+maskheight%2)*rainprop.spatialres[1],lonrange[catx]+(maskwidth/2+maskwidth%2)*rainprop.spatialres[0])/1000.,catmax)
    plt.xlabel("Distance from Area of Interest [km]")
    plt.ylabel("Storm total rainfall [mm]")
    
    
    
#    labels=np.zeros_like(distmat,dtype='bool')
#    labels[domainmask==True]=True
#    outdomain=np.zeros_like(distmat,dtype='float32')
#    outdomain[np.isnan(distmat)]=np.nan
#    latmid=ingridy[ymin,xmin]
#    lonmid=ingridx[ymin,xmin]
#    distarray=RainyDay.latlondistance(aoi_ctr_lat,aoi_ctr_lon,ingridy,ingridx)
#    distarray[labels==False]=np.nan
#    gridarea=RainyDay.latlondistance(latmid,lonmid,latmid+res,lonmid)*RainyDay.latlondistance(latmid,lonmid,latmid,lonmid+res)
#    domarea=gridarea*np.nansum(domainmask)
#    area=0.
#    while np.less_equal(area,domarea) and np.all(labels==False)==False:
#        mindist_ind=np.unravel_index(np.nanargmin(distarray), distarray.shape)
#        if labels[mindist_ind[0],mindist_ind[1]]:
#            print("Processing "+str(area/1000./1000.)+" out of "+str(domarea/1000./1000.)+" km^2")
#            latmid=latrange[mindist_ind[0]]
#            lonmid=lonrange[mindist_ind[1]]
#            area=area+gridarea
#            outdomain[mindist_ind[0],mindist_ind[1]]=area/1000./1000.
#            #area=area+latlondistance(latmid,lonmid,latmid+res,lonmid)*latlondistance(latmid,lonmid,latmid,lonmid+res)
#        labels[mindist_ind[0],mindist_ind[1]]=False
#        distarray[mindist_ind[0],mindist_ind[1]]=np.nan
#
#        
#plt.imshow(outdomain)
#plt.colorbar()   
#    
#plt.scatter(outdomain[caty,catx],catmax)
#plt.xlabel("Size of Transposition Domain [km^2]")
#plt.ylabel("Storm total rainfall [mm]")

#==============================================================================
# STEP 2 (OPTIONAL): STORM TRANSPOSITION
#==============================================================================

if FreqAnalysis:
    print("resampling and transposing...")
    
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
        whichstorms[i,ncounts>=i+1]=np.random.randint(0,nstorms,(len(ncounts[ncounts>=i+1]))) # why was this previously "nstorms-1"??? Bug?
    
    whichrain=np.zeros((whichstorms.shape),dtype='float32')
    testrain=np.zeros((whichstorms.shape),dtype='float32')
    whichx=np.zeros((whichstorms.shape),dtype='int32')
    whichy=np.zeros((whichstorms.shape),dtype='int32')
    
    if durcorrection:
        whichtimeind=np.zeros((whichstorms.shape),dtype='float32')
    
    
    if transpotype=='uniform' and domain_type=='irregular':
        domainmask[-maskheight:,:]=0.
        domainmask[:,-maskwidth:]=0.
        xmask,ymask=np.meshgrid(np.arange(0,domainmask.shape[1],1),np.arange(0,domainmask.shape[0],1))
        xmask=xmask[np.equal(domainmask,True)]
        ymask=ymask[np.equal(domainmask,True)]

    
    #==============================================================================
    # If you're using intensity-dependent resampling, calculate the "intensity factor moments"
    #==============================================================================

    if stochrescale:
        print("reading in rainfall intensity data")
        intenserain,_,_,_=RainyDay.readintensityfile(rescalingfile,inbounds=rainprop.subind)
        intenserain[:,np.equal(domainmask,0.)]=np.nan
        intenserain[np.equal(intenserain,0.)]=np.nan
        intense_quantile=np.random.random_integers(5,high=95,size=whichrain.shape)/100.
        if intenserain.shape[0]!=100:
            sys.exit("RainyDay was expecting the intensity file to have 100 storms per grid cell.")
        else:
            intenserain=intenserain[5:95,:]
                
        #intense_quantile=np.zeros_like(whichrain,dtype='float32')
        #for i in range(0,np.nanmax(ncounts)):
            #intense_quantile[i,ncounts>=i+1]=np.random.random_sample(size=len(ncounts[ncounts>=i+1]))
        #    intense_quantile[i,ncounts>=(i+1)]=np.random.random_integers(5,high=95,size=len(ncounts[ncounts>=i+1]))/100.
        #nzeroesones=1
        
        
        # incredibly, this number generator will create zeros and ones, which will blow up the later statistics! Poor showing on someone's part!
       # while nzeroesones>0:
       #     zerooneindex=np.logical_or(np.isclose(intense_quantile,0.),np.isclose(intense_quantile,1.))
       #     nzeroesones=np.sum(zerooneindex)
       #     intense_quantile[zerooneindex]=np.random.random_sample(size=nzeroesones)
                
    
    # here is the main resampling and transposition loop
    for i in range(0,nstorms):
        print('Resampling and transposing storm '+str(i+1)+' out of '+str(nstorms)+' ('"{0:0.0f}".format(100*(i+1)/nstorms)+'%)')
        # UNIFORM RESAMPLING
        if transpotype=='uniform' and domain_type=='rectangular':
            whichx[whichstorms==i]=np.random.randint(0,np.int(rainprop.subdimensions[1])-maskwidth+1,len(whichx[whichstorms==i]))
            whichy[whichstorms==i]=np.random.randint(0,np.int(rainprop.subdimensions[0])-maskheight+1,len(whichy[whichstorms==i]))
     
        # KERNEL-BASED AND INTENSITY-BASED RESAMPLING (ALSO NEEDED FOR IRREGULAR TRANSPOSITION DOMAINS)
        elif transpotype=='kernel':
            rndloc=np.random.random_sample(len(whichx[whichstorms==i]))
            if numbacheck:
                tempx=np.empty((len(rndloc)),dtype=np.int32)
                tempy=np.empty((len(rndloc)),dtype=np.int32)
                whichx[whichstorms==i],whichy[whichstorms==i]=RainyDay.numbakernel(rndloc,cumkernel,tempx,tempy)
            else:
                whichx[whichstorms==i],whichy[whichstorms==i]=RainyDay.pykernel(rndloc,cumkernel)
                
        if transpotype=='uniform' and domain_type=='irregular':
            rndloc=np.random.randint(0,np.sum(np.equal(domainmask,True)),np.sum(whichstorms==i))
            whichx[whichstorms==i]=xmask[rndloc]
            whichy[whichstorms==i]=ymask[rndloc]
            
        if stochrescale:
            intense_dat=[intense_quantile[whichstorms==i],intenserain,ymin,ymax,xmin,xmax]  
        
        # SET UP MANUAL PDF RESAMPLING
        elif transpotype=='manual':  
            sys.exit("not configured for manually supplied pdf yet!")
    
        if durcorrection:
            #sys.exit("write this!!!")
            passrain=RainyDay.rolling_sum(catrain[i,:], int(duration*60/rainprop.timeres))
            
        else:
            passrain=np.nansum(catrain[i,:],axis=0)         # time-average the rainfall
    
        if rotation: 
            print('rotating storms for transposition, '+str(100*(i+1)/nstorms)+'% complete...')
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
                if stochrescale:
                    print("WARNING: rotation + intensity-based resampling = not tested!")
                    whichrain[np.logical_and(whichstorms==i,anglebins==cbin)]=RainyDay.SSTalt(tpass,whichx[np.logical_and(whichstorms==i,anglebins==cbin)],whichy[np.logical_and(whichstorms==i,anglebins==cbin)],trimmask,maskheight,maskwidth,intense_data=intense_dat,durcheck=durcorrection)*rainprop.timeres/60./mnorm
                else:
                    whichrain[np.logical_and(whichstorms==i,anglebins==cbin)]=RainyDay.SSTalt(tpass,whichx[np.logical_and(whichstorms==i,anglebins==cbin)],whichy[np.logical_and(whichstorms==i,anglebins==cbin)],trimmask,maskheight,maskwidth,durcheck=durcorrection)*rainprop.timeres/60./mnorm     
                binctr=binctr+1
        else:
            if stochrescale:
                whichrain[whichstorms==i]=RainyDay.SSTalt(passrain,whichx[whichstorms==i],whichy[whichstorms==i],trimmask,maskheight,maskwidth,intense_data=intense_dat,durcheck=durcorrection)*rainprop.timeres/60./mnorm
            else:
                whichrain[whichstorms==i]=RainyDay.SSTalt(passrain,whichx[whichstorms==i],whichy[whichstorms==i],trimmask,maskheight,maskwidth,durcheck=durcorrection)*rainprop.timeres/60./mnorm
   
    
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
        sortangle=np.empty((maxind.shape),dtype="float32")
    if stochrescale:
        maxquantile=np.empty((maxind.shape),dtype="float32") 
        sortquantile=np.empty((maxind.shape),dtype="float32") 
        
    for i in range(0,np.max(ncounts)):
        maxx[maxind==i]=whichx[i,maxind==i]
        maxy[maxind==i]=whichy[i,maxind==i]
        maxstorm[maxind==i]=whichstorms[i,maxind==i]
        if rotation:
            maxangles[maxind==i]=randangle[i,maxind==i]
        if stochrescale:
            maxquantile[maxind==i]=intense_quantile[i,maxind==i]
    
    
    # RANK THE STORMS BY INTENSITY AND ASSIGN RETURN PERIODS
    exceedp=np.linspace(1,1./nsimulations,nsimulations)
    returnperiod=1/exceedp
    sortind=np.argsort(maxrain,axis=0)
    sortrain=np.sort(maxrain,axis=0)
    sortx=np.empty((maxind.shape),dtype="int32")
    sorty=np.empty((maxind.shape),dtype="int32")
    sortstorms=np.empty((maxind.shape),dtype="int32")
    
    
    for i in range(0,nrealizations):
        sortx[:,i]=maxx[sortind[:,i],i]
        sorty[:,i]=maxy[sortind[:,i],i]
        sortstorms[:,i]=maxstorm[sortind[:,i],i]
        if rotation:
            sortangle[:,i]=maxangles[sortind[:,i],i]
        if stochrescale:
            sortquantile[:,i]=maxquantile[sortind[:,i],i]
    
        
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
        if rotation:    
            sortangle=sortangle[reducedlevind,:]
        if stochrescale:        
            sortquantile=sortquantile[reducedlevind,:]
        
    
    # what is this?
    nanmask=deepcopy(trimmask)
    nanmask[np.isclose(nanmask,0.)]=np.nan
    nanmask[np.isclose(nanmask,0.)==False]=1.0
        
    
    #################################################################################
    # STEP 2a (OPTIONAL): Find the single storm maximized storm rainfall-added DBW 7/19/2017
    #################################################################################    
    
    if deterministic:
        print("finding maximizing rainfall...")
        
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
    
    
    print("preparing frequency analysis...")
    
    sortrain=sortrain/arfval
    if spreadtype=='ensemble':
        spreadmin=np.nanmin(sortrain,axis=1)
        spreadmax=np.nanmax(sortrain,axis=1)    
    else:
        spreadmin=np.percentile(sortrain,(100-quantilecalc)/2,axis=1)
        spreadmax=np.percentile(sortrain,quantilecalc+(100-quantilecalc)/2,axis=1)

    freqanalysis=np.column_stack((exceedp,returnperiod,spreadmin,RainyDay.mynanmean(sortrain,1),spreadmax))
    
    np.savetxt(FreqFile,freqanalysis,delimiter=',',header='prob.exceed,returnperiod,minrain,meanrain,maxrain',fmt='%6.3f',comments='')
    
    #sys.exit('nopers!')
    
    
    import matplotlib.patches as mpatches
    from matplotlib.font_manager import FontProperties
    from matplotlib import pyplot as plt
   # warnings.filterwarnings('ignore')
    fontP = FontProperties()
    fontP.set_size('xx-small')
    fig, ax = plt.subplots(1)
    line1, = plt.plot(exceedp[exceedp<=0.5], RainyDay.mynanmean(sortrain,1)[exceedp<=0.5], lw=1, label='Average', color='blue')

    ax.fill_between(exceedp[exceedp<=0.5], spreadmin[exceedp<=0.5], spreadmax[exceedp<=0.5], facecolor='dodgerblue', alpha=0.5,label='Ensemble Variability')
    blue_patch = mpatches.Patch(color='dodgerblue', label='Spread')
    plt.legend(handles=[line1,blue_patch],loc='lower right',prop = fontP)

    if np.nanmax(spreadmax[exceedp<=0.5])<10.:
        upperlimit=10.
    elif np.nanmax(spreadmax[exceedp<=0.5])<100.:
        upperlimit=100.
    elif np.nanmax(spreadmax[exceedp<=0.5])<1000.:
        upperlimit=1000.
    else:
        upperlimit=10000.
        
    if np.nanmin(spreadmin[exceedp<=0.5])<1.:
        lowerlimit=0.1
    elif np.nanmin(spreadmin[exceedp<=0.5])<10.:
        lowerlimit=1 
    elif np.nanmin(spreadmin[exceedp<=0.5])<100.:
        lowerlimit=10. 
    else:
        lowerlimit=100.
            
            
    plt.ylim(lowerlimit,upperlimit)
    ax.set_xlabel('Exceedance Probability [-]\n1/(Return Period) [year]')
    ax.set_ylabel('Rainfall [mm]')
    ax.set_yscale('log')
    ax.set_xscale('log')
    plt.gca().invert_xaxis()
    ax.grid()
    plt.tight_layout()
    plt.savefig(fullpath+'/FrequencyAnalysis.png',dpi=250)
    plt.close('all')
           
        
    #################################################################################
    # STEP 4 (OPTIONAL): WRITE RAINFALL SCENARIOS
    #################################################################################
    
    
    if Scenarios:
        print("writing spacetime rainfall scenarios...")
        
        # if desired, this will "pad" the beginning of the scenarios to add a spin-up period.  Not recommended to use long spin-ups, due to memory constraints
        # this scheme basically creates two storm catalogs and mixes them up!
        if prependrain==True:
            print("prepending rainfall for model spin-up...")
            outerextent=np.array(rainprop.subextent,dtype='float32') 
            if CreateCatalog==False:
                # need to generate a file list since it hasn't already been done
                flist,nyears=RainyDay.createfilelist(inpath,includeyears,excludemonths)
                if len(flist)==0:
                    sys.exit("couldn't prepend the rainfall files for spinup period because the input rainfall files weren't available.") 
    
            _,_,inlatitude,inlongitude=RainyDay.readnetcdf(flist[0])
            rainprop.dimensions=[len(inlatitude),len(inlongitude)]
            rainprop.bndbox=[np.min(inlongitude),np.max(inlongitude)+rainprop.spatialres[0],np.min(inlatitude)-rainprop.spatialres[1],np.max(inlatitude)]          
            
            rainprop.subextent,rainprop.subind,rainprop.subdimensions=RainyDay.findsubbox(inarea,rainprop)
            innerextent=deepcopy(rainprop.subextent)
            innerind=deepcopy(rainprop.subind)
                 
            #subgrid,gx,gy=RainyDay.creategrids(rainprop)
        
            tlen=np.int(1440*pretime/rainprop.timeres)
            
            precat=np.zeros((catrain.shape[0],tlen,catrain.shape[2],catrain.shape[3]),dtype='float32')
    
            starttime=np.empty((cattime.shape[0],tlen),dtype='datetime64[m]')
            starttime[:,0]=np.subtract(cattime[:,0],np.timedelta64(np.int32(pretime)*1440,'m'))
            for i in range(0,nstorms):
                starttime[i,:]=np.arange(starttime[i,0],cattime[i,0],np.timedelta64(rainprop.timeres,'m'))   
            cattime=np.concatenate((starttime,cattime),axis=1)
            
            
            # roll 'em back!
            for i in np.arange(0,nstorms):
                print("Creating pre-pending catalog entry for storm "+str(i+1))
                starttime=cattime[i,tlen]-np.timedelta64(np.int32(pretime)*1440,'m')
                startstr=np.str(starttime).replace('-','').split('T')[0]
                prelist=np.core.defchararray.find(flist,startstr)
                preind=np.where(prelist==np.max(prelist))[0][0]
                tlist=flist[preind:preind+np.int(np.round(pretime))] 
    
                for j in range(0,len(tlist)):
                    print('Pre-pending rainfall with file '+tlist[j])
                    inrain,intime,_,_=RainyDay.readnetcdf(tlist[j],inbounds=rainprop.subind)
                    inrain[inrain<0.]=np.nan
                    
                    for k in range(0,int(24*60/rainprop.timeres)):
                        if np.in1d(intime[k],cattime[i,:])[0]: 
                            cattime[i,j*int(24*60/rainprop.timeres):int(j*24*60/rainprop.timeres+24*60/rainprop.timeres)]=intime
                            precat[i,j*int(24*60/rainprop.timeres):int(j*24*60/rainprop.timeres+24*60/rainprop.timeres),:]=np.reshape(inrain,(int(24*60/rainprop.timeres),rainprop.subdimensions[0],rainprop.subdimensions[1]))
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
            if stochrescale:
                writequantile=sortquantile[minind:,:]
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
            if stochrescale:
                writequantile=sortquantile
    
        for rlz in range(0,nrealizations):
            print("writing spacetime rainfall scenarios for realization "+str(rlz+1)+"/"+str(nrealizations))
            
            # this statement is only really needed if you are prepending rainfall, but it is fast so who cares?
            outtime=np.empty((writetimes.shape[0],cattime.shape[1]),dtype='datetime64[m]')
            unqstm=np.unique(writestorm[:,rlz])
            for i in range(0,len(unqstm)):
                outtime[writestorm[:,rlz]==unqstm[i],:]=cattime[unqstm[i],:] 
                
            if stochrescale:
                sys.exit("not finished yet!")
                #intense_dat=[writequantile[:,rlz],murain,stdrain,catorigmu,catorigstd]
        
            
            if rotation:
                if stochrescale:
                    sys.exit("not tested!")
                else:
                    outrain=RainyDay.SSTspin_write_v2(catrain,writex[:,rlz],writey[:,rlz],writestorm[:,rlz],nanmask,maskheight,maskwidth,precat,cattime[:,-1],rainprop,rlzanglebin=binwriteang[:,rlz],delarray=delarray,spin=prependrain,flexspin=False,samptype=transpotype,cumkernel=cumkernel,rotation=rotation,domaintype=domain_type)
            else:
                if stochrescale:
                    outrain=RainyDay.SSTspin_write_v2(catrain,writex[:,rlz],writey[:,rlz],writestorm[:,rlz],nanmask,maskheight,maskwidth,precat,cattime[:,-1],rainprop,spin=prependrain,flexspin=False,samptype=transpotype,cumkernel=cumkernel,rotation=rotation,domaintype=domain_type)
                else:
                    outrain=RainyDay.SSTspin_write_v2(catrain,writex[:,rlz],writey[:,rlz],writestorm[:,rlz],nanmask,maskheight,maskwidth,precat,cattime[:,-1],rainprop,spin=prependrain,flexspin=False,samptype=transpotype,cumkernel=cumkernel,rotation=rotation,domaintype=domain_type)
            
            
            outrain[:,:,np.isclose(trimmask,0.)]=-9999.               # this line produced problems in CUENCAS CONVERSIONS :(
            writename=WriteName+'_SSTrlz'+str(rlz+1)+'.nc'
            subrangelat=latrange[ymin:ymax+1]
            subrangelon=lonrange[xmin:xmax+1]
            
            #print "need to write angles to the realization files"
            RainyDay.writerealization(rlz,nrealizations,writename,outrain,writemax[:,rlz],writestorm[:,rlz],writeperiod,writex[:,rlz],writey[:,rlz],outtime,subrangelat,subrangelon,whichorigstorm[:,rlz])
        
        if deterministic:
            RainyDay.writemaximized(wd+'/'+scenarioname+'/'+scenarioname+'_maximizedstorm.nc',maxstm_rain,maxstm_avgrain,maxstm_ts,max_trnsx,max_trnsy,maxstm_time,subrangelat,subrangelon)
else:
    print("skipping the frequency analysis!")
            
end = time.time()   
print("RainyDay has successfully finished!\n")
print("Elapsed time: "+"{0:0.2f}".format((end - start)/60.)+" minutes")

   
#################################################################################
# THE END
#################################################################################

