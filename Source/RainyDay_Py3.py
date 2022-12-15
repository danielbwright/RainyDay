#!/usr/bin/env python

# give the right permissions (for a Mac or Linux): chmod +x {scriptname}


#==============================================================================
# WELCOME
#==============================================================================
#    Welcome to RainyDay, a framework for coupling gridded precipitation
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

import cartopy.crs as ccrs
import cartopy.feature as cfeature
import cartopy.io.shapereader as shpreader
from cartopy.io.shapereader import Reader
import glob
import geopandas as gp
import xarray as xr
from cartopy.feature import ShapelyFeature
import cartopy.mpl.ticker as cticker
#from numba import njit, prange
numbacheck=True

# plotting stuff, really only needed for diagnostic plots
import matplotlib
#matplotlib.use('Agg')
import matplotlib.pyplot as plt

# import RainyDay functions
import RainyDay_utilities_Py3.RainyDay_functions as RainyDay

from numba.types import int32

import warnings
warnings.filterwarnings("ignore")

#==============================================================================
# RAINFALL CLASS
# THIS CONTAINS INFORMATION ABOUT THE SPECIFIED INPUT RAINFALL DATASET
#==============================================================================
class GriddedRainProperties(object):
    def __init__(self,dataset,bndbox,subind,subextent,dimensions,subdimensions,spatialres,timeres,timeunits,spatialunits,rainunits,nodata,notes):
        self.dataset=dataset                #  INPUT RAINFALL DATA SOURCE (TMPA, STAGE IV, ETC.)
        self.bndbox=bndbox                  # COORDINATES OF FULL DATASET BOUNDING BOX (THIS SHOULD BE THE IN THE ORDER [WEST LON,EAST LON,SOUTH LAT,NOTH LAT])
        self.subind=subind                  # MATRIX INDICIES OF BOUNDING BOX (XMIN,XMAX,YMIN,YMAX)
        self.subextent=subextent            # COORDINATES OF USER-DEFINED BOUNDING BOX (NOTE: THIS WILL BE AUTOMATICALLY CALCULATED)
        self.dimensions=dimensions          # WHAT ARE THE SPATIAL DIMENSIONS OF THE INPUT DATASET (NOTE: EVEN IF THE INPUT DATA YOU ARE USING IS FOR A SUBDOMAIN, THIS SHOULD BE THE DIMENSIONS OF THE FULL DATASET)
        self.subdimensions=subdimensions    # WHAT ARE THE DIMENSIONS OF THE SUBDOMAIN (IN THIS SCRIPT, THIS WILL BE RESET TO THE ACTUAL DIMENSION OF THE INPUT DATA, IF NEEDED)
        self.spatialres=spatialres          # WHAT IS THE SPATIAL RESOLUTION OF THE INPUT DATA [dx,dy]?  CURRENTLY THIS SCRIPT WILL ONLY HANDLE RECTANGULAR GRIDS IN DEGREES
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
    #parameterfile='/Users/daniel/Google_Drive/Presentations/MyPresentations/WisconsinRainfallProject/Wisconsin_24hrIDF_Madison.sst'
except:
    sys.exit("You didn't specify an input ('.sst') file!")

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
    #wd=np.str(sys.argv[2])
    try:
        scenarioname=str(cardinfo[cardinfo[:,0]=="SCENARIONAME",1][0])
    except Exception:
        print("You did't specify SCENARIONAME, defaulting to 'RainyDayAnalysis'!")
        scenarioname='RainyDayAnalysis'
    fullpath=wd+'/'+scenarioname
    if os.path.isdir(fullpath)==False:
        #os.system('mkdir -p -v %s' %(fullpath))
        os.mkdir(fullpath)
    os.chdir(wd)
except ImportError:
    sys.exit("You did't specify MAINPATH, which is a required field!")


# PROPERTIES RELATED TO SST PROCEDURE:
try:
    catalogname=str(cardinfo[cardinfo[:,0]=="CATALOGNAME",1][0])
except Exception:
    catalogname='SSTstormcatalog.nc'
if catalogname.find('/')==-1:
    catalogname=wd+'/'+catalogname
    
if len(catalogname.split('.nc'))==1:
    sys.exit("You didn't specify the '.nc' extension on the storm catalog name!")    

try:
    CreateCatalog=cardinfo[cardinfo[:,0]=="CREATECATALOG",1][0]
except Exception:
    sys.exit("You didn't specify CREATECATALOG, which is required!")  
    
if CreateCatalog.lower()=='true':
    CreateCatalog=True
else:
    CreateCatalog=False
    if os.path.isfile(catalogname)==False:
        sys.exit("You need to create a storm catalog first.")
    else:
        print("Reading an existing storm catalog!")
        try:
            catrain,cattime,latrange,lonrange,catx,caty,catmax,_,domainmask,timeres=RainyDay.readcatalog(catalogname)
        except ValueError:
            catrain,cattime,latrange,lonrange,catx,caty,catmax,_,domainmask=RainyDay.readcatalog(catalogname)
            timeres=RainyDay.readtimeresolution(catalogname)
        #if catrain.shape[1]==1:
        #    timeres=RainyDay.readtimeresolution(catalogname)
        yres=np.abs(np.mean(latrange[1:]-latrange[0:-1]))
        xres=np.abs(np.mean(lonrange[1:]-lonrange[0:-1])) 
        catarea=[lonrange[0],lonrange[-1]+xres,latrange[-1]-yres,latrange[0]]
        if np.isclose(xres,yres)==False:
            sys.exit('Sadly, RainyDay currently only supports equal x and y resolutions.')
        else:
            res=np.min([yres,xres])


try:
    nstorms=np.int(cardinfo[cardinfo[:,0]=="NSTORMS",1][0])
    defaultstorms=False
    print(str(nstorms)+" storms will be identified for storm catalog!")
except Exception:
    #nstorms=100
    print("you didn't specify NSTORMS, defaulting to 20 per year, or whatever is in the catalog!")
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


# this following bit is only needed in the very specific (and generally not recommended) case when the desired analysis/catalog duration is equal to the temporal resolution of the dataset
# try:
#     temptimeres=np.int(cardinfo[cardinfo[:,0]=="TIMERESOLUTION",1][0])
#     print("A resolution of "+str(temptimeres)+" minutes has been provided. Be careful with this, because if it is improperly specified, this will cause errors. Note that TIMERESOLUTION is not needed unless the duration of each storm is to be exactly equal to the temporal resolution of the input data or catalog. In other words, make sure you know what you're doing!")
# except Exception:
#     temptimeres=False
    


try:
    samplingtype=cardinfo[cardinfo[:,0]=="RESAMPLING",1][0]
    if samplingtype.lower()!='poisson' and samplingtype.lower()!='empirical' and samplingtype.lower()!='negbinom':
        sys.exit("unrecognized storm count resampling type, options are: 'poisson', 'empirical', or 'negbinom'")
except Exception:
    samplingtype='poisson'

if samplingtype=='poisson':
    print("Poisson-based temporal resampling scheme will be used!")
elif samplingtype=='empirical':
    print("Empirically-based temporal resampling scheme will be used!")
elif samplingtype=='negbinom':
    print("Negative binomial-based temporal resampling scheme will be used! WARNING: this is not tested!")

try:
    areatype=str(cardinfo[cardinfo[:,0]=="POINTAREA",1][0])
    if areatype.lower()=="basin" or areatype.lower()=="watershed":
        areatype='basin'
        try:
            wsmaskshp=str(cardinfo[cardinfo[:,0]=="WATERSHEDSHP",1][0])
            print("You selected 'watershed' for 'POINTAREA', please note that if the watershed is not in a regular lat/lon projection such as EPSG4326/WGS 84, the results will likely be incorrect!")
        except IndexError:
            sys.exit("You specified 'watershed' for 'POINTAREA' but didn't specify 'WATERSHEDSHP'")            
    elif areatype.lower()=="point" or areatype.lower()=="grid":
        try:
            ptlat=np.float32(cardinfo[cardinfo[:,0]=="POINTLAT",1][0]) 
            ptlon=np.float32(cardinfo[cardinfo[:,0]=="POINTLON",1][0]) 
            ctrlat=ptlat
            ctrlon=ptlon
            areatype="point"
        except ImportError:
            sys.exit("You specified 'point' for 'POINTAREA' but didn't properly specify 'POINTLAT' and 'POINTLON'")
    elif areatype.lower()=="box" or areatype.lower()=="rectangle":
        try:
            box1=np.float32(cardinfo[cardinfo[:,0]=="BOX_YMIN",1][0])
            box2=np.float32(cardinfo[cardinfo[:,0]=="BOX_YMAX",1][0])
            box3=np.float32(cardinfo[cardinfo[:,0]=="BOX_XMIN",1][0])
            box4=np.float32(cardinfo[cardinfo[:,0]=="BOX_XMAX",1][0])
            boxarea=np.array([box3,box4,box1,box2])
            ctrlat=(box1+box2)/2.
            ctrlon=(box3+box4)/2.
            areatype="box"
        except ImportError:
            sys.exit("You specified 'box' for 'POINTAREA' but didn't properly specify 'BOX_YMIN', 'BOX_YMAX', 'BOX_XMIN', and 'BOX_XMAX'")
    elif areatype.lower()=="pointlist":
        if CreateCatalog:
            sys.exit("POINTLIST is currently not available when creating a new storm catalog.")
        ptlistname=str(cardinfo[cardinfo[:,0]=="POINTLIST",1][0])
        #ptlistname=np.str(sys.argv[3])
        ptlistdat=np.loadtxt(ptlistname,skiprows=1,delimiter=',')
        ptlatlist=ptlistdat[:,0]
        ptlonlist=ptlistdat[:,1] 
        npoints_list=len(ptlatlist)
    else:
        sys.exit("unrecognized area type")
except ImportError:
    sys.exit("You didn't specify 'POINTAREA', which needs to either be set to 'watershed', 'point', 'grid', or 'rectangle'")
    
# transposition
try:
    transpotype=cardinfo[cardinfo[:,0]=="TRANSPOSITION",1][0]
    if transpotype.lower()=='kernel' or transpotype.lower()=='nonuniform':
        transpotype='kernel'
        print("You selected the kernel density-based non-uniform storm transposition scheme!")
    elif transpotype.lower()=='uniform' and areatype.lower()=="pointlist":
        print("You selected the spatially uniform storm transposition scheme and to perform IDF for a list of points!")
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
 

# Use stochastic or deterministic ratio multiplier?
rescaletype='none'
try:
    rescalingtype=cardinfo[cardinfo[:,0]=="ENHANCEDSST",1][0]
    if rescalingtype.lower()=='stochastic' or rescalingtype.lower()=='deterministic':
        print("You selected the 'Ratio Rescaling'! This is a more advanced option that should be used with caution")
        if rescalingtype.lower()=='stochastic':
            rescaletype='stochastic'
            print("You selected stochastic ratio rescaling. This has not been thoroughly vetted. Be careful!")
            if areatype.lower()=='pointlist':
                pass
                #sys.exit("You selected 'pointlist' for POINTAREA. This is currently not compatible with stochastic rescaling.")
        if rescalingtype.lower()=='deterministic':
            rescaletype='deterministic'
            print("You selected deterministic ratio rescaling. This has not been thoroughly vetted. Be careful!")
        
        try:
            rescalingfile=cardinfo[cardinfo[:,0]=="RAINDISTRIBUTIONFILE",1][0]
            if os.path.isfile(rescalingfile)==False:
                sys.exit("The precipitation file specified in 'RAINDISTRIBUTIONFILE' cannot be found!")
        except IndexError:
            sys.exit("Even though you 'ratio rescaling', you didn't specify the file of precipitation distributions!")  
    elif rescalingtype.lower()=='dimensionless':
        print("You selected 'dimensionless SST', modeled after Nathan et al. (2016). This has not been thoroughly vetted. Be careful!")
        rescaletype='dimensionless'
        try:
            rescalingfile=cardinfo[cardinfo[:,0]=="RAINDISTRIBUTIONFILE",1][0]
            if os.path.isfile(rescalingfile)==False:
                sys.exit("The precipitation file specified in 'RAINDISTRIBUTIONFILE' cannot be found!")
        except IndexError:
            sys.exit("Even though you 'dimensionless SST', you didn't specify the file of precipitation distributions!")      
    else:
        print("No rescaling will be performed!")
except Exception:
    print("No rescaling will be performed!")

    
try:   
    do_deterministic=cardinfo[cardinfo[:,0]=="MAXTRANSPO",1][0]
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
    if CreateCatalog:
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
                sys.exit("This capability isn't tested. Unclear whether changes are needed due to updating for CF compliance.")
                sys.exit("Can't find the transposition domain NetCDF file!")  
            else:
                sys.exit("This capability isn't tested. Unclear whether changes are needed due to updating for CF compliance.")
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
    else:
        try:
            domainshp=str(cardinfo[cardinfo[:,0]=="DOMAINSHP",1][0])
            if os.path.isfile(domainshp)==False:
                sys.exit("can't find the transposition domain shapefile!")
            else:
                print("You selected 'irregular' for 'DOMAINTYPE', please note that if the domain shapefile is not in a regular lat/lon projection such as EPSG4326/WGS 84, the results will likely be incorrect!")
                shpdom=True
        except Exception:
            print("Trouble finding the domain shapefile. Technically we don't need it, so we'll skip this part.")          
        yres=np.abs(np.mean(latrange[1:]-latrange[0:-1]))
        xres=np.abs(np.mean(lonrange[1:]-lonrange[0:-1]))
        inarea=np.array([lonrange[0],lonrange[-1]+res,latrange[-1]-res,latrange[0]])  
    
    if ncfdom==False and shpdom==False and CreateCatalog:
        sys.exit("You selected 'irregular' for 'DOMAINTYPE', but didn't provide a shapefile or NetCDF file for the domain!")
else:  
    print("rectangular domain type selected!")
    domain_type='rectangular'   
    if CreateCatalog==False:
        inarea=deepcopy(catarea)
    else:
        try:
            lim1=np.float32(cardinfo[cardinfo[:,0]=="LATITUDE_MIN",1][0])
            lim2=np.float32(cardinfo[cardinfo[:,0]=="LATITUDE_MAX",1][0])
            lim3=np.float32(cardinfo[cardinfo[:,0]=="LONGITUDE_MIN",1][0]) 
            lim4=np.float32(cardinfo[cardinfo[:,0]=="LONGITUDE_MAX",1][0]) 
            inarea=np.array([lim3,lim4,lim1,lim2])
        except ImportError:
            sys.exit("need to specify 'LATITUDE_MIN', 'LATITUDE_MAX', 'LONGITUDE_MIN', 'LONGITUDE_MAX'")


if CreateCatalog==False and np.allclose(inarea,catarea)==False:
    if inarea[0]>=catarea[0] and inarea[1]<=catarea[1] and inarea[2]>=catarea[2] and inarea[3]<=catarea[3]:
        sys.exit("You are changing the domain size, but it fits inside the catalog domain. That's a start, but RainyDay still can't handle it.")
    else:
        sys.exit("You are changing the domain size for an existing catalog. RainyDay can't handle that!")
        
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
        #os.system('mkdir %s' %(diagpath))
        try:
            os.mkdir(diagpath)
        except:
            pass
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
    

if areatype.lower()=="pointlist":
    FreqFile_mean=fullpath+'/'+scenarioname+'_mean.FreqAnalysis'
    FreqFile_min=fullpath+'/'+scenarioname+'_min.FreqAnalysis'
    FreqFile_max=fullpath+'/'+scenarioname+'_max.FreqAnalysis'
else:
    FreqFile=fullpath+'/'+scenarioname+'_FreqAnalysis.csv'

try:
    Scenarios=cardinfo[cardinfo[:,0]=="SCENARIOS",1][0]
    if Scenarios.lower()=='true' and areatype.lower()!="pointlist":
        Scenarios=True
        FreqAnalysis=True
        WriteName=wd+'/'+scenarioname+'/Realizations'
        #os.system('mkdir %s' %(WriteName))
        if os.path.isdir(WriteName)==False:
            os.mkdir(WriteName)
        WriteName=WriteName+'/'+scenarioname
        print("RainyDay will write "+str(nrealizations)+" realizations times "+str(nsimulations)+" years worth of output precipitation scenarios. If this is a big number, this could be very slow!")
    elif Scenarios.lower()=='true' and areatype.lower()=="pointlist":
        print("You specified 'POINTAREA pointlist', but want precipitation scenario outputs. The 'pointlist option' does not support precipitation scenarios!")
        Scenarios=False
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
        for i in np.arange(0,len(exclude)):
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
            for i in np.arange(0,len(includeyr)):
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
if CreateCatalog:
    try:
        inpath=str(cardinfo[cardinfo[:,0]=="RAINPATH",1][0])
    except ImportError:
        sys.exit("You didn't specify 'RAINPATH', which is a required field for creating a new storm catalog!")


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

if areatype=='point' or areatype=='pointlist':
    try:
        ARFval=cardinfo[cardinfo[:,0]=="POINTADJUST",1][0]     
        if ARFval.lower()!='false':
            print("Using the POINTADJUST option! Be sure you know what you're doing!")
            if ',' in ARFval:
                ARFval=ARFval.replace(" ","")
                arfval=np.array(ARFval.split(','),dtype='float32')
            else:
                arfval=np.array([np.float(ARFval)])
            
        else:
            FrequencySens=1.
    except Exception:
        arfval=np.array([1.])    
        pass
else:
    arfval=np.array([1.])

try:    
    CalcType=cardinfo[cardinfo[:,0]=="CALCTYPE",1][0]
    if CalcType.lower()=='ams' or CalcType.lower()=='annmax':
        calctype='ams'
        print("You've selected to use Annual Maxima Series.")
    elif CalcType.lower()=='pds' or CalcType.lower()=='partialduration':
        calctype='pds'
        print("You've selected to use Partial Duration Series.")
    else:
        print("Unrecognized entry for CALCTYPE. Defaulting to Annual Maxima Series.")
except Exception:
    calctype='ams'
    print("Nothing provided for CALCTYPE. Defaulting to Annual Maxima Series.")

if calctype=='pds' and Scenarios:
    sys.exit("Unfortunately, RainyDay currently only allows Partial Duration Series for rainfall frequency analysis, not for scenario writing. You can change CALCTYPE to 'AMS' and try again.")
    

# here is where we'll determine if you want to write more than one storm per year to realization files
# ideally would write the if statement better to provide more guidance if the user provides some weird combination, such as 'CALCTYPE pds' and 'NPERYEAR 5'
if Scenarios and areatype.lower()!="pointlist":
    try:
        nperyear=cardinfo[cardinfo[:,0]=="NPERYEAR",1]
        if nperyear!='false' or np.int(nperyear)!=1:
            print("RainyDay will output "+str(np.int(nperyear))+" storms per synthetic year! If this is a big number, this could be very slow!")
            calctype='npyear'
            nperyear=np.int(nperyear)
            if nperyear==1:
            	nperyear='false'
#        else:
#            print("RainyDay will output "+str(nperyear)+" storms per synthetic year!")
    except Exception:
       pass

        

try:    
    userdistr=cardinfo[cardinfo[:,0]=="INTENSDISTR",1][0]
    if userdistr.lower()=='false':
        userdistr=np.zeros((1),dtype='bool')
    elif len(userdistr.split(','))==3:
        print("reading user-defined precipitation intensity distribution...")
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
                print('spinup precipitation will be not included in precipitation scenarios...')
            else:
                try:
                    pretime=np.int(pretime)
                    prependrain=True
                    if pretime==0.:
                        prependrain=False
                        print('the spinup time you provided was 0, no spinup precipitation will be included...')
                    elif pretime<0.:
                        sys.exit("the spin-up time value is negative.")
                    elif pretime<1.:
                        print('the spinup time you provided was short, rounding up to 1 day...')
                        pretime=1.0
                    else:
                        print( np.str(pretime)+' days of spinup precipitation will be included in precipitation scenarios...')
                except:
                    sys.exit('unrecognized value provided for SPINPERIOD.')
               
        else:
            prependrain=False
            print('no SPINPERIOD field was provided.  Spin-up precipitation will be not be included...')
    except Exception:
        print('no SPINPERIOD field was provided.  Spin-up precipitation will be not be included...')

try:
    if np.any(np.core.defchararray.find(cardinfo[:,0],"UNCERTAINTY")>-1):       
        spread=cardinfo[cardinfo[:,0]=="UNCERTAINTY",1][0]
        if spread.lower()=='ensemble':
            spreadtype='ensemble'
            print('"ensemble spread" will be calculated for precipitation frequency analysis...')
        else:
            try:
                int(spread)
            except:
                print('unrecognized value for UNCERTAINTY.')
            if int(spread)>=0 and int(spread)<=100:
                spreadtype='quantile'
                quantilecalc=int(spread)
                printcalc=(100-quantilecalc)//2
                print(str(printcalc)+'th-'+str(quantilecalc+printcalc)+'th interquantile range will be calculated for precipitation frequency analysis...')
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
                    speclevels=np.float32(speclevels)
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
        sys.exit("Not sure I ever finished writing this capability!")

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
   
# Added by DBW, 16 Feb 2021, specifically to support Dr. Emad Habib's team in performing ARF analyses.
# If ARFANALYSIS is used, DURATIONCORRECTION will be turned on automatically, and the scenarios will have a duration equal to DURATION. 
#This is sensible if the objective is to create rainfall scenario files for subsequent analysis (ARFs, scaling properties, etc.). 
# If ARFANALYSIS is turned off and DURATIONCORRECTION is turned on, then the rainfall scenarios will under most circumstances have a longer duration than DURATION.
# This latter approach (DURATIONCORRECTION on, ARFANALYSIS off) is a more justifiable way of generating rainfall scenarios for flood modeling.
# Also, this ARFANALYSIS option isn't set up to work with 'pointlists'. 
try:
    if np.any(np.core.defchararray.find(cardinfo[:,0],"ARFANALYSIS")>-1):
        arfcorr=cardinfo[cardinfo[:,0]=="ARFANALYSIS",1][0]
        if arfcorr.lower()=='false':
            arfcorrection=False
        elif arfcorr.lower()=='true' and areatype.lower()!="pointlist":
            durcorrection=True
            arfcorrection=True
            print("ARFANALYSIS set to 'true'. DURATIONCORRECTION will be used. This is only really recommended in you want to create rainfall scenarios for ARF or other spatial analysis. Make sure you know what you're doing!")
        elif prependrain:
            sys.exit("You've selected to do ARFANALYSIS and to pre-pend rainfall. This suggests that you don't know what you're doing.")
        else:
            print('Invalid option provided for ARFANALYSIS (should be "true" or "false"). Defaulting to "false"!')
            arfcorrection=False
    else:
        arfcorrection=False
except Exception:
    arfcorrection=False

if CreateCatalog and durcorrection:
    catduration=max([72.,3.*duration])    # this will be slow!
    print('Since DURCORRECTION will be used and a storm catalog will be created, the duration of the catalog will be '+"{0:0.2f}".format(catduration)+' hours')
elif CreateCatalog:
    catduration=duration
   
    
#sys.exit('decide what to do with duration correction and timeseparation!')
try:
    timeseparation=np.float32(cardinfo[cardinfo[:,0]=="TIMESEPARATION",1][0])     
except Exception:
    timeseparation=0. 
    
islassiter=False   
try:    
    if np.any(np.core.defchararray.find(cardinfo[:,0],"ISLASSITER")>-1):
        islassiter=cardinfo[cardinfo[:,0]=="ISLASSITER",1][0]
        if islassiter.lower()!="false":
            islassiter=True
            print("Using Lassiter-style Netcdf files!")
        else:
            islassiter=False
except Exception:
    pass
try:    
    if np.any(np.core.defchararray.find(cardinfo[:,0],"ISFITZGERALD")>-1):
        islassiter=cardinfo[cardinfo[:,0]=="ISFITZGERALD",1][0]
        if islassiter.lower()!="false":
            islassiter=True
            print("Using FitzGerald-style Netcdf files!")
        else:
            islassiter=False
except Exception:
    pass        
  

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
    rainprop.spatialres,rainprop.dimensions,rainprop.bndbox,rainprop.timeres,rainprop.nodata=RainyDay.rainprop_setup(flist[0],lassiterfile=islassiter)
    spatres=rainprop.spatialres[0]
    
    
    #==============================================================================
    # SET UP THE SUBGRID INFO
    #==============================================================================
    # 'subgrid' defines the transposition domain
 
    rainprop.subextent,rainprop.subind,rainprop.subdimensions=RainyDay.findsubbox(inarea,rainprop)
    if ncfdom and (np.any(domainmask.shape!=rainprop.subdimensions)):
        sys.exit("Something went terribly wrong :(")        # this shouldn't happen



#==============================================================================
# IF A STORM CATALOG ALREADY EXISTS, USE IT
#==============================================================================  
    
else:
    print("Using the existing storm catalog...")

    rainprop.spatialres=[xres,yres]
    if len(cattime[-1,1:])!=0:
        rainprop.timeres=np.float32(np.mean(cattime[-1,1:]-cattime[-1,0:-1]))
    else:  
        rainprop.timeres=timeres
    
    rainprop.nodata=np.unique(catrain[catrain<0.])
    
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

    rainprop.bndbox=catarea
    rainprop.subextent=rainprop.bndbox
    
    #rainprop.subextent,rainprop.subind,rainprop.subdimensions=RainyDay.findsubbox(inarea,rainprop)
        
    rainprop.subind=np.array([0,len(lonrange)-1,len(latrange)-1,0],dtype='int32')
    rainprop.dimensions=np.array([len(latrange),len(lonrange)],dtype='int32') 
    
    #catrain=catrain[:,:,rainprop.subind[3]:rainprop.subind[2]+1,rainprop.subind[0]:rainprop.subind[1]+1]
    #latrange=latrange[rainprop.subind[0]:rainprop.subind[1]+1]
    #lonrange=lonrange[rainprop.subind[3]:rainprop.subind[2]+1]
    
    rainprop.subdimensions=rainprop.dimensions

if int(duration*60/rainprop.timeres)<=0:
    sys.exit("it appears that you specified a duration shorter than the temporal resolution of the input data!")


if timeseparation<=0. and durcorrection==False:
    timeseparation=duration
elif timeseparation>0. and durcorrection==False:
    timeseparation=timeseparation+duration
elif durcorrection:
    timeseparation=np.max([timeseparation+duration,catduration])

spatres=rainprop.spatialres[0]
ingridx,ingridy=np.meshgrid(np.arange(rainprop.subextent[0],rainprop.subextent[1]-spatres/1000.,spatres),np.arange(rainprop.subextent[3],rainprop.subextent[2]+spatres/1000.,-spatres))        
lonrange=ingridx[0,:]
latrange=ingridy[:,0]

#============================================================================
# Do the setup to run for specific times of day!
#=============================================================================

if CreateCatalog==False:
    tdates = pd.DatetimeIndex(cattime[:,0])
    tdates.year
    nyears=np.max(np.array(tdates.year))-np.min(np.array(tdates.year))+1
if starthour==0 and endhour==24:
    hourinclude=np.ones((int(24*60/rainprop.timeres)),dtype='int32')
else:
    sys.exit("Restrictions to certain hours isn't currently tested or supported")
    try:
        sys.exit("need to fix this")
        _,temptime,_,_=RainyDay.readnetcdf(flist[0],inbounds=rainprop.subind,lassiterfile=islassiter)
    except Exception:
        sys.exit("Can't find the input files necessary for setup to calculate time-of-day-specific IDF curves")

    starthour=starthour+np.float(rainprop.timeres/60)   # because the netcdf file timestamps correspond to the end of the accumulation period
    hourinclude=np.zeros((24*60/rainprop.timeres),dtype='int32')
    temphour=np.zeros((24*60/rainprop.timeres),dtype='float32')

    for i in np.arange(0,len(temptime)):
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
if areatype.lower()=="basin" or areatype.lower()=="watershed":
    if os.path.isfile(wsmaskshp)==False:
        sys.exit("can't find the basin shapefile!")
    else:
        catmask=RainyDay.rastermask(wsmaskshp,rainprop,'fraction')
        #catmask=catmask.reshape(ingridx.shape,order='F')

elif areatype.lower()=="point":
    catmask=np.zeros((rainprop.subdimensions))
    yind=np.where((latrange-ptlat)>0)[0][-1]
    xind=np.where((ptlon-lonrange)>0)[0][-1]  
    if xind==0 or yind==0:
        sys.exit('the point you defined is too close to the edge of the box you defined!')
    else:
        catmask[yind,xind]=1.0
        
elif areatype.lower()=="pointlist":
    catmask=np.zeros((rainprop.subdimensions))
    yind_list=np.zeros_like(ptlatlist,dtype='int32')
    xind_list=np.zeros_like(ptlatlist,dtype='int32')
    for pt in np.arange(0,npoints_list):
        yind_list[pt]=np.where((latrange-ptlatlist[pt])>0)[0][-1]
        xind_list[pt]=np.where((ptlonlist[pt]-lonrange)>0)[0][-1]  
    if np.any(yind_list==0) or np.any(xind_list==0):
        sys.exit('the point you defined is too close to the edge of the box you defined!')
    else:
        catmask[yind_list[1],xind_list[1]]=1.0          # the idea of the catmask gets a little goofy in this situation

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
timestep=np.timedelta64(np.int(rainprop.timeres),'m')

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

elif domain_type.lower()=='irregular' and shpdom and CreateCatalog:
    domainmask=RainyDay.rastermask(domainshp,rainprop,'simple').astype('float32')
    #domainmask=catmask.reshape(ingridx.shape,order='F')
elif domain_type.lower()=='rectangular':
    domainmask=np.ones((catmask.shape),dtype='float32')    
else:
    pass


if catmask.shape!=domainmask.shape:
    sys.exit("Oh dear, 'catmask' and 'domainmask' aren't the same size!")

if np.allclose(np.multiply(catmask,domainmask),0.) and areatype.lower()!="pointlist":
    sys.exit("it looks as if the location specified in 'POINTAREA' is outside of the transposition domain!")

# exclude points that are outside of the transposition domain:
if areatype.lower()=="pointlist" and domain_type.lower()=='irregular':
    keeppoints=np.ones_like(yind_list,dtype='bool')
    for pt in np.arange(0,npoints_list):
        if domainmask[yind_list[pt],xind_list[pt]]==0:
            keeppoints[pt]=False
elif areatype.lower()=="pointlist" and domain_type.lower()=='rectangular':
    keeppoints=np.ones_like(yind_list,dtype='bool')
    for pt in np.arange(0,npoints_list):
        if ptlatlist[pt]>rainprop.subextent[3] or ptlatlist[pt]<rainprop.subextent[2] or ptlonlist[pt]>rainprop.subextent[1] or ptlonlist[pt]<rainprop.subextent[0]:
            keeppoints[pt]=False
            

if areatype.lower()=="pointlist":
    if np.any(keeppoints)==False:
        sys.exit("it looks as if none of the points in specified in 'POINTLIST' are inside the transposition domain!")
    else:
        yind_list=yind_list[keeppoints]
        xind_list=xind_list[keeppoints] 
        ptlatlist=ptlatlist[keeppoints]
        ptlonlist=ptlonlist[keeppoints]
        npoints_list=len(yind_list)   
    
        

#################################################################################
# STEP 1: CREATE STORM CATALOG
#################################################################################

if CreateCatalog: 
    print("reading precipitation files...")
    
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
    #sys.exit("set back!!!")
    #filerange=range(2759,2763)
    
    start = time.time()
    for i in filerange: 
        infile=flist[i]
        inrain,intime,_,_=RainyDay.readnetcdf(infile,inbounds=rainprop.subind,lassiterfile=islassiter)
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
            
            if np.any(np.greater(temparray,np.min(catmax))): # DBW-added this if statement on 10112022. It seems like this should speed things up!
                if domain_type=='irregular':
                    rainmax,ycat,xcat=RainyDay.catalogNumba_irregular(temparray,trimmask,xlen,ylen,maskheight,maskwidth,rainsum,domainmask)
                else:
                    rainmax,ycat,xcat=RainyDay.catalogNumba(temparray,trimmask,xlen,ylen,maskheight,maskwidth,rainsum)
                
                minind=np.argmin(catmax)
                tempmin=catmax[minind]
                if rainmax>tempmin:
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
    RainyDay.writecatalog(scenarioname,catrain,catmax,catx,caty,cattime,latrange,lonrange,catalogname,nstorms,catmask,parameterfile,domainmask,timeresolution=rainprop.timeres)   
        

#################################################################################
# STEP 2: RESAMPLING
#################################################################################
   

    
print("trimming storm catalog...")

origstormsno=np.arange(0,len(catmax),dtype='int32')

if CreateCatalog==False:
    if duration>catrain.shape[1]*rainprop.timeres/60.:
        sys.exit("The specified duration is longer than the length of the storm catalog")

                
# EXLCUDE "BAD" STORMS OR FOR DOING SENSITIVITY ANALYSIS TO PARTICULAR STORMS.  THIS IS PARTICULARLY NEEDED FOR ANY RADAR RAINFALL PRODUCT WITH SERIOUS ARTIFACTS
includestorms=np.ones((len(catmax)),dtype="bool")

try:
    exclude=cardinfo[cardinfo[:,0]=="EXCLUDESTORMS",1][0] 
    exclude=exclude.replace(' ', '')
except Exception:
    exclude='none'
   
if exclude.lower()!="none" and exclude.lower()!="false" and CreateCatalog==False:
	exclude=exclude.split(',')
	for i in range(0,len(exclude)):
		if np.int(exclude[i])-1<len(catmax):
			includestorms[np.int(exclude[i])-1]=False
		else:
			sys.exit("something seems wrong! You are excluding storms that aren't in the catalog.")
elif exclude.lower()!="none" and CreateCatalog:
		sys.exit("You are excluding storms from a new storm catalog, without inspecting them first. This seems like a bad idea.")

    
includestorms[np.isclose(catmax,0.)]=False
        
catrain=catrain[includestorms,:]
catmax=catmax[includestorms]
catx=catx[includestorms]
caty=caty[includestorms]
cattime=cattime[includestorms,:]    
modstormsno=origstormsno[includestorms]  


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
    nstorms_cat=np.sum(catinclude==True)
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
        
else:
    nstorms_cat=len(catmax) 
        
if nstorms<nstorms_cat:
    catrain=catrain[-nstorms:,:]
    catmax=catmax[-nstorms:]
    catx=catx[-nstorms:]
    caty=caty[-nstorms:]
    cattime=cattime[-nstorms:,:]
    nstorms=np.shape(catx)[0]
    modstormsno=modstormsno[-nstorms:]
else:    
    nstorms=np.sum(includestorms==True)
    
catmax=catmax*IntensitySens
catrain=catrain*IntensitySens



#==============================================================================
# If the storm catalog has a different duration than the specified duration, fix it!
# find the max rainfall for the N-hour duration, not the M-day duration
#==============================================================================

durationcheck=60./rainprop.timeres*duration==np.float(catrain.shape[1])

#if (durationcheck==False and durcorrection==True) or (durationcheck==False and DoDiagnostics): 
if (durationcheck==False and durcorrection==False): 
    print("checking storm catalog duration, and adjusting if needed...")
    
    # if you are using a catalog that is longer in duration than your desired analysis, this happens:
    print("Storm catalog duration is longer than the specified duration, finding max precipitation periods for specified duration...")

    # the first run through trims the storm catalog duration to the desired duration, for diagnostic purposes if needed...       
    dur_maxind=np.array((nstorms),dtype='int32')
    dur_x=0
    dur_y=0
    dur_j=0
    temprain=np.zeros((nstorms,int(duration*60/rainprop.timeres),rainprop.subdimensions[0],rainprop.subdimensions[1]),dtype='float32')
    rainsum=np.zeros((rainprop.subdimensions[0]-maskheight+1,rainprop.subdimensions[1]-maskwidth+1),dtype='float32')

    # I think the following commented block was wrong, but haven't fully tested the change-DBW 1/24/2020
#        if durcorrection:
#            catmax_subdur=np.zeros_like(catmax)
#            catx_subdur=np.zeros_like(catx)
#            caty_subdur=np.zeros_like(caty)
#            cattime_subdur=cattime
    
    catmax_subdur=np.zeros_like(catmax)
    catx_subdur=np.zeros_like(catx)
    caty_subdur=np.zeros_like(caty)
    cattime_subdur=cattime

    temptime=np.empty((nstorms,int(duration*60/rainprop.timeres)),dtype='datetime64[m]')
    for i in range(0,nstorms):
        #if (100*((i+1)%(nstorms//10)))==0:
        print('adjusting duration of storms, '+"{0:0.0f}".format(100*(i+1)/nstorms)+'% complete...')
        dur_max=0.
        for j in range(0,catrain.shape[1]-int(duration*60/rainprop.timeres)):
            maxpass=np.nansum(catrain[i,j:j+int(duration*60./rainprop.timeres),:],axis=0)
            
            if domain_type.lower()=='irregular':
                maxtemp,tempy,tempx=RainyDay.catalogNumba_irregular(maxpass,trimmask,xlen,ylen,maskheight,maskwidth,rainsum,domainmask)   
            else:
                maxtemp,tempy,tempx=RainyDay.catalogNumba(maxpass,trimmask,xlen,ylen,maskheight,maskwidth,rainsum)                       
 
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
    
    cattime=cattime_subdur[sind,:]
    catx=catx_subdur[sind]
    caty=caty_subdur[sind]
    catrain=catrain_subdur[sind,:]
    catmax=catmax_subdur[sind]/mnorm*rainprop.timeres/60.        


#==============================================================================
# IF THE USER IS SUPPLYING A DISTRIBUTION FOR THE INTENSITY, NORMALIZE THE FIELDS
# SO THAT THE INTENSITY CAN BE APPLIED PROPERLY
#==============================================================================
if userdistr.all()!=False:   
    print("normalizing precipitation fields...")
    for i in range(0,nstorms):
        tempmax=np.nanmax(np.nansum(catrain[i,:],axis=0))
        catrain[i,:]=catrain[i,:]/tempmax



#==============================================================================
# Create kernel density smoother of transposition probability, even if you don't use it for resampling
#==============================================================================
print("calculating transposition probability map...")

kx,ky=np.meshgrid(np.arange(0,rainprop.subdimensions[1]-maskwidth+1),np.arange(0,rainprop.subdimensions[0]-maskheight+1))
kpositions=np.vstack([ky.ravel(),kx.ravel()])

if domain_type!='irregular':
    # this "checkind" line seems to cause some unrealistic issues with irregular domains, which makes sense
    checkind=np.where(np.logical_and(np.logical_and(caty!=0,catx!=0),np.logical_and(caty!=ylen-1,catx!=xlen-1)))
    invalues=np.vstack([caty[checkind], catx[checkind]])
else:
    invalues=np.vstack([caty, catx])
    

stmkernel=stats.gaussian_kde(invalues,bw_method=RainyDay.my_kde_bandwidth)
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
    cumkernel=np.expand_dims(cumkernel,2)        # do we need this???
elif transpotype=='kernel' or rescaletype!='none':
    smoothsig=5
    # the following is an important fix. All prior kernel-based results are conceptually incorrect-DBW 1/24/2018. But this hasn't been thoroughly vetted!
    if areatype.lower()=="pointlist":
        transpokernel=np.empty(pltkernel.shape,dtype='float64')
        cumkernel=np.empty((tempmask.shape[0],tempmask.shape[1],npoints_list),dtype='float32')
        for pt in range(0,npoints_list):
            phome=pltkernel[yind_list[pt],xind_list[pt]]
            if np.isclose(phome,0.):
                sys.exit("There is a zero probability of transposition to your location of interest. RainyDay's kernel-based non-uniform transposition scheme doesn't know how to cope with this...")
            transpokernel[np.less(pltkernel,phome)]=pltkernel[np.less(pltkernel,phome)]/phome
            transpokernel[np.greater(pltkernel,phome)]=phome/pltkernel[np.greater(pltkernel,phome)]
            transpokernel[np.equal(pltkernel,phome)]=1.0
            transpokernel=RainyDay.mysmoother(transpokernel,sigma=[smoothsig,smoothsig])
            #transpokernel=sp.ndimage.filters.gaussian_filter(transpokernel, [3,3], mode='nearest')
            transpokernel[np.equal(pltkernel,0.)]=0.
            rescaler=np.nansum(transpokernel)
            transpokernel=transpokernel/rescaler  
            
            cumkernel[:,:,pt]=np.array(np.reshape(np.cumsum(transpokernel),(kx.shape)),dtype='float32')
            tempmask[np.equal(tempmask,0.)]=100.
            cumkernel[np.equal(tempmask,100.),pt]=100.         
    else:
        transpokernel=np.empty(pltkernel.shape,dtype='float64')
        phome=pltkernel[ymin,xmin]          # I hope this is right... I think so.
        if np.isclose(phome,0.):
            sys.exit("There is a zero probability of transposition to your location of interest. RainyDay's kernel-based non-uniform transposition scheme doesn't know how to cope with this...")
        transpokernel[np.less(pltkernel,phome)]=pltkernel[np.less(pltkernel,phome)]/phome
        transpokernel[np.greater(pltkernel,phome)]=phome/pltkernel[np.greater(pltkernel,phome)]
        transpokernel[np.equal(pltkernel,phome)]=1.0
        transpokernel=RainyDay.mysmoother(transpokernel,sigma=[smoothsig,smoothsig])
        #transpokernel=sp.ndimage.filters.gaussian_filter(transpokernel, [3,3], mode='nearest')
        transpokernel[np.equal(pltkernel,0.)]=0.
        rescaler=np.nansum(transpokernel)
        transpokernel=transpokernel/rescaler  
        
        cumkernel=np.array(np.reshape(np.cumsum(transpokernel),(kx.shape)),dtype=np.float64)
        tempmask[np.equal(tempmask,0.)]=100.
        cumkernel[np.equal(tempmask,100.)]=100.
        cumkernel=np.expand_dims(cumkernel,2)    # do we need this??? 
        
        
#==============================================================================
# DO YOU WANT TO CREATE DIAGNOSTIC PLOTS?
#==============================================================================

if DoDiagnostics: 
    for f in glob.glob(diagpath+'*.png'):
        os.remove(f)      
    if areatype.lower()=="box":
        from shapely.geometry.polygon import LinearRing
        lons = [boxarea[0], boxarea[0], boxarea[1], boxarea[1]]
        lats = [boxarea[2], boxarea[3], boxarea[3], boxarea[2]]
        ring = LinearRing(list(zip(lons, lats)))
    elif areatype.lower()=="point":
        from shapely.geometry.polygon import LinearRing
        lons = [ptlon-rainprop.spatialres[0]/2., ptlon-rainprop.spatialres[0]/2., ptlon+rainprop.spatialres[0]/2., ptlon+rainprop.spatialres[0]/2.]
        lats = [ptlat-rainprop.spatialres[1]/2., ptlat+rainprop.spatialres[1]/2.,ptlat+rainprop.spatialres[1]/2., ptlat-rainprop.spatialres[1]/2.]
        ring = LinearRing(list(zip(lons, lats)))

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
    
    
    if rainprop.subdimensions[1]>rainprop.subdimensions[0]:
        orientation='horizontal'
    else:
        orientation='vertical'
    proj = ccrs.PlateCarree()

    
    if areatype.lower()=="basin" and os.path.isfile(wsmaskshp):
        try:
            #wmap = shpreader.Reader(wsmaskshp)
            wmap_feature=ShapelyFeature(Reader(wsmaskshp).geometries(),  crs=ccrs.PlateCarree())
        except ValueError:
                print("problem plotting the watershed map; skipping...")
                
    if domain_type.lower()=="irregular" and os.path.isfile(domainshp):
        domain_feature=ShapelyFeature(Reader(domainshp).geometries(),  crs=ccrs.PlateCarree())
#    if BaseMap.lower()!='none':
#        try: 
#            sys.exit("fix this")
#        except ValueError:
#                print("problem plotting the basemap; skipping...")   
                
                
    outerextent=np.array(rainprop.subextent,dtype='float32')
    coast_10m = cfeature.NaturalEarthFeature("physical", "land", "10m", edgecolor="k", facecolor="0.8")
    
    states_provinces = cfeature.NaturalEarthFeature(
        category='cultural',
        name='admin_1_states_provinces_lines',
        scale='50m',
        facecolor='none')
    
    # PLOT STORM OCCURRENCE PROBABILITIES-there is a problem with the "alignment of the raster and the storm locations
    print("     Creating storm probability map...")
    #plot_kernel=np.column_stack([pltkernel,np.zeros((pltkernel.shape[0],catmask.shape[1]-pltkernel.shape[1]))])
    #plot_kernel=np.row_stack([plot_kernel,np.zeros((catmask.shape[0]-pltkernel.shape[0],plot_kernel.shape[1]))])

    #padleft=math.floor(maskwidth/2)-1
    #padright=math.ceil(maskwidth/2)
    padleft=0
    padright=maskwidth-1
    
    plot_kernel=np.column_stack([np.zeros((pltkernel.shape[0],padleft)),pltkernel,np.zeros((pltkernel.shape[0],padright))])

    #padtop=math.floor(maskheight/2)
    #padbottom=math.ceil(maskheight/2)-1
    padtop=0
    padbottom=maskheight-1
    
    
    plot_kernel=np.row_stack([np.zeros((padtop,plot_kernel.shape[1])),plot_kernel,np.zeros((padbottom,plot_kernel.shape[1]))])

    xplot_kernel=xr.Dataset(
        data_vars=dict(plot_kernel=(["y","x"],plot_kernel)),
        coords=dict(
            lat=(["y"],latrange),
            lon=(["x"],lonrange)),
        attrs=dict(description="diagnostic plotting of the storm probability density"),
    )
    
    
    fig = plt.figure(figsize=(figsizex,figsizey))
    ax=plt.axes(projection=proj)
    #ax.set_extent(outerextent)
    if areatype.lower()=="basin" and os.path.isfile(wsmaskshp):
        ax.add_feature(wmap_feature,edgecolor="red",facecolor="None")
    elif areatype.lower()=="box" or areatype.lower()=="point":
        ax.add_geometries([ring], facecolor='none',edgecolor='black',crs=ccrs.PlateCarree())
        
    if domain_type.lower()=="irregular" and os.path.isfile(domainshp):
        ax.add_feature(domain_feature,edgecolor="black",facecolor="None")
    xplot_kernel.plot_kernel.plot(x="lon",y="lat",yincrease=True,cmap='Reds',cbar_kwargs={'orientation':orientation,'label':"Probability of storm occurrence [-]"})
    #xplot_kernel.pltkernel.plot(x="lon",y="lat",yincrease=True,cmap='Reds',cbar_kwargs={'orientation':orientation,'label':"Probability of storm occurrence [-]"})
        
    plt.scatter(lonrange[catx]+(maskwidth/2)*rainprop.spatialres[0],latrange[caty]-(maskheight/2)*rainprop.spatialres[1],s=catmax/2,facecolors='k',edgecolors='none',alpha=0.75)

    ax.add_feature(states_provinces)
    ax.set_xticks(np.linspace(outerextent[0],outerextent[1],2))
    lon_formatter = cticker.LongitudeFormatter()
    ax.xaxis.set_major_formatter(lon_formatter)

# Define the yticks for latitude
    ax.set_yticks(np.linspace(outerextent[2],outerextent[3],2))
    lat_formatter = cticker.LatitudeFormatter()
    ax.set(xlabel=None,ylabel=None)
    ax.yaxis.set_major_formatter(lat_formatter)
    
    ax.set_title('Prob. of storm occurrence from\n'+catalogname.split('/')[-1]+'\nNOTE: The prob. map may not extend to lower/right edges. That is not a mistake!')
    #ax.axes.set_title(xlabel=None)
    
    plt.savefig(diagpath+'ProbabilityOfStorms.png',dpi=250)
    plt.close()   
    
    
    # PLOT AVERAGE STORM RAINFALL
    print ("     Creating mean precipitation map...")
    outerextent=np.array(rainprop.subextent,dtype='float32')
    avgrain=np.nansum(catrain,axis=(0,1))/nstorms*rainprop.timeres/60.
    xplot_avgrain=xr.Dataset(
        data_vars=dict(avgrain=(["y","x"],avgrain)),
        coords=dict(
            lat=(["y"],latrange),
            lon=(["x"],lonrange)),
        attrs=dict(description="diagnostic plotting of the mean storm total rainfall"),
    )
    
    fig = plt.figure(figsize=(figsizex,figsizey))
    ax=plt.axes(projection=proj)
    #ax.set_extent(outerextent)
    if areatype.lower()=="basin" and os.path.isfile(wsmaskshp):
        ax.add_feature(wmap_feature,edgecolor="red",facecolor="None")
    elif areatype.lower()=="box" or areatype.lower()=="point":
        ax.add_geometries([ring], facecolor='none',edgecolor='red',crs=ccrs.PlateCarree())
        
    if domain_type.lower()=="irregular" and os.path.isfile(domainshp):
        ax.add_feature(domain_feature,edgecolor="black",facecolor="None")

    xplot_avgrain.avgrain.plot(x="lon",y="lat",yincrease=True,cmap='Blues',cbar_kwargs={'orientation':orientation,'label':"Mean Storm Total precipitation [mm]"})
    plt.scatter(lonrange[catx]+(maskwidth/2)*rainprop.spatialres[0],latrange[caty]-(maskheight/2)*rainprop.spatialres[1],s=catmax/2,facecolors='k',edgecolors='none',alpha=0.75)

    ax.add_feature(states_provinces)
    #ax.add_feature(coast_10m)
    ax.set_xticks(np.linspace(outerextent[0],outerextent[1],2))
    lon_formatter = cticker.LongitudeFormatter()
    ax.xaxis.set_major_formatter(lon_formatter)

# Define the yticks for latitude
    ax.set_yticks(np.linspace(outerextent[2],outerextent[3],2))
    lat_formatter = cticker.LatitudeFormatter()
    ax.set(xlabel=None,ylabel=None)
    ax.yaxis.set_major_formatter(lat_formatter)
    
    ax.set_title('Mean of Storm Catalog Rainfall from\n'+catalogname.split('/')[-1])
    #ax.axes.set_title(xlabel=None)
    
    plt.savefig(diagpath+'MeanStormRain.png',dpi=250)
    plt.close()   
    
    
    
    
    # PLOT STORM RAINFALL Standard deviation
    print("     Creating precipitation standard deviation map...")
    stdrain=np.nanstd(np.nansum(catrain,axis=1)*rainprop.timeres/60.,axis=0)
    
    xplot_stdrain=xr.Dataset(
        data_vars=dict(stdrain=(["y","x"],stdrain)),
        coords=dict(
            lat=(["y"],latrange),
            lon=(["x"],lonrange)),
        attrs=dict(description="diagnostic plotting of the std dev. of storm total rainfall"),
    )
    
    fig = plt.figure(figsize=(figsizex,figsizey))
    ax=plt.axes(projection=proj)
    #ax.set_extent(outerextent)
    if areatype.lower()=="basin" and os.path.isfile(wsmaskshp):
        ax.add_feature(wmap_feature,edgecolor="red", facecolor='none')
    elif areatype.lower()=="box" or areatype.lower()=="point":
        ax.add_geometries([ring], facecolor='none',edgecolor='red',crs=ccrs.PlateCarree())
        
    if domain_type.lower()=="irregular" and os.path.isfile(domainshp):
        ax.add_feature(domain_feature,edgecolor="black",facecolor="None")
    xplot_stdrain.stdrain.plot(x="lon",y="lat",yincrease=True,cmap='Greens',cbar_kwargs={'orientation':orientation,'label':"Standard Deviation Storm Total precipitation [mm]"})
    plt.scatter(lonrange[catx]+(maskwidth/2)*rainprop.spatialres[0],latrange[caty]-(maskheight/2)*rainprop.spatialres[1],s=catmax/2,facecolors='k',edgecolors='none',alpha=0.75)

    ax.add_feature(states_provinces)
    #ax.add_feature(coast_10m)
    ax.set_xticks(np.linspace(outerextent[0],outerextent[1],2))
    lon_formatter = cticker.LongitudeFormatter()
    ax.xaxis.set_major_formatter(lon_formatter)

# Define the yticks for latitude
    ax.set_yticks(np.linspace(outerextent[2],outerextent[3],2))
    lat_formatter = cticker.LatitudeFormatter()
    ax.set(xlabel=None,ylabel=None)
    ax.yaxis.set_major_formatter(lat_formatter)
    
    ax.set_title('Std. Dev. of Storm Catalog Rainfall from\n'+catalogname.split('/')[-1])
    #ax.axes.set_title(xlabel=None)
    
    plt.savefig(diagpath+'StdDevStormRain.png',dpi=250)
    plt.close()   
    
    
    
    
    # PLOT CATALOG CDF/PDF
    print ("     Creating storm catalog PDF and CDF...")
    fig = plt.figure()
    ax  = fig.add_subplot(111)
    fig.set_size_inches(6,4)
    n, bins, patches = ax.hist(catmax, nstorms, density=True, histtype='stepfilled',cumulative=True, label='Empirical',color='lightgray')
    ax.annotate('Largest Storm: '+"{0:0.1f}".format(np.nanmax(catmax))+" mm\n"+str(cattime[-1,-1]), xy=(np.nanmax(catmax), 1),  xycoords='data',
            xytext=(0.75, 0.75), textcoords='axes fraction',
            arrowprops=dict(facecolor='black', shrink=0.05),
            horizontalalignment='right', verticalalignment='top',
            )
    ax.set_title('CDF of '+str(nstorms)+' Storm Catalog Entries')
    ax.set_xlabel('Storm Total Precipitation [mm]')
    ax.set_ylabel('Nonexceedance Probability [-]')
    plt.savefig(diagpath+'CDF_StormCatalog.png',dpi=250)
    plt.close()
    
    
    # PLOT EACH STORM
    print ("     Creating storm precipitation maps and hyetographs (this could take a while)...")
    
    for i in range(0,nstorms): 
        temprain=np.nansum(catrain[i,:],axis=0)*rainprop.timeres/60.
        
        xplot_temprain=xr.Dataset(
            data_vars=dict(temprain=(["y","x"],temprain)),
            coords=dict(
                lat=(["y"],latrange),
                lon=(["x"],lonrange)),
            attrs=dict(description="diagnostic plotting of storm total rainfall"),
        )
        
        fig = plt.figure(figsize=(figsizex,figsizey))
        ax=plt.axes(projection=proj)
        #ax.set_extent(outerextent)
        if areatype.lower()=="basin" and os.path.isfile(wsmaskshp):
            ax.add_feature(wmap_feature,edgecolor="red", facecolor='none')
        elif areatype.lower()=="box" or areatype.lower()=="point":
            ax.add_geometries([ring], facecolor='none',edgecolor='red',crs=ccrs.PlateCarree())
        if domain_type.lower()=="irregular" and os.path.isfile(domainshp):
            ax.add_feature(domain_feature,edgecolor="black",facecolor="None")
            
        xplot_temprain.temprain.plot(x="lon",y="lat",yincrease=True,cmap='Blues',cbar_kwargs={'orientation':orientation,'label':"Storm Total precipitation [mm]"})

        ax.add_feature(states_provinces)
        #ax.add_feature(coast_10m)
        ax.set_xticks(np.linspace(outerextent[0],outerextent[1],2))
        lon_formatter = cticker.LongitudeFormatter()
        ax.xaxis.set_major_formatter(lon_formatter)
    
    # Define the yticks for latitude
        ax.set_yticks(np.linspace(outerextent[2],outerextent[3],2))
        lat_formatter = cticker.LatitudeFormatter()
        ax.set(xlabel=None,ylabel=None)
        ax.yaxis.set_major_formatter(lat_formatter)
        ax.set_title('Storm '+str(i+1)+': '+str(cattime[i,-1])+' Storm Total Rainfall\nMax Precipitation:'+str(round(catmax[i]))+' mm @ Lat/Lon:'+"{:6.1f}".format(latrange[caty[i]]-(maskheight/2+maskheight%2)*rainprop.spatialres[0])+u'\N{DEGREE SIGN}'+','+"{:6.1f}".format(lonrange[catx[i]]+(maskwidth/2+maskwidth%2)*rainprop.spatialres[0])+u'\N{DEGREE SIGN}')

        
        plt.savefig(diagpath+'Storm'+str(i+1)+'_'+str(cattime[i,-1]).split('T')[0]+'.png',dpi=250)
        plt.close()     
    
    
    # create hyetograph diagnostic plots:
    try:
        maplist=glob.glob(diagpath+'Hyetograph_Storm*.png')
        for filePath in maplist:
            try:
                os.remove(filePath)
            except:
                print("Error while deleting file : ", filePath)
            
    except Exception:
        pass

    for i in range(0,nstorms):    
        temprain=np.nansum(catrain[i,:,caty[i]:caty[i]+maskheight,catx[i]:catx[i]+maskwidth],axis=(1,2))/mnorm*rainprop.timeres/60.
        fig = plt.figure()
        ax  = fig.add_subplot(111)
        fig.set_size_inches(6,4)
        ax.bar(np.arange(0,temprain.shape[0]*rainprop.timeres/60.,rainprop.timeres/60.), temprain)
        ax.set_title('Storm '+str(i+1)+': '+str(cattime[i,-1])+' Hyetograph\nMax Precipitation:'+str(round(catmax[i]))+' mm @ Lat/Lon:'+"{:6.1f}".format(latrange[caty[i]]-(maskheight/2+maskheight%2)*rainprop.spatialres[0])+u'\N{DEGREE SIGN}'+','+"{:6.1f}".format(lonrange[catx[i]]+(maskwidth/2+maskwidth%2)*rainprop.spatialres[0])+u'\N{DEGREE SIGN}')
        ax.set_xlabel('Time [hours]')
        ax.set_ylabel('Precipitation Rate [mm/hr]')
        plt.savefig(diagpath+'Hyetograph_Storm'+str(i+1)+'_'+str(cattime[i,-1]).split('T')[0]+'.png',dpi=250)
        plt.close()


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
        #ncounts[ncounts==0]=1
        if calctype.lower()=='npyear' and lrate<nperyear:   
            sys.exit("You specified to write multiple storms per year, but you specified a number that is too large relative to the resampling rate!")
    elif samplingtype.lower()=='negbinom':
        sys.exit("Sorry, the negative binomial resampling isn't set up yet :(")
        _,yrscount=np.unique(cattime[:,-1].astype('datetime64[Y]').astype(int)+1970,return_counts=True)
        if len(yrscount)<nyears:
            yrscount=np.append(yrscount,np.ones(nyears-len(yrscount),dtype='int32'))
        rparam=np.mean(yrscount)*np.mean(yrscount)/(np.var(yrscount)-np.mean(yrscount))   
        pparam=1.-np.mean(yrscount)/np.var(yrscount)
        ncounts=np.random.negative_binomial(rparam,pparam,size=(nsimulations,nrealizations)) 
        if calctype.lower()=='npyear' and np.mean(yrscount)<nperyear :   
            sys.exit("You specified to write multiple storms per year, but you specified a number that is too large relative to the resampling rate!")
             	
    else:
        _,yrscount=np.unique(cattime[:,-1].astype('datetime64[Y]').astype(int)+1970,return_counts=True)
        if len(yrscount)<nyears:
            yrscount=np.append(yrscount,np.ones(nyears-len(yrscount),dtype='int32'))
        ncounts=np.random.choice(yrscount,(nsimulations,nrealizations),replace=True)   
        if calctype.lower()=='npyear' and np.mean(yrscount)<nperyear :   
            sys.exit("You specified to write multiple storms per year, but you specified a number that is too large relative to the resampling rate!")
        #ncounts[ncounts==0]=1
            
    whichstorms=np.empty((np.nanmax(ncounts),ncounts.shape[0],ncounts.shape[1]),dtype='int32')
    whichstorms[:]=-9999
    
    if rotation==True:
        randangle=(maxangle-minangle)*np.random.random_sample(((np.nanmax(ncounts),ncounts.shape[0],ncounts.shape[1])))+minangle
        
        angbins=np.linspace(minangle,maxangle,nanglebins)
        angs=math.pi/180*angbins
        anglebins=np.digitize(randangle.ravel(),angbins).reshape(np.nanmax(ncounts),ncounts.shape[0],ncounts.shape[1])
    
    
    # DOES THIS PROPERLY HANDLE STORM EXCLUSIONS???  I think so...
    for i in range(0,np.nanmax(ncounts)):
        whichstorms[i,ncounts>=i+1]=np.random.randint(0,nstorms,(len(ncounts[ncounts>=i+1]))) # why was this previously "nstorms-1"??? Bug?
    
    
    # the next three lines were commented out when adding the "pointlist" option
    #whichrain=np.zeros((whichstorms.shape),dtype='float32')
    #whichx=np.zeros((whichstorms.shape),dtype='int32')
    #whichy=np.zeros((whichstorms.shape),dtype='int32')
    
    if areatype.lower()=="pointlist":
        whichx=np.zeros((whichstorms.shape[0],whichstorms.shape[1],whichstorms.shape[2],npoints_list),dtype='int32')
        whichy=np.zeros((whichstorms.shape[0],whichstorms.shape[1],whichstorms.shape[2],npoints_list),dtype='int32')
        whichrain=np.zeros((whichstorms.shape[0],whichstorms.shape[1],whichstorms.shape[2],npoints_list),dtype='float32')
    else:
        whichx=np.zeros((whichstorms.shape[0],whichstorms.shape[1],whichstorms.shape[2],1),dtype='int32')
        whichy=np.zeros((whichstorms.shape[0],whichstorms.shape[1],whichstorms.shape[2],1),dtype='int32')  
        whichrain=np.zeros((whichstorms.shape[0],whichstorms.shape[1],whichstorms.shape[2],1),dtype='float32')
        whichstep=np.zeros((whichstorms.shape[0],whichstorms.shape[1],whichstorms.shape[2],1),dtype='int32')
        
    if durcorrection:
        whichtimeind=np.zeros((whichstorms.shape),dtype='float32')
    
    
    if transpotype=='uniform' and domain_type=='irregular':
        domainmask[-maskheight:,:]=0.
        domainmask[:,-maskwidth:]=0.
        xmask,ymask=np.meshgrid(np.arange(0,domainmask.shape[1],1),np.arange(0,domainmask.shape[0],1))
        xmask=xmask[np.equal(domainmask,True)]
        ymask=ymask[np.equal(domainmask,True)]
        
    if rescaletype=='stochastic' or rescaletype=='deterministic' or rescaletype=='dimensionless':
        whichmultiplier=np.empty_like(whichrain)
        whichmultiplier[:]=np.nan

        
    #==============================================================================
    # If you're using intensity-dependent resampling, get ready for it!
    #==============================================================================


    if rescaletype=='stochastic' or rescaletype=='deterministic':
        smoothsig=5
        
        print("reading in precipitation intensity data...")
        intenserain,_,intenselat,intenselon=RainyDay.readintensityfile(rescalingfile)
        intensemask=np.equal(np.sum(intenserain,axis=0),0.)
        intenserain[:,intensemask]=np.nan
        int_xmin=np.abs(intenselon-rainprop.bndbox[0]).argmin()
        int_ymin=np.abs(intenselat-rainprop.bndbox[3]).argmin()
        int_xmax=np.abs(intenselon-rainprop.bndbox[1]).argmin()
        int_ymax=np.abs(intenselat-rainprop.bndbox[2]).argmin()
        #int_xmax=np.abs(intenselon-rainprop.bndbox[1]).argmin()+1
        #int_ymax=np.abs(intenselat-rainprop.bndbox[2]).argmin()+1
        intenserain=intenserain[:,int_ymin:int_ymax,int_xmin:int_xmax]
        intenselat=intenselat[int_ymin:int_ymax]
        intenselon=intenselon[int_xmin:int_xmax]
        
        #intenserain[np.equal(intenserain,0.)]=np.nan
        nintstorms=np.min((intenserain.shape[0],2*nyears))
        intenserain=intenserain[-nintstorms:,:]

        if np.array_equal(intenselat,latrange)==False or np.array_equal(intenselon,lonrange)==False:  
            intensegridx,intensengridy=np.meshgrid(intenselon,intenselat)        
            ingrid_intense=np.column_stack((intensegridx.flatten(),intensengridy.flatten())) 
            grid_out=np.column_stack((ingridx.flatten(),ingridy.flatten())) 
            delaunay=sp.spatial.qhull.Delaunay(ingrid_intense)
            tempintense=np.empty((intenserain.shape[0],ingridx.shape[0],ingridx.shape[1]),dtype='float32')
            for i in range(0,intenserain.shape[0]):
                interp=sp.interpolate.LinearNDInterpolator(delaunay,intenserain[i,:].flatten(),fill_value=np.nan)
                tempintense[i,:]=np.reshape(interp(grid_out),ingridx.shape)
            intenserain=tempintense 
        
        #intenserain[:,np.equal(domainmask,0.)]=np.nan
      

        # the stochastic multiplier approach uses the log of the rainfall:
        intenserain=np.log(intenserain)
        hometemp=np.nansum(np.multiply(intenserain,catmask),axis=(1,2))/mnorm
        xlen_wmask=rainprop.subdimensions[1]-maskwidth+1
        ylen_wmask=rainprop.subdimensions[0]-maskheight+1
                
        
        # there is some goofy numba stuff below...
        if maskheight>1 or maskwidth>1:
            tempintense=np.empty((intenserain.shape[0],ylen_wmask,xlen_wmask),dtype='float32')
            intenserain=RainyDay.intenseloop(intenserain,tempintense,xlen_wmask,ylen_wmask,maskheight,maskwidth,trimmask,mnorm,domainmask)
        intensecorr=np.empty((ylen_wmask,xlen_wmask),dtype='float32')    
        intensecorr=RainyDay.intense_corrloop(intenserain,intensecorr,hometemp,xlen_wmask,ylen_wmask,mnorm,domainmask)
            
        #homemean=np.mean(homerain)
        #homestd=np.std(homerain)
        intensemean=np.mean(intenserain,axis=0)
        intensestd=np.std(intenserain,axis=0) 
        
        intensemean=RainyDay.mysmoother(intensemean,sigma=[smoothsig,smoothsig])
        intensestd=RainyDay.mysmoother(intensestd,sigma=[smoothsig,smoothsig])
        intensecorr=RainyDay.mysmoother(intensecorr,sigma=[smoothsig,smoothsig])
        
        if maskheight>1 or maskwidth>1:
            homemean=np.nansum(np.multiply(intensemean,catmask[:-maskheight+1,:-maskwidth+1]),axis=(0,1))/mnorm
            homestd=np.nansum(np.multiply(intensestd,catmask[:-maskheight+1,:-maskwidth+1]),axis=(0,1))/mnorm        
        else:    
            homemean=np.nansum(np.multiply(intensemean,catmask),axis=(0,1))/mnorm
            homestd=np.nansum(np.multiply(intensestd,catmask),axis=(0,1))/mnorm
        
        # just in case you don't have any data to inform the rescaling:
        intensemean[np.isneginf(intensemean)]=homemean
        intensestd[np.isneginf(intensestd)]=0.
        intensecorr[np.isneginf(intensecorr)]=1.0
        
        intensemean[np.isnan(intensemean)]=homemean
        intensestd[np.isnan(intensestd)]=0.
        intensecorr[np.isnan(intensecorr)]=1.0
        
    elif rescaletype=='dimensionless':
        print("reading in precipitation map for dimensionless SST...")
        if '.nc' in rescalingfile:
            sys.exit('need to set this up')
            #intenserain,_,intenselat,intenselon=RainyDay.readintensityfile(rescalingfile)
        elif '.asc' in rescalingfile:
            asciigrid,ncols,nrows,xllcorner,yllcorner,cellsize=RainyDay.read_arcascii(rescalingfile)
            dlsstarea=[xllcorner,xllcorner+ncols*cellsize,yllcorner,yllcorner+nrows*cellsize]
            atlasgridx,atlasgridy=np.meshgrid(np.arange(dlsstarea[0],dlsstarea[1]-cellsize/10.,cellsize),np.arange(dlsstarea[3],dlsstarea[2]+cellsize/10.,-cellsize))
            atlas14_domain=np.column_stack((atlasgridx.flatten(),atlasgridy.flatten())) 
        
            delaunay=sp.spatial.qhull.Delaunay(atlas14_domain)
            interp=sp.interpolate.LinearNDInterpolator(delaunay,asciigrid.flatten(),fill_value=np.nan)
            
            grid_out=np.column_stack((ingridx.flatten(),ingridy.flatten())) 
            atlas_regridded=np.reshape(interp(grid_out),ingridx.shape) 
            atlas_regridded=np.log(atlas_regridded)
            if areatype.lower()!='pointlist':
                hometemp=np.nansum(np.multiply(atlas_regridded,catmask))/mnorm
            else:
                hometemp=np.nanmean(atlas_regridded[domainmask==True])
            atlas_regridded[np.isnan(atlas_regridded)]=hometemp
   
        else:
            sys.exit('Unrecognized file format provided for dimensionless SST')   
    else:
        pass
        
   
    # here is the main resampling and transposition loop
    for i in np.arange(0,nstorms):
        print('Resampling and transposing storm '+str(i+1)+' out of '+str(nstorms)+' ('"{0:0.0f}".format(100*(i+1)/nstorms)+'%)')
        # UNIFORM RESAMPLING
        if transpotype=='uniform' and domain_type=='rectangular':
            whichx[whichstorms==i,0]=np.random.randint(0,np.int(rainprop.subdimensions[1])-maskwidth+1,len(whichx[whichstorms==i]))
            whichy[whichstorms==i,0]=np.random.randint(0,np.int(rainprop.subdimensions[0])-maskheight+1,len(whichy[whichstorms==i]))
     
        # KERNEL-BASED AND INTENSITY-BASED RESAMPLING (ALSO NEEDED FOR IRREGULAR TRANSPOSITION DOMAINS)
        elif transpotype=='kernel':
            rndloc=np.array(np.random.random_sample(len(whichx[whichstorms==i])),dtype='float32')
            tempx=np.empty((len(rndloc)),dtype='int32')
            tempy=np.empty((len(rndloc)),dtype='int32')
            for pt in np.arange(0,whichx.shape[3]):
                whichx[whichstorms==i,pt],whichy[whichstorms==i,pt]=RainyDay.numbakernel_fast(rndloc,cumkernel[:,:,pt],tempx,tempy,rainprop.subdimensions[1])

        if transpotype=='uniform' and domain_type=='irregular':
            rndloc=np.random.randint(0,np.sum(np.equal(domainmask,True)),np.sum(whichstorms==i))
            for pt in np.arange(0,whichx.shape[3]):
                whichx[whichstorms==i,pt]=xmask[rndloc].reshape(len(xmask[rndloc]))
                whichy[whichstorms==i,pt]=ymask[rndloc].reshape(len(ymask[rndloc]))
        
        # SET UP MANUAL PDF RESAMPLING
        elif transpotype=='manual':  
            sys.exit("not configured for manually supplied pdf yet!")
    
        if durcorrection:
            passrain=np.array(RainyDay.rolling_sum(catrain[i,:], int(duration*60/rainprop.timeres)),dtype='float32')
            
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
                if rescaletype=='stochastic':
                    sys.exit("WARNING: rotation + intensity-based resampling = not tested!")
                    #whichrain[np.logical_and(whichstorms==i,anglebins==cbin)]=RainyDay.SSTalt(tpass,whichx[np.logical_and(whichstorms==i,anglebins==cbin)],whichy[np.logical_and(whichstorms==i,anglebins==cbin)],trimmask,maskheight,maskwidth,intense_data=intense_dat,durcheck=durcorrection)*rainprop.timeres/60./mnorm
                #else:
                    #whichrain[np.logical_and(whichstorms==i,anglebins==cbin)]=RainyDay.SSTalt(tpass,whichx[np.logical_and(whichstorms==i,anglebins==cbin)],whichy[np.logical_and(whichstorms==i,anglebins==cbin)],trimmask,maskheight,maskwidth,durcheck=durcorrection)*rainprop.timeres/60./mnorm     
                binctr=binctr+1
        else:
            for pt in np.arange(0,whichx.shape[3]):
                if rescaletype=='stochastic' and areatype.lower()!='pointlist' and areatype.lower!='point':                    
                    temprain,whichmultiplier[whichstorms==i,pt],whichstep=RainyDay.SSTalt(passrain,whichx[whichstorms==i,pt],whichy[whichstorms==i,pt],trimmask,maskheight,maskwidth,intensemean=intensemean,intensestd=intensestd,intensecorr=intensecorr,homemean=homemean,homestd=homestd,durcheck=durcorrection)
                    whichrain[whichstorms==i,pt]=temprain*rainprop.timeres/60./mnorm    
                elif rescaletype=='deterministic' and areatype.lower()!='pointlist' and areatype.lower()!='point':
                    temprain,whichmultiplier[whichstorms==i,pt],whichstep=RainyDay.SSTalt(passrain,whichx[whichstorms==i,pt],whichy[whichstorms==i,pt],trimmask,maskheight,maskwidth,intensemean=intensemean,homemean=homemean,durcheck=durcorrection)
                    whichrain[whichstorms==i,pt]=temprain*rainprop.timeres/60./mnorm 
                elif areatype.lower()!='pointlist' and areatype.lower()!='point' and rescaletype=='none':
                    temprain,whichstep[whichstorms==i,pt]=RainyDay.SSTalt(passrain,whichx[whichstorms==i,pt],whichy[whichstorms==i,pt],trimmask,maskheight,maskwidth,durcheck=durcorrection)                
                    whichrain[whichstorms==i,pt]=temprain*rainprop.timeres/60./mnorm 
                elif areatype.lower()=='pointlist':
                    if rescaletype=='deterministic':
                        homemeanpt=intensemean[yind_list[pt],xind_list[pt]]
                        temprain,whichmultiplier[whichstorms==i,pt],_=RainyDay.SSTalt_singlecell(passrain,whichx[whichstorms==i,pt],whichy[whichstorms==i,pt],trimmask,1,1,durcheck=durcorrection,intensemean=intensemean,homemean=homemeanpt)
                    elif rescaletype=='dimensionless':
                        homemeanpt=atlas_regridded[yind_list[pt],xind_list[pt]]
                        temprain,whichmultiplier[whichstorms==i,pt],_=RainyDay.SSTalt_singlecell(passrain,whichx[whichstorms==i,pt],whichy[whichstorms==i,pt],trimmask,1,1,durcheck=durcorrection,intensemean=atlas_regridded,homemean=homemeanpt)
                    elif rescaletype=='stochastic':
                        homemeanpt=intensemean[yind_list[pt],xind_list[pt]]
                        homestdpt=intensestd[yind_list[pt],xind_list[pt]]
                         
                        intensemeanpt=np.mean(intenserain,axis=0)
                        intensestdpt=np.std(intenserain,axis=0) 
                        
                        intensemeanpt=RainyDay.mysmoother(intensemeanpt,sigma=[smoothsig,smoothsig])
                        intensestdpt=RainyDay.mysmoother(intensestdpt,sigma=[smoothsig,smoothsig])
                         
                        intensecorrpt=np.empty((ylen_wmask,xlen_wmask),dtype='float32')    
                        intensecorrpt=RainyDay.intense_corrloop(intenserain,intensecorrpt,intenserain[:,yind_list[pt],xind_list[pt]],xlen_wmask,ylen_wmask,mnorm,domainmask)    
                        intensecorrpt=RainyDay.mysmoother(intensecorrpt,sigma=[smoothsig,smoothsig])
                         
                        intensecorrpt[np.isneginf(intensecorrpt)]=1.0
                    
                        intensemeanpt[np.isnan(intensemeanpt)]=homemeanpt
                        intensestdpt[np.isnan(intensestdpt)]=0.
                        intensecorrpt[np.isnan(intensecorrpt)]=1.0
                        temprain,whichmultiplier[whichstorms==i,pt],_=RainyDay.SSTalt_singlecell(passrain,whichx[whichstorms==i,pt],whichy[whichstorms==i,pt],trimmask,maskheight,maskwidth,intensemean=intensemeanpt,intensestd=intensestdpt,intensecorr=intensecorrpt,homemean=homemeanpt,homestd=homestdpt,durcheck=durcorrection)
                    else:
                        temprain,_=RainyDay.SSTalt_singlecell(passrain,whichx[whichstorms==i,pt],whichy[whichstorms==i,pt],trimmask,maskheight,maskwidth,durcheck=durcorrection)

                    whichrain[whichstorms==i,pt]=temprain*rainprop.timeres/60.   

                   
                elif areatype.lower()=='point':
                    
                    if rescaletype=='deterministic':
                        homemeanpt=intensemean[ymin,xmin]
                        temprain,whichmultiplier[whichstorms==i,pt],whichstep[whichstorms==i,pt]=RainyDay.SSTalt_singlecell(passrain,whichx[whichstorms==i,pt],whichy[whichstorms==i,pt],trimmask,1,1,durcheck=durcorrection,intensemean=intensemean,homemean=homemeanpt)
                        whichrain[whichstorms==i,pt]=temprain*rainprop.timeres/60.
                    elif rescaletype=='dimensionless':
                        homemeanpt=atlas_regridded[ymin,xmin]
                        temprain,whichmultiplier[whichstorms==i,pt],whichstep[whichstorms==i,pt]=RainyDay.SSTalt_singlecell(passrain,whichx[whichstorms==i,pt],whichy[whichstorms==i,pt],trimmask,1,1,durcheck=durcorrection,intensemean=atlas_regridded,homemean=homemeanpt)
                        whichrain[whichstorms==i,pt]=temprain*rainprop.timeres/60.
                    elif rescaletype=='stochastic':
                        homemeanpt=intensemean[ymin,xmin]
                        homestdpt=intensemean[ymin,xmin]
                        temprain,whichmultiplier[whichstorms==i,pt],whichstep[whichstorms==i,pt]=RainyDay.SSTalt(passrain,whichx[whichstorms==i,pt],whichy[whichstorms==i,pt],trimmask,1,1,intensemean=intensemean,intensestd=intensestd,intensecorr=intensecorr,homemean=homemeanpt,homestd=homestdpt,durcheck=durcorrection)
                        whichrain[whichstorms==i,pt]=temprain*rainprop.timeres/60. 
                    else:
                        temprain,whichstep[whichstorms==i,pt]=RainyDay.SSTalt_singlecell(passrain,whichx[whichstorms==i,pt],whichy[whichstorms==i,pt],trimmask,1,1,durcheck=durcorrection)
                        whichrain[whichstorms==i,pt]=temprain*rainprop.timeres/60. 

    if areatype.lower()=='pointlist' or areatype.lower()=='point':
        if len(arfval)==1 and np.isclose(arfval[0],1.):
            sortrain=whichrain*arfval
        else:
            if len(arfval)==1:
                arfrand=np.random.exponential(arfval[0]-1.,size=whichrain.shape)+1.
                arflimit=sp.stats.expon.ppf(0.90,1./(arfval[0]-1.))+1.
                arfmed=sp.stats.expon.ppf(0.5,1./(arfval[0]-1.))+1.
            elif len(arfval)==2:
                arfrand=np.random.gamma(shape=arfval[0],scale=arfval[1],size=whichrain.shape)
                arflimit=sp.stats.gamma.ppf(0.90,a=arfval[0],scale=arfval[1])
                arfmed=sp.stats.gamma.ppf(0.5,a=arfval[0],scale=arfval[1])
            #arfrand[arfrand>1.5]=1.0
            arfrand[arfrand>arflimit]=arfmed
            whichrain=np.multiply(whichrain,arfrand)
            
            
    # modified by DBW to account for k=0 situations, 9/22/2022
    nostorm_index=np.equal(whichstorms,-9999)
    whichrain[nostorm_index]=-9999.
    whichx[nostorm_index]=-9999
    whichy[nostorm_index]=-9999
    try:
        whichstep[nostorm_index]=-9999
    except NameError:
        pass
    try:
        whichmultiplier[nostorm_index]=-9999.
    except NameError:
        pass  
    try:
        whichtimeind[nostorm_index]=-9999
    except NameError:
        pass  
            
    
    # HERE ARE THE ANNUAL MAXIMA!!!
    if areatype.lower()=="pointlist":
        sortrain=np.empty((whichrain.shape[1],whichrain.shape[2],npoints_list),dtype='float32')
        #if rescaletype=='stochastic' or rescaletype=='deterministic':
        #    sortmultiplier=np.empty((whichrain.shape[1],whichrain.shape[2],npoints_list),dtype='float32')
        for pt in np.arange(0,whichx.shape[3]):
            
            if calctype.lower()=='ams':
                maxrain=np.nanmax(whichrain[:,:,:,pt],axis=0)
                
            elif calctype.lower()=='pds':
                temprain=np.squeeze(whichrain[:,:,:,pt])
                temppds=temprain.reshape(-1, temprain.shape[-1])
                maxrain=np.sort(temppds,axis=0)[-nsimulations:,:]
            
            
            # PULL OUT THE CORRESPONDING TRANSPOSITION INFORMATION
            maxind=np.nanargmax(whichrain[:,:,:,pt],axis=0)
            
            #added for k=0 catching, DBW 9/22/22
            maxind[np.equal(maxrain,-9999.)]=-9999
        
            
            # THIS ISN'T VERY ELEGANT
            maxx=np.empty((maxind.shape),dtype="int32")
            maxy=np.empty((maxind.shape),dtype="int32")
            maxstorm=np.empty((maxind.shape),dtype="int32")
            maxx[:]=-9999
            maxy[:]=-9999
            maxstorm[:]=-9999
            #if rescaletype=='stochastic' or rescaletype=='deterministic':
            #    maxmultiplier=np.empty((maxind.shape),dtype="float32") 
                
            for i in range(0,np.max(ncounts)):
                maxx[maxind==i]=whichx[i,maxind==i,pt]
                maxy[maxind==i]=whichy[i,maxind==i,pt]
                maxstorm[maxind==i]=whichstorms[i,maxind==i]
             #   if rescaletype=='stochastic' or rescaletype=='deterministic':
             #       maxmultiplier[maxind==i]=whichmultiplier[i,maxind==i,pt]
            
            
            # RANK THE STORMS BY INTENSITY AND ASSIGN RETURN PERIODS
            exceedp=np.linspace(1,1./nsimulations,nsimulations)
            
            returnperiod=1/exceedp
            sortind=np.argsort(maxrain,axis=0)
            sortrain[:,:,pt]=np.sort(maxrain,axis=0)
            #if rescaletype=='stochastic' or rescaletype=='deterministic':
            #    sortmultiplier[:,:,pt]=maxmultiplier[sortind]
                
                
        if alllevels==False:
            reducedlevind=[]
            for i in range(0,len(speclevels)):
                reducedlevind.append(RainyDay.find_nearest(returnperiod,speclevels[i]))  
            
            returnperiod=returnperiod[reducedlevind]
            sortrain=sortrain[reducedlevind,:]
            exceedp=exceedp[reducedlevind]
                
    else:
        if calctype=='ams' or calctype=='pds':          # this isn't very elegant!
            if calctype.lower()=='ams':
                maxrain=np.nanmax(whichrain[:,:,:,pt],axis=0)
                maxind=np.nanargmax(whichrain,axis=0)
            elif calctype.lower()=='pds':
                temprain=np.squeeze(whichrain)
                temppds=temprain.reshape(-1, temprain.shape[-1])
                maxrain=np.sort(temppds,axis=0)[-nsimulations:,:]
                maxind=np.nanargmax(whichrain,axis=0)
            
            #added for k=0 catching, DBW 9/22/22
            maxind[np.equal(maxrain,-9999.)]=-9999
            
            #elif calctype.lower()=='npyear':
            #    temprain=np.squeeze(whichrain)
            #    maxrain=np.sort(temprain,axis=0)
            #    maxind=np.argsort(temprain,axis=0)
                
                #maxrain=np.sort(temprain,axis=0)[-5:,:]
                #maxind=np.argsort(temprain,axis=0)[-5:,:]
      
            
            # HERE THE OPTIONAL USER SPECIFIED INTENSITY DISTRIBUTION IS APPLIED    
            if userdistr.all()!=False:
                rvs=sp.stats.genextreme.rvs(userdistr[2],loc=userdistr[0],scale=userdistr[1],size=maxrain.shape).astype('float32')
                maxrain=maxrain*rvs
                maxrain[np.equal(maxind,-9999)]=-9999.
                
            # PULL OUT THE CORRESPONDING TRANSPOSITION INFORMATION
            
            
            # THIS ISN'T VERY ELEGANT
            maxx=np.empty((maxind.shape),dtype="int32")
            maxy=np.empty((maxind.shape),dtype="int32")
            maxx[:]=-9999
            maxy[:]=-9999
            maxstorm=np.empty((maxind.shape),dtype="int32")
            maxstorm[:]=-9999
            if arfcorrection:
                maxstep=np.empty((maxind.shape),dtype="int32")
                maxstep[:]=-9999
            if rotation:
                maxangles=np.empty((maxind.shape),dtype="float32")
                sortangle=np.empty((maxind.shape),dtype="float32")
                maxangles[:]=-9999.
                sortangle[:]=-9999.
            if rescaletype=='stochastic' or rescaletype=='deterministic' or rescaletype=='dimensionless':
                maxmultiplier=np.empty((maxind.shape),dtype="float32") 
                sortmultiplier=np.empty((maxind.shape),dtype="float32")
                maxmultiplier[:]=-9999.
                sortmultiplier[:]=-9999.
                
            for i in range(0,np.max(ncounts)):
                maxx[maxind==i]=np.squeeze(whichx[i,np.squeeze(maxind==i)])
                maxy[maxind==i]=np.squeeze(whichy[i,np.squeeze(maxind==i)])
                maxstorm[maxind==i]=np.squeeze(whichstorms[i,np.squeeze(maxind==i)])
                if arfcorrection:
                    maxstep[maxind==i]=np.squeeze(whichstep[i,np.squeeze(maxind==i)])
                
                if rotation:
                    maxangles[maxind==i]=randangle[i,maxind==i]
                if rescaletype=='stochastic' or rescaletype=='deterministic' or rescaletype=='dimensionless':
                    maxmultiplier[maxind==i]=np.squeeze(whichmultiplier[i,maxind==i])
    #            elif calctype.lower()=='npyear':
    #                sys.exit("having problems here")
    #                for stm in range(0,nperyear):
    #                    maxx[maxind==i]=np.squeeze(whichx[i,np.squeeze(maxind==i)])
    #                    maxy[stm,maxind==i]=np.squeeze(whichy[i,np.squeeze(maxind==i)])
    #                    maxstorm[stm,maxind==i]=np.squeeze(whichstorms[i,np.squeeze(maxind==i)])
    #                    if rotation:
    #                        print("Warning: We haven't tested ROTATION with NPERYEAR!")
    #                        maxangles[stm,maxind==i]=randangle[stm,i,maxind==i]
    #                    if rescaletype=='stochastic' or rescaletype=='deterministic' or rescaletype=='dimensionless':
    #                        maxmultiplier[stm,maxind==i]=np.squeeze(whichmultiplier[stm,i,maxind==i])
                    
                
            
            # RANK THE STORMS BY INTENSITY AND ASSIGN RETURN PERIODS
            exceedp=np.linspace(1,1./nsimulations,nsimulations)
            returnperiod=1/exceedp
            #rp_pds=1./(1.-np.exp(-1./(returnperiod)))
            #rp_pds=1./np.log(returnperiod/(returnperiod-1))
            sortind=np.argsort(maxrain,axis=0)
            sortrain=np.sort(maxrain,axis=0)
            sortx=np.empty((maxind.shape),dtype="int32")
            sorty=np.empty((maxind.shape),dtype="int32")
            sortstorms=np.empty((maxind.shape),dtype="int32")
            if arfcorrection:
                sortstep=np.empty((maxind.shape),dtype="int32")
            
            
            for i in range(0,nrealizations):
                sortx[:,i]=maxx[sortind[:,i],i]
                sorty[:,i]=maxy[sortind[:,i],i]
                sortstorms[:,i]=maxstorm[sortind[:,i],i]
                if arfcorrection:
                    sortstep[:,i]=maxstep[sortind[:,i],i]
                if rotation:
                    sortangle[:,i]=maxangles[sortind[:,i],i]
                if rescaletype=='stochastic' or rescaletype=='deterministic' or rescaletype=='dimensionless':
                    sortmultiplier[:,i]=maxmultiplier[sortind[:,i],i]
            
                
            # FIND THE TIMES:
            if arfcorrection==False:
                sorttimes=np.zeros((maxind.shape[0],maxind.shape[1],cattime.shape[1]),dtype="datetime64[m]")
            else:       # using ARFANALYSIS
                sorttimes=np.zeros((maxind.shape[0],maxind.shape[1],int(duration*60/rainprop.timeres)),dtype="datetime64[m]")
            whichorigstorm=np.zeros((maxind.shape[0],maxind.shape[1]),dtype='int32')
            for i in range(0,nstorms):
                if np.sum(np.squeeze(sortstorms==i))>0:
                    if arfcorrection==False:
                        sorttimes[np.squeeze(sortstorms==i),:]=cattime[i,:]
                        
                    else:  # using ARFANALYSIS
                        tstep=sortstep[np.squeeze(sortstorms==i),:]
                        if len(tstep)>1:
                            tempstep=np.squeeze(sortstep[np.squeeze(sortstorms==i),:])
                            temptime=np.empty((tempstep.shape[0],int(duration*60/rainprop.timeres)),dtype="datetime64[m]")
                            for j in range(0,tempstep.shape[0]):
                                temptime[j,:]=cattime[i,tempstep[j]:tempstep[j]+int(duration*60/rainprop.timeres)]
                        else:
                            tempstep=sortstep[np.squeeze(sortstorms==i),:][0]
                            temptime=np.empty((tempstep.shape[0],int(duration*60/rainprop.timeres)),dtype="datetime64[m]")
                            for j in range(0,tempstep.shape[0]):
                                temptime[j,:]=cattime[i,tempstep[j]:tempstep[j]+int(duration*60/rainprop.timeres)]
    
                        sorttimes[np.squeeze(sortstorms==i),:]=temptime
                    whichorigstorm[np.squeeze(sortstorms==i)]=modstormsno[i]+1
                else:
                    continue
                
                
            if alllevels==False:
                reducedlevind=[]
                for i in range(0,len(speclevels)):
                    reducedlevind.append(RainyDay.find_nearest(returnperiod,speclevels[i]))  
                
                returnperiod=returnperiod[reducedlevind]
                sortrain=sortrain[reducedlevind,:]
                sortstorms=sortstorms[reducedlevind,:]
                sorttimes=sorttimes[reducedlevind,:]
                exceedp=exceedp[reducedlevind]
                sortx=sortx[reducedlevind,:]
                sorty=sorty[reducedlevind,:]
                if arfcorrection:
                    sortstep=sortstep[reducedlevind,:]

                whichorigstorm=whichorigstorm[reducedlevind,:]
                if rotation:    
                    sortangle=sortangle[reducedlevind,:]
                if rescaletype=='stochastic' or rescaletype=='deterministic' or rescaletype=='dimensionless':        
                    sortmultiplier=sortmultiplier[reducedlevind,:]
            
        
        #################################################################################
        # STEP 2a (OPTIONAL): Find the single storm maximized storm rainfall-added DBW 7/19/2017
        #################################################################################    
        
        nanmask=deepcopy(trimmask)
        nanmask[np.isclose(nanmask,0.)]=np.nan
        nanmask[np.isclose(nanmask,0.)==False]=1.0
        
        if deterministic:
            print("finding maximizing precipitation...")
            
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
    
    if calctype!='npyear':
        print("preparing frequency analysis...")
    
        if areatype.lower()=='pointlist':
            spreadmean=np.nanmean(sortrain,1)
            
            if spreadtype=='ensemble':
                spreadmin=np.nanmin(sortrain,axis=1)
                spreadmax=np.nanmax(sortrain,axis=1)    
            else:
                spreadmin=np.percentile(sortrain,(100-quantilecalc)/2,axis=1)
                spreadmax=np.percentile(sortrain,quantilecalc+(100-quantilecalc)/2,axis=1)
            
            if spreadmean.shape[0]!=returnperiod.shape[0] or spreadmax.shape[0]!=returnperiod.shape[0] or spreadmean.shape[1]!=ptlatlist.shape[0]:
                sys.exit("There is some dimension inconsistency in the pointlist scheme!")
            
            #fmean=open(FreqFile_mean,'w')
            #fmin=open(FreqFile_min,'w')
            #fmax=open(FreqFile_max,'w')
            
            #fmean.write('#prob.exceed,returnperiod,meanrain\n'+ptlistname+'\n')
            #fmax.write('#prob.exceed,returnperiod,maxrain\n'+ptlistname+'\n')
            #fmin.write('#prob.exceed,returnperiod,minrain\n'+ptlistname+'\n')
    
            freqanalysis_mean=np.column_stack((exceedp,returnperiod,spreadmean))
            freqanalysis_min=np.column_stack((exceedp,returnperiod,spreadmin))
            freqanalysis_max=np.column_stack((exceedp,returnperiod,spreadmax))
            
            outlat_line=np.append([-999.,-999.],ptlatlist)
            outlon_line=np.append([-999.,-999.],ptlonlist)
            coordline=np.row_stack((outlat_line,outlon_line))
            freqanalysis_mean=np.row_stack((coordline,freqanalysis_mean))
            freqanalysis_min=np.row_stack((coordline,freqanalysis_min))
            freqanalysis_max=np.row_stack((coordline,freqanalysis_max))
            
            np.savetxt(FreqFile_mean,freqanalysis_mean,delimiter=',',header='prob.exceed,returnperiod,meanrain',fmt='%6.2f',comments='#',footer=ptlistname)
            np.savetxt(FreqFile_min,freqanalysis_min,delimiter=',',header='prob.exceed,returnperiod,minrain',fmt='%6.2f',comments='#',footer=ptlistname)
            np.savetxt(FreqFile_max,freqanalysis_max,delimiter=',',header='prob.exceed,returnperiod,maxrain',fmt='%6.2f',comments='#',footer=ptlistname)
            
        else:
            if spreadtype=='ensemble':
                spreadmin=np.nanmin(sortrain,axis=1)
                spreadmax=np.nanmax(sortrain,axis=1)    
            else:
                spreadmin=np.percentile(sortrain,(100-quantilecalc)/2,axis=1)
                spreadmax=np.percentile(sortrain,quantilecalc+(100-quantilecalc)/2,axis=1)
        
            freqanalysis=np.column_stack((exceedp,returnperiod,spreadmin,np.nanmean(sortrain,1),spreadmax))
            
            np.savetxt(FreqFile,freqanalysis,delimiter=',',header='prob.exceed,returnperiod,minrain,meanrain,maxrain',fmt='%6.2f',comments='')
            
            import matplotlib.patches as mpatches
            from matplotlib.font_manager import FontProperties
            from matplotlib import pyplot as plt
           # warnings.filterwarnings('ignore')
            fontP = FontProperties()
            fontP.set_size('xx-small')
            fig, ax = plt.subplots(1)
            line1, = plt.plot(exceedp[exceedp<=0.5], RainyDay.np.nanmean(sortrain,1)[exceedp<=0.5], lw=1, label='Average', color='blue')
        
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
            ax.set_xlabel('Annual Exceed. Prob. [-]\n1/(Return Period) [year]')
            ax.set_ylabel('Precip. Depth [mm]')
            ax.set_yscale('log')
            ax.set_xscale('log')
            plt.gca().invert_xaxis()
            ax.grid()
            plt.tight_layout()
            plt.savefig(fullpath+'/'+scenarioname+'_FrequencyAnalysis.png',dpi=250)
            plt.close('all')
    else:
        print("You selected NPERYEAR greater than 1. RainyDay isn't configured to do a rainfall frequency analysis for this, but will write output scenarios. If you want a rainfall frequency analysis, set NPERYEAR to '1','false' or omit it from the .sst file!")
       
    
    #################################################################################
    # STEP 4 (OPTIONAL): WRITE RAINFALL SCENARIOS
    #################################################################################

    subrangelat=latrange[ymin:ymax+1]
    subrangelon=lonrange[xmin:xmax+1]
    
    if deterministic:
        RainyDay.writemaximized(scenarioname,wd+'/'+scenarioname+'/'+scenarioname+'_maximizedstorm.nc',maxstm_rain,maxstm_avgrain,maxstm_ts,max_trnsx,max_trnsy,maxstm_time,subrangelat,subrangelon)
       
    if Scenarios and calctype!='npyear':
        print("writing spacetime precipitation scenarios...")
        
        # if desired, this will "pad" the beginning of the scenarios to add a spin-up period.  Not recommended to use long spin-ups, due to memory constraints
        # this scheme basically creates two storm catalogs and mixes them up!
        if prependrain==True:
            print("prepending precipitation for model spin-up...")
            outerextent=np.array(rainprop.subextent,dtype='float32') 
            if CreateCatalog==False:
                # need to generate a file list since it hasn't already been done
                flist,nyears=RainyDay.createfilelist(inpath,includeyears,excludemonths)
                if len(flist)==0:
                    sys.exit("couldn't prepend the precipitation files for spinup period because the input files weren't available.") 
        
            tlen=np.int(1440*pretime/rainprop.timeres)
            
            precat=np.zeros((catrain.shape[0],tlen,catrain.shape[2],catrain.shape[3]),dtype='float32')
    
            starttime=np.empty((cattime.shape[0],tlen),dtype='datetime64[m]')
            starttime[:,0]=np.subtract(cattime[:,0],np.timedelta64(np.int32(pretime)*1440,'m'))
            for i in range(0,nstorms):
                starttime[i,:]=np.arange(starttime[i,0],cattime[i,0],np.timedelta64(int(rainprop.timeres),'m'))   
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
                    print('Pre-pending precipitation with file '+tlist[j])
                    inrain,intime,_,_=RainyDay.readnetcdf(tlist[j],inbounds=rainprop.subind,lassiterfile=islassiter)
                    inrain[inrain<0.]=np.nan
                    
                    for k in range(0,int(24*60/rainprop.timeres)):
                        if np.in1d(intime[k],cattime[i,:])[0]: 
                            cattime[i,j*int(24*60/rainprop.timeres):int(j*24*60/rainprop.timeres+24*60/rainprop.timeres)]=intime
                            precat[i,j*int(24*60/rainprop.timeres):int(j*24*60/rainprop.timeres+24*60/rainprop.timeres),:]=np.reshape(inrain,(int(24*60/rainprop.timeres),rainprop.subdimensions[0],rainprop.subdimensions[1]))
        else:
            precat=np.zeros((catrain.shape[0],0,catrain.shape[2],catrain.shape[3]),dtype='float32')
            tlen=0
            
        if alllevels:
            minind=RainyDay.find_nearest(returnperiod,RainfallThreshYear)
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
            if rescaletype=='stochastic' or rescaletype=='deterministic' or rescaletype=='dimensionless':
                writemultiplier=sortmultiplier[minind:,:]
            if arfcorrection:
                writesteps=sortstep[minind:,:]
        else:
            writemax=sortrain
            writex=sortx
            writey=sorty
            writestorm=sortstorms
            writeperiod=returnperiod
            writeexceed=exceedp
            writetimes=sorttimes
            if arfcorrection:
                writesteps=sortstep
            if rotation:
                writeangle=sortangle
                binwriteang=np.digitize(writeangle.ravel(),angbins).reshape(writeangle.shape)
            if rescaletype=='stochastic' or rescaletype=='deterministic' or rescaletype=='dimensionless':
                writemultiplier=sortmultiplier
    
        for rlz in range(0,nrealizations):
            print("writing scenarios for realization "+str(rlz+1)+"/"+str(nrealizations))
            
            # this statement is only really needed if you are prepending rainfall, but it is fast so who cares?
            if arfcorrection!=True:
                outtime=np.empty((writetimes.shape[0],cattime.shape[1]),dtype='datetime64[m]')
                unqstm=np.unique(writestorm[:,rlz])
                for i in range(0,len(unqstm)):
                    outtime[np.squeeze(writestorm[:,rlz])==unqstm[i],:]=cattime[unqstm[i],:]
            else:
                # I'm not sure why the above statements for outtime are needed, rather than the following line... I should have commented my code better back in 2015 or whenever I wrote that!
                outtime=writetimes[:,rlz,:]
            
            if rotation:
                if rescaletype=='stochastic':
                    sys.exit("not tested!")
                elif arfcorrection==True:
                    sys.exit("not tested!")
                else:
                    outrain=RainyDay.SSTspin_write_v2(catrain,writex[:,rlz],writey[:,rlz],writestorm[:,rlz],nanmask,maskheight,maskwidth,precat,cattime[:,-1],rainprop,rlzanglebin=binwriteang[:,rlz],delarray=delarray,spin=prependrain,flexspin=False,samptype=transpotype,cumkernel=cumkernel,rotation=rotation,domaintype=domain_type)
            else:
                if rescaletype=='stochastic' or rescaletype=='deterministic' or rescaletype=='dimensionless':
                    outrain=RainyDay.SSTspin_write_v2(catrain,np.squeeze(writex[:,rlz]),np.squeeze(writey[:,rlz]),np.squeeze(writestorm[:,rlz]),nanmask,maskheight,maskwidth,precat,cattime[:,-1],rainprop,spin=prependrain,flexspin=False,samptype=transpotype,cumkernel=cumkernel,rotation=rotation,domaintype=domain_type)
                    for rl in range(0,outrain.shape[0]):
                        outrain[rl,tlen:,:]=outrain[rl,tlen:,:]*writemultiplier[rl,rlz,0]
                else:
                    outrain=RainyDay.SSTspin_write_v2(catrain,np.squeeze(writex[:,rlz]),np.squeeze(writey[:,rlz]),np.squeeze(writestorm[:,rlz]),nanmask,maskheight,maskwidth,precat,cattime[:,-1],rainprop,spin=prependrain,flexspin=False,samptype=transpotype,cumkernel=cumkernel,rotation=rotation,domaintype=domain_type)
            
            outrain[:,:,np.isclose(trimmask,0.)]=-9999.               # this line produced problems in CUENCAS CONVERSIONS :(
            writename=WriteName+'_SSTrealization'+str(rlz+1)+'.nc'
            
            if arfcorrection:
                temprain=np.empty((outrain.shape[0],writetimes.shape[2],outrain.shape[2],outrain.shape[3]),dtype='float32')
                for j in range(0,temprain.shape[0]):
                    #sys.exit("need to check this carefully!")
                    temprain[j,:]=outrain[j,writesteps[j,rlz,0]:writesteps[j,rlz,0]+int(duration*60/rainprop.timeres),:,:]
                outrain=temprain
                
            
            #print "need to write angles to the realization files"
            RainyDay.writerealization(scenarioname,rlz,nrealizations,writename,outrain,writemax[:,rlz],np.squeeze(writestorm[:,rlz]),writeperiod,np.squeeze(writex[:,rlz]),np.squeeze(writey[:,rlz]),outtime,subrangelat,subrangelon,whichorigstorm[:,rlz])
    
    elif calctype=='npyear':
        # this code is adapted from Guo Yu's version of RainyDay
        # this isn't pretty, but it works. It doesn't have all the functionality of the "1 per year" scenarios above.
        print("Writing top N precipitation scenarios")
            
        if rescaletype=='none':
            whichmultiplier=np.ones_like(whichrain)

        for i in range(0,nrealizations):
            outrain_large = np.zeros((nsimulations,nperyear,int(catduration),maskheight,maskwidth),dtype='float32')
            outrain_large[:] = -9999.
            outtime_large =  np.empty((nsimulations,nperyear,int(catduration)),dtype='datetime64[m]')
            outtime_large[:] =  np.datetime64(datetime(1900,1,1,0,0,0))
            
            rlz_rain = whichrain[:,:,i]*whichmultiplier[:,:,i]
            rlz_whichstorms = whichstorms[:,:,i]
            rlz_order = np.argsort(-rlz_rain,axis=0)
            rlz_order[rlz_whichstorms<0]=-9999
        
            rlz_whichx = whichx[:,:,i]
            rlz_whichy = whichy[:,:,i]
        
            for j in range(0,nsimulations):
                # loop through synthetic year
                if np.sum(rlz_order[:,j]>=0) >= nperyear:
                    n_largest = nperyear
                else:
                    n_largest = np.sum(rlz_order[:,j]>=0)
                
                for k in range(0,n_largest):
                    
                    y1 = rlz_whichy[rlz_order[k,j],j][0][0]
                    y2 = rlz_whichy[rlz_order[k,j],j][0][0]+ maskheight 
                    x1 = rlz_whichx[rlz_order[k,j],j][0][0]
                    x2 = rlz_whichx[rlz_order[k,j],j][0][0]+ maskwidth
                    
                    sst_storm=rlz_whichstorms[rlz_order[k,j],j][0]
                    if sst_storm<=0:
                        continue
                    
                    SST_rain = np.array(catrain[sst_storm,:,y1:y2,x1:x2])*whichmultiplier[rlz_order[k,j],j,i,0]
                    SST_rain[:,np.isclose(trimmask,0.)]=-9999.
                    
                    outrain_large[j,k,:,:,:] = SST_rain[:]
                    outtime_large[j,k,:] = cattime[sst_storm,:]
            
            writename=WriteName+'_SSTrealization'+str(i+1)+'.nc'
            RainyDay.writerealization_nperyear(scenarioname,writename,i,nperyear,nrealizations,outrain_large,outtime_large,subrangelat,subrangelon,rlz_order,nsimulations)
    
            #### THIS COMMENTED SECTION CAN BE DELETED ONCE 'writerealization_nperyear()' is tested
            # dataset=Dataset(filename, 'w', format='NETCDF4')
        
            # # create dimensions
        
            # outlats=dataset.createDimension('outlat',len(subrangelat))
            # outlons=dataset.createDimension('outlon',len(subrangelon))
            # time=dataset.createDimension('time',outtime_large.shape[2])
            # nyears=dataset.createDimension('nyears',nsimulations)
            # topN=dataset.createDimension('topN',nperyear)
        
        
            # # create variables
            # times=dataset.createVariable('time',np.float64, ('nyears','topN','time'))
            # latitudes=dataset.createVariable('latitude',np.float32, ('outlat'))
            # longitudes=dataset.createVariable('longitude',np.float32, ('outlon'))
            # rainrate=dataset.createVariable('rainrate',np.float32,('nyears','topN','time','outlat','outlon'),zlib=True,complevel=4,least_significant_digit=2)
            # top_event=dataset.createVariable('top_event',np.int16, ('nyears'))
            # # Global Attributes
        
            # dataset.history = 'Created ' + str(datetime.now())
           
            # # Variable Attributes (time since 1970-01-01 00:00:00.0 in numpys)
        
            # rainrate.units = 'mm/h'
            # times.units = 'minutes since 1970-01-01 00:00.0'
            # times.calendar = 'gregorian'
        
            # # fill the netcdf file
            # latitudes[:]=subrangelat
            # longitudes[:]=subrangelon
            # rainrate[:]=outrain_large 
            # times[:]=outtime_large
            # n_evnet = np.nansum(rlz_order>=0,axis=0)
            # n_evnet[n_evnet>=nperyear]=nperyear
            # top_event[:]= n_evnet
           
            # dataset.close()
else:
    print("skipping the frequency analysis!")
    
import time     # reimporting due to goofy issue with nperyear writing above      
end = time.time()   
print("RainyDay has successfully finished!\n")
print("Elapsed time: "+"{0:0.2f}".format((end - start)/60.)+" minutes")

   
#################################################################################
# THE END
#################################################################################

