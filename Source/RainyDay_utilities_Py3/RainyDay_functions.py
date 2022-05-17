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
# THIS DOCUMENT CONTAINS VARIOUS FUNCTIONS NEEDED TO RUN RainyDay
#==============================================================================
                                               
import os
import sys
import numpy as np
import scipy as sp
import glob
import math     
from datetime import datetime, date, time, timedelta      
import time
from copy import deepcopy
import fiona

import cartopy
from matplotlib.patches import Polygon   
from scipy import stats
from netCDF4 import Dataset, num2date, date2num
import rasterio
from rasterio.transform import from_origin
from rasterio.shutil import delete
from rasterio.mask import mask
import pandas as pd
from numba import prange,jit

import shapely
import geopandas as gp


from scipy.stats import norm
from scipy.stats import lognorm

# plotting stuff, really only needed for diagnostic plots
import matplotlib.pyplot as plt
import matplotlib
from matplotlib.colors import LogNorm 

import subprocess
try:
    os.environ.pop('PYTHONIOENCODING')
except KeyError:
    pass

import warnings
warnings.filterwarnings("ignore")

from numba.types import int32,int64,float32,uint32
import linecache
GEOG="+proj=longlat +ellps=WGS84 +datum=WGS84 +no_defs"



# =============================================================================
# Smoother that is compatible with nan values. Adapted from https://stackoverflow.com/questions/18697532/gaussian-filtering-a-image-with-nan-in-python
# =============================================================================

def mysmoother(inarray,sigma=[3,3]):
    if len(sigma)!=len(inarray.shape):
        sys.exit("there seems to be a mismatch between the sigma dimension and the dimension of the array you are trying to smooth")
    V=inarray.copy()
    V[np.isnan(inarray)]=0.
    VV=sp.ndimage.gaussian_filter(V,sigma=sigma)

    W=0.*inarray.copy()+1.
    W[np.isnan(inarray)]=0.
    WW=sp.ndimage.gaussian_filter(W,sigma=sigma)
    outarray=VV/WW
    outarray[np.isnan(inarray)]=np.nan
    return outarray

def my_kde_bandwidth(obj, fac=1):     # this 1.5 choice is completely subjective :(
    #We use Scott's Rule, multiplied by a constant factor
    return np.power(obj.n, -1./(obj.d+4)) * fac



def convert_3D_2D(geometry):
    '''
    Takes a GeoSeries of 3D Multi/Polygons (has_z) and returns a list of 2D Multi/Polygons
    '''
    new_geo = []
    for p in geometry:
        if p.has_z:
            if p.geom_type == 'Polygon':
                lines = [xy[:2] for xy in list(p.exterior.coords)]
                new_p = shapely.geometry.Polygon(lines)
                new_geo.append(new_p)
            elif p.geom_type == 'MultiPolygon':
                new_multi_p = []
                for ap in p:
                    lines = [xy[:2] for xy in list(ap.exterior.coords)]
                    new_p = shapely.geometry.Polygon(lines)
                    new_multi_p.append(new_p)
                new_geo.append(shapely.geometry.MultiPolygon(new_multi_p))
    return new_geo



#==============================================================================
# LOOP TO DO SPATIAL SEARCHING FOR MAXIMUM RAINFALL LOCATION AT EACH TIME STEP
# THIS IS THE CORE OF THE STORM CATALOG CREATION TECHNIQUE
#==============================================================================
    
#def catalogweave(temparray,trimmask,xlen,ylen,maskheight,maskwidth,rainsum):
#    rainsum[:]=0.
#    code= """
#        #include <stdio.h>
#        int i,j,x,y;
#        for (x=0;x<xlen;x++) {
#            for (y=0;y<ylen;y++) {
#                for (j=0;j<maskheight;j++) {
#                    for (i=0;i<maskwidth;i++) {
#                        rainsum(y,x)=rainsum(y,x)+temparray(y+j,x+i)*trimmask(j,i);                     
#                    }                               
#                }
#            }                      
#        }
#    """
#    vars=['temparray','trimmask','xlen','ylen','maskheight','maskwidth','rainsum']
#    sp.weave.inline(code,vars,type_converters=converters.blitz,compiler='gcc')
#    rmax=np.nanmax(rainsum)
#    wheremax=np.where(rainsum==rmax)
#    return rmax, wheremax[0][0], wheremax[1][0]
#    


def catalogAlt(temparray,trimmask,xlen,ylen,maskheight,maskwidth,rainsum,domainmask):
    rainsum[:]=0.
    for i in range(0,(ylen)*(xlen)):
        y=i//xlen
        x=i-y*xlen
        #print x,
        rainsum[y,x]=np.nansum(np.multiply(temparray[(y):(y+maskheight),(x):(x+maskwidth)],trimmask))
    #wheremax=np.argmax(rainsum)
    rmax=np.nanmax(rainsum)
    wheremax=np.where(rainsum==rmax)
    return rmax, wheremax[0][0], wheremax[1][0]

def catalogAlt_irregular(temparray,trimmask,xlen,ylen,maskheight,maskwidth,rainsum,domainmask):
    rainsum[:]=0.
    for i in range(0,(ylen)*(xlen)):
        y=i//xlen
        x=i-y*xlen
        #print x,y
        if np.any(np.equal(domainmask[y+maskheight/2,x:x+maskwidth],1.)) and np.any(np.equal(domainmask[y:y+maskheight,x+maskwidth/2],1.)):
            rainsum[y,x]=np.nansum(np.multiply(temparray[(y):(y+maskheight),(x):(x+maskwidth)],trimmask))
        else:
            rainsum[y,x]=0.
    #wheremax=np.argmax(rainsum)
    rmax=np.nanmax(rainsum)
    wheremax=np.where(rainsum==rmax)
    
    return rmax, wheremax[0][0], wheremax[1][0]



@jit(nopython=True,fastmath=True)  
def catalogNumba_irregular(temparray,trimmask,xlen,ylen,maskheight,maskwidth,rainsum,domainmask):
    rainsum[:]=0.
    halfheight=int32(np.ceil(maskheight/2))
    halfwidth=int32(np.ceil(maskwidth/2))
    for i in range(0,ylen*xlen):
        y=i//xlen
        x=i-y*xlen
        #print x,y
        if np.any(np.equal(domainmask[y+halfheight,x:x+maskwidth],1.)) and np.any(np.equal(domainmask[y:y+maskheight,x+halfwidth],1.)):
            rainsum[y,x]=np.nansum(np.multiply(temparray[y:(y+maskheight),x:(x+maskwidth)],trimmask))
        else:
            rainsum[y,x]=0.
    #wheremax=np.argmax(rainsum)
    rmax=np.nanmax(rainsum)
    wheremax=np.where(np.equal(rainsum,rmax))
    return rmax, wheremax[0][0], wheremax[1][0]


@jit(nopython=True)
def catalogNumba(temparray,trimmask,xlen,ylen,maskheight,maskwidth,rainsum):
    rainsum[:]=0.
    for i in range(0,(ylen)*(xlen)):
        y=i//xlen
        x=i-y*xlen
        #print x,y
        rainsum[y,x]=np.nansum(np.multiply(temparray[(y):(y+maskheight),(x):(x+maskwidth)],trimmask))

    #wheremax=np.argmax(rainsum)
    rmax=np.nanmax(rainsum)
    wheremax=np.where(np.equal(rainsum,rmax))
    return rmax, wheremax[0][0], wheremax[1][0]


@jit(nopython=True)
def DistributionBuilder(intenserain,tempmax,xlen,ylen,checksep):
    for y in np.arange(0,ylen):
        for x in np.arange(0,xlen):
            if np.any(checksep[:,y,x]):
                #fixind=np.where(checksep[:,y,x]==True)
                for i in np.arange(0,checksep.shape[0]):
                    if checksep[i,y,x]==True:
                        fixind=i
                        break
                if tempmax[y,x]>intenserain[fixind,y,x]:
                    intenserain[fixind,y,x]=tempmax[y,x]
                    checksep[:,y,x]=False
                    checksep[fixind,y,x]=True
                else:
                    checksep[fixind,y,x]=False
            elif tempmax[y,x]>np.min(intenserain[:,y,x]):
                fixind=np.argmin(intenserain[:,y,x])
                intenserain[fixind,y,x]=tempmax[y,x]
                checksep[fixind,y,x]=True
    return intenserain,checksep

# slightly faster numpy-based version of above
def DistributionBuilderFast(intenserain,tempmax,xlen,ylen,checksep):
    minrain=np.min(intenserain,axis=0)
    if np.any(checksep):
        
        flatsep=np.any(checksep,axis=0)
        minsep=np.argmax(checksep[:,flatsep],axis=0)
        
        islarger=np.greater(tempmax[flatsep],intenserain[minsep,flatsep])
        if np.any(islarger):
            intenserain[minsep,flatsep][islarger]=tempmax[flatsep][islarger]
            checksep[:]=False
            checksep[minsep,flatsep]=True
        else:
            checksep[minsep,flatsep]=False
    elif np.any(np.greater(tempmax,minrain)):
        #else:
        fixind=np.greater(tempmax,minrain)
        minrainind=np.argmin(intenserain,axis=0)
        
        intenserain[minrainind[fixind],fixind]=tempmax[fixind]
        checksep[minrainind[fixind],fixind]=True
    return intenserain,checksep



#def SSTalt(passrain,sstx,ssty,trimmask,maskheight,maskwidth,intense_data=False):
#    rainsum=np.zeros((len(sstx)),dtype='float32')
#   nreals=len(rainsum)
#
#    for i in range(0,nreals):
#        rainsum[i]=np.nansum(np.multiply(passrain[(ssty[i]) : (ssty[i]+maskheight) , (sstx[i]) : (sstx[i]+maskwidth)],trimmask))
#    return rainsum


@jit(fastmath=True)
def SSTalt(passrain,sstx,ssty,trimmask,maskheight,maskwidth,intensemean=None,intensestd=None,intensecorr=None,homemean=None,homestd=None,durcheck=False):
    maxmultiplier=1.5
    
    rainsum=np.zeros((len(sstx)),dtype='float32')
    whichstep=np.zeros((len(sstx)),dtype='int32')
    nreals=len(rainsum)
    nsteps=passrain.shape[0]
    multiout=np.empty_like(rainsum)
    if (intensemean is not None) and (homemean is not None):
        domean=True
    else:
        domean=False

    if (intensestd is not None) and (intensecorr is not None) and (homestd is not None):
        #rquant=np.random.random_integers(5,high=95,size=nreals)/100.
        rquant=np.random.random_sample(size=nreals)
        doall=True
    else:
        doall=False
        rquant=np.nan
        
    
    if durcheck==False:
        exprain=np.expand_dims(passrain,0)
    else:
        exprain=passrain
        

    for k in range(0,nreals):
        y=int(ssty[k])
        x=int(sstx[k])
        if np.all(np.less(exprain[:,y:y+maskheight,x:x+maskwidth],0.5)):
            rainsum[k]=0.
            multiout[k]=-999.
        else:
            if domean:
                #sys.exit('need to fix short duration part')
                muR=homemean-intensemean[y,x]
                if doall:
                    stdR=np.sqrt(np.power(homestd,2)+np.power(intensestd[y,x],2)-2.*intensecorr[y,x]*homestd*intensestd[y,x])
                   # multiplier=sp.stats.lognorm.ppf(rquant[k],stdR,loc=0,scale=np.exp(muR))     
                    #multiplier=10.
                    #while multiplier>maxmultiplier:       # who knows what the right number is to use here...
                    inverrf=sp.special.erfinv(2.*rquant-1.)
                    multiplier=np.exp(muR+np.sqrt(2.*np.power(stdR,2))*inverrf[k])
                    
                    #multiplier=np.random.lognormal(muR,stdR)
                    if multiplier>maxmultiplier:
                        multiplier=1.    
                else:
                    multiplier=np.exp(muR)
                    if multiplier>maxmultiplier:
                        multiplier=1.
            else:
                multiplier=1.
#            print("still going!")
            if multiplier>maxmultiplier:
                sys.exit("Something seems to be going horribly wrong in the multiplier scheme!")
            else:
                multiout[k]=multiplier
        
            if durcheck==True:            
                storesum=0.
                storestep=0
                for kk in range(0,nsteps):
                    #tempsum=numba_multimask_calc(passrain[kk,:],rsum,train,trimmask,ssty[k],maskheight,sstx[k],maskwidth)*multiplier
                    tempsum=numba_multimask_calc(passrain[kk,:],trimmask,y,x,maskheight,maskwidth)*multiplier
                    if tempsum>storesum:
                        storesum=tempsum
                        storestep=kk
                rainsum[k]=storesum
                whichstep[k]=storestep
            else:
                rainsum[k]=numba_multimask_calc(passrain,trimmask,y,x,maskheight,maskwidth)*multiplier
    if domean:
        return rainsum,multiout,whichstep
    else:
        return rainsum,whichstep


#@jit(nopython=True,fastmath=True,parallel=True)
@jit(nopython=True,fastmath=True)
def numba_multimask_calc(passrain,trimmask,ssty,sstx,maskheight,maskwidth):
    train=np.multiply(passrain[ssty : ssty+maskheight , sstx : sstx+maskwidth],trimmask)
    rainsum=np.sum(train)       
    return rainsum


@jit(fastmath=True)
def SSTalt_singlecell(passrain,sstx,ssty,trimmask,maskheight,maskwidth,intensemean=None,intensestd=None,intensecorr=None,homemean=None,homestd=None,durcheck=False):
    rainsum=np.zeros((len(sstx)),dtype='float32')
    whichstep=np.zeros((len(sstx)),dtype='int32')
    nreals=len(rainsum)
    nsteps=passrain.shape[0]
    multiout=np.empty_like(rainsum)

    # do we do deterministic or dimensionless rescaling?
    if (intensemean is not None) and (homemean is not None):
        domean=True
    else:
        domean=False       

    # do we do stochastic rescaling?    
    if (intensestd is not None) and (intensecorr is not None) and (homestd is not None):
        rquant=np.random.random_sample(size=nreals)
        inverrf=sp.special.erfinv(2.*rquant-1.)
        doall=True
    else:
        doall=False
        #rquant=np.nan

    if durcheck==False:
        passrain=np.expand_dims(passrain,0)
       
    # deterministic or dimensionless:
    if domean and doall==False:
        rain,multi,step=killerloop_singlecell(passrain,rainsum,whichstep,nreals,ssty,sstx,nsteps,durcheck=durcheck,intensemean=intensemean,homemean=homemean,multiout=multiout)
        return rain,multi,step
    
    # stochastic:
    elif doall:
        rain,multi,step=killerloop_singlecell(passrain,rainsum,whichstep,nreals,ssty,sstx,nsteps,durcheck=durcheck,intensemean=intensemean,intensestd=intensestd,intensecorr=intensecorr,homemean=homemean,homestd=homestd,multiout=multiout,inverrf=inverrf)
        return rain,multi,step
    
    # no rescaling:
    else:
        rain,_,step=killerloop_singlecell(passrain,rainsum,whichstep,nreals,ssty,sstx,nsteps,durcheck=durcheck,multiout=multiout)
        return rain,step
    


#@jit(nopython=True,fastmath=True,parallel=True)
@jit(nopython=True,fastmath=True)
def killerloop_singlecell(passrain,rainsum,whichstep,nreals,ssty,sstx,nsteps,durcheck=False,intensemean=None,homemean=None,homestd=None,multiout=None,rquant=None,intensestd=None,intensecorr=None,inverrf=None):
    maxmultiplier=1.5  # who knows what the right number is to use here...
    for k in prange(nreals):
        y=int(ssty[k])
        x=int(sstx[k])
        
        # deterministic or dimensionless:
        if (intensemean is not None) and (homemean is not None) and (homestd is None):
            if np.less(homemean,0.001) or np.less(intensemean[y,x],0.001):
                multiplier=1.           # or maybe this should be zero     
            else:
                multiplier=np.exp(homemean-intensemean[y,x])
                if multiplier>maxmultiplier:           
                    multiplier=1.        # or maybe this should be zero
                    
        # stochastic:
        elif (intensemean is not None) and (homemean is not None) and (homestd is not None):
            if np.less(homemean,0.001) or np.less(intensemean[y,x],0.001):
                multiplier=1.          # or maybe this should be zero
            else:
                muR=homemean-intensemean[y,x]
                stdR=np.sqrt(np.power(homestd,2)+np.power(intensestd[y,x],2)-2*intensecorr[y,x]*homestd*intensestd[y,x])

                multiplier=np.exp(muR+np.sqrt(2.*np.power(stdR,2))*inverrf[k])
                if multiplier>maxmultiplier:
                    multiplier=1.        # or maybe this should be zero
        
        # no rescaling:
        else:
            multiplier=1.
            
        if durcheck==False:
            rainsum[k]=np.nansum(passrain[:,y, x])
        else:
            storesum=0.
            storestep=0
            for kk in range(nsteps):
                tempsum=passrain[kk,y,x]
                if tempsum>storesum:
                    storesum=tempsum
                    storestep=kk
            rainsum[k]=storesum*multiplier
            multiout[k]=multiplier
            whichstep[k]=storestep
            
    return rainsum,multiout,whichstep



#@jit(nopython=True,fastmath=True,parallel=True)
#def killerloop(passrain,rainsum,nreals,ssty,sstx,maskheight,maskwidth,trimmask,nsteps,durcheck):
#    for k in prange(nreals):
#        spanx=int64(sstx[k]+maskwidth)
#        spany=int64(ssty[k]+maskheight)
#        if np.all(np.less(passrain[:,ssty[k]:spany,sstx[k]:spanx],0.5)):
#            rainsum[k]=0.
#        else:
#            if durcheck==False:
#                rainsum[k]=np.nansum(np.multiply(passrain[ssty[k] : spany , sstx[k] : spanx],trimmask))
#            else:
#                storesum=float32(0.)
#                for kk in range(nsteps):
#                    tempsum=np.nansum(np.multiply(passrain[kk,ssty[k]:spany,sstx[k]:spanx],trimmask))
#                    if tempsum>storesum:
#                        storesum=tempsum
#                rainsum[k]=storesum
#    return rainsum
    
    
                    #whichstep[k]=storestep
#return rainsum,whichstep



# this function below never worked for some unknown Numba problem-error messages indicated that it wasn't my fault!!! Some problem in tempsum
#@jit(nopython=True,fastmath=True,parallel=True)
#def killerloop(passrain,rainsum,nreals,ssty,sstx,maskheight,maskwidth,masktile,nsteps,durcheck):
#    for k in prange(nreals):
#        spanx=sstx[k]+maskwidth
#        spany=ssty[k]+maskheight
#        if np.all(np.less(passrain[:,ssty[k]:spany,sstx[k]:spanx],0.5)):
#            rainsum[k]=0.
#        else:
#            if durcheck==False:
#                #tempstep=np.multiply(passrain[:,ssty[k] : spany , sstx[k] : spanx],trimmask)
#                #xnum=int64(sstx[k])
#                #ynum=int64(ssty[k])
#                #rainsum[k]=np.nansum(passrain[:,ssty[k], sstx[k]])
#                rainsum[k]=np.nansum(np.multiply(passrain[:,ssty[k] : spany , sstx[k] : spanx],masktile))
#            else:
#                storesum=float32(0.)
#                for kk in range(nsteps):
#                    #tempsum=0.
#                    #tempsum=np.multiply(passrain[kk,ssty[k]:spany,sstx[k]:spanx],masktile[0,:,:])
#                    tempsum=np.nansum(np.multiply(passrain[kk,ssty[k]:spany,sstx[k]:spanx],masktile[0,:,:]))
#    return rainsum


#==============================================================================
# THIS VARIANT IS SIMPLER AND UNLIKE SSTWRITE, IT ACTUALLY WORKS RELIABLY!
#==============================================================================
#def SSTwriteAlt(catrain,rlzx,rlzy,rlzstm,trimmask,xmin,xmax,ymin,ymax,maskheight,maskwidth):
#    nyrs=np.int(rlzx.shape[0])
#    raindur=np.int(catrain.shape[1])
#    outrain=np.zeros((nyrs,raindur,maskheight,maskwidth),dtype='float32')
#    unqstm,unqind,unqcnts=np.unique(rlzstm,return_inverse=True,return_counts=True)
#    #ctr=0
#    for i in range(0,len(unqstm)):
#        unqwhere=np.where(unqstm[i]==rlzstm)[0]
#        for j in unqwhere:
#            #ctr=ctr+1
#            #print ctr
#            outrain[j,:]=np.multiply(catrain[unqstm[i],:,(rlzy[j]) : (rlzy[j]+maskheight) , (rlzx[j]) : (rlzx[j]+maskwidth)],trimmask)
#    return outrain
       

#==============================================================================
# THIS VARIANT IS SAME AS ABOVE, BUT HAS A MORE INTERESTING RAINFALL PREPENDING PROCEDURE
#==============================================================================

#def SSTwriteAltPreCat(catrain,rlzx,rlzy,rlzstm,trimmask,xmin,xmax,ymin,ymax,maskheight,maskwidth,precat,ptime):    
#    catyears=ptime.astype('datetime64[Y]').astype(int)+1970
#    ptime=ptime.astype('datetime64[M]').astype(int)-(catyears-1970)*12+1
#    nyrs=np.int(rlzx.shape[0])
#    raindur=np.int(catrain.shape[1]+precat.shape[1])
#    outrain=np.zeros((nyrs,raindur,maskheight,maskwidth),dtype='float32')
#    unqstm,unqind,unqcnts=np.unique(rlzstm,return_inverse=True,return_counts=True)
#
#    for i in range(0,len(unqstm)):
#        unqwhere=np.where(unqstm[i]==rlzstm)[0]
#        unqmonth=ptime[unqstm[i]]
#        pretimeind=np.where(np.logical_and(ptime>unqmonth-2,ptime<unqmonth+2))[0]
#        for j in unqwhere:
#            temprain=np.concatenate((np.squeeze(precat[np.random.choice(pretimeind, 1),:,(rlzy[j]) : (rlzy[j]+maskheight) , (rlzx[j]) : (rlzx[j]+maskwidth)],axis=0),catrain[unqstm[i],:,(rlzy[j]) : (rlzy[j]+maskheight) , (rlzx[j]) : (rlzx[j]+maskwidth)]),axis=0)
#            outrain[j,:]=np.multiply(temprain,trimmask)
#    return outrain
#    

#==============================================================================
# SAME AS ABOVE, BUT HANDLES STORM ROTATION
#==============================================================================    
    
#def SSTwriteAltPreCatRotation(catrain,rlzx,rlzy,rlzstm,trimmask,xmin,xmax,ymin,ymax,maskheight,maskwidth,precat,ptime,delarray,rlzanglebin,rainprop):
##def SSTwriteAltPreCatRotation(catrain,rlzx,rlzy,rlzstm,trimmask,xmin,xmax,ymin,ymax,maskheight,maskwidth,precat,ptime,delarray,rlzanglebin):
#    catyears=ptime.astype('datetime64[Y]').astype(int)+1970
#    ptime=ptime.astype('datetime64[M]').astype(int)-(catyears-1970)*12+1
#    nyrs=np.int(rlzx.shape[0])
#    raindur=np.int(catrain.shape[1]+precat.shape[1])
#    outrain=np.zeros((nyrs,raindur,maskheight,maskwidth),dtype='float32')
#    unqstm,unqind,unqcnts=np.unique(rlzstm,return_inverse=True,return_counts=True)      # unqstm is the storm number
#
#    for i in range(0,len(unqstm)):
#        unqwhere=np.where(unqstm[i]==rlzstm)[0]
#        unqmonth=ptime[unqstm[i]]
#        pretimeind=np.where(np.logical_and(ptime>unqmonth-2,ptime<unqmonth+2))[0]
#        for j in unqwhere:
#            inrain=catrain[unqstm[i],:].copy()
#            
#            xctr=rlzx[j]+maskwidth/2.
#            yctr=rlzy[j]+maskheight/2.
#            xlinsp=np.linspace(-xctr,rainprop.subdimensions[1]-xctr,rainprop.subdimensions[1])
#            ylinsp=np.linspace(-yctr,rainprop.subdimensions[0]-yctr,rainprop.subdimensions[0])
#    
#            ingridx,ingridy=np.meshgrid(xlinsp,ylinsp)
#            ingridx=ingridx.flatten()
#            ingridy=ingridy.flatten()
#            outgrid=np.column_stack((ingridx,ingridy))  
#            
#            for k in range(0,inrain.shape[0]):
#                interp=sp.interpolate.LinearNDInterpolator(delarray[unqstm[i]][rlzanglebin[j]-1],inrain[k,:].flatten(),fill_value=0.)
#                inrain[k,:]=np.reshape(interp(outgrid),rainprop.subdimensions)
#                #inrain[k,:]=temprain
#            
#            temprain=np.concatenate((np.squeeze(precat[np.random.choice(pretimeind, 1),:,(rlzy[j]) : (rlzy[j]+maskheight) , (rlzx[j]) : (rlzx[j]+maskwidth)],axis=0),inrain[:,(rlzy[j]) : (rlzy[j]+maskheight) , (rlzx[j]) : (rlzx[j]+maskwidth)]),axis=0)
#
#            outrain[j,:]=np.multiply(temprain,trimmask)
#    return outrain
       
@jit(fastmath=True)
def SSTspin_write_v2(catrain,rlzx,rlzy,rlzstm,trimmask,maskheight,maskwidth,precat,ptime,rainprop,rlzanglebin=None,delarray=None,spin=False,flexspin=True,samptype='uniform',cumkernel=None,rotation=False,domaintype='rectangular'):
    catyears=ptime.astype('datetime64[Y]').astype(int)+1970
    ptime=ptime.astype('datetime64[M]').astype(int)-(catyears-1970)*12+1
    nyrs=np.int(rlzx.shape[0])
    raindur=np.int(catrain.shape[1]+precat.shape[1])
    outrain=np.zeros((nyrs,raindur,maskheight,maskwidth),dtype='float32')
    unqstm,unqind,unqcnts=np.unique(rlzstm,return_inverse=True,return_counts=True)      # unqstm is the storm number
    
    for i in range(0,len(unqstm)):
        unqwhere=np.where(unqstm[i]==rlzstm)[0]
        unqmonth=ptime[unqstm[i]]
        pretimeind=np.where(np.logical_and(ptime>unqmonth-1,ptime<unqmonth+1))[0]
        
        # flexspin allows you to use spinup rainfall from anywhere in transposition domain, rather than just storm locations, but it doesn't seem to be very useful based on initial testing
        if spin==True and flexspin==True:       
            if samptype=='kernel' or domaintype=='irregular':
                rndloc=np.random.random_sample(len(unqwhere))
                shiftprex,shiftprey=numbakernel(rndloc,cumkernel)
            else:
                shiftprex=np.random.random_integers(0,np.int(rainprop.subdimensions[1])-maskwidth-1,len(unqwhere))
                shiftprey=np.random.random_integers(0,np.int(rainprop.subdimensions[0])-maskheight-1,len(unqwhere))
            
        ctr=0   
        for j in unqwhere:
            inrain=catrain[unqstm[i],:].copy()
                        
            # this doesn't rotate the prepended rainfall
            if rotation==True:
                xctr=rlzx[j]+maskwidth/2.
                yctr=rlzy[j]+maskheight/2.
                xlinsp=np.linspace(-xctr,rainprop.subdimensions[1]-xctr,rainprop.subdimensions[1])
                ylinsp=np.linspace(-yctr,rainprop.subdimensions[0]-yctr,rainprop.subdimensions[0])
        
                ingridx,ingridy=np.meshgrid(xlinsp,ylinsp)
                ingridx=ingridx.flatten()
                ingridy=ingridy.flatten()
                outgrid=np.column_stack((ingridx,ingridy))  
                
                for k in range(0,inrain.shape[0]):
                    interp=sp.interpolate.LinearNDInterpolator(delarray[unqstm[i]][rlzanglebin[j]-1],inrain[k,:].flatten(),fill_value=0.)
                    inrain[k,:]=np.reshape(interp(outgrid),rainprop.subdimensions)
                    
            if spin==True and flexspin==True:
                temprain=np.concatenate((np.squeeze(precat[np.random.choice(pretimeind, 1),:,(shiftprey[ctr]) : (shiftprey[ctr]+maskheight) , (shiftprex[ctr]) : (shiftprex[ctr]+maskwidth)],axis=0),inrain[:,(rlzy[j]) : (rlzy[j]+maskheight) , (rlzx[j]) : (rlzx[j]+maskwidth)]),axis=0)
            elif spin==True and flexspin==False:
                temprain=np.concatenate((np.squeeze(precat[np.random.choice(pretimeind, 1),:,(rlzy[j]) : (rlzy[j]+maskheight) , (rlzx[j]) : (rlzx[j]+maskwidth)],axis=0),inrain[:,(rlzy[j]) : (rlzy[j]+maskheight) , (rlzx[j]) : (rlzx[j]+maskwidth)]),axis=0)
            elif spin==False:
                temprain=inrain[:,(rlzy[j]) : (rlzy[j]+maskheight) , (rlzx[j]) : (rlzx[j]+maskwidth)]
            else:
                sys.exit("what else is there?")
            ctr=ctr+1

            outrain[j,:]=np.multiply(temprain,trimmask)
    return outrain


##==============================================================================
## SAME AS ABOVE, BUT A BIT MORE DYNAMIC IN TERMS OF SPINUP
##==============================================================================    
#def SSTspin_write_v2(catrain,rlzx,rlzy,rlzstm,trimmask,xmin,xmax,ymin,ymax,maskheight,maskwidth,precat,ptime,rainprop,rlzanglebin=None,delarray=None,spin=False,flexspin=True,samptype='uniform',cumkernel=None,rotation=False,domaintype='rectangular',intense_data=False):
#    catyears=ptime.astype('datetime64[Y]').astype(int)+1970
#    ptime=ptime.astype('datetime64[M]').astype(int)-(catyears-1970)*12+1
#    nyrs=np.int(rlzx.shape[0])
#    raindur=np.int(catrain.shape[1]+precat.shape[1])
#    outrain=np.zeros((nyrs,raindur,maskheight,maskwidth),dtype='float32')
#    unqstm,unqind,unqcnts=np.unique(rlzstm,return_inverse=True,return_counts=True)      # unqstm is the storm number
#    
#    if intense_data!=False:
#        sys.exit("Scenario writing for intensity-based resampling not tested!")
#        intquant=intense_data[0]
#        fullmu=intense_data[1]
#        fullstd=intense_data[2]
#        muorig=intense_data[3]
#        stdorig=intense_data[4]
#    
#    for i in range(0,len(unqstm)):
#        unqwhere=np.where(unqstm[i]==rlzstm)[0]
#        unqmonth=ptime[unqstm[i]]
#        pretimeind=np.where(np.logical_and(ptime>unqmonth-1,ptime<unqmonth+1))[0]
#        
#        if transpotype=='intensity':
#            origmu=np.multiply(murain[caty[i]:caty[i]+maskheight,catx[i]:catx[i]+maskwidth],trimmask)
#            origstd=np.multiply(stdrain[caty[i]:caty[i]+maskheight,catx[i]:catx[i]+maskwidth],trimmask)
#            #intense_dat=[intquant[],murain,stdrain,origmu,origstd]
#        
#        # flexspin allows you to use spinup rainfall from anywhere in transposition domain, rather than just storm locations, but it doesn't seem to be very useful based on initial testing
#        if spin==True and flexspin==True:       
#            if samptype=='kernel' or domaintype=='irregular':
#                rndloc=np.random.random_sample(len(unqwhere))
#                shiftprex,shiftprey=numbakernel(rndloc,cumkernel)
#            else:
#                shiftprex=np.random.random_integers(0,np.int(rainprop.subdimensions[1])-maskwidth-1,len(unqwhere))
#                shiftprey=np.random.random_integers(0,np.int(rainprop.subdimensions[0])-maskheight-1,len(unqwhere))
#            
#        ctr=0   
#        for j in unqwhere:
#            inrain=catrain[unqstm[i],:].copy()
#            
#            if intense_data!=False:
#                transmu=np.multiply(fullmu[(rlzy[i]) : (rlzy[i]+maskheight) , (rlzx[i]) : (rlzx[i]+maskwidth)],trimmask)
#                transtd=np.multiply(fullstd[(rlzy[i]) : (rlzy[i]+maskheight) , (rlzx[i]) : (rlzx[i]+maskwidth)],trimmask)
#                mu_multi=transmu/muorig
#                std_multi=np.abs(transtd-stdorig)/stdorig
#                multipliermask=norm.ppf(intquant[i],loc=mu_multi,scale=std_multi)
#                multipliermask[multipliermask<0.]=0.
#                multipliermask[np.isnan(multipliermask)]=0.
#            
#            # this doesn't rotate the prepended rainfall
#            if rotation==True:
#                xctr=rlzx[j]+maskwidth/2.
#                yctr=rlzy[j]+maskheight/2.
#                xlinsp=np.linspace(-xctr,rainprop.subdimensions[1]-xctr,rainprop.subdimensions[1])
#                ylinsp=np.linspace(-yctr,rainprop.subdimensions[0]-yctr,rainprop.subdimensions[0])
#        
#                ingridx,ingridy=np.meshgrid(xlinsp,ylinsp)
#                ingridx=ingridx.flatten()
#                ingridy=ingridy.flatten()
#                outgrid=np.column_stack((ingridx,ingridy))  
#                
#                for k in range(0,inrain.shape[0]):
#                    interp=sp.interpolate.LinearNDInterpolator(delarray[unqstm[i]][rlzanglebin[j]-1],inrain[k,:].flatten(),fill_value=0.)
#                    inrain[k,:]=np.reshape(interp(outgrid),rainprop.subdimensions)
#                    
#            if spin==True and flexspin==True:
#                temprain=np.concatenate((np.squeeze(precat[np.random.choice(pretimeind, 1),:,(shiftprey[ctr]) : (shiftprey[ctr]+maskheight) , (shiftprex[ctr]) : (shiftprex[ctr]+maskwidth)],axis=0),inrain[:,(rlzy[j]) : (rlzy[j]+maskheight) , (rlzx[j]) : (rlzx[j]+maskwidth)]),axis=0)
#            elif spin==True and flexspin==False:
#                temprain=np.concatenate((np.squeeze(precat[np.random.choice(pretimeind, 1),:,(rlzy[j]) : (rlzy[j]+maskheight) , (rlzx[j]) : (rlzx[j]+maskwidth)],axis=0),inrain[:,(rlzy[j]) : (rlzy[j]+maskheight) , (rlzx[j]) : (rlzx[j]+maskwidth)]),axis=0)
#            elif spin==False:
#                temprain=inrain[:,(rlzy[j]) : (rlzy[j]+maskheight) , (rlzx[j]) : (rlzx[j]+maskwidth)]
#            else:
#                sys.exit("what else is there?")
#            ctr=ctr+1
#            if intense_data!=False:
#                outrain[j,:]=np.multiply(temprain,multipliermask)
#            else:
#                outrain[j,:]=np.multiply(temprain,trimmask)
#    return outrain
    
    
#==============================================================================
# LOOP FOR KERNEL BASED STORM TRANSPOSITION
# THIS FINDS THE TRANSPOSITION LOCATION FOR EACH REALIZATION IF YOU ARE USING THE KERNEL-BASED RESAMPLER
# IF I CONFIGURE THE SCRIPT SO THE USER CAN PROVIDE A CUSTOM RESAMPLING SCHEME, THIS WOULD PROBABLY WORK FOR THAT AS WELL
#==============================================================================    
#def weavekernel(rndloc,cumkernel):
#    nlocs=len(rndloc)
#    nrows=cumkernel.shape[0]
#    ncols=cumkernel.shape[1]
#    tempx=np.empty((len(rndloc)),dtype="int32")
#    tempy=np.empty((len(rndloc)),dtype="int32")
#    code= """
#        #include <stdio.h>
#        int i,x,y,brklp;
#        double prevprob;
#        for (i=0;i<nlocs;i++) {
#            prevprob=0.0;
#            brklp=0;
#            for (y=0; y<nrows; y++) {
#                for (x=0; x<ncols; x++) {
#                    if ( (rndloc(i)<=cumkernel(y,x)) && (rndloc(i)>prevprob) ) {
#                        tempx(i)=x;
#                        tempy(i)=y;
#                        prevprob=cumkernel(y,x);
#                        brklp=1;
#                        break;
#                    }                     
#                }
#                if (brklp==1) {
#                    break;                    
#                }                         
#            }   
#        }
#    """
#    vars=['rndloc','cumkernel','nlocs','nrows','ncols','tempx','tempy']
#    sp.weave.inline(code,vars,type_converters=converters.blitz,compiler='gcc')
#    return tempx,tempy
    
    
def pykernel(rndloc,cumkernel):
    nlocs=len(rndloc)
    ncols=cumkernel.shape[1]
    tempx=np.empty((len(rndloc)),dtype="int32")
    tempy=np.empty((len(rndloc)),dtype="int32")
    flatkern=np.append(0.,cumkernel.flatten())
    
    for i in range(0,nlocs):
        x=rndloc[i]-flatkern
        x[np.less(x,0.)]=1000.
        whereind = np.argmin(x)
        y=whereind//ncols
        x=whereind-y*ncols        
        tempx[i]=x
        tempy[i]=y
    return tempx,tempy

@jit 
def numbakernel(rndloc,cumkernel,tempx,tempy,ncols):
    nlocs=len(rndloc)
    #ncols=xdim
    flatkern=np.append(0.,cumkernel.flatten())
    #x=np.zeros_like(rndloc,dtype='float64')
    for i in np.arange(0,nlocs):
        x=rndloc[i]-flatkern
        x[np.less(x,0.)]=10.
        whereind=np.argmin(x)
        y=whereind//ncols
        x=whereind-y*ncols 
        tempx[i]=x
        tempy[i]=y
    return tempx,tempy


@jit 
def numbakernel_fast(rndloc,cumkernel,tempx,tempy,ncols):
    nlocs=int32(len(rndloc))
    ncols=int32(cumkernel.shape[1])
    flatkern=np.append(0.,cumkernel.flatten()) 
    return kernelloop(nlocs,rndloc,flatkern,ncols,tempx,tempy)

#@jit(nopython=True,fastmath=True,parallel=True)
@jit(nopython=True,fastmath=True)
def kernelloop(nlocs,rndloc,flatkern,ncols,tempx,tempy):
    for i in prange(nlocs):
        diff=rndloc[i]-flatkern
        diff[np.less(diff,0.)]=10.
        whereind=np.argmin(diff)
        y=whereind//ncols
        x=whereind-y*ncols 
        tempx[i]=x
        tempy[i]=y
    return tempx,tempy



#==============================================================================
# FIND THE BOUNDARY INDICIES AND COORDINATES FOR THE USER-DEFINED SUBAREA
# NOTE THAT subind ARE THE MATRIX INDICIES OF THE SUBBOX, STARTING FROM UPPER LEFT CORNER OF DOMAIN AS (0,0)
# NOTE THAT subcoord ARE THE COORDINATES OF THE OUTSIDE BORDER OF THE SUBBOX
# THEREFORE THE DISTANCE FROM THE WESTERN (SOUTHERN) BOUNDARY TO THE EASTERN (NORTHERN) BOUNDARY IS NCOLS (NROWS) +1 TIMES THE EAST-WEST (NORTH-SOUTH) RESOLUTION
#============================================================================== 
def findsubbox(inarea,rainprop):
    outind=np.empty([4],dtype='int')
    outextent=np.empty([4])
    outdim=np.empty([2])
    inbox=deepcopy(inarea)

    rangex=np.arange(rainprop.bndbox[0],rainprop.bndbox[1]-rainprop.spatialres[0]/1000,rainprop.spatialres[0])
    rangey=np.arange(rainprop.bndbox[3],rainprop.bndbox[2]+rainprop.spatialres[1]/1000,-rainprop.spatialres[1])

    if rangex.shape[0]<rainprop.dimensions[1]:
        rangex=np.append(rangex,rangex[-1])
    if rangey.shape[0]<rainprop.dimensions[0]:
        rangey=np.append(rangey,rangey[-1])
    if rangex.shape[0]>rainprop.dimensions[1]:
        rangex=rangex[0:-1]
    if rangey.shape[0]>rainprop.dimensions[0]:
        rangey=rangey[0:-1]
    
    outextent=inbox
    
    # "SNAP" output extent to grid
    outind[0]=np.abs(rangex-outextent[0]).argmin()
    outind[1]=np.abs(rangex-outextent[1]).argmin()-1
    outind[2]=np.abs(rangey-outextent[2]).argmin()-1
    outind[3]=np.abs(rangey-outextent[3]).argmin()
    outextent[0]=rangex[outind[0]]
    outextent[1]=rangex[outind[1]+1]
    outextent[2]=rangey[outind[2]+1]
    outextent[3]=rangey[outind[3]]

    outdim[1]=np.shape(np.arange(outind[0],outind[1]+1))[0]
    outdim[0]=np.shape(np.arange(outind[3],outind[2]+1))[0]
    outdim=np.array(outdim,dtype='int32')
    return outextent,outind,outdim
    

#==============================================================================
# THIS RETURNS A LOGICAL GRID THAT CAN THEN BE APPLIED TO THE GLOBAL GRID TO EXTRACT
# A USEER-DEFINED SUBGRID
# THIS HELPS TO KEEP ARRAY SIZES SMALL
#==============================================================================
def creategrids(rainprop):
    globrangex=np.arange(0,rainprop.dimensions[1],1)
    globrangey=np.arange(0,rainprop.dimensions[0],1)
    subrangex=np.arange(rainprop.subind[0],rainprop.subind[1]+1,1)
    subrangey=np.arange(rainprop.subind[3],rainprop.subind[2]+1,1)
    subindx=np.logical_and(globrangex>=subrangex[0],globrangex<=subrangex[-1])
    subindy=np.logical_and(globrangey>=subrangey[0],globrangey<=subrangey[-1])
    gx,gy=np.meshgrid(subindx,subindy)
    outgrid=np.logical_and(gx==True,gy==True)
    return outgrid,subindx,subindy


#==============================================================================
# FUNCTION TO CREATE A MASK ACCORDING TO A USER-DEFINED POLYGON SHAPEFILE AND PROJECTION
#==============================================================================
def rastermask(shpname,shpproj,rainprop,masktype):            
    bndbox=np.array(rainprop.subind)
    bndcoords=np.array(rainprop.subextent)
    
    xdim=rainprop.subdimensions[0]  
    ydim=rainprop.subdimensions[1]  

    with fiona.open(shpname, "r") as shapefile:
        shapes = [feature["geometry"] for feature in shapefile]
    if len(shapes)!=1:
        sys.exit("something is wrong with the basin shapefile! It has either zero or >1 features!")
    
    
    
    if masktype=='simple':
        print('creating simple mask (0s and 1s)')
        transform = from_origin(bndcoords[0], bndcoords[3], rainprop.spatialres[0], rainprop.spatialres[1])
        rastertemplate=np.ones((ydim,xdim),dtype='float32')
        rastermask = rasterio.open('temp9999.tif', 'w', driver='GTiff',
                                height = rastertemplate.shape[1], width = rastertemplate.shape[0],
                                count=1, dtype=str(rastertemplate.dtype),
                                crs='+proj=longlat +datum=WGS84 +no_defs',
                                transform=transform)
        rastermask.write(rastertemplate, 1)
        rastermask.close()
        
        with rasterio.open('temp9999.tif') as src:
            simplemask, out_transform = mask(src, shapes, crop=False,all_touched=True)
            out_meta = src.meta
        rastertemplate =simplemask[0,:]



    elif masktype=="fraction":
        print('creating fractional mask (range from 0.0-1.0)')
        
        transform = from_origin(bndcoords[0], bndcoords[3], rainprop.spatialres[0]/10., rainprop.spatialres[1]/10.)
        rastertemplate=np.ones((ydim,xdim),dtype='float32')
        rastermask = rasterio.open('temp9999.tif', 'w', driver='GTiff',
                                height = 10*rastertemplate.shape[1], width = 10*rastertemplate.shape[0],
                                count=1, dtype=str(rastertemplate.dtype),
                                crs='+proj=longlat +datum=WGS84 +no_defs',
                                transform=transform)
        rastermask.write(rastertemplate, 1)
        rastermask.close()
        
        with rasterio.open('temp9999.tif') as src:
            simplemask, out_transform = mask(src, shapes, crop=False,all_touched=True)
            out_meta = src.meta
        rastertemplate=simplemask[0,:]
        from scipy.signal import convolve2d
        n=10
        kernel = np.ones((n, n))
        convolved = convolve2d(rastertemplate, kernel, mode='valid')
        rastertemplate=convolved[::n, ::n] / n /n
        
    else:
        sys.exit("You entered an incorrect mask type, options are 'simple' or 'fraction'")
    delete('temp9999.tif')   
    return rastertemplate   


#==============================================================================
# WRITE SCENARIOS TO NETCDF ONE REALIZATION AT A TIME
#==============================================================================
def writerealization(rlz,nrealizations,writename,outrain,writemax,writestorm,writeperiod,writex,writey,writetimes,latrange,lonrange,whichorigstorm):
    # SAVE outrain AS NETCDF FILE
    dataset=Dataset(writename, 'w', format='NETCDF4')

    # create dimensions
    outlats=dataset.createDimension('outlat',len(latrange))
    outlons=dataset.createDimension('outlon',len(lonrange))
    time=dataset.createDimension('time',writetimes.shape[1])
    nyears=dataset.createDimension('nyears',len(writeperiod))

    # create variables
    times=dataset.createVariable('time',np.float64, ('nyears','time'))
    latitudes=dataset.createVariable('latitude',np.float32, ('outlat'))
    longitudes=dataset.createVariable('longitude',np.float32, ('outlon'))
    rainrate=dataset.createVariable('rainrate',np.float32,('nyears','time','outlat','outlon'),zlib=True,complevel=4,least_significant_digit=2) 
    basinrainfall=dataset.createVariable('basinrainfall',np.float32,('nyears')) 
    xlocation=dataset.createVariable('xlocation',np.int32,('nyears')) 
    ylocation=dataset.createVariable('ylocation',np.int32,('nyears')) 
    returnperiod=dataset.createVariable('returnperiod',np.float32,('nyears')) 
    stormnumber=dataset.createVariable('stormnumber',np.int32,('nyears'))
    original_stormnumber=dataset.createVariable('original_stormnumber',np.int32,('nyears'))
    #stormtimes=dataset.createVariable('stormtimes',np.float64,('nyears'))          
    
    # Global Attributes
    dataset.description = 'SST Rainfall Scenarios Realization: '+str(rlz+1)+' of '+str(nrealizations)

    dataset.history = 'Created ' + str(datetime.now())
    dataset.source = 'Storm Catalog for (FILL IN THE BLANK)'
    
    # Variable Attributes (time since 1970-01-01 00:00:00.0 in numpys)
    latitudes.units = 'degrees north'
    longitudes.units = 'degrees east'
    rainrate.units = 'mm/h'
    times.units = 'minutes since 1970-01-01 00:00.0'
    times.calendar = 'gregorian'
    
    #print dataset.description
    #print dataset.history
    
    # fill the netcdf file
    latitudes[:]=latrange
    longitudes[:]=lonrange
    rainrate[:]=outrain 
    basinrainfall[:]=writemax
    times[:]=writetimes
    xlocation[:]=writex
    ylocation[:]=writey
    stormnumber[:]=writestorm
    returnperiod[:]=writeperiod
    original_stormnumber[:]=whichorigstorm
    #stormtimes[:]=writetimes
    
    dataset.close()
    
    
#==============================================================================
# WRITE The maximized storm
#==============================================================================
def writemaximized(writename,outrain,writemax,write_ts,writex,writey,writetimes,latrange,lonrange):
    # SAVE outrain AS NETCDF FILE
    dataset=Dataset(writename, 'w', format='NETCDF4')

    # create dimensions
    outlats=dataset.createDimension('outlat',len(latrange))
    outlons=dataset.createDimension('outlon',len(lonrange))
    time=dataset.createDimension('time',len(writetimes))

    # create variables
    times=dataset.createVariable('time',np.float64, ('time'))
    latitudes=dataset.createVariable('latitude',np.float32, ('outlat'))
    longitudes=dataset.createVariable('longitude',np.float32, ('outlon'))
    rainrate=dataset.createVariable('rainrate',np.float32,('time','outlat','outlon'),zlib=True,complevel=4,least_significant_digit=2) 
    basinrainfall=dataset.createVariable('basinrainfall',np.float32) 
    xlocation=dataset.createVariable('xlocation',np.int32) 
    ylocation=dataset.createVariable('ylocation',np.int32) 
    #stormtimes=dataset.createVariable('stormtimes',np.float64,('nyears'))          
    
    # Global Attributes
    dataset.description = 'SST Rainfall Maximum Storm'

    dataset.history = 'Created ' + str(datetime.now())
    dataset.source = 'Storm Catalog for (FILL IN THE BLANK)'
    
    # Variable Attributes (time since 1970-01-01 00:00:00.0 in numpys)
    latitudes.units = 'degrees north'
    longitudes.units = 'degrees east'
    rainrate.units = 'mm/h'
    times.units = 'minutes since 1970-01-01 00:00.0'
    times.calendar = 'gregorian'
    
    #print dataset.description
    #print dataset.history
    
    # fill the netcdf file
    latitudes[:]=latrange
    longitudes[:]=lonrange
    rainrate[:]=outrain 
    basinrainfall[:]=writemax
    times[:]=writetimes
    xlocation[:]=writex
    ylocation[:]=writey
    
    dataset.close()
        
        

#==============================================================================
# READ RAINFALL FILE FROM NETCDF (ONLY FOR RAINYDAY NETCDF-FORMATTED DAILY FILES!
#==============================================================================
def readnetcdf(rfile,inbounds=False):
    infile=Dataset(rfile,'r')
    if np.any(inbounds!=False):
        outrain=np.array(infile.variables['rainrate'][:,inbounds[3]:inbounds[2]+1,inbounds[0]:inbounds[1]+1])
        outlatitude=np.array(infile.variables['latitude'][inbounds[3]:inbounds[2]+1])
        outlongitude=np.array(infile.variables['longitude'][inbounds[0]:inbounds[1]+1])         
    else:
        outrain=np.array(infile.variables['rainrate'][:])
        outlatitude=np.array(infile.variables['latitude'][:])
        outlongitude=np.array(infile.variables['longitude'][:])
    outtime=np.array(infile.variables['time'][:],dtype='datetime64[m]')
    infile.close()
    return outrain,outtime,outlatitude,outlongitude
    
    
#==============================================================================
# READ RAINFALL FILE FROM NETCDF
#==============================================================================
def readcatalog(rfile):
    infile=Dataset(rfile,'r')
    outrain=np.array(infile.variables['rainrate'][:])
    outtime=np.array(infile.variables['time'][:],dtype='datetime64[m]')
    outlatitude=np.array(infile.variables['latitude'][:])
    outlongitude=np.array(infile.variables['longitude'][:])
    outlocx=np.array(infile.variables['xlocation'][:])
    outlocy=np.array(infile.variables['ylocation'][:])
    outmax=np.array(infile.variables['basinrainfall'][:])
    outmask=np.array(infile.variables['gridmask'][:])
    domainmask=np.array(infile.variables['domainmask'][:])
    try:
        timeresolution=np.int(infile.timeresolution)
        resexists=True
    except:
        resexists=False
    infile.close()
    
    if resexists:
        return outrain,outtime,outlatitude,outlongitude,outlocx,outlocy,outmax,outmask,domainmask,timeresolution
    else:
        return outrain,outtime,outlatitude,outlongitude,outlocx,outlocy,outmax,outmask,domainmask
    
def readtimeresolution(rfile):
    infile=Dataset(rfile,'r')
    try:
        timeresolution=np.int(infile.timeresolution)
    except:
        sys.exit("The time resolution of your storm catalog is ambiguous. This only appears in very specific circumstances. You can contact Dr. Daniel Wright if you need help!")
    
    return timeresolution
 

#==============================================================================
# READ RAINFALL FILE FROM NETCDF: LEGACY VERSION! ONLY NEEDED IF READING AN OLDER DATASET
#==============================================================================
def readcatalog_LEGACY(rfile):
    infile=Dataset(rfile,'r')
    outrain=np.array(infile.variables['rainrate'][:])
    outtime=np.array(infile.variables['time'][:],dtype='datetime64[m]')
    outlatitude=np.array(infile.variables['latitude'][:])
    outlongitude=np.array(infile.variables['longitude'][:])
    outlocx=np.array(infile.variables['xlocation'][:])
    outlocy=np.array(infile.variables['ylocation'][:])
    outmax=np.array(infile.variables['basinrainfall'][:])
    outmask=np.array(infile.variables['gridmask'][:])
    #domainmask=np.array(infile.variables['domainmask'][:])
    infile.close()
    return outrain,outtime,outlatitude,outlongitude,outlocx,outlocy,outmax,outmask


    
#==============================================================================
# WRITE RAINFALL FILE TO NETCDF
#==============================================================================
def writecatalog(catrain,catmax,catx,caty,cattime,latrange,lonrange,catalogname,nstorms,gridmask,parameterfile,dmask,timeresolution=False):
    # SAVE outrain AS NETCDF FILE
    dataset=Dataset(catalogname, 'w', format='NETCDF4')
    
    # create dimensions
    outlats=dataset.createDimension('outlat',len(latrange))
    outlons=dataset.createDimension('outlon',len(lonrange))
    time=dataset.createDimension('time',cattime.shape[1])
    nstorms=dataset.createDimension('nstorms',nstorms)

    # create variables
    times=dataset.createVariable('time',np.float64, ('nstorms','time',))
    latitudes=dataset.createVariable('latitude',np.float32, ('outlat',))
    longitudes=dataset.createVariable('longitude',np.float32, ('outlon',))
    rainrate=dataset.createVariable('rainrate',np.float32,('nstorms','time','outlat','outlon',),zlib=True,complevel=4,least_significant_digit=2) 
    basinrainfall=dataset.createVariable('basinrainfall',np.float32,('nstorms')) 
    xlocation=dataset.createVariable('xlocation',np.int32,('nstorms')) 
    ylocation=dataset.createVariable('ylocation',np.int32,('nstorms')) 
    gmask=dataset.createVariable('gridmask',np.float32,('outlat','outlon',)) 
    domainmask=dataset.createVariable('domainmask',np.float32,('outlat','outlon',)) 
    
    
    # Global Attributes
    with open(parameterfile, "r") as myfile:
        params=myfile.read()
    myfile.close
    dataset.description=params
    if timeresolution!=False:
        dataset.timeresolution=timeresolution

    dataset.history = 'Created ' + str(datetime.now())
    dataset.source = 'RainyDay Storm Catalog'
    
    # Variable Attributes (time since 1970-01-01 00:00:00.0 in numpys)
    latitudes.units = 'degrees north'
    longitudes.units = 'degrees east'
    rainrate.units = 'mm/h'
    times.units = 'minutes since 1970-01-01 00:00.0'
    times.calendar = 'gregorian'
    gmask.units="N/A"

    # fill the netcdf file
    latitudes[:]=latrange
    longitudes[:]=lonrange
    rainrate[:]=catrain 
    basinrainfall[:]=catmax
    times[:]=cattime
    xlocation[:]=catx
    ylocation[:]=caty
    gmask[:]=gridmask
    domainmask[:]=dmask
    
    dataset.close()


def writeintensityfile(intenserain,filename,latrange,lonrange,intensetime):
    # SAVE outrain AS NETCDF FILE

    dataset=Dataset(filename, 'w', format='NETCDF4')
    
    # create dimensions
    outlats=dataset.createDimension('outlat',intenserain.shape[1])
    outlons=dataset.createDimension('outlon',intenserain.shape[2])
    nstorms=dataset.createDimension('nstorms',intenserain.shape[0])

    # create variables
    latitudes=dataset.createVariable('latitude',np.float32, ('outlat',))
    longitudes=dataset.createVariable('longitude',np.float32, ('outlon',))
    stormtotals=dataset.createVariable('stormtotals',np.float32,('nstorms','outlat','outlon',))
    times=dataset.createVariable('time',np.float64, ('nstorms','outlat','outlon',))


    dataset.history = 'Created ' + str(datetime.now())
    dataset.source = 'RainyDay Storm Intensity File'
    
    # Variable Attributes (time since 1970-01-01 00:00:00.0 in numpys)
    latitudes.units = 'degrees north'
    longitudes.units = 'degrees east'
    stormtotals.units = 'mm'
    times.units = 'minutes since 1970-01-01 00:00.0'

    # fill the netcdf file
    latitudes[:]=latrange
    longitudes[:]=lonrange
    stormtotals[:]=intenserain
    times[:]=intensetime
    
    dataset.close()
    
    
def readintensityfile(rfile,inbounds=False):
    infile=Dataset(rfile,'r')
    if np.any(inbounds!=False):
        outrain=np.array(infile.variables['stormtotals'][:,inbounds[3]:inbounds[2]+1,inbounds[0]:inbounds[1]+1])
        outtime=np.array(infile.variables['time'][:,inbounds[3]:inbounds[2]+1,inbounds[0]:inbounds[1]+1],dtype='datetime64[m]')
        outlat=np.array(infile.variables['latitude'][inbounds[3]:inbounds[2]+1])
        outlon=np.array(infile.variables['longitude'][inbounds[0]:inbounds[1]+1])
    else:
        outrain=np.array(infile.variables['stormtotals'][:])
        outtime=np.array(infile.variables['time'][:],dtype='datetime64[m]')
        outlat=np.array(infile.variables['latitude'][:])
        outlon=np.array(infile.variables['longitude'][:])        
    infile.close()
    return outrain,outtime,outlat,outlon

def readmeanfile(rfile,inbounds=False):
    infile=Dataset(rfile,'r')
    if np.any(inbounds!=False):
        outrain=np.array(infile.variables['stormtotals'][inbounds[3]:inbounds[2]+1,inbounds[0]:inbounds[1]+1])
        outlat=np.array(infile.variables['latitude'][inbounds[3]:inbounds[2]+1])
        outlon=np.array(infile.variables['longitude'][inbounds[0]:inbounds[1]+1])
    else:
        outrain=np.array(infile.variables['stormtotals'][:])
        outlat=np.array(infile.variables['latitude'][:])
        outlon=np.array(infile.variables['longitude'][:])        
    infile.close()
    return outrain,outlat,outlon


def writedomain(domain,mainpath,latrange,lonrange,parameterfile):
    # SAVE outrain AS NETCDF FILE
    dataset=Dataset(mainpath, 'w', format='NETCDF4')

    # create dimensions
    outlats=dataset.createDimension('outlat',domain.shape[0])
    outlons=dataset.createDimension('outlon',domain.shape[1])

    # create variables
    latitudes=dataset.createVariable('latitude',np.float32, ('outlat',))
    longitudes=dataset.createVariable('longitude',np.float32, ('outlon',))
    domainmap=dataset.createVariable('domain',np.float32,('outlat','outlon',))
    
    dataset.history = 'Created ' + str(datetime.now())
    dataset.source = 'RainyDay Storm Transposition Domain Map File'
    
    # Variable Attributes (time since 1970-01-01 00:00:00.0 in numpys)
    latitudes.units = 'degrees north'
    longitudes.units = 'degrees east'
    domainmap.units = '-'
    
    # fill the netcdf file
    latitudes[:]=latrange
    longitudes[:]=lonrange
    domainmap[:]=domain
    
    with open(parameterfile, "r") as myfile:
        params=myfile.read()
    myfile.close
    dataset.description=params
    
    dataset.close()



#==============================================================================    
# http://stackoverflow.com/questions/10106901/elegant-find-sub-list-in-list 
#============================================================================== 
def subfinder(mylist, pattern):
    matches = []
    for i in range(len(mylist)):
        if mylist[i] == pattern[0] and mylist[i:i+len(pattern)] == pattern:
            matches.append(i)
    return matches
    
    
#==============================================================================
# CREATE FILE LIST
#==============================================================================
def createfilelist(inpath,includeyears,excludemonths):
    flist=glob.glob(inpath)
    flist=np.array(flist)
        
    if len(flist)==0:
        sys.exit("couldn't find any input rainfall files!")
    
    numbers=[]
    for c in flist[0]:
        numbers.append(c.isdigit())
    if sum(numbers)<8:
        sys.exit("There is something wrong with your input rainfall file names, the date must appear in the file name in the YYYYMMDD format.")
    datechecklist=[True,True,True,True,True,True,True,True]              

    fstrind=subfinder(numbers,datechecklist)
    if len(fstrind)<1:
        sys.exit("We could not parse a file date in YYYYMMDD format from the filenames.")
    elif len(fstrind)>1:
        print("Warning: the file date in the YYYYMMDD format was ambiguous.")
        fstrind=fstrind[-1]
    else:
        fstrind=fstrind[0]


    # THIS IS UGLY BUT YOLO
    ctr=0
    fmonth=np.zeros(flist.shape,dtype="int")
    fyear=np.zeros(flist.shape,dtype="int")
    ftime=np.zeros(flist.shape,dtype="int")
    finclude=np.ones(flist.shape,dtype="bool")
    for f in flist:
        ftime[ctr]=f[fstrind:(fstrind+8)]
        fmonth[ctr]=np.int(f[fstrind:(fstrind+8)][4:6])
        fyear[ctr]=np.int(f[fstrind:(fstrind+8)][0:4])
        ctr=ctr+1
    if isinstance(includeyears, (bool))==False:  
        allyears=np.arange(min(fyear),max(fyear)+1)
        excludeyears=set(allyears)^set(includeyears)
        for j in excludeyears:
            finclude[fyear==j]=False
        nyears=len(allyears)-len(excludeyears)
    else:
        nyears=len(np.unique(fyear))
    
    if isinstance(excludemonths, (bool))==False:
        for j in excludemonths:
            finclude[fmonth==j]=False
        
    flist=flist[finclude==True]
    ftime=ftime[finclude==True]
        
    fsort=np.array(sorted(enumerate(ftime), key=lambda x: x[1]))
    sortind=fsort[:,0]
    flist=flist[sortind]
    return flist,nyears


#==============================================================================
# Get things set up
#==============================================================================
def rainprop_setup(infile,catalog=False):
    if catalog:
        inrain,intime,inlatitude,inlongitude,catx,caty,catmax,_,domainmask=readcatalog(infile)
    else:
        inrain,intime,inlatitude,inlongitude=readnetcdf(infile)
    
    if len(inlatitude.shape)>1 or len(inlongitude.shape)>1:
        inlatitude=inlatitude[:,0]          # perhaps would be safer to have an error here...
        inlongitude=inlongitude[0,:]        # perhaps would be safer to have an error here...
    subdiff=np.abs(np.subtract(inlatitude[1:],inlatitude[0:-1]))
    yres=np.mean(subdiff[subdiff>0.0001])
    if np.allclose(subdiff[subdiff>0.0001],yres,rtol=1e-03)==False:
        sys.exit("North-South resolution is not constant. RainyDay cannot support that.")
    subdiff=np.abs(np.subtract(inlongitude[1:],inlongitude[0:-1]))
    xres=np.mean(subdiff[subdiff>0.0001])
    if np.allclose(subdiff[subdiff>0.0001],xres,rtol=1e-03)==False:
        sys.exit("East-West resolution is not constant. RainyDay cannot support that.")
    

    unqtimes=np.unique(intime)   
    if len(unqtimes)>1:
        tempres=np.min(unqtimes[1:]-unqtimes[0:-1])   # temporal resolution
    else:
        tempres=np.float32(1440.)
        tempres=tempres.astype('timedelta64[m]')      # temporal resolution in minutes-haven't checked to make sure this works right
        
    if len(intime)*np.float32(tempres)!=1440. and catalog==False:
        sys.exit("RainyDay requires daily input files, but has detected something different.")
    if np.allclose(np.array(np.subtract(unqtimes[1:],unqtimes[0:-1]),dtype='float32'),np.float32(tempres),rtol=1e-03)==False and catalog==False:
        sys.exit("Temporal resolution is not constant. RainyDay cannot support that.") 
    tempres=np.int(np.float32(tempres))
        
    nodata=np.unique(inrain[inrain<0.])
    if len(nodata)>1:
        sys.exit("More than one missing value flag.")
    elif len(nodata)==0 and catalog==False:
        print("Warning: Missing data flag is ambiguous.")
        nodata==-999.
    elif catalog:
        nodata=-999.
    else:
        nodata=nodata[0]

    if catalog:
        return [xres,yres], [len(inlatitude),len(inlongitude)],[np.min(inlongitude),np.max(inlongitude),np.min(inlatitude),np.max(inlatitude)],tempres,nodata,inrain,intime,inlatitude,inlongitude,catx,caty,catmax,domainmask       
    else:
        return [xres,yres], [len(inlatitude),len(inlongitude)],[np.min(inlongitude),np.max(inlongitude)+xres,np.min(inlatitude)-yres,np.max(inlatitude)],tempres,nodata
    
#==============================================================================
# Get things set up_LEGACY VERSION
#==============================================================================
def rainprop_setup_LEGACY(infile,catalog=False):
    if catalog:
        inrain,intime,inlatitude,inlongitude,catx,caty,catmax,_=readcatalog_LEGACY(infile)
    else:
        inrain,intime,inlatitude,inlongitude=readnetcdf(infile)
    
    if len(inlatitude.shape)>1 or len(inlongitude.shape)>1:
        inlatitude=inlatitude[:,0]          # perhaps would be safer to have an error here...
        inlongitude=inlongitude[0,:]        # perhaps would be safer to have an error here...
    subdiff=np.abs(np.subtract(inlatitude[1:],inlatitude[0:-1]))
    yres=np.mean(subdiff[subdiff>0.0001])
    if np.allclose(subdiff[subdiff>0.0001],yres,rtol=1e-03)==False:
        sys.exit("North-South resolution is not constant. RainyDay cannot support that.")
    subdiff=np.abs(np.subtract(inlongitude[1:],inlongitude[0:-1]))
    xres=np.mean(subdiff[subdiff>0.0001])
    if np.allclose(subdiff[subdiff>0.0001],xres,rtol=1e-03)==False:
        sys.exit("East-West resolution is not constant. RainyDay cannot support that.")
    

    unqtimes=np.unique(intime)   
    if len(unqtimes)>1:
        tempres=np.min(unqtimes[1:]-unqtimes[0:-1])   # temporal resolution
    else:
        tempres=np.float32(1440.)
        tempres=tempres.astype('timedelta64[m]')      # temporal resolution in minutes-haven't checked to make sure this works right
        
    if len(intime)*np.float32(tempres)!=1440. and catalog==False:
        sys.exit("RainyDay requires daily input files, but has detected something different.")
    if np.allclose(np.array(np.subtract(unqtimes[1:],unqtimes[0:-1]),dtype='float32'),np.float32(tempres),rtol=1e-03)==False and catalog==False:
        sys.exit("Temporal resolution is not constant. RainyDay cannot support that.") 
    tempres=np.int(np.float32(tempres))
        
    nodata=np.unique(inrain[inrain<0.])
    if len(nodata)>1:
        sys.exit("More than one missing value flag.")
    elif len(nodata)==0 and catalog==False:
        print("Warning: Missing data flag is ambiguous.")
        nodata==-999.
    elif catalog:
        nodata=-999.
    else:
        nodata=nodata[0]

    if catalog:
        return [xres,yres], [len(inlatitude),len(inlongitude)],[np.min(inlongitude),np.max(inlongitude),np.min(inlatitude),np.max(inlatitude)],tempres,nodata,inrain,intime,inlatitude,inlongitude,catx,caty,catmax      
    else:
        return [xres,yres], [len(inlatitude),len(inlongitude)],[np.min(inlongitude),np.max(inlongitude)+xres,np.min(inlatitude)-yres,np.max(inlatitude)],tempres,nodata
    


#==============================================================================
# READ REALIZATION
#==============================================================================

def readrealization(rfile):
    infile=Dataset(rfile,'r')
    outrain=np.array(infile.variables['rainrate'][:])
    outtime=np.array(infile.variables['time'][:],dtype='datetime64[m]')
    outlatitude=np.array(infile.variables['latitude'][:])
    outlongitude=np.array(infile.variables['longitude'][:])
    outlocx=np.array(infile.variables['xlocation'][:])
    outlocy=np.array(infile.variables['ylocation'][:])
    outmax=np.array(infile.variables['basinrainfall'][:])
    outreturnperiod=np.array(infile.variables['returnperiod'][:])
    outstormnumber=np.array(infile.variables['stormnumber'][:])
    origstormnumber=np.array(infile.variables['original_stormnumber'][:])
    #outstormtime=np.array(infile.variables['stormtimes'][:],dtype='datetime64[m]')
    
    infile.close()
    return outrain,outtime,outlatitude,outlongitude,outlocx,outlocy,outmax,outreturnperiod,outstormnumber,origstormnumber


#==============================================================================
# READ A PREGENERATED SST DOMAIN FILE
#==============================================================================

def readdomainfile(rfile,inbounds=False):
    infile=Dataset(rfile,'r')
    if np.any(inbounds!=False):
        outmask=np.array(infile.variables['domain'][inbounds[3]:inbounds[2]+1,inbounds[0]:inbounds[1]+1])
        outlatitude=np.array(infile.variables['latitude'][inbounds[3]:inbounds[2]+1])
        outlongitude=np.array(infile.variables['longitude'][inbounds[0]:inbounds[1]+1])         
    else:
        outmask=np.array(infile.variables['domain'][:])
        outlatitude=np.array(infile.variables['latitude'][:])
        outlongitude=np.array(infile.variables['longitude'][:])
    infile.close()
    return outmask,outlatitude,outlongitude


#==============================================================================
# "Rolling sum" function to correct for short-duration biases
#==============================================================================
    
def rolling_sum(a, n):
    ret = np.nancumsum(a, axis=0, dtype=float)
    ret[n:,:] = ret[n:,:] - ret[:-n,: ]
    return ret[n - 1:,: ]


#==============================================================================
# Distance between two points
#==============================================================================
    
def latlondistance(lat1,lon1,lat2,lon2):    
    #if len(lat1)>1 or len(lon1)>1:
    #    sys.exit('first 2 sets of points must be length 1');

    R=6371000;
    dlat=np.radians(lat2-lat1)
    dlon=np.radians(lon2-lon1)
    a=np.sin(dlat/2.)*np.sin(dlat/2.)+np.cos(np.radians(lat1))*np.cos(np.radians(lat2))*np.sin(dlon/2.)*np.sin(dlon/2.);
    c=2.*np.arctan2(np.sqrt(a),np.sqrt(1-a))
    return R*c
 
#==============================================================================
# rescaling functions
#==============================================================================
        
@jit(fastmath=True)
def intenseloop(intenserain,tempintense,xlen_wmask,ylen_wmask,maskheight,maskwidth,trimmask,mnorm,domainmask):
    for i in range(0,xlen_wmask*ylen_wmask):
        y=i//xlen_wmask
        x=i-y*xlen_wmask
        if np.equal(domainmask[y,x],1.) and  np.any(np.isnan(intenserain[:,y,x]))==False:
        # could probably get this working in nopython if I coded the multiplication explicitly, rather than using using the axis argument of nansum, which isn't numba-supported
            tempintense[:,y,x]=np.sum(np.multiply(intenserain[:,y:(y+maskheight),x:(x+maskwidth)],trimmask),axis=(1,2))/mnorm    
        else:
            tempintense[:,y,x]=np.nan
    return tempintense

@jit(nopython=True,fastmath=True)
def intense_corrloop(intenserain,intensecorr,homerain,xlen_wmask,ylen_wmask,mnorm,domainmask):   
    for i in range(0,xlen_wmask*ylen_wmask): 
        y=i//xlen_wmask
        x=i-y*xlen_wmask
        if np.equal(domainmask[y,x],1.) and  np.any(np.isnan(intenserain[:,y,x]))==False:
            intensecorr[y,x]=np.corrcoef(homerain,intenserain[:,y,x])[0,1]
        else:
            intensecorr[y,x]=np.nan
    return intensecorr


#==============================================================================
# read arcascii files
#==============================================================================

def read_arcascii(asciifile):
    temp=linecache.getline(asciifile, 1)
    temp=linecache.getline(asciifile, 2)
    xllcorner=linecache.getline(asciifile, 3)
    yllcorner=linecache.getline(asciifile, 4)
    cellsize=linecache.getline(asciifile, 5)
    nodata=linecache.getline(asciifile, 6)
    
    #ncols=np.int(ncols.split('\n')[0].split(' ')[-1])
    #nrows=np.int(nrows.split('\n')[0].split(' ')[-1])
    
    xllcorner=np.float(xllcorner.split('\n')[0].split(' ')[-1])
    yllcorner=np.float(yllcorner.split('\n')[0].split(' ')[-1])
    
    cellsize=np.float(cellsize.split('\n')[0].split(' ')[-1])
    nodata=np.float(nodata.split('\n')[0].split(' ')[-1])
    
    #asciigrid = np.loadtxt(asciifile, skiprows=6)
    asciigrid = np.array(pd.read_csv(asciifile, skiprows=6,delimiter=' ', header=None),dtype='float32')
    nrows=asciigrid.shape[0]
    ncols=asciigrid.shape[1]
    
    asciigrid[np.equal(asciigrid,nodata)]=np.nan

    return asciigrid,ncols,nrows,xllcorner,yllcorner,cellsize