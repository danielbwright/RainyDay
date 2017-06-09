#==============================================================================
# WELCOME
#==============================================================================


#    Welcome to RainyDay, a framework for coupling remote sensing precipitation
#    fields with Stochastic Storm Transposition for assessment of rainfall-driven hazards.
#    Copyright (C) 2015  Daniel Benjamin Wright (danielb.wright@wisc.edu)
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




#==============================================================================
# THIS DOCUMENT CONTAINS VARIOUS FUNCTIONS NEEDED TO RUN RainyDay
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
#from scipy import weave
#from scipy.weave import converters
#import pylab
from pytz import timezone
import pytz
utc = pytz.utc  
import pickle
from mpl_toolkits.basemap import Basemap, addcyclic
from matplotlib.patches import Polygon   
from scipy import stats
from netCDF4 import Dataset, num2date, date2num
import gdal
from numba import jit

# plotting stuff, really only needed for diagnostic plots
import matplotlib.pyplot as plt
import matplotlib
from matplotlib.colors import LogNorm 


HRAP="+proj=stere +lat_0=90 +lat_ts=60 +lon_0=-105 +k=1 +x_0=0 +y_0=0 +ellps=WGS84 +units=m +no_defs"          # RainyDay doesn't currently support anything other than geographic projections
GEOG="+proj=longlat +ellps=WGS84 +datum=WGS84 +no_defs"





# np.nanmean doesn't exist on some older builds of numpy
def mynanmean(x,axis):
    x = np.ma.filled(np.ma.masked_array(x,np.isnan(x)).mean(axis), fill_value=np.nan)
    return x
    
def find_nearest(array,value):
    idx = (np.abs(array-value)).argmin()
    return idx


#==============================================================================
# LOOP TO DO SPATIAL SEARCHING FOR MAXIMUM RAINFALL LOCATION AT EACH TIME STEP
# THIS IS THE CORE OF THE STORM CATALOG CREATION TECHNIQUE
#==============================================================================
    
def catalogweave(temparray,trimmask,xlen,ylen,maskheight,maskwidth,rainsum):
    rainsum[:]=0.
    code= """
        #include <stdio.h>
        int i,j,x,y;
        for (x=0;x<xlen;x++) {
            for (y=0;y<ylen;y++) {
                for (j=0;j<maskheight;j++) {
                    for (i=0;i<maskwidth;i++) {
                        rainsum(y,x)=rainsum(y,x)+temparray(y+j,x+i)*trimmask(j,i);                     
                    }                               
                }
            }                      
        }
    """
    vars=['temparray','trimmask','xlen','ylen','maskheight','maskwidth','rainsum']
    sp.weave.inline(code,vars,type_converters=converters.blitz,compiler='gcc')
    rmax=np.nanmax(rainsum)
    wheremax=np.where(rainsum==rmax)
    return rmax, wheremax[0][0], wheremax[1][0]
    


def catalogAlt(temparray,trimmask,xlen,ylen,maskheight,maskwidth,rainsum):
    rainsum[:]=0.
    for i in range(0,(ylen)*(xlen)):
        y=i/xlen
        x=i-y*xlen
        #print x,y
        rainsum[y,x]=np.nansum(np.multiply(temparray[(y):(y+maskheight),(x):(x+maskwidth)],trimmask))
    #wheremax=np.argmax(rainsum)
    rmax=np.nanmax(rainsum)
    wheremax=np.where(rainsum==rmax)
    
    return rmax, wheremax[0][0], wheremax[1][0]

@jit
def catalogNumba(temparray,trimmask,xlen,ylen,maskheight,maskwidth,rainsum):
    rainsum[:]=0.
    for i in range(0,(ylen)*(xlen)):
        y=i/xlen
        x=i-y*xlen
        #print x,y
        rainsum[y,x]=np.nansum(np.multiply(temparray[(y):(y+maskheight),(x):(x+maskwidth)],trimmask))
    #wheremax=np.argmax(rainsum)
    rmax=np.nanmax(rainsum)
    wheremax=np.where(rainsum==rmax)
    return rmax, wheremax[0][0], wheremax[1][0]

def SSTalt(passrain,whichx,whichy,trimmask,xmin,xmax,ymin,ymax,maskheight,maskwidth):
    rainsum=np.zeros((len(whichx)),dtype='float32')
    nreals=len(rainsum)

    for i in range(0,nreals):
        rainsum[i]=np.nansum(np.multiply(passrain[(whichy[i]) : (whichy[i]+maskheight) , (whichx[i]) : (whichx[i]+maskwidth)],trimmask))
    return rainsum
      

#==============================================================================
# THIS VARIANT IS SIMPLER AND UNLIKE SSTWRITE, IT ACTUALLY WORKS RELIABLY!
#==============================================================================
def SSTwriteAlt(catrain,rlzx,rlzy,rlzstm,trimmask,xmin,xmax,ymin,ymax,maskheight,maskwidth):
    nyrs=np.int(rlzx.shape[0])
    raindur=np.int(catrain.shape[1])
    outrain=np.zeros((nyrs,raindur,maskheight,maskwidth),dtype='float32')
    unqstm,unqind,unqcnts=np.unique(rlzstm,return_inverse=True,return_counts=True)
    #ctr=0
    for i in range(0,len(unqstm)):
        unqwhere=np.where(unqstm[i]==rlzstm)[0]
        for j in unqwhere:
            #ctr=ctr+1
            #print ctr
            outrain[j,:]=np.multiply(catrain[unqstm[i],:,(rlzy[j]) : (rlzy[j]+maskheight) , (rlzx[j]) : (rlzx[j]+maskwidth)],trimmask)
    return outrain
       

#==============================================================================
# THIS VARIANT IS SAME AS ABOVE, BUT HAS A MORE INTERESTING RAINFALL PREPENDING PROCEDURE
#==============================================================================

def SSTwriteAltPreCat(catrain,rlzx,rlzy,rlzstm,trimmask,xmin,xmax,ymin,ymax,maskheight,maskwidth,precat,ptime):    
    catyears=ptime.astype('datetime64[Y]').astype(int)+1970
    ptime=ptime.astype('datetime64[M]').astype(int)-(catyears-1970)*12+1
    nyrs=np.int(rlzx.shape[0])
    raindur=np.int(catrain.shape[1]+precat.shape[1])
    outrain=np.zeros((nyrs,raindur,maskheight,maskwidth),dtype='float32')
    unqstm,unqind,unqcnts=np.unique(rlzstm,return_inverse=True,return_counts=True)

    for i in range(0,len(unqstm)):
        unqwhere=np.where(unqstm[i]==rlzstm)[0]
        unqmonth=ptime[unqstm[i]]
        pretimeind=np.where(np.logical_and(ptime>unqmonth-2,ptime<unqmonth+2))[0]
        for j in unqwhere:
            temprain=np.concatenate((np.squeeze(precat[np.random.choice(pretimeind, 1),:,(rlzy[j]) : (rlzy[j]+maskheight) , (rlzx[j]) : (rlzx[j]+maskwidth)],axis=0),catrain[unqstm[i],:,(rlzy[j]) : (rlzy[j]+maskheight) , (rlzx[j]) : (rlzx[j]+maskwidth)]),axis=0)
            outrain[j,:]=np.multiply(temprain,trimmask)
    return outrain
    

#==============================================================================
# SAME AS ABOVE, BUT HANDLES STORM ROTATION
#==============================================================================    
    
def SSTwriteAltPreCatRotation(catrain,rlzx,rlzy,rlzstm,trimmask,xmin,xmax,ymin,ymax,maskheight,maskwidth,precat,ptime,delarray,rlzanglebin,rainprop):
#def SSTwriteAltPreCatRotation(catrain,rlzx,rlzy,rlzstm,trimmask,xmin,xmax,ymin,ymax,maskheight,maskwidth,precat,ptime,delarray,rlzanglebin):
    catyears=ptime.astype('datetime64[Y]').astype(int)+1970
    ptime=ptime.astype('datetime64[M]').astype(int)-(catyears-1970)*12+1
    nyrs=np.int(rlzx.shape[0])
    raindur=np.int(catrain.shape[1]+precat.shape[1])
    outrain=np.zeros((nyrs,raindur,maskheight,maskwidth),dtype='float32')
    unqstm,unqind,unqcnts=np.unique(rlzstm,return_inverse=True,return_counts=True)      # unqstm is the storm number

    for i in range(0,len(unqstm)):
        unqwhere=np.where(unqstm[i]==rlzstm)[0]
        unqmonth=ptime[unqstm[i]]
        pretimeind=np.where(np.logical_and(ptime>unqmonth-2,ptime<unqmonth+2))[0]
        for j in unqwhere:
            inrain=catrain[unqstm[i],:].copy()
            
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
                #inrain[k,:]=temprain
            
            temprain=np.concatenate((np.squeeze(precat[np.random.choice(pretimeind, 1),:,(rlzy[j]) : (rlzy[j]+maskheight) , (rlzx[j]) : (rlzx[j]+maskwidth)],axis=0),inrain[:,(rlzy[j]) : (rlzy[j]+maskheight) , (rlzx[j]) : (rlzx[j]+maskwidth)]),axis=0)

            outrain[j,:]=np.multiply(temprain,trimmask)
    return outrain
       

#==============================================================================
# SAME AS ABOVE, BUT A BIT MORE DYNAMIC IN TERMS OF SPINUP
#==============================================================================    
def SSTspin_write_v2(catrain,rlzx,rlzy,rlzstm,trimmask,xmin,xmax,ymin,ymax,maskheight,maskwidth,precat,ptime,rainprop,rlzanglebin=None,delarray=None,spin=False,flexspin=True,samptype='uniform',cumkernel=None,rotation=False):
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
        
        if spin==True and flexspin==True:
            if samptype=='kernel':
                rndloc=np.random.random_sample(len(unqwhere))
                shiftprex,shiftprey=weavekernel(rndloc,cumkernel)
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
    
    
#==============================================================================
# LOOP FOR KERNEL BASED STORM TRANSPOSITION
# THIS FINDS THE TRANSPOSITION LOCATION FOR EACH REALIZATION IF YOU ARE USING THE KERNEL-BASED RESAMPLER
# IF I CONFIGURE THE SCRIPT SO THE USER CAN PROVIDE A CUSTOM RESAMPLING SCHEME, THIS WOULD PROBABLY WORK FOR THAT AS WELL
#==============================================================================    
def weavekernel(rndloc,cumkernel):
    nlocs=len(rndloc)
    nrows=cumkernel.shape[0]
    ncols=cumkernel.shape[1]
    tempx=np.empty((len(rndloc)),dtype="int32")
    tempy=np.empty((len(rndloc)),dtype="int32")
    code= """
        #include <stdio.h>
        int i,x,y,brklp;
        double prevprob;
        for (i=0;i<nlocs;i++) {
            prevprob=0.0;
            brklp=0;
            for (y=0; y<nrows; y++) {
                for (x=0; x<ncols; x++) {
                    if ( (rndloc(i)<=cumkernel(y,x)) && (rndloc(i)>prevprob) ) {
                        tempx(i)=x;
                        tempy(i)=y;
                        prevprob=cumkernel(y,x);
                        brklp=1;
                        break;
                    }                     
                }
                if (brklp==1) {
                    break;                    
                }                         
            }   
        }
    """
    vars=['rndloc','cumkernel','nlocs','nrows','ncols','tempx','tempy']
    sp.weave.inline(code,vars,type_converters=converters.blitz,compiler='gcc')
    return tempx,tempy
    
    
def pykernel(rndloc,cumkernel):
    nlocs=len(rndloc)
    nrows=cumkernel.shape[0]
    ncols=cumkernel.shape[1]
    tempx=np.empty((len(rndloc)),dtype="int32")
    tempy=np.empty((len(rndloc)),dtype="int32")
    #flatkern=cumkernel.flatten()
    flatkern=np.append(0.,cumkernel.flatten())
    
    for i in range(0,nlocs):
        whereind=np.where(np.logical_and(rndloc[i]>flatkern[0:-1],rndloc[i]<=flatkern[1:]))[0][0]
        y=whereind/ncols
        x=whereind-y*ncols        
        tempx[i]=x
        tempy[i]=y
    return tempx,tempy

@jit
def numbakernel(rndloc,cumkernel):
    nlocs=len(rndloc)
    nrows=cumkernel.shape[0]
    ncols=cumkernel.shape[1]
    tempx=np.empty((len(rndloc)),dtype="int32")
    tempy=np.empty((len(rndloc)),dtype="int32")
    #flatkern=cumkernel.flatten()
    flatkern=np.append(0.,cumkernel.flatten())
    
    for i in range(0,nlocs):
        whereind=np.where(np.logical_and(rndloc[i]>flatkern[0:-1],rndloc[i]<=flatkern[1:]))[0][0]
        y=whereind/ncols
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

    rangex=np.arange(rainprop.bndbox[0],rainprop.bndbox[1]+rainprop.spatialres[0]/1000,rainprop.spatialres[0])
    rangey=np.arange(rainprop.bndbox[3],rainprop.bndbox[2]-rainprop.spatialres[1]/1000,-rainprop.spatialres[1])

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
# THIS USES GDAL COMMANDS FROM THE OS TO RASTERIZE
#==============================================================================
def rastermaskGDAL(shpname,shpproj,rainprop,masktype,fullpath):            
    bndbox=np.array(rainprop.subind)
    bndcoords=np.array(rainprop.subextent)
    
    if rainprop.projection==GEOG:
        xdim=np.shape(np.linspace(bndcoords[0],bndcoords[1],rainprop.subind[1]-rainprop.subind[0]))[0]     
        ydim=np.shape(np.linspace(bndcoords[2],bndcoords[3],rainprop.subind[2]-rainprop.subind[3]))[0]    
    elif rainprop.projection==HRAP:
        xdim=np.int((bndcoords[1]-bndcoords[0])/rainprop.spatialres[0])
        ydim=np.int((bndcoords[3]-bndcoords[2])/rainprop.spatialres[1])
        
    rastertemplate=np.zeros((ydim,xdim),dtype='float32')

    if masktype=='simple':
        print 'creating simple mask (0s and 1s)'
        #os.system('gdal_rasterize -at -burn 1.0 -te '+str(rainprop.subextent[0])+' '+str(rainprop.subextent[2])+' '+str(rainprop.subextent[1])+' '+str(rainprop.subextent[3])+' -tr '+str(rainprop.spatialres[0])+' '+str(rainprop.spatialres[1])+' -ts '+str(np.int(rainprop.subdimensions[1]))+' '+str(np.int(rainprop.subdimensions[0]))+' -ot Float32 '+shpname+' '+fullpath+'/temp.tiff');
        
        os.system('gdal_rasterize -at -burn 1.0 -te '+"%.9f"%(rainprop.subextent[0])+' '+"%.9f"%(rainprop.subextent[2])+' '+"%.9f"%(rainprop.subextent[1])+' '+"%.9f"%(rainprop.subextent[3])+' -tr '+"%.9f"%(rainprop.spatialres[0])+' '+"%.9f"%(rainprop.spatialres[1])+' -ts '+"%.9f"%(np.int(rainprop.subdimensions[1]))+' '+"%.9f"%(np.int(rainprop.subdimensions[0]))+' -ot Float32 '+shpname+' '+fullpath+'/temp.tiff')
        
        ds=gdal.Open(fullpath+'/temp.tiff')
        rastertemplate=np.array(ds.GetRasterBand(1).ReadAsArray(),dtype='float32')
        os.system('rm '+fullpath+'/temp.tiff')
    elif masktype=="fraction":
        print 'creating fractional mask (range from 0.0-1.0)'
        #os.system('gdal_rasterize -at -burn 1.0 -te '+str(rainprop.subextent[0])+' '+str(rainprop.subextent[2])+' '+str(rainprop.subextent[1])+' '+str(rainprop.subextent[3])+' -tr '+str(rainprop.spatialres[0]/10.)+' '+str(rainprop.spatialres[1]/10.)+' -ts '+str(np.int(rainprop.subdimensions[1])*10)+' '+str(np.int(rainprop.subdimensions[0])*10)+' -ot Float32 '+shpname+' '+fullpath+'/temp.tiff');
        #os.system('gdalwarp -r average -te '+str(rainprop.subextent[0])+' '+str(rainprop.subextent[2])+' '+str(rainprop.subextent[1])+' '+str(rainprop.subextent[3])+' -ts '+str(np.int(rainprop.subdimensions[1]))+' '+str(np.int(rainprop.subdimensions[0]))+' -overwrite '+fullpath+'/temp.tiff '+fullpath+'/tempAGG.tiff');

        os.system('gdal_rasterize -at -burn 1.0 -te '+"%.9f"%(rainprop.subextent[0])+' '+"%.9f"%(rainprop.subextent[2])+' '+"%.9f"%(rainprop.subextent[1])+' '+"%.9f"%(rainprop.subextent[3])+' -tr '+"%.9f"%(rainprop.spatialres[0]/10.)+' '+"%.9f"%(rainprop.spatialres[1]/10.)+' -ts '+"%.9f"%(np.int(rainprop.subdimensions[1])*10)+' '+"%.9f"%(np.int(rainprop.subdimensions[0])*10)+' -ot Float32 '+shpname+' '+fullpath+'/temp.tiff')
        os.system('gdalwarp -r average -te '+"%.9f"%(rainprop.subextent[0])+' '+"%.9f"%(rainprop.subextent[2])+' '+"%.9f"%(rainprop.subextent[1])+' '+"%.9f"%(rainprop.subextent[3])+' -ts '+"%.9f"%(np.int(rainprop.subdimensions[1]))+' '+"%.9f"%(np.int(rainprop.subdimensions[0]))+' -overwrite '+fullpath+'/temp.tiff '+fullpath+'/tempAGG.tiff')

        ds=gdal.Open(fullpath+'/tempAGG.tiff')
        rastertemplate=np.array(ds.GetRasterBand(1).ReadAsArray(),dtype='float32')
        os.system('rm '+fullpath+'/temp.tiff')
        os.system('rm '+fullpath+'/tempAGG.tiff')
    else:
        sys.exit("You entered an incorrect mask type, options are 'simple' or 'fraction'")
        
    rastertemplate=np.array(rastertemplate[:])
    
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
    rainrate=dataset.createVariable('rainrate',np.float32,('nyears','time','outlat','outlon'),zlib=True,complevel=4) 
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
# READ RAINFALL FILE FROM NETCDF
#==============================================================================
def readnetcdf(rfile):
    infile=Dataset(rfile,'r')
    outrain=np.array(infile.variables['rainrate'][:])
    outtime=np.array(infile.variables['time'][:],dtype='datetime64[m]')
    outlatitude=np.array(infile.variables['latitude'][:])
    outlongitude=np.array(infile.variables['longitude'][:])
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
    infile.close()
    return outrain,outtime,outlatitude,outlongitude,outlocx,outlocy,outmax,outmask

    
#==============================================================================
# WRITE RAINFALL FILE TO NETCDF
#==============================================================================
def writecatalog(catrain,catmax,catx,caty,cattime,latrange,lonrange,catalogname,nstorms,gridmask,parameterfile):
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
    rainrate=dataset.createVariable('rainrate',np.float32,('nstorms','time','outlat','outlon',),zlib=True,complevel=4) 
    basinrainfall=dataset.createVariable('basinrainfall',np.float32,('nstorms')) 
    xlocation=dataset.createVariable('xlocation',np.int32,('nstorms')) 
    ylocation=dataset.createVariable('ylocation',np.int32,('nstorms')) 
    gmask=dataset.createVariable('gridmask',np.float32,('outlat','outlon',)) 
    
    # Global Attributes
    with open(parameterfile, "r") as myfile:
        params=myfile.read()
    myfile.close
    dataset.description=params

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
        print "Warning: the file date in the YYYYMMDD format was ambiguous."
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
    if isinstance(excludemonths, (bool))==False:
        for j in excludemonths:
            finclude[fmonth==j]=False
        
    flist=flist[finclude==True]
    ftime=ftime[finclude==True]
        
    fsort=np.array(sorted(enumerate(ftime), key=lambda x: x[1]))
    sortind=fsort[:,0]
    flist=flist[sortind]
    return flist


#==============================================================================
# READ REALIZATION
#==============================================================================
def rainprop_setup(infile,catalog=False):
    if catalog:
        inrain,intime,inlatitude,inlongitude,catx,caty,catmax,_=readcatalog(infile)
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
        print "Warning: Missing data flag is ambiguous."
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
# DIFFERENT PROJECTIONS
#==============================================================================
HRAP="+proj=stere +lat_0=90 +lat_ts=60 +lon_0=-105 +k=1 +x_0=0 +y_0=0 +ellps=WGS84 +units=m +no_defs"
GEOG="+proj=longlat +ellps=WGS84 +datum=WGS84 +no_defs"


