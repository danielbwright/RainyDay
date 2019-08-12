Here is an example application of RainyDay to Big Thompson watershed
upstream of Olympus Dam, in northern Colorado. This is a small-to-medium
sized watershed, and so a duration of 72 hours is used. Diagnostic plots reveal significant variations in storm catalog properties over the region, which led us to select the 'nonuniform' option for the TRANSPOSITION field. Visual inspection of the 350 storm rainfall maps led to the decision to exclude a small number from subsequent frequency analysis, since they appeared to have unrealistic radar 'artifacts'. In the .sst file, this exclusion looks like: EXCLUDESTORMS	45,130,132,168,235,333,343

DURATIONCORRECTION is not used for this, since the watershed is moderately large and the rainfall duration is long. But, it could be used!


This example uses the Stage IV radar+gage dataset, which is known to have some deficiencies in mountainous terrain. This example is for illustrative purposes only.

The Stage IV data can be downloaded here: https://drive.google.com/open?id=0B6s2whbrq8qEb29FelgtU3A1bUk. Note that without this data, you cannot create a new storm catalog, but you can use the existing storm catalog (see below) within RainyDay.

Contents of Example Directory:
1. BigThompson72hr_example.sst: this is where the parameters and options of the analysis are defined

2.Two irregular domains are provided:
2a. "Hand-drawn" shapefile (using the instructions given in the RainyDay User's Guide): BigThompsonSSTdomain-polygon.shp 
This was based upon the Google Earth 'BigThompsonSSTdomain.kmz'
For those inexperienced with GIS data: the other files of the same name (i.e. BigThompsonSSTdomain-polygon.shx, etc.) are necessary components of the shapefile and should not be deleted.
2b. Experimentally-determined netcdf-based domain: PRISM_domain_BigThompson.nc

3. Storm catalog file: StageIV_72hour_BigThompson_Example.nc. This was generated using the hand-drawn shapefile described above.

4. Subdirectory of diagnostic figures. The storm total maps for the 350 storms have been visually screened

5. RainyDay_BigThompsonExample.FreqAnalysis: the estimates of the specific return levels specified in the RETURNLEVELS field: 'RETURNLEVELS	2,5,10,25,50,100,200,500'. If one wanted to include all return levels, use 'RETURNLEVELS	all'

6. FrequencyAnalysis.png: the IDF info, plotted (really this is the same info as the .FreqAnalysis file

