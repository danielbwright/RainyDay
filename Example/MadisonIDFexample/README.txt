Here is an example application of RainyDay to IDF estimation in Madison, Wisconsin. Results of a 24-hour duration IDF analysis are provided, but this could easily be changed to other durations. A rectangular domain is used, though irregular domains could be used instead. Visual inspection of the storm rainfall maps led to the decision to exclude a small number from subsequent frequency analysis, since they appeared to have unrealistic radar 'artifacts'. In the .sst file, this exclusion looks like: EXCLUDESTORMS	460,459,452,435,430,417,416,412,297,292,274,273,230,218,215,213,145,102,101,86,72,45,21,14

DURATIONCORRECTION is used for this analysis, and is highly recommended for most IDF estimation applications, along with a large storm catalog.


This example uses the Stage IV radar+gage dataset, supplemented with additional Stage II data going back to 1996. Both are known to have some deficiencies. This example is for illustrative purposes only. However, it is interesting to compare these IDF results against those from NOAA Atlas 14 for the same location: https://hdsc.nws.noaa.gov/hdsc/pfds/pfds_map_cont.html

The Stage IV data can be downloaded here: https://drive.google.com/open?id=0B6s2whbrq8qEb29FelgtU3A1bUk. Note that without this data, you cannot create a new storm catalog, but you can use the existing storm catalog (see below) within RainyDay.

Contents of Example Directory:
1. Madison24hrIDF_example.sst: this is where the parameters and options of the analysis are defined

2. Storm catalog file: StageIV_Madison_ExampleCatalog.nc. This was generated using a rectangular domain, as specified in the .sst file

3. Subdirectory of diagnostic figures. The storm total maps for the 350 storms have been visually screened

4. RainyDay_MadisonExample.FreqAnalysis: the estimates of the specific return levels specified in the RETURNLEVELS field: 'RETURNLEVELS	2,5,10,25,50,100,200,500'. If one wanted to include all return levels, use 'RETURNLEVELS	all'

5. FrequencyAnalysis.png: the IDF info, plotted (really this is the same info as the .FreqAnalysis file

