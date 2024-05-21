Download and visualize Images and Metadata from Mars2020 EDL procedure

## Download

To download the descent images paste the following into your terminal:

``` source res/pdsimg-atlas-wget_2024-05-16T05_54_47_719_clean.bat ```

To download the HiRise Orthomosaic and DTM go to:

``` https://astrogeology.usgs.gov/search/map/mars_2020_terrain_relative_navigation_hirise_orthorectified_image_mosaic``` 
``` https://astrogeology.usgs.gov/search/map/Mars/Mars2020/JEZ_hirise_soc_006_DTM_MOLAtopography_DeltaGeoid_1m_Eqc_latTs0_lon0_blend40``` 

## Viewer

To view images run:

```python3 view.py```

## Requirements

Install requirements using conda/mamba

```mamba install numpy opencv pdr```

