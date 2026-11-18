.. Template documentation master file, created by
   sphinx-quickstart on Tue Oct 31 17:10:19 2023.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Find HI and deblend
======================================




Source finding
----------------
AGE finds HI sources in data cubes by automatically running SoFiA (https://gitlab.com/SoFiA-Admin/SoFiA-2) on the provided data. After running SoFiA, it collects the detected sources and prepares them for further analysis, including deblending overlapping sources.

Deblending
----------------
Once the sources are identified, AGE performs deblending through the deblend_sofia_detections package (https://github.com/PeterKamphuis/deblend_sofia_detections) to separate connected HI sources. 
This step ensures that each source can be analyzed individually, improving the accuracy of subsequent measurements and analyses.
The code then moves on to seperate the regularly rotating disc from anomalous gas components.
