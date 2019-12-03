# Description
The goal of this project is to perform anomaly detection in IP traffic. The methodology in this
case is to build a profile of each IP address as graphlets. We will first generate the graphlets in
the initial traffic classification according to Karagiannis et al [1] and then further
generating the profile graphlets to perform anomaly detection. With these graphlets, we will then
build a model using Support Vector Machine to distinguish normal from malicious end hosts
from an annotated trace. The last step will be to try to detect attack in a not annotated trace.
This project is an implementation of Profiling the End Host paper [2].



# References
[1] Thomas Karagiannis, Konstantina Papagiannaki, and Michalis Faloutsos. BLINC: Multilevel Traffic Classification in the Dark

[2] Thomas Karagiannis, Konstantina Papagiannaki, Nina Taft, and Michalis Faloutsos. Profiling the End Host
