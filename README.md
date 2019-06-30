
## Title: Is your clock-face cozie? A smartwatch methodology for the in-situ collection of occupant comfort data

P Jayathissa, M Quintana, T Sood, N. Narzarian, C. Miller 

Building and Urban Data Science Group, National University of Singapore (NUS), Singapore

University of New South Wales (UNSW), Australia

![Overviewimage](https://github.com/buds-lab/cisbat-cozie-paper/blob/master/Images/cozie-overview.jpg)

### Abstract:
Labelled human comfort data can be a valuable resource in optimising the built environment, and improving the wellbeing of individual occupants. The acquisition of labelled data however remains a challenge. This paper presents a methodology for the collection of in-situ occupant feedback data using a Fitbit smartwatch. The clock-face application cozie can be downloaded free-of-charge on the Fitbit store and tailored to fit a range of occupant comfort related experiments. In the initial trial of the app, fifteen users were given a smartwatch for one month and were prompted to give feedback on their thermal preferences. In one month, with minimal administrative overhead, 1460 labelled responses were collected. This paper demonstrates how these large data sets of human feedback can be analysed to reveal a range of results from building anomalies, occupant behaviour, occupant personality clustering, and general feedback related to the building. The paper also discusses limitations in the approach and the next phase of design of the platform.



## Reproducibility
In an effort to provide reproducible results, the `Data` folder includes the Python code and anonymized participant data for reproducing the graphics of this paper

Please use the ipython notebooks, which querry `cozie.csv`, which is a snapshop of the database when this paper was written. 

note that the `computeCozie.py ` is written as an exmample code for future researchers when querrying the influx database directly. If you wish to use this, please contact the authors, or submit an issue within this repository, and we will send you the necessary credentials file to access the data. 