# SLOSS Simulator

## Description
The single large or several small (SLOSS) debate is a classic problem in conservation biology which originated in the 1970s but remains an important consideration today. The dilemma central to the SLOSS debate is whether conservationists should consolidate the protected habitat into a single large reserve or several small reserves, given the same amount of total land area available for protection. Today, scientists generally agree that neither approach is uniformly better than the other. Rather, different goals and unique species characteristics determine the best approach to distributing protected habitat. However, despite advances in our understanding of habitat planning, it can still be difficult to know whether SLOSS reserves are better for conserving species under different sets of conditions. This interactive visualization tool helps users intuitively understand how SLOSS reserves perform as system parameters change.

## Features
* Highly visual to support intuition of population dynamics
* Two-column GUI to compare SLOSS reserves
* System parameter sliders are easy to use and encourage user exploration
* Logistic function models growth rate and carrying capacity
* Edge effect parameter captures species preferences for edge or inland habitat
* Patchiness represents potential habitat heterogeneity and irregularity
* Dispersal and metapopulation dynamics modeled using a Gaussian kernel to distribute migrants
* Localized disturbances simulate real-world events such as diseases and natural disasters
* Preset examples to show clear examples of when single large or several small may be favored

## Getting Started
### Dependencies
* numpy
* scipy
* plotly
* streamlit

### Executing program
Navigate to the folder where the app is saved and run 
```
streamlit run app.py
```
The SLOSS simulator will open in your browser.

## Authors
Megan Lim
Ai Omae
