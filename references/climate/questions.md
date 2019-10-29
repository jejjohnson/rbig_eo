# Q n A


## Peer - Oct 26th

> What does CMIP5 stand for here? One climate model from the archive? A multi-model-mean? Which type of simulation (AMIP, coupled ocean)?

The exact specification of the CMIP5 is: 
* experiment RCP 8.5
* the GISS E2 R model 
* the ensemble member R1i1P1. 
* 
* I had no reason to choose RCP over any other but I believe I remember seeing that the RCP 8.5 tended to be the most extreme outcome. I also chose the GISS E2 R and R1i1P1 because thet combination had the most overlap in time with the other reanalysis methods. So I could look into using other models/experiments if you have any suggestions. 

> From a climate science point of view, this would intuitively rather be the other way around. I guess this motivates your further work suggestions?

So that actually makes sense and after talking with Valero about that, we were thinking about inverting the exact scenario I mentioned to what you mentioned: take one sample as a point in time and each feature would be the surrounding spatial locations. That way we can see how the entropy evolves with time spatially. The only reason I did what I did was to promote looking at from a spatial vs temporal feature perspective. But I personally think the spatial features argument makes more sense in this case.

A hypothesis could be: **I think that there are more spatial "extreme events" therefore there is more expected uncertainty. Do the climate models (or reanalysis) respect this property.**

> It would be interesting to understand why for mean sea level pressure ERA and NCEP don't agree quite as well. Do you take spatial weighting of the grid cells into account when doing your calculations?


MSLP - NCEP vs ERA: So perhaps I’m not seeing it but I think the MSLP agrees a bit better than the Surface Pressure (SP). Regarding the reweighing, I used the Earth System Modeling Framework (ESMF). It was one that was recommended to me by a climate scientist. But after browsing through the docs I decided to use the simple nearest neighbour interpolation [method](http://www.earthsystemmodeling.org/esmf_releases/public/ESMF_7_1_0r/ESMF_refdoc/node5.html#SECTION05012300000000000000) because it was the simplest and most intuitive. But based on what you said and if I understood correctly, perhaps I could use the [conservative](http://www.earthsystemmodeling.org/esmf_releases/public/ESMF_7_1_0r/ESMF_refdoc/node5.html#SECTION05012500000000000000) regridding technique which uses a spatial area I believe. I'll look into that.

> Validation

So when I mean validation, I mean purely from a signal processing point of view. In many of the papers I've seen, most are looking at it from a physical perspective. Understandable. But I was wondering what does the community do in order to compare the output signals of two climate model. I think the Taylor paper was more or less what I was looking for. But I was expecting a bit more in terms of perhaps local correlation measures with respect to space and time; not just the entire signal.
