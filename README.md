# BOLAR
A Python implementation of BOLAR, developed by <cite>[Zelinsky (2003)][1]</cite>.

## What it is

**BOLAR** -- Bank Of Local Analyzer Responses -- Is a model for measuring image difference. This is useful in visual Psychology/Neuroscience experiments when some measure of image differences is needed.

It works by filtering images through banks of oriented filters at different spatial scales. In this way, it is similar to, for example, the <cite>[HMAx][2]</cite> model by <cite>[Riesenhuber & Poggio (1999)][3]</cite>, a computational model of neural responses in visual cortex. However, BOLAR is not developed as a model of cortical responses, but as a simple and flexible model that can account for some experimental effects in Psychology on vison.

## How it works
The images are filtered with <cite>[Gaussian derivative filters][4]</cite> up to the 4rd order, at 4 spatial scales. Each order provided order + 1 filters, so 9 (2 + 3 + 4) filters. Nine f 
![ ](https://github.com/kalleknast/BOLAR/blob/master/bolar_filters.png  =250x275 "Filters")


[1]:http://www.psychology.sunysb.edu/gzelinsky-/index_htm_files/Z2003.pdf
[2]:http://maxlab.neuro.georgetown.edu/hmax.html
[3]:http://maxlab.neuro.georgetown.edu/docs/publications/nn99.pdf
[4]:http://campar.in.tum.de/Chair/HaukeHeibelGaussianDerivatives