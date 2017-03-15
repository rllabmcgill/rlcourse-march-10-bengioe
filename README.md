# March 10 assignment

## Implementation of KBSF

I implemented KBSF, by [Baretto, Precup & Pineau](http://jmlr.org/papers/volume17/13-134/13-134.pdf). Tested it on Mountain Car, Catch and Atari.

If your patience is high, it is theoretically possible to make it "work" on Catch and Atari, but my implementation not being very optimized, running this code with a lot of support vectors of high dimension will take considerable time.  
For example on breakout, after a few hours the agent learns to survive for 5000 frames (instead of dying in <600 frames like a random agent does).

See `KBSF.ipynb` for details, `main.py` for the implementation