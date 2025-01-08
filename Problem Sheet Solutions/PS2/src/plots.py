#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jun  8 13:07:23 2023

@author: karan_bania
"""

import util
import matplotlib.pyplot as plt

Xa, Ya = util.load_csv('../data/ds1_a.csv', add_intercept=False)
Xb, Yb = util.load_csv('../data/ds1_b.csv', add_intercept=False)

#print(-1 in Yb)
plt.close('all')
util.plot_points(Xa, Ya)
plt.savefig("./datasetA.png")
plt.close('all')
util.plot_points(Xb, Yb)
plt.savefig("./datasetB.png")
plt.close('all')