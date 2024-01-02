# -*- coding: utf-8 -*-
"""
Created on Sun Dec 31 11:50:01 2023

@author: LuisJa
"""

import numpy as np
import py4dgeo

epoch1, epoch2 = py4dgeo.read_from_las(
    "ahk_2017_652900_5189100_gnd_subarea.laz",
    "ahk_2018A_652900_5189100_gnd_subarea.laz",
    additional_dimensions={"point_source_id": "scanpos_id"},
)

with open(py4dgeo.find_file("sps.json"), "r") as load_f:
    scanpos_info_dict = eval(load_f.read())
epoch1.scanpos_info = scanpos_info_dict
epoch2.scanpos_info = scanpos_info_dict

corepoints = py4dgeo.read_from_las("ahk_cp_652900_5189100_subarea.laz").cloud

Cxx = np.loadtxt(py4dgeo.find_file("Cxx.csv"), dtype=np.double, delimiter=",")
tfM = np.loadtxt(py4dgeo.find_file("tfM.csv"), dtype=np.double, delimiter=",")
refPointMov = np.loadtxt(
    py4dgeo.find_file("redPoint.csv"), dtype=np.float64, delimiter=","
)


m3c2_ep = py4dgeo.M3C2EP(
    epochs=(epoch1, epoch2),
    corepoints=corepoints,
    normal_radii=(0.5, 1.0, 2.0),
    cyl_radii=(0.5,),
    max_distance=3.0,
    Cxx=Cxx,
    tfM=tfM,
    refPointMov=refPointMov,
)

# distances, uncertainties, covariance = m3c2_ep.run()

# distances[0:8]
# uncertainties["lodetection"][0:8]

# import matplotlib.cm as cm
# import matplotlib.pyplot as plt


# def plt_3d(corepoints, distances):
#     fig, ax = plt.subplots(figsize=(10, 10), subplot_kw={"projection": "3d"})

#     # add axis labels
#     ax.set_xlabel("X [m]")
#     ax.set_ylabel("Y [m]")
#     ax.set_zlabel("Z [m]")

#     # plot the corepoints colored by their distance
#     x, y, z = np.transpose(corepoints)
#     vmin = np.min(distances)
#     vmax = np.max(distances)
#     pts = ax.scatter(
#         x, y, z, s=10, c=distances, vmin=vmin, vmax=vmax, cmap=cm.seismic_r
#     )

#     # add colorbar
#     cmap = plt.colorbar(pts, shrink=0.5, label="Distance [m]", ax=ax)

#     # add title
#     ax.set_title("Visualize Changes")

#     ax.set_aspect("equal")
#     ax.view_init(22, 113)
#     plt.show()
    
# plt_3d(corepoints, distances)