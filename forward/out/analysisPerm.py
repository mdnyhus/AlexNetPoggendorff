import numpy as np
import argparse
import glob, os
import matplotlib.pyplot as plt
from matplotlib.cbook import get_sample_data
from matplotlib.offsetbox import (TextArea, DrawingArea, OffsetImage,
                                  AnnotationBbox)
import matplotlib.lines
from matplotlib.transforms import Bbox, TransformedBbox
from matplotlib.legend_handler import HandlerBase
from matplotlib.image import BboxImage
import matplotlib.transforms as transforms
from matplotlib.cbook import get_sample_data
from matplotlib.offsetbox import (TextArea, DrawingArea, OffsetImage,
                                  AnnotationBbox)

font = {'size' : 15}

plt.rc('font', **font)

fileInEnding = ".csv"

numImgs = 121
folders = [
  "gapLineMiddleHalves2",
  "gapLineMiddleHalves",
  "gapLineMiddleHalvesFlipped2",
  "gapLineMiddleHalvesFlipped",
  "gapMiddleHalves2",
  "gapMiddleHalves",
  "gapMiddleHalvesFlipped2",
  "gapMiddleHalvesFlipped"]
  
def formatTitle(fileName):
  title = ""
  
  if "Line" in fileName:
    title += "Vert"
  else:
    title += "Line"
    
  if "Flipped" in fileName:
    title += "Down"
  else:
    title += "Up"
  
  if fileName[-1] == "2":
    title += "LR"
  else:
    title += "RL"
  
  return title
  
centered = False
title = "Permutations of connected Poggendorffs against\ndifferent controls in different orientations"
fileTitle = "permutations"
if centered:
  title += " - centered"
  fileTitle += "Centered"
  
plt.close('all')
fig = plt.figure()
fig.set_size_inches(12, 7, forward=True)
ax = plt.subplot(111)
fig.subplots_adjust(left=0.11, bottom=0.25)
plt.xlabel("Image index", labelpad=70)
ylabel = "Cosine similarity with control"
if centered:
  ylabel += ", centered"
plt.ylabel(ylabel)



handles = []
handler_map = {}
x_axis = np.arange(numImgs)
folder = "gapLineMiddle"
for fileName in folders:
  for line in reversed(list(open(fileName + fileInEnding))):
    line = line.split(",")
    label = line[0]
    # account for label and trailing ","
    line = np.asarray(line[1:-1]).astype(float)
    if centered:
      line -= np.mean(line)
    
    if label == "fc8":
      t, = ax.plot(x_axis, line, label=formatTitle(fileName))
      handles.append(t)      
      
      imgs = [0,(numImgs-1)//6,2*(numImgs-1)//6,3*(numImgs-1)//6,4*(numImgs-1)//6,5*(numImgs-1)//6,numImgs-1]
      for i in imgs:
        xy = [i,line[0]]
        fn = get_sample_data(os.getcwd() + "\\..\\" + folder + "\\" + str(i) + "Wide.png", asfileobj=False)
        arr_img = plt.imread(fn, format='png')
        imagebox = OffsetImage(arr_img, zoom=0.2)
        imagebox.image.axes = ax
        ab = AnnotationBbox(imagebox, xy,
                      xybox=(xy[0], -0.17),
                      xycoords='data',
                      boxcoords=("data","axes fraction"),
                      pad=0.1,
                      frameon=False)
        ax.add_artist(ab)
  
box = ax.get_position()
ax.set_position([box.x0, box.y0, box.width * 0.7, box.height])   
leg = ax.legend(loc='center left', bbox_to_anchor=(1,0.5), handles=handles, frameon=False,labelspacing=2)

# Get the bounding box of the original legend
bb = leg.get_bbox_to_anchor().inverse_transformed(ax.transAxes)

# Change to location of the legend. 
xOffset = 0.03
bb.x0 += xOffset
bb.x1 += xOffset
leg.set_bbox_to_anchor(bb, transform = ax.transAxes)

# ax.legend(handles, ["","","","","","","","",""], handler_map=handler_map, 
   # labelspacing=0.0, fontsize=36, borderpad=0, loc='center left', bbox_to_anchor=(1,0.5),
    # handletextpad=0, borderaxespad=0)
# ax.legend(handles, ["","","","","","","","",""], handler_map=handler_map, 
   # labelspacing=0.0, fontsize=40, borderpad=0.15, loc='center left', bbox_to_anchor=(1,0.5),
    # handletextpad=0.2, borderaxespad=0.15)
    
plt.title(title)
# plt.legend(handles=handles)

# add images to legend
if centered:
  for i in range(len(folders)):
    fileName = folders[i]
    # add legend images
    xy = [x_axis[0],line[0]]
    fn = get_sample_data(os.getcwd() + "\\..\\" + fileName + "\\averageWide.png", asfileobj=False)
    arr_img = plt.imread(fn, format='png')
    imagebox = OffsetImage(arr_img, zoom=0.15)
    imagebox.image.axes = ax
    ab = AnnotationBbox(imagebox, xy,
                  xybox=(1.4, 1 - 1.*i/(len(folders) - 1)),
                  xycoords='data',
                  boxcoords=("axes fraction","axes fraction"),
                  pad=0.1)
    ax.add_artist(ab)
    
    ci = 75
    plt.axvline(x=ci,color='black')
    trans = transforms.blended_transform_factory(
      ax.transData, ax.transAxes)
    ax.text(ci+1, 0.01, 'straight\ndiagonal\nat index '+str(ci), transform=trans)
    
    fn = get_sample_data(os.getcwd() + "\\..\\" + fileName + "\\controlWide.png", asfileobj=False)
    arr_img = plt.imread(fn, format='png')
    imagebox = OffsetImage(arr_img, zoom=0.15)
    imagebox.image.axes = ax
    ab = AnnotationBbox(imagebox, xy,
                  xybox=(1.5, 1 - 1.*i/(len(folders) - 1)),
                  xycoords='data',
                  boxcoords=("axes fraction","axes fraction"),
                  pad=0.1)
    ax.add_artist(ab)
else:
  # add images to legend
  for i in range(len(folders)):
    fileName = folders[i]
    # add legend images
    xy = [x_axis[0],0.85]
    fn = get_sample_data(os.getcwd() + "\\..\\" + fileName + "\\averageWide.png", asfileobj=False)
    arr_img = plt.imread(fn, format='png')
    imagebox = OffsetImage(arr_img, zoom=0.15)
    imagebox.image.axes = ax
    ab = AnnotationBbox(imagebox, xy,
                  xybox=(1.4, 1 - 1.*i/(len(folders) - 1)),
                  xycoords='data',
                  boxcoords=("axes fraction","axes fraction"),
                  pad=0.1)
    ax.add_artist(ab)
    
    ci = 75
    plt.axvline(x=ci,color='black')
    trans = transforms.blended_transform_factory(
      ax.transData, ax.transAxes)
    ax.text(ci+1, 0.01, 'straight\ndiagonal\nat index '+str(ci), transform=trans)
    
    fn = get_sample_data(os.getcwd() + "\\..\\" + fileName + "\\controlWide.png", asfileobj=False)
    arr_img = plt.imread(fn, format='png')
    imagebox = OffsetImage(arr_img, zoom=0.15)
    imagebox.image.axes = ax
    ab = AnnotationBbox(imagebox, xy,
                  xybox=(1.5, 1 - 1.*i/(len(folders) - 1)),
                  xycoords='data',
                  boxcoords=("axes fraction","axes fraction"),
                  pad=0.1)
    ax.add_artist(ab)


plt.savefig(os.path.join(".\\figs", fileTitle + ".png"))
plt.show() 
     
    
      