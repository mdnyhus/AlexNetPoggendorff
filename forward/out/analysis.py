import numpy as np
import argparse
import glob, os
import matplotlib.pyplot as plt
import shutil
import matplotlib.transforms as transforms
from matplotlib.cbook import get_sample_data
from matplotlib.offsetbox import (TextArea, DrawingArea, OffsetImage,
                                  AnnotationBbox)
from re import *

font = {'size' : 15}

plt.rc('font', **font)

fileInEnding = ".csv"

fileNames = [
  # "diagonal",
  # "diagonalLine",
  # "gap2",
  "gapLine",
  # "gapLineHalves",
  # "gapLineHalves2",
  "gapLineMiddle",
  "gapLineMiddleHalves",
  "gapLineMiddleHalves2",
  "gapLineMiddleHalvesFlipped",
  "gapLineMiddleHalvesFlipped2",
  "gapMiddleHalves",
  "gapMiddleHalves2",
  "gapMiddleHalvesFlipped",
  "gapMiddleHalvesFlipped2",
  "vertical"]
  
fileNames = [x + fileInEnding for x in fileNames]
  
controlIndex = [
  # 42,
  # 42,
  # -1,
  75,
  # -1,
  # -1,
  -1,
  -1,
  -1,
  -1,
  -1,
  -1,
  -1,
  -1,
  -1,
  113]
  
specialIndices = {"gapLineMiddle": 98, "gapLineMiddleHalves": 98}
  
def camel_case_split(identifier):
    matches = finditer('.+?(?:(?<=[a-z])(?=[A-Z])|(?<=[A-Z])(?=[A-Z][a-z])|$)', identifier)
    return [m.group(0) for m in matches]
  
def formatTitle(fileName,fc8=False):
  # parse title that is in CamelCase
  titleArr = camel_case_split(fileName)
  print(titleArr)
  
  fc8Text = "\n- fc8 only -"
  
  if fileName == "vertical":
    title = "Cosine similarity for vertical lines"
    if fc8:
      title += fc8Text
    return title
  
  if fileName == "gapLine":
    title = 'Cosine similarity for centered, connected Poggendorffs\nagainst a control with diagonal at bottom'
    if fc8:
      title += fc8Text
    return title
  
  if fileName == "gapLineMiddle":
    title = 'Cosine similarity for centered, connected Poggendorffs\nagainst a control with centered diagonal'
    if fc8:
      title += fc8Text
    return title
  
  withWord = "without"
  if "Line" in titleArr:
    withWord = "with"
  title = 'Cosine similarity for connected Poggendorffs\nagainst a control {} verticals and\na centered diagonal '.format(withWord)
  
  ori = "middle"
  if "Halves" in titleArr:
    if fileName[-1] == "2":
      ori = "left-right"
    else:
      ori = "right-left"
  title += "in a {} orientation".format(ori)
  
  if fc8:
    title += fc8Text
  return title
  
def copy_rename(folder, filename):
  filenameTemp = filename
  if filenameTemp == "average" and os.path.isfile(os.getcwd() + "\\..\\" + folder + "\\" + filenameTemp + "Colour.png"):
    filenameTemp = filenameTemp + "Colour"
  src_file = os.path.join(os.getcwd() + "\\..\\" + folder + "\\" + filenameTemp + ".png")
  shutil.copy(src_file,os.path.join(".\\figs",folder + " - " + filename + ".png"))
  # os.rename(os.path.join(".\\figs",filename+".png"),)

for i in range(len(fileNames)):
  fileName = fileNames[i]
  folder = fileName[:-1*len(fileInEnding)]
  ci = controlIndex[i]
  with open(fileName) as f:
    # TODO - better graph title
    fileTitle = fileName[:-1*len(fileInEnding)]
    print(fileTitle)
    
    plt.close('all')
    
    fig = plt.figure()
    fig.set_size_inches(7, 7, forward=True)
    ax = plt.subplot(111)
    bottom = 0.25
    fig.subplots_adjust(left=0.16, bottom=bottom,top=0.8)
    plt.xlabel("Image index", labelpad=70)
    plt.ylabel("cosine similarity with control")
   
    handles = []
    x_axis = None
    
    if ci >= 0:
      plt.axvline(x=ci,color='black')
      trans = transforms.blended_transform_factory(
        ax.transData, ax.transAxes)
      ax.text(ci+1, 0.01, '$test = control$\nat index '+str(ci), transform=trans)
    else:
      ci = 75
      plt.axvline(x=ci,color='black')
      trans = transforms.blended_transform_factory(
        ax.transData, ax.transAxes)
      if fileTitle in specialIndices:
        ax.text(ci-1, 0.01, 'straight\ndiagonal\nat index '+str(ci), transform=trans, ha="right")
      else:
        ax.text(ci+1, 0.01, 'straight\ndiagonal\nat index '+str(ci), transform=trans)
      
    if fileTitle in specialIndices:
      index = specialIndices[fileTitle]
      plt.axvline(x=index,color='black')
      trans = transforms.blended_transform_factory(
        ax.transData, ax.transAxes)
      ax.text(index+1, 0.01, '$test = control$\nof right arm\nat index '+str(index), transform=trans)
    
    for line in reversed(list(open(fileName))):
      line = line.split(",")
      label = line[0]
      # account for label and trailing ","
      line = np.asarray(line[1:-1]).astype(float)
      
      if x_axis is None:
        x_axis = np.arange(len(line))
      
      t, = ax.plot(x_axis, line, label=label)
      handles.append(t)
      
      if label == "fc8":
        print("fc8 max: {} at index {}".format(np.max(line), np.argmax(line)))
        print("fc8 at 75: {}".format(line[75]))
        print(np.argsort(1 - line)[:40])
        
        numImgs = len(line)
        imgs = [0,(numImgs-1)//4,(numImgs-1)//2,3*(numImgs-1)//4,numImgs-1]
        if fileName == "vertical":
          imgs = [3,(numImgs-1)//4,(numImgs-1)//2,3*(numImgs-1)//4,numImgs-1-3]
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
                        pad=0.1)

          ax.add_artist(ab)
        
        plt.title(formatTitle(fileTitle,True))
        plt.savefig(os.path.join(".\\figs", fileTitle + "fc8.png"))
    
    plt.title(formatTitle(fileTitle))
    fig.set_size_inches(9, 7, forward=True)
    fig.subplots_adjust(left=0.1, bottom=bottom,top=0.85)
    box = ax.get_position()
    ax.set_position([box.x0, box.y0, box.width * 0.85, box.height])  
    plt.legend(loc='center left', bbox_to_anchor=(1,0.5), handles=handles)
    plt.savefig(os.path.join(".\\figs", fileTitle + ".png"))
    
    
    copy_rename(folder, "control")
    copy_rename(folder, "average")
    
# plt.show()
     
    
      