from PIL import Image, ImageDraw
import math
import numpy as np
import colorsys

# compare from 0 to 226

folder="gapLineMiddle"
angle_degrees = 55
angle = math.radians(angle_degrees)

dimension = 227
width = 3
widthW = 9
gap = 21 # chosen with angle so that height is close to an int
margin = 31 # chosen so seg_height is close to an int

image = Image.new('RGB', (dimension, dimension), (255,255,255))
draw = ImageDraw.Draw(image)

seg_height = round((dimension - 2*margin - math.tan(angle) * gap) / 3)
center = round((dimension - 1)/2)
lcenterx = center - math.floor(gap/2)
lcentery = dimension - 1 - margin - seg_height

ly = dimension - 1 - margin
# tan(theta) = (ly - center) / (center - lx)
lx = round(lcenterx - (ly - lcentery) / math.tan(angle))

draw.line((lx, ly - seg_height/2, lcenterx, lcentery - seg_height/2), fill=0, width=width)

rcenterx = center + math.ceil(gap/2)
rcentery = margin + 2*seg_height - 1

# ry = margin + seg_height - 1
ry = rcentery - seg_height
# tan(theta) = seg_height / (lx - rcenterx)
rx = round(seg_height / math.tan(angle) + rcenterx)
draw.line((rx, ry - seg_height/2, rcenterx, rcentery - seg_height/2), fill=0, width=width)

draw.line((lcenterx, lcentery - seg_height/2, rcenterx, rcentery - seg_height/2), fill=0, width=width)
draw.line((lcenterx, margin, lcenterx, dimension - 1 - margin), fill=0, width=width)
draw.line((rcenterx, margin, rcenterx, dimension - 1 - margin), fill=0, width=width)

image.save(folder + "/control.png")

# extension is the height of the line segment, plus the width of the line
extension = center - ry + int(width / 2) - 1

avg = Image.new('RGB', (dimension, dimension), (255,255,255))
drAvg = ImageDraw.Draw(avg)

len = seg_height*2 + round(math.tan(angle) * gap) + 1
for i in range(seg_height*2 + round(math.tan(angle) * gap) + 1):
  im = Image.new('RGB', (dimension, dimension), (255,255,255))
  dr = ImageDraw.Draw(im)
  imW = Image.new('RGB', (dimension, dimension), (255,255,255))
  drW = ImageDraw.Draw(imW)
  
  # keep left line the same
  dr.line((lx, ly, lcenterx, lcentery), fill=0, width=width)
  drW.line((lx, ly, lcenterx, lcentery), fill=0, width=widthW)
  
  # right line will be moved vertically by one
  rcenterx = center + math.ceil(gap/2)
  rcentery = dimension - 1 - margin - i
  
  ry = rcentery - seg_height
  # tan(theta) = seg_height / (lx - rcenterx)
  rx = round(seg_height / math.tan(angle) + rcenterx)
  dr.line((rx, ry, rcenterx, rcentery), fill=0, width=width)
  
  dr.line((lcenterx, lcentery, rcenterx, rcentery), fill=0, width=width)
  dr.line((lcenterx, margin, lcenterx, dimension - 1 - margin), fill=0, width=width)
  dr.line((rcenterx, margin, rcenterx, dimension - 1 - margin), fill=0, width=width)
  
  drW.line((rx, ry, rcenterx, rcentery), fill=0, width=widthW)
  
  drW.line((lcenterx, lcentery, rcenterx, rcentery), fill=0, width=widthW)
  drW.line((lcenterx, margin, lcenterx, dimension - 1 - margin), fill=0, width=widthW)
  drW.line((rcenterx, margin, rcenterx, dimension - 1 - margin), fill=0, width=widthW)
  
  im.save(folder + "/" + str(i) + ".png")
  imW.save(folder + "/" + str(i) + "Wide.png")
  
  colour = tuple(int(255*x) for x in colorsys.hsv_to_rgb(0.7*i/len,1,0.8))
  drAvg.line((rx, ry, rcenterx, rcentery), fill=colour, width=width)
  drAvg.line((lcenterx, lcentery, rcenterx, rcentery), fill=colour, width=width)

  
drAvg.line((lx, ly, lcenterx, lcentery), fill=0, width=width)
drAvg.line((lcenterx, margin, lcenterx, dimension - 1 - margin), fill=0, width=width)
drAvg.line((rcenterx, margin, rcenterx, dimension - 1 - margin), fill=0, width=width)
avg.save(folder + "/average.png")
avg.show()
