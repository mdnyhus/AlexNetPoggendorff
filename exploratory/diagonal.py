from PIL import Image, ImageDraw
import math
import numpy as np
import colorsys

# compare from 0 to 226

folder="diagonal"
angle_degrees = 55
angle = math.radians(angle_degrees)

dimension = 227
margin = 50
width = 3
widthW = 9

image = Image.new('RGB', (dimension, dimension), (255,255,255))
draw = ImageDraw.Draw(image)

seg_height = round((dimension - 2*margin) / 3)
centerx = int((dimension - 1)/2)
centery = dimension - 1 - margin - seg_height

ly = dimension - 1 - margin
# tan(theta) = (ly - center) / (center - lx)
lx = round(centerx - (ly - centery) / math.tan(angle))
draw.line((lx, ly, centerx, centery), fill=0, width=width)

ry = centery - seg_height
rx = round(seg_height / math.tan(angle) + centerx)
draw.line((rx, ry, centerx, centery), fill=0, width=width)

image.save("diagonal/control.png")

avg = Image.new('RGB', (dimension, dimension), (255,255,255))
drAvg = ImageDraw.Draw(avg)
drAvg.line((lx, ly, centerx, centery), fill=0, width=width)

for i in range(seg_height * 2 + 1):
  im = Image.new('RGB', (dimension, dimension), (255,255,255))
  dr = ImageDraw.Draw(im)
  imW = Image.new('RGB', (dimension, dimension), (255,255,255))
  drW = ImageDraw.Draw(imW)
  
  dr.line((lx, ly, centerx, centery), fill=0, width=width)
  drW.line((lx, ly, centerx, centery), fill=0, width=widthW)

  rcenterx = centerx
  rcentery = dimension - 1 - margin - i
  
  ry = rcentery - seg_height
  rx = round(seg_height / math.tan(angle) + rcenterx)
  dr.line((rx, ry, rcenterx, rcentery), fill=0, width=width)
  drW.line((rx, ry, rcenterx, rcentery), fill=0, width=widthW)

  im.save("diagonal/" + str(i) + ".png")
  imW.save("diagonal/" + str(i) + "Wide.png")
  
  colour = tuple(int(255*x) for x in colorsys.hsv_to_rgb(0.7*i/(seg_height * 2 + 1),1,0.8))
  drAvg.line((rx, ry, rcenterx, rcentery), fill=colour, width=width)
  
avg.save(folder + "/average.png")
avg.show()

