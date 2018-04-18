from PIL import Image, ImageDraw
import math
import numpy as np
import colorsys

# compare from 0 to 226

folder="gapLineMiddleHalves"
angle_degrees = 55
angle = math.radians(angle_degrees)

dimension = 227
width = 3
widthW = 9
gap = 21 # chosen with angle so that height is close to an int
margin = 31 # chosen so seg_height is close to an int

# want control and test images to be in two different halves of the image
shift = math.ceil((dimension - 1.0) / 4)

image = Image.new('RGB', (dimension, dimension), (255,255,255))
draw = ImageDraw.Draw(image)
imageW = Image.new('RGB', (dimension, dimension), (255,255,255))
drawW = ImageDraw.Draw(imageW)

seg_height = round((dimension - 2*margin - math.tan(angle) * gap) / 3)
center = round((dimension - 1)/2)
lcenterx = center - math.floor(gap/2)
lcentery = dimension - 1 - margin - seg_height

ly = dimension - 1 - margin
# tan(theta) = (ly - center) / (center - lx)
lx = round(lcenterx - (ly - lcentery) / math.tan(angle))

draw.line((lx - shift, ly - seg_height/2, lcenterx - shift, lcentery - seg_height/2), fill=0, width=width)
drawW.line((lx - shift, ly - seg_height/2, lcenterx - shift, lcentery - seg_height/2), fill=0, width=widthW)

rcenterx = center + math.ceil(gap/2)
rcentery = margin + 2*seg_height - 1

# ry = margin + seg_height - 1
ry = rcentery - seg_height
# tan(theta) = seg_height / (lx - rcenterx)
rx = round(seg_height / math.tan(angle) + rcenterx)
draw.line((rx - shift, ry - seg_height/2, rcenterx - shift, rcentery - seg_height/2), fill=0, width=width)

draw.line((lcenterx - shift, lcentery - seg_height/2, rcenterx - shift, rcentery - seg_height/2), fill=0, width=width)
draw.line((lcenterx - shift, margin, lcenterx - shift, dimension - 1 - margin), fill=0, width=width)
draw.line((rcenterx - shift, margin, rcenterx - shift, dimension - 1 - margin), fill=0, width=width)

image.save(folder + "/control.png")

drawW.line((rx - shift, ry - seg_height/2, rcenterx - shift, rcentery - seg_height/2), fill=0, width=widthW)

drawW.line((lcenterx - shift, lcentery - seg_height/2, rcenterx - shift, rcentery - seg_height/2), fill=0, width=widthW)
drawW.line((lcenterx - shift, margin, lcenterx - shift, dimension - 1 - margin), fill=0, width=widthW)
drawW.line((rcenterx - shift, margin, rcenterx - shift, dimension - 1 - margin), fill=0, width=widthW)

imageW.save(folder + "/controlWide.png")

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
  dr.line((lx + shift, ly, lcenterx + shift, lcentery), fill=0, width=width)
  drW.line((lx + shift, ly, lcenterx + shift, lcentery), fill=0, width=widthW)
  
  # right line will be moved vertically by one
  rcenterx = center + math.ceil(gap/2)
  rcentery = dimension - 1 - margin - i
  
  ry = rcentery - seg_height
  # tan(theta) = seg_height / (lx - rcenterx)
  rx = round(seg_height / math.tan(angle) + rcenterx)
  dr.line((rx + shift, ry, rcenterx + shift, rcentery), fill=0, width=width)
  
  dr.line((lcenterx + shift, lcentery, rcenterx + shift, rcentery), fill=0, width=width)
  dr.line((lcenterx + shift, margin, lcenterx + shift, dimension - 1 - margin), fill=0, width=width)
  dr.line((rcenterx + shift, margin, rcenterx + shift, dimension - 1 - margin), fill=0, width=width)
  
  drW.line((rx + shift, ry, rcenterx + shift, rcentery), fill=0, width=widthW)
  
  drW.line((lcenterx + shift, lcentery, rcenterx + shift, rcentery), fill=0, width=widthW)
  drW.line((lcenterx + shift, margin, lcenterx + shift, dimension - 1 - margin), fill=0, width=widthW)
  drW.line((rcenterx + shift, margin, rcenterx + shift, dimension - 1 - margin), fill=0, width=widthW)
  
  im.save(folder + "/" + str(i) + ".png")
  imW.save(folder + "/" + str(i) + "Wide.png")
  
  colour = tuple(int(255*x) for x in colorsys.hsv_to_rgb(0.7*i/len,1,0.8))
  drAvg.line((rx + shift, ry, rcenterx + shift, rcentery), fill=colour, width=width)
  drAvg.line((lcenterx + shift, lcentery, rcenterx + shift, rcentery), fill=colour, width=width)

  
drAvg.line((lx + shift, ly, lcenterx + shift, lcentery), fill=0, width=width)
drAvg.line((lcenterx + shift, margin, lcenterx + shift, dimension - 1 - margin), fill=0, width=width)
drAvg.line((rcenterx + shift, margin, rcenterx + shift, dimension - 1 - margin), fill=0, width=width)
avg.save(folder + "/average.png")
avg.show()
