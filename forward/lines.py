from PIL import Image, ImageDraw
import numpy as np
import colorsys

# compare from 0 to 226
folder="vertical"

dimension = 227
gap = 10
width = 3
widthW = 9
extension = int(width / 2)

image = Image.new('RGB', (dimension, dimension), (255,255,255))
draw = ImageDraw.Draw(image)
draw.line((int(dimension/2) + 1, gap, int(dimension/2) + 1, dimension - 1 - gap), fill=0, width=width)

image.save(folder + "/control.png")
# result = Image.fromarray()

avg = Image.new('RGB', (dimension, dimension), (255,255,255))
drAvg = ImageDraw.Draw(avg)

len = dimension - 2*extension
for i in range(len):
  im = Image.new('RGB', (dimension, dimension), (255,255,255))
  dr = ImageDraw.Draw(im)
  dr.line((i + extension, gap, i + extension, dimension - 1 - gap), fill=0, width=width)
  im.save(folder + "/" + str(i) + ".png")
  
  im = Image.new('RGB', (dimension, dimension), (255,255,255))
  dr = ImageDraw.Draw(im)
  dr.line((i + extension, gap, i + extension, dimension - 1 - gap), fill=0, width=widthW)
  im.save(folder + "/" + str(i) + "Wide.png")
  
  colour = tuple(int(255*x) for x in colorsys.hsv_to_rgb(0.7*i/len,1,0.8))
  drAvg.line((i + extension, gap, i + extension, dimension - 1 - gap), fill=colour, width=width)
  
avg.save(folder + "/average.png")
avg.show()

