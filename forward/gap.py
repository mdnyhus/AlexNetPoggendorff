from PIL import Image, ImageDraw
import math

# compare from 0 to 226

folder="gap"
angle_degrees = 55
angle = math.radians(angle_degrees)

dimension = 227
margin = 50
width = 3
gap = 21

image = Image.new('RGB', (dimension, dimension), (255,255,255))
draw = ImageDraw.Draw(image)

center = int(dimension/2)
ly = dimension - 1 - margin
# tan(theta) = (ly - center + 1) / (center - lx + 1)
lx = center - int(gap/2) - (ly - center + 1) / math.tan(angle) + 1
draw.line((lx, ly, center - int(gap/2), center), fill=0, width=width)

ry = margin
rx = (center - ry + 1) / math.tan(angle) + center + int(gap/2) + 1
shift = round(math.tan(angle) * gap)
draw.line((rx, ry - shift, center + int(gap/2), center - shift), fill=0, width=width)

draw.line((center - int(gap/2), center, center + int(gap/2), center - shift), fill=0, width=width)

image.save(folder + "/control.png")

# extension is the height of the line segment, plus the width of the line
extension = center - ry + int(width / 2) - 1

for i in range(dimension - 2*extension):
  im = Image.new('RGB', (dimension, dimension), (255,255,255))
  dr = ImageDraw.Draw(im)
  
  # keep left line the same
  ly = dimension - 1 - margin
  lx = center - int(gap/2) - (ly - center + 1) / math.tan(angle) + 1
  dr.line((lx, ly, center - int(gap/2), center), fill=0, width=width)
  
  # right line will be moved vertically by one
  ry = i
  rx = (center - margin + 1) / math.tan(angle) + center + int(gap/2) + 1
  dr.line((rx, ry, center + int(gap/2), i + center - margin), fill=0, width=width)
  
  dr.line((center - int(gap/2), center, center + int(gap/2), i + center - margin), fill=0, width=width)

  im.save(folder + "/" + str(i) + ".png")
