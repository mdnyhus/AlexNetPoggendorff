from PIL import Image, ImageDraw
import math
import scipy
import numpy as np
import argparse
import sys
import os

# commandl ine arguments
parser = argparse.ArgumentParser(formatter_class=argparse.RawTextHelpFormatter)
# background_192_6_04_num_5_05
# line type
types = ['default','gap','gap_connect']
default = 'default'
parser.add_argument('-t', nargs=1, metavar='type', choices=types, default=[default], required=False, 
                    help='Type of line to be generated.\nAllowed values are: '+', '.join(types)+'\nDefault: '+default)
# joint angle
default = ['6', '0.75']
parser.add_argument('-ja', nargs=2, metavar=('gamma_a','gamma_b'), default=default, required=False, 
                    help='Gamma used for joint angles\nDefault: ('+str(default[0]) + ',' + str(default[1]) + ')')
# background
background = parser.add_mutually_exclusive_group(required=True)
background.add_argument('-s', action='store_true', help="No background lines")
background.add_argument('-bg', nargs=3, metavar=('colour', 'gamma_a', 'gamma_b'), 
                        help="Background lines generated with a gamma function")
background.add_argument('-bc', nargs=2, metavar=('colour', 'num'), 
                        help="Constant number of background lines")
background.add_argument('-v', nargs=1, metavar=('colour'),
                        help="Two vertical background lines")
background.add_argument('-lc', nargs=2, metavar=('colour', 'num'),
                        help="Background is entirely lines, number decided by gamma")

# normalization
parser.add_argument('-pc', action='store_true', help='For gap/gap_connect, use a constant length for\nthe distance between lines parallel to the line')

args = parser.parse_args()
print(args)

type = args.t[0]
ja_a = float(args.ja[0])
ja_b = float(args.ja[1])

single = args.s
bc = args.bc
bg = args.bg
v = args.v
lc = args.lc

pc = args.pc

save_folder = "./" + type + "_" + str(ja_a) + "_" + str(ja_b)
colour = 255
if single:
  save_folder += "_single"
elif bc != None:
  colour = int(bc[0])
  save_folder += "_background_" + str(bc[0]) + "_const_" + str(bc[1])
elif bg != None:
  colour = int(bg[0])
  save_folder += "_background_" + str(bg[0]) + "_gamma_" + str(bg[1]) + "_" + str(bg[2])
elif v != None:
  colour = int(v[0])
  save_folder += "_background_" + str(v[0]) + "_vert"
else:
  colour = int(lc[0])
  save_folder += "_background_" + str(lc[0]) + "_const_" + str(lc[1])

if pc:
  save_folder += "_pc"  

print(save_folder)

margin = 10
dimension = 227
dimension_eff = 227 - 2*margin
width = 3
gap = 21
pc_gap = 13

# returns new x, y that bring them in bounds while mainting the same line
def in_bound(x, y, s_x, s_y):
  ratio = (1.0 * (y - s_y) / (x - s_x))
  # get a point that is in bounds
  x_n = max(margin, min(dimension - margin, x))
  # y_n - s_y = (x_n - s_x) * ratio
  y_n = s_y + (x_n - s_x) * ratio
  # now do the same for y
  y_n = max(margin, min(dimension - margin, y_n))
  x_n = s_x + (y_n - s_y) * ratio 
  
  return x_n, y_n
  
# returns translations to keep x, y in bounds and maintain length and angle
def translations(s_x, s_y, x, y):
  x_diff = 0
  if x < margin:
    x_diff = margin - x
  elif x > dimension - margin:
    x_diff = dimension - margin - x
  
  s_x = s_x + x_diff
  x = x + x_diff
  
  y_diff = 0
  if y < margin:
    y_diff = margin - y
  elif y > dimension - margin:
    y_diff = dimension - margin - y
  
  s_y = s_y + y_diff
  y = y + y_diff
      
  return x_diff, y_diff

def get_points(x, y, length, angle):
  # x_n - x = length * cos(angle)
  x_n = x + length * math.cos(angle)
  # y_n - y = -1 * length * sin(angle); -1 because 0,0 is TOP left corner
  y_n = y - length * math.sin(angle)
  return x_n, y_n
  
def draw_line(dr, s_x, s_y, length, angle, fill, vert, vertsC): 
  # x, y = in_bound(get_points(s_x, s_y, length, angle), s_x, s__y)
  # if math.sqrt((s_x - x)**2 + (s_y - y)**2) < length:
    # # line got cut off, try extending in other direction
    # s_x, s_y = in_bound(get_points(x, y, length, (angle - math.pi) % (2 * math.pi)), x, y)
    
  if vert and length < 2*gap:
    length = 2*gap
    
  x, y = get_points(s_x, s_y, length, angle)
  x_diff, y_diff = translations(s_x, s_y, x, y)
  s_x += x_diff
  x += x_diff
  s_y += y_diff
  y += y_diff
  
  if vert:  
    min_y = min(margin,s_y,y)
    max_y = max(dimension-margin,s_y,y)
    outside_len = int((length * 1.0 - gap)/2)
    inx1, iny1 = get_points(s_x, s_y, outside_len, angle)
    inx2, iny2 = get_points(inx1, iny1, gap, angle)
    
    # condition for a good line: gap must be at least pc_gap
    if abs(inx1 - inx2) < pc_gap:
      # it goes backwards; reject it
      return False
    
    dr.line((inx1, min_y, inx1, max_y), fill=vertsC, width=width)
    dr.line((inx2, min_y, inx2, max_y), fill=vertsC, width=width)
  
  dr.line((s_x, s_y, x, y), fill=fill, width=width)
  return True
  
def draw_line_joint(dr, s_x, s_y, length, angle, joint_angle, fill):
  # to keep this in bounds, simply translate
  x, y = get_points(s_x, s_y, length / 2, angle)
  x_diff, y_diff = translations(s_x, s_y, x, y)
  s_x += x_diff
  x += x_diff
  s_y += y_diff
  y += y_diff
  
  x2, y2 = get_points(x, y, length/2, (angle + joint_angle) % (2 * math.pi))
  x_diff, y_diff = translations(x, y, x2, y2)
  s_x += x_diff
  x += x_diff
  x2 += x_diff
  s_y += y_diff
  y += y_diff
  y2 += y_diff
  
  dr.line((s_x, s_y, x, y), fill=fill, width=width)
  dr.line((x, y, x2, y2), fill=fill, width=width)
  return True
  
def draw_line_gap_connected(dr, s_x, s_y, length, angle, joint_angle, fill, vert, vertsC):
  if pc:
    joint_angle = math.atan2(1.0 * gap * math.sin(joint_angle), 1.0*pc_gap)
  
  # want line of at least 2*gap
  if length < 2*gap:
    length = 2*gap
  
  line_length = int((length * 1.0 - gap)/2)
  
  # to keep this in bounds, simply translate
  x, y = get_points(s_x, s_y, line_length, angle)
  x_diff, y_diff = translations(s_x, s_y, x, y)
  s_x += x_diff
  x += x_diff
  s_y += y_diff
  y += y_diff
  
  x2, y2 = get_points(x, y, gap, (angle + joint_angle) % (2 * math.pi))
  x_diff, y_diff = translations(x, y, x2, y2)
  s_x += x_diff
  x += x_diff
  x2 += x_diff
  s_y += y_diff
  y += y_diff
  y2 += y_diff
  
  x3, y3 = get_points(x2, y2, line_length, angle % (2 * math.pi))
  x_diff, y_diff = translations(x2, y2, x3, y3)
  s_x += x_diff
  x += x_diff
  x2 += x_diff
  x3 += x_diff
  s_y += y_diff
  y += y_diff
  y2 += y_diff
  y3 += y_diff  
  
  if vert:
    # condition for a good line: that the "zag" doesn't go backwards
    if (s_x - x) * (x - x2) <= 0 or abs(x2 - x) < pc_gap:
      # it goes backwards; reject it
      return False
    
    min_y = min(margin,s_y,y,y2,y3)
    max_y = max(dimension-margin,s_y,y,y2,y3)
    dr.line((x, min_y, x, max_y), fill=vertsC, width=width)
    dr.line((x2, min_y, x2, max_y), fill=vertsC, width=width)
  
  dr.line((s_x, s_y, x, y), fill=fill, width=width)
  dr.line((x, y, x2, y2), fill=fill, width=width)
  dr.line((x2, y2, x3, y3), fill=fill, width=width)
  return True
  
def draw_line_gap(dr, s_x, s_y, length, angle, joint_angle, fill, vert, vertsC):
  if pc:
    joint_angle = math.atan2(1.0 * gap * math.sin(joint_angle), 1.0*pc_gap)
  
  # want line of at least 2*gap
  if length < 2*gap:
    length = 2*gap
  
  line_length = int((length * 1.0 - gap)/2)
  
  # to keep this in bounds, simply translate
  x, y = get_points(s_x, s_y, line_length, angle)
  x_diff, y_diff = translations(s_x, s_y, x, y)
  s_x += x_diff
  x += x_diff
  s_y += y_diff
  y += y_diff
  
  x2, y2 = get_points(x, y, gap, (angle + joint_angle) % (2 * math.pi))
  x_diff, y_diff = translations(x, y, x2, y2)
  s_x += x_diff
  x += x_diff
  x2 += x_diff
  s_y += y_diff
  y += y_diff
  y2 += y_diff
  
  x3, y3 = get_points(x2, y2, line_length, angle % (2 * math.pi))
  x_diff, y_diff = translations(x2, y2, x3, y3)
  s_x += x_diff
  x += x_diff
  x2 += x_diff
  x3 += x_diff
  s_y += y_diff
  y += y_diff
  y2 += y_diff
  y3 += y_diff  
  
  if vert:
    # condition for a good line: that the "zag" doesn't go backwards
    # take min gap to be pc_gap
    if (s_x - x) * (x - x2) <= 0 or abs(x2 - x) < pc_gap:
      # it goes backwards; reject it
      return False
    
    min_y = min(margin,s_y,y,y2,y3)
    max_y = max(dimension-margin,s_y,y,y2,y3)
    dr.line((x, min_y, x, max_y), fill=vertsC, width=width)
    dr.line((x2, min_y, x2, max_y), fill=vertsC, width=width)
  
  dr.line((s_x, s_y, x, y), fill=fill, width=width)
  # dr.line((x, y, x2, y2), fill=fill, width=width)
  dr.line((x2, y2, x3, y3), fill=fill, width=width)
  return True
  
folder_train ="train"
total_train = 100

folder_val = "validation"
total_val = 2000

folder_test = "test"
total_test = 2000

folders = {folder_train: total_train}#, folder_val: total_val, folder_test: total_test}

if not os.path.exists(save_folder):
  os.makedirs(save_folder)
for key in folders:
  folder = os.path.join(save_folder, key)
  if not os.path.exists(folder):
    os.makedirs(folder)

total = total_train + total_val + total_test
  
# use gamma distributions for number of extra lines
# used http://homepage.divms.uiowa.edu/~mbognar/applets/gamma.html to pick a, b for gamma distributions
if single:
  num_lines = np.ndarray.astype(np.zeros(total) + 1, int)
elif bc != None:
  num_lines = np.ndarray.astype(np.zeros(total) + 1 + int(bc[1]), int)
elif bg != None:
  num_lines = 1 + np.ndarray.astype(np.random.gamma(float(bg[1]), float(bg[2]), total), int)
elif v != None:
  # lines will all be added in one go, but create a buffer for "bad" lines
  buffer = 20
  num_lines = np.ndarray.astype(np.zeros(total) + 1 + buffer, int)
else:
  num_lines = np.ndarray.astype(np.zeros(total) + 1 + int(lc[1]), int)
  
# length gamma based on testing using this function:
# def func(a, b):
  # x = np.random.gamma(a, b, 100000)
  # x = 320 * x / 20
  # bin = max(x)/200.0
  # plt.hist(x, bins=np.arange(0, max(x) + bin, bin))
  # plt.savefig(str(a) + "-" + str(b) + ".png")
  # plt.clf()
# found that a = 8, b = 1.0 works well; 
# gives peak around 100 and doesn't have too long a tail
a_length = 9
b_length = 0.65
# want to use gamma for length since it provides a much nicer distribution
# picking random points has too many short and long points, rather than "nice" length lines

# angle gamma also based on testing using the following function:
# def angle(a,b):
  # x = np.random.gamma(a,b,100000)
  # x = math.pi*(x / 20)
  # bin = max(x)/200.0
  # plt.hist(x, bins=np.arange(0, max(x)+bin, bin))
  # plt.savefig("angle-" + str(a) + "-" + str(b) + ".png")
  # plt.clf()
# based on inspection, chose the following:
a_angle = ja_a
b_angle = ja_b
# want to use gamma for angle to minimize number of instances very close to 0, but also weight 
# highly values that are relatively close to 0

start_points = margin + dimension_eff * np.random.rand(total, max(num_lines), 2)

lengths = np.random.gamma(a_length, b_length, (total, max(num_lines)))
# normalize
lengths = math.sqrt(2) * dimension_eff * lengths / 20

joint_angles = np.random.gamma(a_angle, b_angle, (total, max(num_lines)))
# normalize, and randomly flip
random_flips = np.random.rand(total, max(num_lines))
random_flips = np.ndarray.astype(2*random_flips, int)
random_flips = random_flips + random_flips%2 - 1
joint_angles = (math.pi * joint_angles / 20) * random_flips

angles = 2 * math.pi * np.random.rand(total, max(num_lines))

# uniform distribution between the number of different line types for background lines
numTypes = 2
if type == 'default':
  numTypes = 3
if lc != None:
  numTypes = 1
line_types = np.ndarray.astype(numTypes * np.random.rand(total, max(num_lines)), int)

types = ['default','gap','gap_connect']

count = 0
for key in folders:
  folder = os.path.join(save_folder, key)
  total_folder = folders[key]

  for i in range(total_folder):
    curIndex = i + count
    im = Image.new('RGB', (dimension, dimension), (255,255,255))
    dr = ImageDraw.Draw(im)
    
    bound = 0
    while bound < num_lines[curIndex]:
      if v != None:
        # vert - pass through from 0 -> num_lines[curIndex]
        j = bound
      else:
        # all others, draw in reverse order
        j = num_lines[curIndex] - 1 - bound
      fill = (colour, colour, colour)
      verts = v != None
      vertsC = (colour, colour, colour)
        
      if j == 0:
        # main line
        fill = (0,0,0)
        goodLine = False
        while not goodLine:
          if i < total_folder // 2:
            if type == 'gap':
              goodLine = draw_line_gap(dr, start_points[curIndex,j,0], start_points[curIndex,j,1], lengths[curIndex,j], angles[curIndex,j], 0, fill, verts, vertsC)
            else:
              goodLine = draw_line(dr, start_points[curIndex,j,0], start_points[curIndex,j,1], lengths[curIndex,j], angles[curIndex,j], fill, verts, vertsC)
          elif type == 'gap':
            goodLine = draw_line_gap(dr, start_points[curIndex,j,0], start_points[curIndex,j,1], lengths[curIndex,j], angles[curIndex,j], joint_angles[curIndex,j], fill, verts, vertsC)
          elif type == 'gap_connect':
            goodLine = draw_line_gap_connected(dr, start_points[curIndex,j,0], start_points[curIndex,j,1], lengths[curIndex,j], angles[curIndex,j], joint_angles[curIndex,j], fill, verts, vertsC)
          else:
            if i - total_folder // 2 < total_folder // 4:
              goodLine = draw_line_joint(dr, start_points[curIndex,j,0], start_points[curIndex,j,1], lengths[curIndex,j], angles[curIndex,j], joint_angles[curIndex,j], fill)
            else:
              goodLine = draw_line_gap_connected(dr, start_points[curIndex,j,0], start_points[curIndex,j,1], lengths[curIndex,j], angles[curIndex,j], joint_angles[curIndex,j], fill, verts, vertsC)
          if v == None:
            # goodLines don't matter
            goodLine = True
          else:
            j += 1
            if j > buffer:
              # out of lines...
              break
              
        if v != None:
          # no background lines for vert
          break
      
      else:
        # background lines
        line_type = line_types[curIndex,j]
        if line_type == 0:
          if type == 'gap' and lc == None:
            draw_line_gap(dr, start_points[curIndex,j,0], start_points[curIndex,j,1], lengths[curIndex,j], angles[curIndex,j], 0, fill, verts, vertsC)
          else:
            draw_line(dr, start_points[curIndex,j,0], start_points[curIndex,j,1], lengths[curIndex,j], angles[curIndex,j], fill, verts, vertsC)
        elif line_type == 1:
          if type == 'gap':
            draw_line_gap(dr, start_points[curIndex,j,0], start_points[curIndex,j,1], lengths[curIndex,j], angles[curIndex,j], joint_angles[curIndex,j], fill, verts, vertsC)
          elif type == 'gap_connect':
            draw_line_gap_connected(dr, start_points[curIndex,j,0], start_points[curIndex,j,1], lengths[curIndex,j], angles[curIndex,j], joint_angles[curIndex,j], fill, verts, vertsC)
          else:
            draw_line_joint(dr, start_points[curIndex,j,0], start_points[curIndex,j,1], lengths[curIndex,j], angles[curIndex,j], joint_angles[curIndex,j], fill)
        else:
          draw_line_gap_connected(dr, start_points[curIndex,j,0], start_points[curIndex,j,1], lengths[curIndex,j], angles[curIndex,j], joint_angles[curIndex,j], fill, verts, vertsC)
    
      bound += 1
    
    im.save(folder + "/" + str(i) + ".png")
  
  count += total_folder
