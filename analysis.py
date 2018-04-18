import numpy as np
import argparse
import glob, os
import matplotlib.pyplot as plt
import matplotlib.transforms as transforms
from matplotlib.cbook import get_sample_data
from matplotlib.offsetbox import (TextArea, DrawingArea, OffsetImage,
                                  AnnotationBbox)

font = {'size' : 15}

plt.rc('font', **font)

fileInEnding = ".out"
fileOut = "analysis.txt"

folderToProcess = ".\\out"

lossErr_tests = []
rawErr_tests = []
classErr_tests = []

def formatTitle(fileName, preFor):
  # parse title
  # format is layer_lr_end_type_jak_jatheta_backgroundType_colour_distr_vars
  titleArr = fileName[:fileName.find('.sh')].split("_")
  if titleArr[4] == "connect":
    titleArr[3:5] = [''.join(titleArr[3:5])]
    
  if titleArr[0] == "cedar":
    titleArr.remove(0)
  elif titleArr[0] == "graham":
    titleArr.remove(0)
  
  print(titleArr)
  
  title = preFor + ' for training and validation\nsets for '  
  if len(preFor) > 30:
    title = preFor + '\nfor training and validation sets for\n'  
  if "dropout" in titleArr:
    title = preFor + ' for training and validation\nsets with dropout for '
 
  if "." not in titleArr[1] and float(titleArr[1]) == 1:
    titleArr[1] = "0." + titleArr[1]
  else:
    titleArr[1] = titleArr[1].replace("-", "/")
  if titleArr[3] == "gapconnect" or titleArr[3] == "default":
    titleArr[3] = "connected"
  else:
    titleArr[3] = "disconnected"
  
  if "dropout" in titleArr:
    title += '$layer={}$ $lr={}$,\n$final={},$'.format(titleArr[0], titleArr[1], titleArr[2])
  else:
    title += '$layer={}$ $lr={}$, $final={},$\n'.format(titleArr[0], titleArr[1], titleArr[2]) 
  if "." not in titleArr[5]:
    titleArr[5] = titleArr[5][0] + "." + titleArr[5][1:]
  title += '$lt={}$, $ja_k={}$, $ja_{{\Theta}}={}$,'.format(titleArr[3],titleArr[4],titleArr[5])
  
  if titleArr[6].split(".")[0] == "single":
    title += " $bg={}$".format(titleArr[6])
  elif titleArr[8] == "const":
    title += "\n$bg={}$, $bg_{{num}}={}$, $bg_{{colour}}={}$".format(titleArr[8],titleArr[9],titleArr[7])
  elif titleArr[8] == "vert":
    title += "\n$bg={}$, $bg_{{colour}}={}$".format(titleArr[8],titleArr[7])
  else:
    if "." not in titleArr[10]:
      titleArr[10] = titleArr[10][0] + "." + titleArr[10][1:]
    title += "\n$bg={}$, $bg_k={}$, $bg_{{\Theta}}={}$, $bg_{{colour}}={}$".format(titleArr[8],titleArr[9],titleArr[10],titleArr[7])
    
  return title

with open(fileOut, 'w+') as out:
  for fileNamePath in glob.glob(os.path.join(folderToProcess,"*" + fileInEnding)):
    length = 5000 * 2

    with open(fileNamePath) as f:
      print(fileNamePath)
      fileName = fileNamePath[len(folderToProcess) + 1:-1*len('.out')]
      
      globalSteps = [None] * length
      lossErr_train = [None] * length
      rawErr_train = [None] * length
      classErr_train = [None] * length
      lossErr_val = [None] * length
      rawErr_val = [None] * length
      classErr_val = [None] * length
      lossErr_test = ""
      rawErr_test = ""
      classErr_test = ""
      
      # -1: N/A
      # 0: Train
      # 1: Validation
      # 2: Test
      lineType = -1
      j = -1
      
      out.write(fileName + "\n")
      for i, line in enumerate(f):
        line = line.rstrip()
        # get info messages
        info = "INFO "
        if line[:len(info)] == info:
          write(line[len(info):])
        # get error category (train, val, test)
        trainLine = "Train error for global step "
        if line[:len(trainLine)] == trainLine:
          j += 1
          globalSteps[j] = int(line[len(trainLine):-1])
          lineType = 0
          continue
        valLine = "Validation error after epoch "
        if line[:len(valLine)] == valLine:
          lineType = 1
          continue
        if line == "Test error:":
          lineType = 2
          continue
        
        # get error type (loss, raw, class)
        lossErrLine = "\tloss_error: "
        if line[:len(lossErrLine)] == lossErrLine:
          if lineType == 0:
            lossErr_train[j] = float(line[len(lossErrLine):])
          elif lineType == 1:
            lossErr_val[j] = float(line[len(lossErrLine):])
          else:
            lossErr_test = float(line[len(lossErrLine):])
          continue
        rawErrLine = "\traw_error: "
        if line[:len(rawErrLine)] == rawErrLine:
          if lineType == 0:
            rawErr_train[j] = float(line[len(rawErrLine):])
          elif lineType == 1:
            rawErr_val[j] = float(line[len(rawErrLine):])
          else:
            rawErr_test = float(line[len(rawErrLine):])
          continue
        classErrLine = "\tclassification_error: "
        if line[:len(classErrLine)] == classErrLine:
          if lineType == 0:
            classErr_train[j] = float(line[len(classErrLine):])
          elif lineType == 1:
            classErr_val[j] = float(line[len(classErrLine):])
          else:
            classErr_test = float(line[len(classErrLine):])
          continue
      
      # there are j entries in the array
      globalSteps = globalSteps[:j+1]
      lossErr_train = lossErr_train[:j+1]
      rawErr_train = rawErr_train[:j+1]
      classErr_train = classErr_train[:j+1]
      lossErr_val = lossErr_val[:j+1]
      rawErr_val = rawErr_val[:j+1]
      classErr_val = classErr_val[:j+1]
      
      # output number of epochs
      out.write("Ran for {} epochs\n".format(j//2 + 1))
      # output test error
      out.write("Final test error:\n\tloss_error: {}\n\traw_error: {}\n\tclassification_error: {}\n".format(lossErr_test, rawErr_test, classErr_test))
      lossErr_tests.append(lossErr_test)
      rawErr_tests.append(rawErr_test)
      classErr_tests.append(classErr_test)      
      
      # calculate best epoch based on validation error
      np_lossErr_val = np.array([i for i in lossErr_val if i != None])
      np_rawErr_val = np.array([i for i in rawErr_val if i != None])
      np_classErr_val = np.array([i for i in classErr_val if i != None])
      
      np_val = np.concatenate((
        np.expand_dims(np_lossErr_val,0),
        np.expand_dims(np_rawErr_val,0),
        np.expand_dims(np_classErr_val,0)), axis=0)
      
      num_increase = 10
      opt = -1 + np.zeros((np_val.shape[0],num_increase+1))
      
      # get global minimum
      opt[:,0] = np.argmin(np_val, axis=1)
      
      cats, elems = np.shape(np_val)
      for i in range(cats):
        cat = np_val[i]
        opt_i = opt[i]
        for k in range(elems):
          next = cat[k]
          cur = next - 1
          next_index = k+1
          count = 0
          while next > cur and next_index < elems and count <= num_increase:
            if count > 0 and opt_i[count] == -1:
              opt_i[count] = k
            count += 1
            cur = next
            next = cat[next_index]
            next_index += 1
      
      opt = np.ndarray.astype(opt, int)
      
      out.write("\n")
      out.write("Min Validation error value and epoch:\n")
      out.write("\tGlobal")
      for i in range(num_increase):
        out.write("\t\t{} inc".format(i+1))
      out.write("\n")
      
      for i in range(num_increase+1):
        out.write("\tVal\tEpoch")
      out.write("\n")
      
      row = np.zeros(2*(num_increase+1))
      row[0::2] = [np_lossErr_val[i] if i >= 0 else 0 for i in opt[0]]
      row[1::2] = opt[0]
      out.write("Loss\t" + "\t".join(['{0:.4f}'.format(x) if int(x) != x else str(int(x)) for x in row]) + "\n")
      
      row[0::2] = [np_classErr_val[i] if i >= 0 else 0 for i in opt[0]]
      row[1::2] = opt[0]
      out.write("C@L\t" + "\t".join(['{0:.4f}'.format(x) if int(x) != x else str(int(x)) for x in row]) + "\n")
      
      
      row[0::2] = [np_rawErr_val[i] if i >= 0 else 0 for i in opt[1]]
      row[1::2] = opt[1]
      out.write("Raw\t" + "\t".join(['{0:.4f}'.format(x) if int(x) != x else str(int(x)) for x in row]) + "\n")
      row[0::2] = [np_classErr_val[i] if i >= 0 else 0 for i in opt[2]]
      row[1::2] = opt[2]
      out.write("Class\t" + "\t".join(['{0:.4f}'.format(x) if int(x) != x else str(int(x)) for x in row]) + "\n")
      
      # Classification
      plt.close('all')
      fig = plt.figure()
      fig.set_size_inches(7, 7, forward=True)
      ax = plt.subplot(111)
      box = ax.get_position()
      ax.set_position([box.x0, box.y0, box.width, box.height * 0.9])
      fig.subplots_adjust(top=0.78)
      
      t, = ax.plot(np.arange(j//2 + 1), classErr_train[0::2], label='Training')
      v, = ax.plot(np.arange(j//2 + 1), np_classErr_val, label='Validation')
      plt.legend(handles=[t, v])
      plt.xlabel("Epoch")
      plt.ylabel("Classification Error")
      
      plt.title(formatTitle(fileName, 'Classification error'), y=1.04)  
      plt.savefig(os.path.join(".\\figs", fileName + ".png"))
      
      # Raw
      plt.close('all')
      fig = plt.figure()
      fig.set_size_inches(7, 7, forward=True)
      ax = plt.subplot(111)
      box = ax.get_position()
      ax.set_position([box.x0, box.y0, box.width, box.height * 0.9])
      fig.subplots_adjust(left=0.16, top=0.78)
      
      t, = ax.plot(np.arange(j//2 + 1), rawErr_train[0::2], label='Training')
      v, = ax.plot(np.arange(j//2 + 1), np_rawErr_val, label='Validation')
      plt.legend(handles=[t, v])
      plt.xlabel("Epoch")
      plt.ylabel("Raw Error")
      
      plt.title(formatTitle(fileName, 'Raw error'), y=1.04)
      plt.savefig(os.path.join(".\\figs", fileName + "_raw.png"))
      
      # Loss
      plt.close('all')
      fig = plt.figure()
      fig.set_size_inches(7, 7, forward=True)
      ax = plt.subplot(111)
      box = ax.get_position()
      ax.set_position([box.x0, box.y0, box.width, box.height * 0.9])
      fig.subplots_adjust(left=0.16, top=0.78)
      
      t, = ax.plot(np.arange(j//2 + 1), lossErr_train[0::2], label='Training')
      v, = ax.plot(np.arange(j//2 + 1), np_lossErr_val, label='Validation')
      plt.legend(handles=[t, v])
      plt.xlabel("Epoch")
      plt.ylabel("Loss Error")
      
      plt.title(formatTitle(fileName, 'Loss error'), y=1.04)
      plt.savefig(os.path.join(".\\figs", fileName + "_loss.png"))
      
      # Both
      plt.close('all')
      fig = plt.figure()
      fig.set_size_inches(7, 7, forward=True)
      ax = plt.subplot(111)
      
      titleArr = fileName[:fileName.find('.sh')].split("_")
      if 'es' in titleArr:
        indexEs = titleArr.index("es")
        if indexEs >= 0:
          ci = int(titleArr[indexEs+1])
          print(ci)
          plt.axvline(x=ci,color='black')
          trans = transforms.blended_transform_factory(
            ax.transData, ax.transAxes)
          ax.text(ci+10, 0.1, '$lr=0.001$', transform=trans)
          trans = transforms.blended_transform_factory(
            ax.transData, ax.transAxes)
          ax.text(ci-75, 0.8, '$lr=0.01$', transform=trans)
      
      
      box = ax.get_position()
      ax.set_position([box.x0, box.y0, box.width, box.height * 0.9])
      fig.subplots_adjust(top=0.78)
      
      t, = ax.plot(np.arange(j//2 + 1), classErr_train[0::2], label='Training Classification')
      v, = ax.plot(np.arange(j//2 + 1), np_classErr_val, label='Validation Classification')
      x, = ax.plot(np.arange(j//2 + 1), lossErr_train[0::2], label='Training Loss')
      y, = ax.plot(np.arange(j//2 + 1), np_lossErr_val, label='Validation Loss')
      plt.legend(handles=[t, v,x,y])
      plt.xlabel("Epoch")
      plt.ylabel("Error")
      
      plt.title(formatTitle(fileName, 'Classification and loss error'), y=1.04)  
      plt.savefig(os.path.join(".\\figs", fileName + "_both.png"))
      
      # All three
      plt.close('all')
      fig = plt.figure()
      fig.set_size_inches(7, 7, forward=True)
      ax = plt.subplot(111)
      
      titleArr = fileName[:fileName.find('.sh')].split("_")
      if 'es' in titleArr:
        indexEs = titleArr.index("es")
        if indexEs >= 0:
          ci = int(titleArr[indexEs+1])
          print(ci)
          plt.axvline(x=ci,color='black')
          trans = transforms.blended_transform_factory(
            ax.transData, ax.transAxes)
          ax.text(ci+10, 0.1, '$lr=0.001$', transform=trans)
          trans = transforms.blended_transform_factory(
            ax.transData, ax.transAxes)
          ax.text(ci-75, 0.8, '$lr=0.01$', transform=trans)
      
      
      box = ax.get_position()
      ax.set_position([box.x0, box.y0, box.width, box.height * 0.9])
      fig.subplots_adjust(top=0.78)
      
      t, = ax.plot(np.arange(j//2 + 1), classErr_train[0::2], label='Training Classification')
      v, = ax.plot(np.arange(j//2 + 1), np_classErr_val, label='Validation Classification')
      x, = ax.plot(np.arange(j//2 + 1), lossErr_train[0::2], label='Training Loss')
      y, = ax.plot(np.arange(j//2 + 1), np_lossErr_val, label='Validation Loss')
      z, = ax.plot(np.arange(j//2 + 1), rawErr_train[0::2], label='Training Raw')
      w, = ax.plot(np.arange(j//2 + 1), np_rawErr_val, label='Validation Raw')
      plt.legend(handles=[t, v,x,y,z,w])
      plt.xlabel("Epoch")
      plt.ylabel("Error")
      
      plt.title(formatTitle(fileName, 'Classification, loss and raw error'), y=1.04)  
      plt.savefig(os.path.join(".\\figs", fileName + "_all.png"))
      
      # break between files
      out.write("\n")
      out.write("======================================================================\n")
      # break
      
plt.close('all')
fig = plt.figure()
fig.set_size_inches(7, 7, forward=True)
ax = plt.subplot(111)
box = ax.get_position()
ax.set_position([box.x0, box.y0, box.width, box.height])
fig.subplots_adjust(left=0.16, )

x_axis = np.arange(len(lossErr_tests))

l, = ax.plot(x_axis, lossErr_tests, label='Loss Error')
r, = ax.plot(x_axis, rawErr_tests, label='Raw Error')
c, = ax.plot(x_axis, classErr_tests, label='Classification Error')
plt.legend(handles=[l,r,c])
plt.xlabel("Run")
plt.ylabel("Error")
plt.title("Final test loss, raw and classification\nerror for different retraining runs")
plt.savefig(os.path.join(".\\figs", "errorComp.png"))