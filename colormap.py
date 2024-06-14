import numpy as np
 
def make_colormap(num=256):
   def bit_get(val, idx):
       return (val >> idx) & 1
 
   colormap = np.zeros((num, 3), dtype=int)
   ind = np.arange(num, dtype=int)
 
   for shift in reversed(list(range(8))):
       for channel in range(3):
           colormap[:, channel] |= bit_get(ind, channel) << shift
       ind >>= 3
 
   return colormap
 
cmap = make_colormap(256).tolist()
palette = [value for color in cmap for value in color]
print(cmap, "\n", palette)