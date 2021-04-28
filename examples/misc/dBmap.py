from matplotlib.pyplot import figure, imshow, colorbar, show, title
from sys import argv
from numpy import log10, load
from time import time
from tqdm import tqdm

def dBmap(File, interpolation):
	z, chosenDimension, nOfBlocks = 0, 2, 1
	df = load(File)
	a = []
	sample = df[[x for x in df if x[-4:] == " (E)"][nOfBlocks-1]]
	for i in tqdm(range(len(sample[0]))):
		a.append([])
		for j in range(len(sample[0][0])):
			temp = [x[i][j][z][chosenDimension] for x in sample]
			a[i].append(max(temp) - min(temp))
	
	peakVal, minVal = max(map(max, a)), min(map(min, a))
	print("Peak at:", [[[i, j] for j, y in enumerate(x) if y == peakVal] for i, x in enumerate(a) if peakVal in x])
	a = 10*log10([[y/minVal for y in x] for x in a])
	
	#figure(figsize=(15, 15))
	title("dB map of Electrical waves in detector region")
	imshow(a, cmap="inferno", interpolation=interpolation)
	cbar = colorbar()
	cbar.ax.set_ylabel("dB scale", rotation=270)
	show()

if __name__ == "__main__":
	start_time = time()
	if argv[1][0:7] == "file://":
		argv[1] = argv[1][7:]
	if len(argv) == 2:
		argv.append("spline16")
	dBmap(argv[1], argv[2])
	end_time = time()
	print("Runtime:", end_time-start_time)
