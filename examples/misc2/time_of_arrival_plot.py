from numpy import array, where, arcsin, degrees, load
from matplotlib.pyplot import subplot, plot, show, title, suptitle, figure, legend, xlabel, ylabel
from scipy.signal import hilbert
#from pandas import read_csv
from sys import argv

"""
Hilbert plot for detector responses.
Warning: ALL detectors are considered be *1 CELL* in size. In long detectors, first cell in detector is considered.

Arguments:-
	filename (string)
	(optional) specificPlot (from {"Ex", "Ey", "Ez", "Hx", "Hy", "Hz"}): if you want to see only a specific field plot
"""

def plotDetection(File, specificPlot, angleParams):
	detectorElement = 0		# cell to consider in each detectors
	#df = read_csv(File)
	df = load(File)
	#dic = df.to_string()
	maxArray = {}

	for detector in df:
		if specificPlot is not None:
			if detector[-2] != specificPlot[0]:
				continue
		if detector[-2] == "E":
			figure(0, figsize=(15, 15))
		elif detector[-2] == "H":
			figure(1, figsize=(15, 15))
		#for dimension in range(len(df[detector][0][1:-1].split("\n")[0][1:-1].split())):
		for dimension in range(len(df[detector][0][0])):
			if specificPlot is not None:
				if ["x", "y", "z"].index(specificPlot[1]) != dimension:
					continue
			# if specificPlot, plot on 1x1, else plot on 2x2
			subplot(2 - int(specificPlot is not None), 2 - int(specificPlot is not None), dimension + 1 if specificPlot is None else 1)
			#plot(abs(hilbert([float(x[2:-2].split()[dimension]) for x in df[detector]])), label=detector)
			#hilbertPlot = abs(hilbert([float(x[1:-1].split("\n ")[detectorElement][1:-1].split()[dimension]) for x in df[detector]]))
			hilbertPlot = abs(hilbert([x[0][dimension] for x in df[detector]]))
			plot(hilbertPlot, label=detector)
			title(detector[-2] + "(" + ["x", "y", "z"][dimension] + ")")
			if detector[-2] not in maxArray:
				maxArray[detector[-2]] = {}
			if str(dimension) not in maxArray[detector[-2]]:
				maxArray[detector[-2]][str(dimension)] = []
			maxArray[detector[-2]][str(dimension)].append([detector, where(hilbertPlot == max(hilbertPlot))[0][0]])
		#suptitle(detector)
		#show()

# Loop same as above, only to add axes labels
	for i in range(2):
		if specificPlot is not None:
			if ["E", "H"][i] != specificPlot[0]:
				continue
		figure(i)
		#for dimension in range(len(df[detector][0][1:-1].split("\n")[0][1:-1].split())):
		for dimension in range(len(df[detector][0][0])):
			if specificPlot is not None:
				if ["x", "y", "z"].index(specificPlot[1]) != dimension:
					continue
			subplot(2 - int(specificPlot is not None), 2 - int(specificPlot is not None), dimension + 1 if specificPlot is None else 1)
			xlabel("Time steps")
			ylabel("Magnitude")
	legend()
	show()
	
	for item in maxArray:
		figure(figsize=(15, 15))
		for dimension in maxArray[item]:
			arrival = array(maxArray[item][dimension])
			plot([int(x) for x in arrival.T[1]], arrival.T[0], label=["x", "y", "z"][int(dimension)])
		title(item)
		xlabel("Time of arrival (time steps)")
		legend()
		show()

# Printing angle (under construction)
	if angleParams is not None:
		specificPlot = specificPlot[0] + str(["x", "y", "z"].index(specificPlot[1]))
		print(maxArray[specificPlot[0]][specificPlot[1]])
		angleParams[1] = (angleParams[1] + len(maxArray[specificPlot[0]][specificPlot[1]])) % len(maxArray[specificPlot[0]][specificPlot[1]])
		angleParams[0] = (angleParams[0] + len(maxArray[specificPlot[0]][specificPlot[1]])) % len(maxArray[specificPlot[0]][specificPlot[1]])
		angle = degrees(arcsin(0.7*(maxArray[specificPlot[0]][specificPlot[1]][angleParams[1]][1] - maxArray[specificPlot[0]][specificPlot[1]][angleParams[0]][1]) / ((angleParams[1] - angleParams[0]) * angleParams[2])))
		print("Angle (based on " + specificPlot[0]+["x", "y", "z"][int(specificPlot[1])] + "):", angle)
		return angle


if __name__ == "__main__":
	if argv[1][0:7] == "file://":
		argv[1] = argv[1][7:]
	if len(argv) < 3:
		argv.append(None)
	if len(argv) > 3:
		argv[3] = [int(x) for x in argv[3][1:-1].split(",")]
	if len(argv) < 4:
		#argv.append([0, 2, 2])
		argv.append(None)
	plotDetection(argv[1], argv[2], argv[3])
