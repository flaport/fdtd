# import only necessary functions from modules to reduce load
from fdtd_venv import fdtd_mod as fdtd
from numpy import sin, radians, tan, meshgrid, arange, flip, savez, array
from os import path, mkdir, chdir, remove
from subprocess import call
from glob import glob
from datetime import datetime
#from pandas import DataFrame
from sys import argv
from matplotlib.pyplot import figure
from time import time

start_time = time()

run = True
saveStuff = True

if saveStuff:
	if not path.exists("./fdtd_output"):  # Output folder declaration
		mkdir("fdtd_output")
	simTitle = str(datetime.now().year) + "-" + str(
		datetime.now().month) + "-" + str(datetime.now().day) + "-" + str(
			datetime.now().hour) + "-" + str(datetime.now().minute) + "-" + str(datetime.now().second)
	if len(argv) > 1:  # Simulation name (optional)
		simTitle = simTitle + " (" + argv[1] + ")"
	folder = "fdtd_output_" + simTitle
	if path.exists(path.join("./fdtd_output", folder)):  # Overwrite protocol
		yn = input("File", folder, "exists. Overwrite? [Y/N]: ")
		if yn.capitalize() == "N":
			exit()
	else:
		mkdir(path.join("./fdtd_output", folder))


# Generate video
def generate_video(delete_frames=False):
	chdir(path.join("./fdtd_output", folder))
	call([
		'ffmpeg', '-y', '-framerate', '8', '-i', 'file%02d.png', '-r', '30',
		'-pix_fmt', 'yuv420p', 'fdtd_sim_video_' + simTitle + '.mp4'
	])
	if delete_frames:  # delete frames
		for file_name in glob("*.png"):
			remove(file_name)
	chdir("../..")


# Save detector readings
def save_data(detectors):
	dic = {}
	for detector in detectors:
		dic[detector.name + " (E)"] = [x for x in detector.detector_values()["E"]]
		dic[detector.name + " (H)"] = [x for x in detector.detector_values()["H"]]
	#df = DataFrame(dic)
	#df.to_csv(path.join("./fdtd_output", folder, "detector_readings.csv"), index=None)
	savez(path.join("./fdtd_output", folder, "detector_readings"), **dic)


# grid
grid = fdtd.Grid(shape=(260, 15.5e-6, 1), grid_spacing=77.5e-9)

# objects
x, y = arange(-200, 200, 1), arange(190, 200, 1)
X, Y = meshgrid(x, y)
lens_mask = X**2 + Y**2 <= 40000
#lens_mask = lens_mask*1.4**2
#for i, row in enumerate(lens_mask):
	#for j, val in enumerate(row):
		#if val:
			#grid[60-i, 21+j, 0] = fdtd.Object(permittivity=1.5**2, name=str(i)+","+str(j))
			#grid[61+i, 21+j, 0] = fdtd.Object(permittivity=1.5**2, name="i"+str(i)+","+str(j))
for j, col in enumerate(lens_mask.T):
	for i, val in enumerate(flip(col)):
		if val:
			grid[30+i:50-i, j-100:j-99, 0] = fdtd.Object(permittivity=1.5**2, name=str(i)+","+str(j))
			break

# source
grid[15, 50:150, 0] = fdtd.LineSource(period = 1550e-9 / (3e8), name="source")
#grid[15, 1.5e-6:14e-6, 0] = fdtd.LineSource(period = 1550e-9 / (3e8), name="source", pulse=True, cycle=3, hanning_dt=4e-15)

# detectors
#for i in range(130):
	#grid[120+i, 80:120, 0] = fdtd.LineDetector(name="detector"+str(i))
grid[80:200, 80:120, 0] = fdtd.BlockDetector(name="detector")

# x boundaries
grid[0:10, :, :] = fdtd.PML(name="pml_xlow")
grid[-10:, :, :] = fdtd.PML(name="pml_xhigh")

# y boundaries
grid[:, 0:10, :] = fdtd.PML(name="pml_ylow")
grid[:, -10:, :] = fdtd.PML(name="pml_yhigh")

# Saving grid geometry
if saveStuff:
	with open(path.join("./fdtd_output", folder, "grid.txt"), "w") as f:
		f.write(str(grid))
		wavelength = 3e8/grid.source.frequency
		wavelengthUnits = wavelength/grid.grid_spacing
		GD = array([grid.x, grid.y, grid.z])
		gridRange = [arange(x/grid.grid_spacing) for x in GD]
		objectRange = array([[gridRange[0][x.x], gridRange[1][x.y], gridRange[2][x.z]] for x in grid.objects]).T
		f.write("\n\nGrid details (in wavelength scale):")
		f.write("\n\tGrid dimensions: ")
		f.write(str(GD/wavelength))
		f.write("\n\tSource dimensions: ")
		f.write(str(array([grid.source.x[-1] - grid.source.x[0] + 1, grid.source.y[-1] - grid.source.y[0] + 1, grid.source.z[-1] - grid.source.z[0] + 1])/wavelengthUnits))
		f.write("\n\tObject dimensions: ")
		f.write(str([(max(map(max, x)) - min(map(min, x)) + 1)/wavelengthUnits for x in objectRange]))

#figure(figsize=(15, 15))
for i in range(400):
	grid.run(total_time=1)
	#grid.visualize(z=0, animate=True)
	grid.visualize(z=0, animate=True, index=i, save=True, folder=folder)
generate_video(delete_frames=True)
save_data(grid.detectors)

#if run:
	#grid.run(total_time=400)
#if saveStuff:
	#grid.visualize(z=0, show=True, index=0, save=True, folder=folder)
	#save_data(grid.detectors)
#else:
	#grid.visualize(z=0, show=True)

end_time = time()
print("Runtime:", end_time-start_time)
