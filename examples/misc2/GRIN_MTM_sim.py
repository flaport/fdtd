# import only necessary functions from modules to reduce load
from fdtd_venv import fdtd_mod as fdtd
from numpy import sin, radians, tan, savez
from os import path, mkdir, chdir, remove
from subprocess import call
from glob import glob
from datetime import datetime
#from pandas import DataFrame
from sys import argv
from matplotlib.pyplot import figure
from time import time

start_time = time()

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
grid = fdtd.Grid(shape=(9.3e-6, 15.5e-6, 1), grid_spacing=77.5e-9)

# objects
n0, theta, t = 1, 30, 0.5
for i in range(50):
	x = i*0.08
	epsilon = n0 + x*sin(radians(theta))/t
	epsilon = epsilon**0.5
	grid[5.1e-6:5.6e-6, (5 + i*0.08)*1e-6:(5.08 + i*0.08)*1e-6, 0] = fdtd.Object(permittivity=epsilon, name="object"+str(i))
#grid[5.1e-6:5.6e-6, 5e-6:9e-6, 0] = fdtd.Object(permittivity=1.4, name="dielectric")	# dielectric
#n0, a, theta, t = 10, 2, 30, 0.5												# GRIN-MTM
#for i in range(50):
	#x = i*0.08
	#epsilon = n0 - ((x**2 + a**2)**0.5 - a + x*sin(radians(theta))) / (t)
	#epsilon = n0 - ((x**2 + a**2)**0.5 - a) / (t)
	#epsilon = epsilon**2
	#grid[5.1e-6:5.6e-6, (5 + i*0.08)*1e-6:(5.08 + i*0.08)*1e-6, 0] = fdtd.Object(permittivity=epsilon, name="object"+str(i))
grid[5.1e-6:5.6e-6, 0.775e-6:5e-6, 0] = fdtd.Object(permittivity=n0**2, name="objectLeft")
grid[5.1e-6:5.6e-6, 9e-6:(15.5 - 0.775)*1e-6, 0] = fdtd.Object(permittivity=epsilon, name="objectRight")

# source
#grid[3.1e-6, 5e-6, 0] = fdtd.PointSource(period=1550e-9 / (3e8), name="source", pulse=True, cycle=3, dt=4e-15)
grid[3.1e-6, 1.5e-6:14e-6, 0] = fdtd.LineSource(period = 1550e-9 / (3e8), name="source", pulse=True, cycle=3, hanning_dt=4e-15)
#angleI = 30
#grid[50:20, 50:50+round(30/tan(radians(angleI))), 0] = fdtd.LineSource(period = 1550e-9 / (3e8), name="source", pulse=True, cycle=3, dt=4e-15)

# detectors
#grid[12e-6, :, 0] = fdtd.LineDetector(name="detector")
for i in range(-4, 8):
	grid[5.8e-6, 84+4*i:86+4*i, 0] = fdtd.LineDetector(name="detector"+str(i))

# x boundaries
grid[0:10, :, :] = fdtd.PML(name="pml_xlow")
grid[-10:, :, :] = fdtd.PML(name="pml_xhigh")

# y boundaries
grid[:, 0:10, :] = fdtd.PML(name="pml_ylow")
grid[:, -10:, :] = fdtd.PML(name="pml_yhigh")

# Saving grid geometry
f = open(path.join("./fdtd_output", folder, "grid.txt"), "w")
f.write(str(grid))
f.close()

#figure(figsize=(15, 15))
for i in range(100):
	grid.run(total_time=1)
	#grid.visualize(z=0, animate=True)
	grid.visualize(z=0, animate=True, index=i, save=True, folder=folder)
generate_video(delete_frames=True)

#grid.run(total_time=120)
#grid.visualize(z=0, show=True, index=0, save=True, folder=folder)
save_data(grid.detectors)

end_time = time()
print("Runtime:", end_time-start_time)
