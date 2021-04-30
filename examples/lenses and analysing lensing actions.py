# import only necessary functions from modules to reduce load
import fdtd
from numpy import sin, radians, tan, meshgrid, arange, flip, array, load
from os import path
from time import time

start_time = time()		# to compute run-time

# grid
grid = fdtd.Grid(shape=(260, 15.5e-6, 1), grid_spacing=77.5e-9)
grid.saveSimulation("Example1")		# initializing environment to save simulation data

# objects
# defining a biconvex lens
x, y = arange(-200, 200, 1), arange(190, 200, 1)
X, Y = meshgrid(x, y)
lens_mask = X**2 + Y**2 <= 40000
for j, col in enumerate(lens_mask.T):
	for i, val in enumerate(flip(col)):
		if val:
			grid[30+i:50-i, j-100:j-99, 0] = fdtd.Object(permittivity=1.5**2, name=str(i)+","+str(j))
			break

# source
# using a continuous source (not pulse)
grid[15, 50:150, 0] = fdtd.LineSource(period = 1550e-9 / (3e8), name="source")

# detectors
# usingff BlockDetector
grid[80:200, 80:120, 0] = fdtd.BlockDetector(name="detector")

# x boundaries
grid[0:10, :, :] = fdtd.PML(name="pml_xlow")
grid[-10:, :, :] = fdtd.PML(name="pml_xhigh")

# y boundaries
grid[:, 0:10, :] = fdtd.PML(name="pml_ylow")
grid[:, -10:, :] = fdtd.PML(name="pml_yhigh")

# Saving grid geometry for future references
with open(path.join("./fdtd_output", grid.folder, "grid.txt"), "w") as f:
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

# running simulation
for i in range(400):
	grid.run(total_time=1)		# running simulation 1 timestep a time and animating
	#grid.visualize(z=0, animate=True)		# live animation without saving frames
	grid.visualize(z=0, animate=True, index=i, save=True, folder=grid.folder)	# saving frames during visualization
grid.generate_video(delete_frames=True)		# rendering video from saved frames
grid.save_data()		# saving detector readings

#grid.run(total_time=400)		# example without live animation
#grid.visualize(z=0, show=True, index=0, save=True, folder=grid.folder)
#grid.save_data()
#grid.visualize(z=0, show=True)

# analysing data stored by above simulation by plotting a 2D decibel map
df = load(path.join("./fdtd_output", grid.folder, "detector_readings.npz"))
fdtd.dBmap2D(df["detector (E)"])

end_time = time()
print("Runtime:", end_time-start_time)
