import meshio
import subprocess

# Convert JMESH to Gmsh MSH using Gmsh
subprocess.run(['C:/Program Files (x86)/gmsh', '-format', 'msh2', '-o', 'temp.msh', 'data/BrainWeb_Subject05.jmsh'])

# Read Gmsh MSH file
mesh = meshio.read('temp.msh', file_format='gmsh2')

# Extract node coordinates and element connectivity
nodes = mesh.points
elements = mesh.cells['tetra']

# Create new mesh object for FEBio INP
new_mesh = meshio.Mesh(points=nodes, cells=[('tetra', elements)])

# Write FEBio INP file
meshio.write('output.inp', new_mesh, file_format='abaqus')
