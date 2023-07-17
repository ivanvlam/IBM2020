% Path to Gmsh executable
gmshPath = 'C:/Program Files (x86)/gmsh/gmsh.exe';

% JMESH file path
jmeshFile = 'data/BrainWeb_Subject05.jmsh';

% Convert JMESH to Gmsh MSH using Gmsh
mshFile = 'output.msh';
cmd = sprintf('"%s" -format msh2 -o "%s" "%s"', gmshPath, mshFile, jmeshFile);
status = system(cmd);

if status == 0
    % Read Gmsh MSH file using meshio
    mesh = meshio_read(mshFile, 'FileType', 'gmsh2');
    
    % Extract node coordinates and element connectivity
    nodes = mesh.points;
    elements = mesh.cells{1}.data;
    
    % Create new mesh object for FEBio INP
    new_mesh = struct();
    new_mesh.points = nodes;
    new_mesh.cells = struct();
    new_mesh.cells{1}.type = 'tetra';
    new_mesh.cells{1}.data = elements;
    
    % Write FEBio INP file using meshio
    febioFile = 'output.inp';
    meshio_write(febioFile, new_mesh, 'FileType', 'abaqus');
    
    disp('Conversion completed successfully.');
else
    disp('Conversion failed.');
end