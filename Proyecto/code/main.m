data=loadjson('data/BrainWeb_Subject05.jmsh')

% Obtain Head sections from data
Scalp = data.MeshElem(data.MeshElem(:,5)==1,:);
Skull = data.MeshElem(data.MeshElem(:,5)==2,:);
CSF = data.MeshElem(data.MeshElem(:,5)==3,:);   % Cerebrospinal fluid
GM = data.MeshElem(data.MeshElem(:,5)==4,:);    % Grey matter
WM = data.MeshElem(data.MeshElem(:,5)==5,:);    % White matter

% Plot GM and WM
plotmesh(data.MeshNode, GM, WM);

% Save head sections as different msh files
savemsh(data.MeshNode, Scalp, 'msh_files/01_Scalp.msh')
savemsh(data.MeshNode, Skull, 'msh_files/02_Skull.msh')
savemsh(data.MeshNode, CSF, 'msh_files/03_CSF.msh')
savemsh(data.MeshNode, GM, 'msh_files/04_GM.msh')
savemsh(data.MeshNode, WM, 'msh_files/05_WM.msh')
