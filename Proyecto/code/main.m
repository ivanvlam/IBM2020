data=loadjson('data/BrainWeb_Subject04.jmsh')

% plotmesh(data.MeshNode, data.MeshElem, 'x<100');
plotmesh(data.MeshNode, data.MeshElem(data.MeshElem(:,5)==4,:),data.MeshElem(data.MeshElem(:,5)==5,:));
% xmlmesh(data.MeshNode, data.MeshElem(data.MeshElem(:,5)==4,:), 'hola.xml')
% stlwrite(data.MeshElem, 'filename.stl');