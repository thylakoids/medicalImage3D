#this is the programe for multi-dimentioanl medical image process 
#homework1
from vtk import *
from numpy import *
import numpy as np
from scipy import spatial
def linecreater(pt1,pt2,i):
	# create source
	source = vtk.vtkLineSource()
	source.SetPoint1(pt1)
	source.SetPoint2(pt2)
	# mapper
	mapper = vtk.vtkPolyDataMapper()
	mapper.SetInputConnection(source.GetOutputPort())
	 
	# actor
	actor = vtk.vtkActor()
	actor.SetMapper(mapper)
	 
	# color actor
	color=[0,0,0]
	color[i]=1
	actor.GetProperty().SetColor(color)
	return actor
def cubeActorcreater():
	#source
	cubeData=vtkCubeSource()
	cubeData.SetXLength(100)
	cubeData.SetYLength(100)
	cubeData.SetZLength(100)
	cubeData.SetCenter(50,50,50)
	#mapper
	cubeMapper=vtkPolyDataMapper()
	cubeMapper.SetInputConnection(cubeData.GetOutputPort())
	#actor
	cubeActor=vtkActor()
	cubeActor.SetMapper(cubeMapper)
	cubeActor.GetProperty().SetColor(1,1,1)
	cubeActor.GetProperty().SetOpacity(0.1)
	return cubeActor
def sphereActorcreater(center,radius,color):
	# create source
	source = vtk.vtkSphereSource()
	source.SetPhiResolution(50)    
	source.SetThetaResolution(50) 
	source.SetCenter(center)
	source.SetRadius(radius)
	 
	# mapper
	mapper = vtk.vtkPolyDataMapper()
	if vtk.VTK_MAJOR_VERSION <= 5:
	    mapper.SetInput(source.GetOutput())
	else:
	    mapper.SetInputConnection(source.GetOutputPort())
	 
	# actor
	actor = vtk.vtkActor()
	actor.SetMapper(mapper)
	actor.GetProperty().SetColor(color)
	return actor
def surfacecreater(coeffi):
	#create an ellipsoid using a implicit quadric
	quadric = vtk.vtkQuadric()
	quadric.SetCoefficients(coeffi)
	 
	# The sample function generates a distance function from the implicit
	# function. This is then contoured to get a polygonal surface.
	sample = vtk.vtkSampleFunction()
	sample.SetImplicitFunction(quadric)
	sample.SetModelBounds(0, 100, 0, 100, 0, 100)
	sample.SetSampleDimensions(40, 40, 40)
	sample.ComputeNormalsOff()
	 
	# contour
	surface = vtk.vtkContourFilter()
	surface.SetInputConnection(sample.GetOutputPort())
	surface.SetValue(0, 0)
	 
	# mapper
	mapper = vtk.vtkPolyDataMapper()
	mapper.SetInputConnection(surface.GetOutputPort())
	#mapper.ScalarVisibilityOff()
	actor = vtk.vtkActor()
	actor.SetMapper(mapper)
	#actor.GetProperty().EdgeVisibilityOn()
	#actor.GetProperty().SetEdgeColor(.2, .2, .5)

	return actor	
def sphere_eval(coeffi,center,radius,colori):
	quadric = vtk.vtkQuadric()
	quadric.SetCoefficients(coeffi)
	index_new=[]
	for i in range(50):
		if quadric.EvaluateFunction(center[i])<=0:
			index_new.append(i)
	return center[index_new,:],radius[index_new],colori[index_new]


def main1():
	global coeffi
	#render
	ren = vtk.vtkRenderer()
	renWin = vtk.vtkRenderWindow()
	renWin.AddRenderer(ren)
	iren = vtk.vtkRenderWindowInteractor()
	iren.SetRenderWindow(renWin)
	#cube
	ren.AddActor(cubeActorcreater())
	#axes
	transform = vtk.vtkTransform()
	transform.Translate(0.0, 0.0, 0.0)
	axes = vtk.vtkAxesActor()
	axes.SetTotalLength(130,130,130)
	axes.SetConeRadius(0.1)
	#  The axes are positioned with a user transform
	axes.SetUserTransform(transform)
	ren.AddActor(axes)
	#sphere*50
	center=np.random.random((50,3))*100
	radius=np.random.random(50)*5+5
	colori=np.random.randint(3,size=50)
	color=[[1,0,0],[1,1,65.0/255],[0,0,1]]
	for i in range(50):
		ren.AddActor(sphereActorcreater(center[i],radius[i],color[colori[i]]))
	#surface
	#ren.AddActor(surfacecreater(coeffi))
	#background,size,bla...bla
	ren.SetBackground(0,0,0)
	renWin.SetSize(500, 500)
	iren.Initialize()
	iren.Start()
	return center,radius,colori
def main2(center,radius,colori):
	global coeffi
	#render
	ren = vtk.vtkRenderer()
	renWin = vtk.vtkRenderWindow()
	renWin.AddRenderer(ren)
	iren = vtk.vtkRenderWindowInteractor()
	iren.SetRenderWindow(renWin)
	#cube
	ren.AddActor(cubeActorcreater())
	##axes
	#transform = vtk.vtkTransform()
	#transform.Translate(0.0, 0.0, 0.0)
	#axes = vtk.vtkAxesActor()
	#axes.SetTotalLength(130,130,130)
	#axes.SetConeRadius(0.1)
	##  The axes are positioned with a user transform
	#axes.SetUserTransform(transform)
	#ren.AddActor(axes)
	#sphere*50
	center,radius,colori=sphere_eval(coeffi,center,radius,colori)
	color=[[1,0,0],[1,1,65.0/255],[0,0,1]]
	for i in range(np.size(colori)):
		ren.AddActor(sphereActorcreater(center[i],radius[i],color[colori[i]]))
	#surface
	ren.AddActor(surfacecreater(coeffi))

	#line
	datacenter,evals,evecs = pca(center,3)  
	for i in range(3):
		ax=.1*evecs[:,i]*evals[i]
		ren.AddActor(linecreater(datacenter,datacenter+ax,i))
	#background,size,bla...bla
	ren.SetBackground(0,0,0)
	renWin.SetSize(500, 500)
	iren.Initialize()
	iren.Start()
	return center,radius,colori
def pca(data,nRedDim=0):
	m=mean(data,axis=0)
	data-=m
	C=cov(transpose(data))
	evals,evecs=linalg.eig(C)
	indices = argsort(evals)  
 	indices = indices[::-1]  
 	evecs = evecs[:,indices]  
 	evals = evals[indices]  
 	if nRedDim>0:  
 		evecs = evecs[:,:nRedDim]  
 	# new matrix
 	x = dot(transpose(evecs),transpose(data))  
 	# original  
 	y=transpose(dot(evecs,x))+m  
 	return m,evals,evecs
def kdtree(pt):
	global center,radius,colori
	si=spatial.KDTree(center).query(pt)[1]
	scenter=center[si]
	sradius=radius[si]
	color=[[1,0,0],[1,1,65.0/255],[0,0,1]]
	scolor=color[colori[si]]
	return scenter,sradius,scolor
if __name__=='__main__':
	coeffi =(1, 1, 1,0,0,0, 0, 0, 0, -7500)
	center,radius,colori=main1()
	pt=np.random.random(3)*100
	scenter,sradius,scolor=kdtree(pt)
	centern,radiusn,colorin=main2(center,radius,colori)


	
	





