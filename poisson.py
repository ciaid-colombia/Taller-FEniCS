from dolfin import *
import matplotlib.pyplot as plt

# Malla
mesh = UnitSquareMesh(8,8)
# Espacio de funciones para la incognita y test
V= FunctionSpace(mesh,"Lagrange",1)
# Expresion para las condiciones de contorno
u0=Expression("1+ x[0]*x[1] + 2*x[1]*x[1]",degree=2)
# Definicion de las condiciones de contorno
bc = DirichletBC ( V , u0 , "on_boundary")
# Fuente
f=Constant(-6.0)

# Funcion de trial
u= TrialFunction(V)
# Funcion de test
v= TestFunction(V)
# Forma bilineal del lado izquierdo
a = inner(grad(u), grad(v))*dx
# Forma lineal del lado derecho
L=f*v*dx
# Funcion donde se almacena el resultado
u = Function(V)
# Solucion del sistema algebraico
solve(a==L, u, bc)

#Impresion usando paraview
vtkfile = File('poisson_solution.pvd')
vtkfile << u
