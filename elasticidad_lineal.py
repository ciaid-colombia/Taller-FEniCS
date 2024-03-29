from dolfin import *
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

# Opciones de compilacion
parameters["form_compiler"]["cpp_optimize"] = True
parameters["form_compiler"]["optimize"] = True

# Dominio
L = 1; W = 0.1; H =0.04
# Malla
mesh = BoxMesh(Point(0., 0., 0.), Point(L, W, H), 60, 10, 5)
x = SpatialCoordinate(mesh)

# Propiedades del material
# Constantes de Lame
E  = 1000.0
nu = 0.3
mu    = Constant(E / (2.0*(1.0 + nu)))
lmbda = Constant(E*nu / ((1.0 + nu)*(1.0 - 2.0*nu)))
# Densidad
rho = Constant(1.0)
# Coeficiente de amortiguacion
eta = Constant(0.1)

# Subdominio
# Izquierda
def left(x, on_boundary):
    return near(x[0], 0.) and on_boundary

# Derecha
def right(x, on_boundary):
    return near(x[0], 1.) and on_boundary

# Espacio de funciones para desplazamiento, velocidad y aceleraciones
V = VectorFunctionSpace(mesh, "Lagrange", 1)
# Espacio de funciones para esfuerzos
Vsig = TensorFunctionSpace(mesh, "DG", 0)

# Funciones
u = TrialFunction(V)
v = TestFunction(V)


# Formas variacionales
# Tensor de esfuerzos
def sigma(u):
    return 2.0*mu*sym(grad(u)) + lmbda*tr(sym(grad(u)))*Identity(len(u))

# Matriz de masa
def mmat(u, v):
    return rho*inner(u, v)*dx

# Matriz de rigidez elastica
def kmat(u, v):
    return inner(sigma(u), sym(grad(v)))*dx

# Amortiguacion de Rayleigh
def cmat(u, v):
    return eta*mmat(u, v)

# Proyeccion local
def local_project(v, V, u=None):
    dv = TrialFunction(V)
    v_ = TestFunction(V)
    a_proj = inner(dv, v_)*dx
    b_proj = inner(v, v_)*dx
    solver = LocalSolver(a_proj, b_proj)
    solver.factorize()
    if u is None:
        u = Function(V)
        solver.solve_local_rhs(u)
        return u
    else:
        solver.solve_local_rhs(u)
        return

# Condicion de contorno izquierda (Dirichlet)
zero = Constant((0.0, 0.0, 0.0))
bc = DirichletBC(V, zero, left)

# Condicion de contorno derecha
boundary_subdomains = MeshFunction("size_t", mesh, mesh.topology().dim() - 1)
boundary_subdomains.set_all(0)
force_boundary = AutoSubDomain(right)
force_boundary.mark(boundary_subdomains, 3)
dss = ds(subdomain_data=boundary_subdomains)

# Cargas
k = 0.25
s0 = 1.
s = Expression(("0", "s0*sin(k*pi*t)","0"), t=0, k=k, s0=s0, degree=0)
g = Constant((0.,-1,0.))

# Integrador temporal
T  = 25.0
dt = 0.25
Nt = int(T/dt)
ti = np.linspace(0, T, Nt)

# Campos para el paso de tiempo anterior
u_old = Function(V, name="Desplazamiento")
v_old = Function(V, name="Velocidad")
a_old = Function(V, name="Aceleracion")

# Metodo de Newmark (punto medio)
gamma   = Constant(0.5)
beta    = Constant((gamma+0.5)**2/4.)

# Aceleracion
# a = 1/(2*beta)*((u - u0 - v0*dt)/(0.5*dt*dt) - (1-2*beta)*a0)
def update_a(u, u_old, v_old, a_old, ufl=True):
    if ufl:
        dt_ = dt
        beta_ = beta
    else:
        dt_ = float(dt)
        beta_ = float(beta)
    return (u-u_old-dt_*v_old)/beta_/dt_**2 - (1-2*beta_)/2/beta_*a_old

# Velocidad
# v = dt * ((1-gamma)*a0 + gamma*a) + v0
def update_v(a, u_old, v_old, a_old, ufl=True):
    if ufl:
        dt_ = dt
        gamma_ = gamma
    else:
        dt_ = float(dt)
        gamma_ = float(gamma)
    return v_old + dt_*((1-gamma_)*a_old + gamma_*a)

# Actualizar campos
def update_fields(u, u_old, v_old, a_old):
    """Update fields at the end of each time step."""

    # Vectores de referencia
    u_vec, u0_vec  = u.vector(), u_old.vector()
    v0_vec, a0_vec = v_old.vector(), a_old.vector()

    # Actualizacion de velocidad y aceleracion
    a_vec = update_a(u_vec, u0_vec, v0_vec, a0_vec, ufl=False)
    v_vec = update_v(a_vec, u0_vec, v0_vec, a0_vec, ufl=False)

    # Actualizacion de desplazamiento
    v_old.vector()[:], a_old.vector()[:] = v_vec, a_vec
    u_old.vector()[:] = u.vector()


# Ecuacion de elasticidad en forma residual
a_new = update_a(u, u_old, v_old, a_old, ufl=True)
v_new = update_v(a_new, u_old, v_old, a_old, ufl=True)
res = mmat(a_new, v) + cmat(v_new, v) \
       + kmat(u, v) - rho*dot(s,v)*dss(3) - dot(rho*g,v)*dx
# Ensamble del sistema de ecuaciones
B_form = lhs(res)
L_form = rhs(res)
K, b = assemble_system(B_form, L_form, bc)

# Inicializacion de variables de salida
u_tip = np.zeros((Nt,))
energies = np.zeros((Nt, 4))
E_damp = 0

# Creacion de archivos de salida (ParaView)
xdmf_file = XDMFFile("bar.xdmf")
xdmf_file.parameters["flush_output"] = True
xdmf_file.parameters["functions_share_mesh"] = True
xdmf_file.parameters["rewrite_function_mesh"] = False

# Solucion
u = Function(V, name="Desplazamiento")
sig = Function(Vsig, name="sigma")

# Propiedades del solver
solver = LUSolver(K, "mumps")
solver.parameters["symmetric"] = True

# Bucle temporal
for i in tqdm(range(Nt),desc='Time loop'):

    t = ti[i]

    # Fuerza evaluada en t_{n+1}=t_{n+1}-dt
    s.t = t

    # Solucionar desplazamientos
    b = assemble(L_form)
    bc.apply(b)
    solver.solve(K, u.vector(), b)

    # Actualizar campos
    update_fields(u, u_old, v_old, a_old)

    # Calcular esfuerzos
    local_project(sigma(u), Vsig, sig)

    # Guardar solucion (vizualizacion)
    xdmf_file.write(u_old, t)
    xdmf_file.write(v_old, t)
    xdmf_file.write(a_old, t)
    xdmf_file.write(sig, t)

    # Guardar solucion (desplazamiento y energia)
    u_tip[i] = u(1., 0.05, 0.02)[1]
    E_elas = assemble(0.5*kmat(u_old, u_old))
    E_kin = assemble(0.5*mmat(v_old, v_old))
    E_damp += dt*assemble(cmat(v_old, v_old))
    E_tot = E_elas+E_kin+E_damp
    energies[i, :] = np.array([E_elas, E_kin, E_damp, E_tot])

# Imprimir desplazamiento en x=(1,0.05,0.02)
plt.figure()
plt.plot(ti, u_tip)
plt.xlabel("Time")
plt.ylabel("Desplazamiento")
plt.savefig('images/tip.png',format='png',dpi=400)
plt.show()

# Imprimir energia
plt.figure()
plt.plot(ti, energies)
plt.legend(("elastica", "cinetica", "amortiguacion", "total"))
plt.xlabel("Time")
plt.ylabel("Energia")
plt.savefig('images/energy.png',format='png',dpi=400)
plt.show()
