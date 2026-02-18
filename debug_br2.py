import numpy as np
from src.dg.viscous_br2 import BR2ViscousFlux
from src.dg.mesh.triangle_mesh import TriangleMesh

# Simple mesh
mesh = TriangleMesh.rectangle((0.0, 1.0), (0.0, 1.0), nx=4, ny=4)
br2 = BR2ViscousFlux(mesh, p=2, nu=1.0, periodic=True, x_range=(0.0, 1.0), y_range=(0.0, 1.0))

# Check some values
print("K (num elements):", br2.K)
print("Np (nodes per element):", br2.Np)
print("Nfp (nodes per face):", br2.Nfp)
print()
print("Fscale range:", br2.Fscale.min(), "to", br2.Fscale.max())
print("LIFT shape:", br2.LIFT.shape)
print()

# Check solution at a simple field
u = np.sin(2*np.pi*br2.x) * np.sin(2*np.pi*br2.y)
print("u range:", u.min(), "to", u.max())

# Compute gradient
ux, uy = br2.compute_gradient(u)
print("Volume gradient ux range:", ux.min(), "to", ux.max())
print("Volume gradient uy range:", uy.min(), "to", uy.max())

# Compute lifted gradient
ux_br2, uy_br2 = br2.compute_lifted_gradient(u)
print("BR2 gradient ux range:", ux_br2.min(), "to", ux_br2.max())
print("BR2 gradient uy range:", uy_br2.min(), "to", uy_br2.max())

# Compute laplacian
lap = br2.compute_laplacian(u)
print("Laplacian range:", lap.min(), "to", lap.max())
print("Expected ~ -8*pi^2 * u ~", -8*np.pi**2)

# Look at a single element's data
print("\n=== Element 0 details ===")
k = 0
print("Fscale for element 0:", br2.Fscale[k])
print("nx for element 0:", br2.nx[k])
print("ny for element 0:", br2.ny[k])

# Check the LIFT matrix more carefully
print("\nLIFT row sums:", br2.LIFT.sum(axis=1))
print("LIFT col sums:", br2.LIFT.sum(axis=0))
