import pyvista as pv

topo = pv.read("capsule_ctrl_pts.vts")

p = pv.Plotter()
p.add_mesh(topo, show_edges=True)
p.view_xy()
p.show()
