
import fresnel
import mdtraj

import numpy as np
# from PySide2.QtWidgets import QMainWindow, QApplication
import PIL
from matplotlib.colors import to_rgb
import os
import itertools
import scipy.linalg
from PAanalysis import utils


def get_bonds_per_molecule(itpfile):
    """Return array of bonded atom pairs of a MARTINI molecule
    using the itpfile
    """
    
    bonds = []
    start = False
    with open(itpfile, 'r') as f:
        for line in f:
            if ('[ bonds ]' in line) or ('[ constraints ]' in line):
                start = True
                continue
            if start:
                words = line.split()
                if words == []:
                    start = False
                    continue
                if not words[0].isdigit():
                    continue
                bonds = bonds + [[int(words[0]),int(words[1])]]

    # starting the index from zero
    bonds = np.array(bonds)-1
    
    return bonds            



def get_bonds_fromtopfile(topfile, molnames):
    """Reads the bonds for each molname given in the topfile
    For atomistic GROMACS simulation
    """

    bondss = []
    for molname in molnames:
        with open(topfile, 'r') as f:
            lines = f.readlines()

        bonds=[]
        start1=False
        start2=False
        for i,line in enumerate(lines):
            if not start1:
                if '[ moleculetype ]' in line:
                    if molname in lines[i+1] or molname in lines[i+2]:
                        start1=True
                        continue
            else:
                if not start2:
                    if '[ bonds ]' in line:
                        start2=True
                        continue
                else:
                    words = line.split()
                    if len(words) == 0:
                        continue
                    if words[0] =='[':
                        bondss += [np.array(bonds)]
                        break
                    if words[0].isdigit():
                        bonds += [ [int(words[0])-1, int(words[1])-1] ]

    
    return bondss




def draw_Hbonds(grofile, trajfile, topfile, itpfiles, 
    molnames, draw_box, frame, filename='snapshot_hbonds.png', preview=True):
    """
    For atomistic simulations.

    Draw snapshots with hbonds along the three gyration eigenvectors
    
    """
    
    #---------------------------------------- Parameters ---------------------------------
    
    radius_dir = dict(H=0, C=0.15, N=0.12, S=0.15, O=0.15)
    radius_cylinder = 0.03
    outline_width = 0.0001

    color_C16  = to_rgb('#0D2D6C')
    color_12C  = to_rgb('#0D2D6C')
    color_pep =  to_rgb('#00FF00')
    color_pep2 = to_rgb('#00FF00')
    color_BMP  = to_rgb('#e41a1c')


    #--------------------------------------------------------------------------------------
    # Calculate bonds and colors

    traj = mdtraj.load(trajfile, top=grofile)

    num_atomss  = []
    nmols       = []
    positions   = []
    start = 0
    for i,molname in enumerate(molnames):
        num_atomss  += [utils.get_num_atoms_fromGRO(itpfiles[i].replace('.itp','')+'.gro')]
        nmols       += [utils.get_num_molecules_fromtop(topfile, molname)]
        positions   += list(traj.xyz[frame,start:start+num_atomss[i]*nmols[i]])
        start += num_atomss[i]*nmols[i]
    positions = np.array(positions)

    N = len(positions)

    Lx, Ly, Lz = traj.unitcell_lengths[frame]

    residues = np.array([r for r in traj.top.residues])[:N]
    atoms = np.array([a for a in traj.top.atoms])[:N]

    # Collect backbone bonds i.e., only between C,N,CA (not C16) 
    bonds=[] 
    for b in traj.top.bonds:
        if b[0].residue.name in ['C16','12C', 'HOH'] or b[1].residue.name in ['C16','12C', 'HOH']:
            continue
        if b[0].name in ['C','N','CA'] and b[1].name in ['C','N','CA']:
            bonds += [ [b[0].index, b[1].index] ]
    bonds = np.array(bonds)

    radius = []
    for atom in atoms:
        radius += [ radius_dir[atom.name[0]] ]

    # Hbonds
    hbonds = mdtraj.baker_hubbard(traj[frame])
    hbonds_=[]
    for b in hbonds:
        if traj.top.atom(b[0]).name == 'N' and traj.top.atom(b[2]).name == 'O':
            hbonds_ += [b]
    hbonds = np.array(hbonds_)
    posH = positions[hbonds[:,1]]
    posA = utils.unwrap_points(positions[hbonds[:,2]], posH, Lx, Ly, Lz)

    # Define custom colors
    colors = np.zeros((N,3))
    colors[0:num_atomss[0]*nmols[0]] = color_pep
    # colors[num_atomss[0]*nmols[0]:num_atomss[0]*nmols[0]+num_atomss[1]*nmols[1]] = color_pep2
    # c_=[]
    # for i in range(nmols[1]):
    #     c_+=[color_BMP]*128+[color_pep2]*(num_atomss[1]-128)
    # colors[num_atomss[0]*nmols[0]:] = c_
    for i,atom in enumerate(atoms):
        if atom.residue.name == 'C16':
            colors[i] = color_C16
        elif atom.residue.name == '12C':
            colors[i] = color_12C
    colors = np.array(colors)
    # alpha = 0.2
    # colors = np.append(colors, [[alpha]]*len(colors), axis=1)
    positions_backbone = positions[ np.sort(list(set(bonds.reshape(-1)))) ]
    center = np.mean(positions_backbone, axis=0)

    #--------------------------------------------------------------------------------------
    # Calculate gyration eigenvectors

    # Gij = 1/N SUM_n:1_N [ rn_i * rn_j ]
    points = positions_backbone - np.mean(positions_backbone, axis=0)
    G = np.zeros((3,3))
    for i,j in itertools.product(range(3),range(3)):
        G[i,j] = np.sum(points[:,i]*points[:,j])
    G /= len(positions_backbone)

    w,v = scipy.linalg.eig(G)

    args = np.argsort(w)

    v1 = v[:,args[2]]
    v2 = v[:,args[1]]
    v3 = v[:,args[0]]

    
    #--------------------------------------------------------------------------------------
    # Draw fresnel scene

    # Sphere
    # scene = fresnel.Scene()
    # geometry = fresnel.geometry.Sphere(scene, N=len(positions), radius=np.array(radius),  outline_width = 0.01)
    # geometry.position[:] = positions
    # geometry.color[:] = colors #fresnel.color.linear(colors)

    for i,vi,vj in [['v3',v3,v1]]:
        # Cylinder
        scene = fresnel.Scene()
        # scene.background_color = [1, 1, 1]
        # scene.background_alpha = 1
        
        geometry = fresnel.geometry.Cylinder(scene, N=len(bonds), radius=radius_cylinder, outline_width=outline_width)
        geometry.points[:] = list(zip(positions[bonds[:,0]], positions[bonds[:,1]]))
        geometry.color[:] = list(zip(colors[bonds[:,0]], colors[bonds[:,1]]))
        geometry.material = fresnel.material.Material(
            # color=fresnel.color.linear([0.1, 0.1, 0.6]),
            roughness=0.5, 
            primitive_color_mix=1,
            # specular=0,
            spec_trans=1
            )
        

        # hbonds
        geometry = fresnel.geometry.Cylinder(scene, N=len(hbonds), radius=0.015, outline_width=0.005)
        geometry.points[:] = list(zip(posH, posA))
        geometry.color[:] = to_rgb('#325288')
        geometry.material = fresnel.material.Material(
            # color=fresnel.color.linear([0.1, 0.1, 0.6]),
            roughness=0.5, 
            primitive_color_mix=1,
            # specular=1,
            # spec_trans=1
            )


        scene.lights = fresnel.light.lightbox()

        scene.camera = fresnel.camera.Orthographic.fit(scene)

        scene.camera.position = 5 * vi + center
        scene.camera.look_at = center
        scene.camera.up = vj

        # scene.lights = fresnel.light.lightbox()
        if preview:
            out = fresnel.preview(scene)
        else:
            out = fresnel.pathtrace(scene, w=800, h=800)
            # out = fresnel.pathtrace(scene, w=800, h=800, samples=16, light_samples=32)
        
        PIL.Image.fromarray(out[:], mode='RGBA').save(i+filename)
        
        # os.system(f'open {i+filename}')

    # view = fresnel.interact.SceneView(scene)
    # view.show()
    # app = QApplication.instance()
    # app.exec_()

    #--------------------------------------------------------------------------------------



def draw_snapshot1(grofile, trajfile, topfile, itpfiles, 
    molnames, draw_box, frame, filename='snapshot.png', preview=True):
    """
    For atomistic simulations.
    
    Single PA system

    Draw snapshots along the three gyration eigenvectors
    
    """

    #---------------------------------------- Parameters ---------------------------------
    
    radius_dir = dict(H=0, C=0.15, N=0.12, S=0.15, O=0.15)
    radius_cylinder = 0.03
    outline_width = 0.005

    color_C16  = to_rgb('#0D2D6C')
    color_12C  = to_rgb('#0D2D6C')
    color_pep = to_rgb('#00FF00')
    color_pep2 = to_rgb('#00FF00')
    color_BMP  = to_rgb('#e41a1c')


    #--------------------------------------------------------------------------------------
    # Calculate bonds and colors

    traj = mdtraj.load(trajfile, top=grofile)

    num_atomss  = []
    nmols       = []
    positions   = []
    start = 0
    for i,molname in enumerate(molnames):
        num_atomss  += [utils.get_num_atoms_fromGRO(itpfiles[i].replace('.itp','')+'.gro')]
        nmols       += [utils.get_num_molecules_fromtop(topfile, molname)]
        positions   += list(traj.xyz[frame,start:start+num_atomss[i]*nmols[i]])
        start += num_atomss[i]*nmols[i]
    positions = np.array(positions)

    N = len(positions)

    Lx, Ly, Lz = traj.unitcell_lengths[frame]

    residues = np.array([r for r in traj.top.residues])[:N]
    atoms = np.array([a for a in traj.top.atoms])[:N]

    # Collect backbone bonds i.e., only between C,N,CA (not C16) 
    bonds=[] 
    for b in traj.top.bonds:
        if b[0].residue.name in ['C16','12C', 'HOH'] or b[1].residue.name in ['C16','12C', 'HOH']:
            continue
        if b[0].name in ['C','N','CA'] and b[1].name in ['C','N','CA']:
            bonds += [ [b[0].index, b[1].index] ]
    bonds = np.array(bonds)

    radius = []
    for atom in atoms:
        radius += [ radius_dir[atom.name[0]] ]


    # Define custom colors
    colors = np.zeros((N,3))
    colors[0:num_atomss[0]*nmols[0]] = color_pep
    # colors[num_atomss[0]*nmols[0]:num_atomss[0]*nmols[0]+num_atomss[1]*nmols[1]] = color_pep2
    # c_=[]
    # for i in range(nmols[1]):
    #     c_+=[color_BMP]*128+[color_pep2]*(num_atomss[1]-128)
    # colors[num_atomss[0]*nmols[0]:] = c_
    for i,atom in enumerate(atoms):
        if atom.residue.name == 'C16':
            colors[i] = color_C16
        elif atom.residue.name == '12C':
            colors[i] = color_12C
    colors = np.array(colors)

    positions_backbone = positions[ np.sort(list(set(bonds.reshape(-1)))) ]
    center = np.mean(positions_backbone, axis=0)

    #--------------------------------------------------------------------------------------
    # Calculate gyration eigenvectors

    # Gij = 1/N SUM_n:1_N [ rn_i * rn_j ]
    points = positions_backbone - np.mean(positions_backbone, axis=0)
    G = np.zeros((3,3))
    for i,j in itertools.product(range(3),range(3)):
        G[i,j] = np.sum(points[:,i]*points[:,j])
    G /= len(positions_backbone)

    w,v = scipy.linalg.eig(G)

    args = np.argsort(w)

    v1 = v[:,args[2]]
    v2 = v[:,args[1]]
    v3 = v[:,args[0]]

    
    #--------------------------------------------------------------------------------------
    # Draw fresnel scene

    # Sphere
    # scene = fresnel.Scene()
    # geometry = fresnel.geometry.Sphere(scene, N=len(positions), radius=np.array(radius),  outline_width = 0.01)
    # geometry.position[:] = positions
    # geometry.color[:] = colors #fresnel.color.linear(colors)

    for i,vi,vj in [['v1',v1,v3], ['v2',v2,v3], ['v3',v3,v1]]:
        # Cylinder
        scene = fresnel.Scene()
        geometry = fresnel.geometry.Cylinder(scene, N=len(bonds), radius=radius_cylinder, outline_width=outline_width)
        geometry.points[:] = list(zip(positions[bonds[:,0]], positions[bonds[:,1]]))
        geometry.color[:] = list(zip(colors[bonds[:,0]], colors[bonds[:,1]]))

        geometry.material = fresnel.material.Material(
            roughness=0.5, 
            primitive_color_mix=1,
            # specular=1,
            # spec_trans=0
            )
        

        scene.camera = fresnel.camera.Orthographic.fit(scene)

        scene.camera.position = 5 * vi + center
        scene.camera.look_at = center
        scene.camera.up = vj

        # scene.lights = fresnel.light.lightbox()
        if preview:
            out = fresnel.preview(scene)
        else:
            out = fresnel.pathtrace(scene, w=800, h=800, samples=16, light_samples=32)
        
        PIL.Image.fromarray(out[:], mode='RGBA').save(i+filename)
        

        # os.system(f'open {i+filename}')

    # view = fresnel.interact.SceneView(scene)
    # view.show()
    # app = QApplication.instance()
    # app.exec_()

    #--------------------------------------------------------------------------------------



def draw_snapshot(grofile, trajfile, topfile, itpfiles, molnames, 
    molattributes=[], hbondattributes=[], slice_slab=[],
    draw_box=False, frame=-1, filename='snapshot.png', preview=True,
    view=[['axono']], primitive='cylinder', lights=''):
    """
    For gromacs atomistic simulations.

    Draw snapshots with hbonds shown
    
    molattributes = [ [particle_id_start, particle_id_end, hexcolor, radius, outline_width], [particle_id_start, particle_id_end, hexcolor, radius, outline_width],... ]
    particle ids not covered in the colors list are not draw
    
    hbondattributes = [hexcolor, radius, outline_width]

    slice_slab = ['eig',i,j, thickness (nm)] | []
    from center cut the slice along vi and slicing in vj direction using thickness

    preview: fresnel preview or pathtrace
    view: ['axono','name']  axonometric view
          ['axis', i, j, name] - looking from axis i, axis j points upwards
          ['eig', i,j, name] - looking from vi to system center, vj is upward direction
    vi -> {0,1,2} are gyration vectors ids, 0 and 2 correspond to largest and smallest eigenvale
          [] - diagonal - axonometric view
          snapshots are saved for each view in the list
          name is the optional name appended before filename for each view snapshot

    primitive: sphere | cylinder

    lights: fresnel lights - lightbox, cloudy, rembrandt
    Notes: no water drawn | only backbone atoms are drawn
    """

    #-------------------------------------- default parameters ---------------------------------
    
    # radius_dir = dict(H=0, C=0.15, N=0.12, S=0.15, O=0.15) # for sphere primitive
    # radius_cylinder = 0.03
    # outline_width_cylinder = 0.00
    # outline_width_sphere = 0.01
    # radius_cylinder_hbond = 0.015
    # outline_width_cylinder_hbonds = 0.005

    radius = 0.1
    outline_width = 0.01
    
    # color_C16  = to_rgb('#0D2D6C') #542e71
    # color_12C  = to_rgb('#0D2D6C')
    # color_pep = to_rgb('#00FF00')
    # color_pep2 = to_rgb('#00FF00')
    # color_BMP  = to_rgb('#e41a1c')

    #------------------------------------ Read system attributes -----------------------------

    traj = mdtraj.load(trajfile, top=grofile)
    
    Lx, Ly, Lz = traj.unitcell_lengths[frame]

    
    atoms = np.array([r for r in traj.top.atoms])

    num_atomss  = []
    nmols       = []
    positionss  = []
    atomss = []
    start = 0
    for i,molname in enumerate(molnames):
        num_atomss  += [utils.get_num_atoms_fromGRO(itpfiles[i].replace('.itp','')+'.gro')]
        nmols       += [utils.get_num_molecules_fromtop(topfile, molname)]
        positionss  += [ traj.xyz[frame,start:start+num_atomss[i]*nmols[i]] ]
        atomss += [ atoms[start:start+num_atomss[i]*nmols[i]] ]
        start += num_atomss[i]*nmols[i]
    

    # c: green, blue, red
    c = [to_rgb('#00FF00'), to_rgb('#2978b5'), to_rgb('#fb3640')]
    # if molattributes == []:
    #     for i,molname in enumerate(molnames):
    #         molattributes += [ [ [0,num_atomss[i]*nmols[i], c[i], radius, outline_width] ] ]
    if molattributes != []:
        for i,molattribute in enumerate(molattributes):
            for j,m in enumerate(molattribute):
                molattributes[i][j][2] = to_rgb(m[2])

    
    # Read bonds per molecule type
    bondss = get_bonds_fromtopfile(topfile, molnames)
    backbone_bondss = []
    alkyl_bondss = []
    bb_and_alkyl_bondss = []
    start = 0
    for i,molname in enumerate(molnames):
        # Collect backbone bonds i.e., only between C,N,CA (not C16) 
        bonds_=[]
        for b in bondss[i]:
            resname0 = atomss[i][b[0]].residue.name
            resname1 = atomss[i][b[1]].residue.name
            atomname0 = atomss[i][b[0]].name
            atomname1 = atomss[i][b[1]].name
            if resname0 in ['C16','12C', 'HOH'] or resname1 in ['C16','12C', 'HOH']:
                continue
            if atomname0 in ['C','N','CA'] and atomname1 in ['C','N','CA']:
                bonds_ += [ [b[0], b[1]] ]
        backbone_bondss += [np.array(bonds_)]
        
        # Collect alkyl bonds i.e., bonds between C-C
        bonds_=[]
        for b in bondss[i]:
            resname0 = atomss[i][b[0]].residue.name
            resname1 = atomss[i][b[1]].residue.name
            atomname0 = atomss[i][b[0]].name
            atomname1 = atomss[i][b[1]].name
            if resname0 in ['C16','12C'] and resname1 in ['C16','12C']:
                if atomname0[0] == 'C' and atomname1[0] == 'C':
                    bonds_ += [ [b[0], b[1]] ]
        alkyl_bondss += [np.array(bonds_)]

        # collect both backbone and alkyl bonds that can be draw
        bb_and_alkyl_bondss += [ np.append(backbone_bondss[i], alkyl_bondss[i], axis=0) ]


    # Backbone positions
    backbone_positionss=[]
    for i,bb in enumerate(backbone_bondss):
        backbone_positions = np.empty((0,3))
        for j in range(nmols[i]):
            args = np.sort(list(set((bb + j * num_atomss[i]).reshape(-1))))
            backbone_positions = np.append(backbone_positions, positionss[i][args], axis=0)
    backbone_positionss += [backbone_positions]



    #------------------------------------  gyration eigenvectors ------------------------
    
    points = np.empty((0,3))
    for p in backbone_positionss:
        points = np.append(points, p, axis=0)
    
    center = np.mean(points, axis=0)

    # Calculate gyration eigenvectors using only backbone atoms
            
    # Gij = 1/N SUM_n:1_N [ rn_i * rn_j ]
    points -= center
    G = np.zeros((3,3))
    for i,j in itertools.product(range(3),range(3)):
        G[i,j] = np.mean(points[:,i]*points[:,j])
    

    w,v = scipy.linalg.eig(G)

    args = np.argsort(w)[::-1]

    eigvec = v[:,args]


    #------------------------------------  slice or not ------------------------

    if slice_slab != []:
        if slice_slab[0]=='eig':
            
            vi = eigvec[:,slice_slab[1]]
            vj = eigvec[:,slice_slab[2]]
            thickness = slice_slab[3]
            
            slice_filtrs = [] 
            for pos in positionss:
                t = (pos-center).dot(vj.reshape(-1,1)).reshape(-1)
                slice_filtrs += [ (t <= thickness/2) * (t >= -thickness/2) ]

            

    #------------------------------------ draw fresnel scene --------------------------------------------------

    
    # Start scene    
    scene = fresnel.Scene()

    # Primitive
    if primitive == 'cylinder':    
        
        for k,molattribute in enumerate(molattributes):
            
            for id1, id2, color_, radius, outline_width in molattribute:
                
                filtr = (bb_and_alkyl_bondss[k][:,0]>=id1) * \
                        (bb_and_alkyl_bondss[k][:,0]<=id2) * \
                        (bb_and_alkyl_bondss[k][:,1]>=id1) * \
                        (bb_and_alkyl_bondss[k][:,1]<=id2)

                bb_ = bb_and_alkyl_bondss[k][filtr]
                bb = []
                for m in range(nmols[k]):
                    bb += list( bb_ + m * num_atomss[k] )
                bb = np.array(bb).astype(int)

                if slice_slab != []:
                    filtr = slice_filtrs[k][bb[:,0]] * \
                             slice_filtrs[k][bb[:,1]]
                    bb = bb[filtr] 

                points1 = positionss[k][bb[:,0]]
                points2 = utils.unwrap_points(positionss[k][bb[:,1]], points1, Lx, Ly, Lz)
                
                geometry                  = fresnel.geometry.Cylinder(scene, N = len(points1))
                geometry.radius[:]        = radius
                geometry.outline_width    = outline_width
                geometry.points[:]        = list(zip(points1, points2))
                geometry.color[:]         = color_
                geometry.material         = fresnel.material.Material(
                                            roughness=0.5, 
                                            primitive_color_mix=1)

    elif primitive == 'sphere':    

        for k,molattribute in enumerate(molattributes):
            
            for id1, id2, color_, radius, outline_width in molattribute:
                
                args_ = np.array(list(set(bb_and_alkyl_bondss[k].reshape(-1))))
                args_ = args_[(args_>=id1) * (args_<=id2)]
                args = []
                for m in range(nmols[k]):
                     args += list( args_ + m * num_atomss[k] )

                if slice_slab != []:
                    filtr = slice_filtrs[k][args] * \
                             slice_filtrs[k][args]
                    args = args[filtr]

                points = positionss[k][args]
                
                geometry                  = fresnel.geometry.Sphere(scene, N = len(points))
                geometry.radius[:]        = radius
                geometry.outline_width    = outline_width
                geometry.position[:]      = points
                geometry.color[:]         = color_
                geometry.material         = fresnel.material.Material(
                                            roughness=0.5, 
                                            primitive_color_mix=1)


    # Hbonds
    if hbondattributes != []:
        color_, radius, outline_width = hbondattributes
        color_ = to_rgb(color_)

        hbonds = mdtraj.baker_hubbard(traj[frame])
        hbonds_=[]
        for b in hbonds:
            if traj.top.atom(b[0]).name == 'N' and traj.top.atom(b[2]).name == 'O':
                hbonds_ += [b]
        hbonds = np.array(hbonds_)
        
        if slice_slab != []:
            slice_filtr = np.empty(0, dtype=bool)
            for f in slice_filtrs:
                slice_filtr = np.append(slice_filtr, f)
            filtr = slice_filtr[hbonds[:,0]] * \
                     slice_filtr[hbonds[:,1]]
            hbonds = hbonds[filtr] 

        pos = np.empty((0,3))
        for p in positionss:
            pos = np.append(pos,p,axis=0)
        posH = pos[hbonds[:,1]]
        posA = utils.unwrap_points(pos[hbonds[:,2]], posH, Lx, Ly, Lz)

        points1 = posH
        points2 = posA
        
        geometry                  = fresnel.geometry.Cylinder(scene, N = len(points1))
        geometry.radius[:]        = radius
        geometry.outline_width    = outline_width
        geometry.points[:]        = list(zip(points1, points2))
        geometry.color[:]         = color_
        geometry.material         = fresnel.material.Material(
                                    roughness=0.5, 
                                    primitive_color_mix=1)



    # Lights
    if lights == 'lightbox':
        scene.lights = fresnel.light.lightbox()
    elif lights == 'cloudy':
        scene.lights = fresnel.light.cloudy()
    elif lights == 'rembrandt':
        scene.lights = fresnel.light.rembrandt()


    # Camera    
    for view_ in view:
        if view_[0] == 'eig':
            try:
                vname = view_[3]
            except:
                vname = ''
            
            vi = eigvec[:,view_[1]]
            vj = eigvec[:,view_[2]]
            
            scene.camera = fresnel.camera.Orthographic.fit(scene)
            scene.camera.position = 5 * vi + center
            scene.camera.look_at = center
            scene.camera.up = vj

        elif view_[0] == 'axono': # use axonometric view
            try:
                vname = view_[1]
            except:
                vname = ''

            scene.camera = fresnel.camera.Orthographic.fit(scene)
            scene.camera.position = center + [4, 4, 5]
            scene.camera.look_at = center    

        if preview:
            out = fresnel.preview(scene, w=800, h=800)
        else:
            out = fresnel.pathtrace(scene, w=800, h=800, samples=16, light_samples=32)
        
        PIL.Image.fromarray(out[:], mode='RGBA').save(vname+filename)


    # os.system(f'open {"axono"+filename}')
    # view = fresnel.interact.SceneView(scene)
    # view.show()
    # app = QApplication.instance()
    # app.exec_()

    #--------------------------------------------------------------------------------------



def draw_snapshot_(grofile, trajfile, topfile, itpfiles, molnames, 
    molattributes = [], draw_box=False, frame=-1, filename='snapshot.png', preview=True,
    view=[['axono']], primitive='cylinder', lights=''):
    """
    For gromacs atomistic simulations.

    Draw snapshots for each view entry
    
    molattributes = [ [particle_id_start, particle_id_end, hexcolor, radius, outline_width], [particle_id_start, particle_id_end, hexcolor, radius, outline_width],... ]
    particle ids not covered in the colors list are not draw
    
    preview: fresnel preview or pathtrace
    view: ['axono','name']  axonometric view
          ['axis', i, j, name] - looking from axis i, axis j points upwards
          ['eig', i,j, name] - looking from vi to system center, vj is upward direction
    vi -> {0,1,2} are gyration vectors ids, 0 and 2 correspond to largest and smallest eigenvale
          [] - diagonal - axonometric view
          snapshots are saved for each view in the list
          name is the optional name appended before filename for each view snapshot

    primitive: sphere | cylinder

    lights: fresnel lights - lightbox, cloudy, rembrandt
    Notes: no water drawn | only backbone atoms are drawn
    """

    #-------------------------------------- default parameters ---------------------------------
    
    # radius_dir = dict(H=0, C=0.15, N=0.12, S=0.15, O=0.15) # for sphere primitive
    # radius_cylinder = 0.03
    # outline_width_cylinder = 0.00
    # outline_width_sphere = 0.01
    # radius_cylinder_hbond = 0.015
    # outline_width_cylinder_hbonds = 0.005

    radius = 0.1
    outline_width = 0.01
    
    # color_C16  = to_rgb('#0D2D6C') #542e71
    # color_12C  = to_rgb('#0D2D6C')
    # color_pep = to_rgb('#00FF00')
    # color_pep2 = to_rgb('#00FF00')
    # color_BMP  = to_rgb('#e41a1c')

    #------------------------------------ Read system attributes -----------------------------

    traj = mdtraj.load(trajfile, top=grofile)
    
    Lx, Ly, Lz = traj.unitcell_lengths[frame]

    
    atoms = np.array([r for r in traj.top.atoms])

    num_atomss  = []
    nmols       = []
    positionss  = []
    atomss = []
    start = 0
    for i,molname in enumerate(molnames):
        num_atomss  += [utils.get_num_atoms_fromGRO(itpfiles[i].replace('.itp','')+'.gro')]
        nmols       += [utils.get_num_molecules_fromtop(topfile, molname)]
        positionss  += [ traj.xyz[frame,start:start+num_atomss[i]*nmols[i]] ]
        atomss += [ atoms[start:start+num_atomss[i]*nmols[i]] ]
        start += num_atomss[i]*nmols[i]
    

    # c: green, blue, red
    c = [to_rgb('#00FF00'), to_rgb('#2978b5'), to_rgb('#fb3640')]
    if molattributes == []:
        for i,molname in enumerate(molnames):
            molattributes += [ [ [0,num_atomss[i]*nmols[i], c[i], radius, outline_width] ] ]
    else:
        for i,molattribute in enumerate(molattributes):
            for j,m in enumerate(molattribute):
                molattributes[i][j][2] = to_rgb(m[2])


    # Read bonds per molecule type
    bondss = get_bonds_fromtopfile(topfile, molnames)
    backbone_bondss = []
    alkyl_bondss = []
    bb_and_alkyl_bondss = []
    start = 0
    for i,molname in enumerate(molnames):
        # Collect backbone bonds i.e., only between C,N,CA (not C16) 
        bonds_=[]
        for b in bondss[i]:
            resname0 = atomss[i][b[0]].residue.name
            resname1 = atomss[i][b[1]].residue.name
            atomname0 = atomss[i][b[0]].name
            atomname1 = atomss[i][b[1]].name
            if resname0 in ['C16','12C', 'HOH'] or resname1 in ['C16','12C', 'HOH']:
                continue
            if atomname0 in ['C','N','CA'] and atomname1 in ['C','N','CA']:
                bonds_ += [ [b[0], b[1]] ]
        backbone_bondss += [np.array(bonds_)]
        
        # Collect alkyl bonds i.e., bonds between C-C
        bonds_=[]
        for b in bondss[i]:
            resname0 = atomss[i][b[0]].residue.name
            resname1 = atomss[i][b[1]].residue.name
            atomname0 = atomss[i][b[0]].name
            atomname1 = atomss[i][b[1]].name
            if resname0 in ['C16','12C'] and resname1 in ['C16','12C']:
                if atomname0[0] == 'C' and atomname1[0] == 'C':
                    bonds_ += [ [b[0], b[1]] ]
        alkyl_bondss += [np.array(bonds_)]

        # collect both backbone and alkyl bonds that can be draw
        bb_and_alkyl_bondss += [ np.append(backbone_bondss[i], alkyl_bondss[i], axis=0) ]


    # Backbone positions
    backbone_positionss=[]
    for i,bb in enumerate(backbone_bondss):
        backbone_positions = np.empty((0,3))
        for j in range(nmols[i]):
            args = np.sort(list(set((bb + j * num_atomss[i]).reshape(-1))))
            backbone_positions = np.append(backbone_positions, positionss[i][args], axis=0)
    backbone_positionss += [backbone_positions]




    #------------------------------------  gyration eigenvectors ------------------------
    
    points = np.empty((0,3))
    for p in backbone_positionss:
        points = np.append(points, p, axis=0)
    
    center = np.mean(points, axis=0)

    # Calculate gyration eigenvectors using only backbone atoms
            
    # Gij = 1/N SUM_n:1_N [ rn_i * rn_j ]
    points -= center
    G = np.zeros((3,3))
    for i,j in itertools.product(range(3),range(3)):
        G[i,j] = np.mean(points[:,i]*points[:,j])
    

    w,v = scipy.linalg.eig(G)

    args = np.argsort(w)[::-1]

    eigvec = v[:,args]

    #------------------------------------ draw fresnel scene --------------------------------------------------

    
    # Start scene    
    scene = fresnel.Scene()

    # Primitive
    if primitive == 'cylinder':    
        
        for k,molattribute in enumerate(molattributes):
            
            for id1, id2, color_, radius, outline_width in molattribute:
                
                filtr = (bb_and_alkyl_bondss[k][:,0]>=id1) * \
                        (bb_and_alkyl_bondss[k][:,0]<=id2) * \
                        (bb_and_alkyl_bondss[k][:,1]>=id1) * \
                        (bb_and_alkyl_bondss[k][:,1]<=id2)
                bb_ = bb_and_alkyl_bondss[k][filtr]
                bb = []
                for m in range(nmols[k]):
                    bb += list( bb_ + m * num_atomss[k] )
                bb = np.array(bb).astype(int)

                points1 = positionss[k][bb[:,0]]
                points2 = utils.unwrap_points(positionss[k][bb[:,1]], points1, Lx, Ly, Lz)
                
                geometry                  = fresnel.geometry.Cylinder(scene, N = len(points1))
                geometry.radius[:]        = radius
                geometry.outline_width    = outline_width
                geometry.points[:]        = list(zip(points1, points2))
                geometry.color[:]         = color_
                geometry.material         = fresnel.material.Material(
                                            roughness=0.5, 
                                            primitive_color_mix=1)

    elif primitive == 'sphere':    

        for k,molattribute in enumerate(molattributes):
            
            for id1, id2, color_, radius, outline_width in molattribute:
                
                args_ = np.array(list(set(bb_and_alkyl_bondss[k].reshape(-1))))
                args_ = args_[(args_>=id1) * (args_<=id2)]
                args = []
                for m in range(nmols[k]):
                     args += list( args_ + m * num_atomss[k] )
                
                
                points = positionss[k][args]
                
                geometry                  = fresnel.geometry.Sphere(scene, N = len(points))
                geometry.radius[:]        = radius
                geometry.outline_width    = outline_width
                geometry.position[:]      = points
                geometry.color[:]         = color_
                geometry.material         = fresnel.material.Material(
                                            roughness=0.5, 
                                            primitive_color_mix=1)



    # Lights
    if lights == 'lightbox':
        scene.lights = fresnel.light.lightbox()
    elif lights == 'cloudy':
        scene.lights = fresnel.light.cloudy()
    elif lights == 'rembrandt':
        scene.lights = fresnel.light.rembrandt()


    # Camera    
    for view_ in view:
        if view_[0] == 'eig':
            try:
                vname = view_[3]
            except:
                vname = ''
            
            vi = eigvec[:,view_[1]]
            vj = eigvec[:,view_[2]]
            
            scene.camera = fresnel.camera.Orthographic.fit(scene)
            scene.camera.position = 5 * vi + center
            scene.camera.look_at = center
            scene.camera.up = vj

        elif view_[0] == 'axono': # use axonometric view
            try:
                vname = view_[1]
            except:
                vname = ''

            scene.camera = fresnel.camera.Orthographic.fit(scene)
            scene.camera.position = center + [4, 4, 5]
            scene.camera.look_at = center    

        if preview:
            out = fresnel.preview(scene, w=800, h=800)
        else:
            out = fresnel.pathtrace(scene, w=800, h=800, samples=16, light_samples=32)
        
        PIL.Image.fromarray(out[:], mode='RGBA').save(vname+filename)


    # os.system(f'open {"axono"+filename}')
    # view = fresnel.interact.SceneView(scene)
    # view.show()
    # app = QApplication.instance()
    # app.exec_()

    #--------------------------------------------------------------------------------------

