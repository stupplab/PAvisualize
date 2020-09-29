"""visualization routines for the CG MARTINI simulations using mayavi
"""

import numpy as np
import os
import mdtraj as md

from mayavi import mlab
from traits.api import HasTraits, Range, Instance, on_trait_change
from mayavi.core.api import PipelineBase
from traitsui.api import View, Item, Group
from mayavi.core.ui.api import MayaviScene, SceneEditor, MlabSceneModel




def unwrap_pos(positions, Lx, Ly, Lz, buffer):
    
    #TODO

    # for i in range(len(positions)):
    #     if positions[i,0] - minx >= Lx/2:
    #         pos[i,0] -= Lx
    #     if pos[i,1] - miny >= Ly/2:
    #         pos[i,1] -= Ly

    return positions




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

                



def get_num_atoms(itpfile):
    """
    Return number of atoms of a MARTINI molecule
    using the itpfile
    """
    
    num_atoms = 0
    start = False
    with open(itpfile, 'r') as f:
        for line in f:
            if '[ atoms ]' in line:
                start = True
                continue
            if start:
                words = line.split()
                if words == []:
                    break
                num_atoms += 1

    
    return num_atoms




def get_num_molecules(topfile, name):
    """Return the number of molecules in the simulation.
    Curretnly returns PA
    """

    with open(topfile, 'r') as f:
        for line in f:
            if line.split() != []:
                if line.split()[0]==name:
                    return int(line.split()[1])




def get_positions(gro, trr, atom_range):
    """Returns all the wrapped positions the MARTINI simulation trajectory
    for a molecule. 
    Returns trajectory of atom_range = (start, end)
    """



    traj = md.load_trr(trr, top=gro)
    # print(traj)
    # raise

    # positions = []
    # u = MDAnalysis.Universe(gro, trr)
    # for frame in u.trajectory:
    #     positions+= [frame.positions]
    # positions = np.array(positions)[:,atom_range[0]:atom_range[1]]
    
    
    # residues = traj.topology.residues

    # atoms = list(traj.topology.atoms) #[:num_atoms*nmol]
    
    # atoms = np.compress([not ('-W' in str(atom)) and not ('ION' in str(atom)) for atom in atoms], atoms, axis=0)
    

    positions = traj.xyz[:,atom_range[0]:atom_range[1]]


    return positions




def mlab_box(color_, box):
    # draw an outline box
    # color = (r,g,b,alpha), rgb -> [0,255]

    Lx = box['Lx']
    Ly = box['Ly']
    Lz = box['Lz']    

    X = np.array([
        ([0,0,0],[1,0,0]), ([0,0,0],[0,1,0]), ([0,0,0],[0,0,1]),
        ([1,0,0],[1,1,0]), ([1,0,0],[1,0,1]), ([0,1,0],[1,1,0]),
        ([0,1,0],[0,1,1]), ([0,0,1],[1,0,1]), ([0,0,1],[0,1,1]),
        ([1,1,0],[1,1,1]), ([1,0,1],[1,1,1]), ([0,1,1],[1,1,1])], dtype=np.float32)
        
    X[:,:,0] *= Lx
    X[:,:,1] *= Ly
    X[:,:,2] *= Lz

    color   = tuple(np.array(color_[:3])/255)
    opacity = color_[3]

    for x1,x2 in X:
        mlab.plot3d([x1[0],x2[0]], [x1[1],x2[1]], [x1[2],x2[2]], 
                    color=color, opacity=opacity, tube_radius=None)    





def mlab_drawaxes(color_, box):
    # draw x,y,z axis lines near (0,0,1)

    Lx = box['Lx']
    Ly = box['Ly']
    Lz = box['Lz']    

    r  = np.array([-1,-1,1+Lz], dtype=np.float32) # axes position
    X  = np.array([([0,0,0],[1,0,0]),([0,0,0],[0,1,0]),([0,0,0],[0,0,1])], dtype=np.float32)
    X += r
    text = ['x','y','z']

    color   = tuple(np.array([204,204,204])/255)
    opacity = 0.6
    scale   = 0.5
    
    # draw arrows
    color   = tuple(np.array(color_[:3])/255)
    opacity = color_[3]
    mlab.quiver3d([r[0]]*3,[r[1]]*3,[r[2]]*3,[1,0,0],[0,1,0],[0,0,1], 
                  color=color, opacity=opacity)

    
    # write text
    for i,(x1,x2) in enumerate(X):
        buff=1.4
        x2 = (x2-r)*buff+r + np.array([-1,-1,1])*scale/2
        mlab.text3d(x2[0], x2[1], x2[2], text=text[i], scale=scale, color=color, opacity=opacity)




def draw_points_and_lines(points, lines, color):
    ''' 
    points: N X M X 3
    lines: N X K X 2  (defines connections between points)
    '''

    bonds_permol =[]
    num_atoms    =[]
    nmol         =[]
    positions    =[]

    
    itpname = os.path.basename(itpfile).strip('.itp')

    bonds_permol = get_bonds_per_molecule(itpfile)
    num_atoms    = get_num_atoms(itpfile)
    nmol         = get_num_molecules(topfile, itpname)
    positions    = get_positions(grofile, trrfile, (0,num_atoms*nmol))
   
    
    num_frames = positions.shape[0]  
    
    

    if centralize:
        # Doesnt work correctly for now
        pos_ = np.copy(positions.reshape(num_frames, nmol, num_atoms, 3))
        pos_[:,:] = pos_[:,:,0].reshape(num_frames,nmol,1,3)
        pos_ = pos_.reshape(num_frames,-1,3)

        positions = PAanalysis.unwrap_points(positions, pos_, Lx,Ly,Lz)
        
        positions -= np.mean(positions, axis=1).reshape(-1,1,3)
        # positions += [Lx/2, Ly/2, Lz/2]

    

    mol_color = [84,39,143,0.6] #[117,107,177, 0.6]
    light_green = [161,215,106]
    yellow = [255, 217, 0]
    purple = [118,42,131]
    pink = [197,27,125]
    orange = [255,127,0]
    blue = [84,39,143]
    red = [165,15,21]
    mol_color = [84,39,143,0.6]
    alkyl_color = purple+[0.6]
    pep_color = blue+[0.7]

    colors = [yellow, blue, red, light_green]

    
    def dump_frame(frame,index):

        mlab.options.offscreen = True
        fig=mlab.figure(1, bgcolor=(1, 1, 1), fgcolor=(0, 0, 0), size=(800, 800))
        mlab.clf()


        connections = []
        connections_alkyl = []
        connections_pep = []
        for i in range(nmol):
            for bond in bonds_permol:
                X1 = positions[ frame, bond[0] + num_atoms*i ]
                X2 = positions[ frame, bond[1] + num_atoms*i ]
                if not np.linalg.norm(X1-X2) > 0.5*min([Lx,Ly,Lz]):
                    connections = connections + [bond+num_atoms*i]
                    if set(bond) in bonds_alkyl:
                        connections_alkyl += [bond+num_atoms*i]
                    else:
                        connections_pep   += [bond+num_atoms*i]
                    
        
        X = positions[frame]

        connections = np.array(connections)
        connections_alkyl = np.array(connections_alkyl)
        connections_pep = np.array(connections_pep)
        

        x=X[:,0]; y=X[:,1]; z=X[:,2]
        src = mlab.pipeline.scalar_scatter(x, y, z)
        src.mlab_source.dataset.lines = connections_alkyl
        src.update()
        mlab.pipeline.surface(src, color=tuple(np.array(alkyl_color[:3])/255),
                opacity=alkyl_color[3], line_width=2)

        
        src = mlab.pipeline.scalar_scatter(x, y, z)
        src.mlab_source.dataset.lines = connections_pep
        src.update()
        mlab.pipeline.surface(src, color=tuple(np.array(pep_color[:3])/255),
                opacity=pep_color[3], line_width=2)
        

        if draw_box:
            box_color = [0,0,0,0.6]#[204,204,204, 0.7]
            mlab_box(box_color)
            mlab_drawaxes(box_color)
            
        mlab.view(azimuth=35, elevation=35, roll=0, distance=40, focalpoint=None)
        fig.scene.parallel_projection = True # orthogonal projection
        
        
        mlab.savefig(sim_path+'%04d.png'%index)
        mlab.close()





    for i,frame in enumerate(range(0,num_frames,10)):
        dump_frame(frame,i)
           

    filename = sim_path+'movie.mp4'
    os.system(f'ffmpeg -framerate 10 -i {sim_path}/%04d.png -s:v 1080x1080 -c:v libx264 -profile:v high -crf 20 -pix_fmt yuv420p \
        {filename}')
    
    os.system('rm %s'%sim_path+'0*.png')






def draw_snapshot(itpfile, topfile, grofile, trrfile, bonds_alkyl, frame, filename, box, draw_box=True):
    """Draw the minimal form of the simulation 
    Takes in atom coordinates and bonds
    """
    
    bonds_permol =[]
    num_atoms    =[]
    nmol         =[]
    positions    =[]

    
    itpname = os.path.basename(itpfile).strip('.itp')

    bonds_permol = get_bonds_per_molecule(itpfile)
    num_atoms    = get_num_atoms(itpfile)
    nmol         = get_num_molecules(topfile, itpname)
    positions    = get_positions(grofile, trrfile, (0,num_atoms*nmol))
    
    
    num_frames = positions.shape[0]  
    
    
    Lx = box['Lx']
    Ly = box['Ly']
    Lz = box['Lz']    


    light_green = [161,215,106]
    purple = [118,42,131]
    alkyl_color = purple+[0.6]
    pep_color = light_green+[0.6]

    
    def dump_frame(frame):

        mlab.options.offscreen = True
        fig=mlab.figure(1, bgcolor=(1, 1, 1), fgcolor=(0, 0, 0), size=(800, 800))
        mlab.clf()


        connections = []
        connections_alkyl = []
        connections_pep = []
        for i in range(nmol):
            for bond in bonds_permol:
                X1 = positions[ frame, bond[0] + num_atoms*i ]
                X2 = positions[ frame, bond[1] + num_atoms*i ]
                if not np.linalg.norm(X1-X2) > 0.5*min([Lx,Ly,Lz]):
                    connections = connections + [bond+num_atoms*i]
                    if set(bond) in bonds_alkyl:
                        connections_alkyl += [bond+num_atoms*i]
                    else:
                        connections_pep   += [bond+num_atoms*i]
                    
        
        X = positions[frame]

        connections = np.array(connections)
        connections_alkyl = np.array(connections_alkyl)
        connections_pep = np.array(connections_pep)
        

        x=X[:,0]; y=X[:,1]; z=X[:,2]
        src = mlab.pipeline.scalar_scatter(x, y, z)
        src.mlab_source.dataset.lines = connections_alkyl
        src.update()
        mlab.pipeline.surface(src, color=tuple(np.array(alkyl_color[:3])/255),
                opacity=alkyl_color[3], line_width=2)

        
        src = mlab.pipeline.scalar_scatter(x, y, z)
        src.mlab_source.dataset.lines = connections_pep
        src.update()
        mlab.pipeline.surface(src, color=tuple(np.array(pep_color[:3])/255),
                opacity=pep_color[3], line_width=2)
        

        if draw_box:
            box_color = [0,0,0,0.6]#[204,204,204, 0.7]
            mlab_box(box_color, box)
            mlab_drawaxes(box_color, box)
            
        mlab.view(azimuth=35, elevation=35, roll=0, distance=40, focalpoint=None)
        fig.scene.parallel_projection = True # orthogonal projection
        
        
        mlab.savefig(filename)
        mlab.close()



    
    dump_frame(frame)
        





def draw_movie(itpfile, topfile, grofile, trrfile, bonds_alkyl, filename, box, draw_box=True, centralize=False):
    """Draw the minimal form of the simulation 
    Takes in atom coordinates and bonds
    """
    
    bonds_permol =[]
    num_atoms    =[]
    nmol         =[]
    positions    =[]

    
    itpname = os.path.basename(itpfile).strip('.itp')

    bonds_permol = get_bonds_per_molecule(itpfile)
    num_atoms    = get_num_atoms(itpfile)
    nmol         = get_num_molecules(topfile, itpname)
    positions    = get_positions(grofile, trrfile, (0,num_atoms*nmol))
   
    
    num_frames = positions.shape[0]  
    
    
    
    Lx = box['Lx']
    Ly = box['Ly']
    Lz = box['Lz']    

    sim_path = os.path.dirname(filename)+'/'


    if centralize:
        # Doesnt work correctly for now
        pos_ = np.copy(positions.reshape(num_frames, nmol, num_atoms, 3))
        pos_[:,:] = pos_[:,:,0].reshape(num_frames,nmol,1,3)
        pos_ = pos_.reshape(num_frames,-1,3)

        positions = PAanalysis.unwrap_points(positions, pos_, Lx,Ly,Lz)
        
        positions -= np.mean(positions, axis=1).reshape(-1,1,3)
        # positions += [Lx/2, Ly/2, Lz/2]

    

    light_green = [161,215,106]
    purple = [118,42,131]
    alkyl_color = purple+[0.6]
    pep_color = light_green+[0.6]

    
    def dump_frame(frame,index):

        mlab.options.offscreen = True
        fig=mlab.figure(1, bgcolor=(1, 1, 1), fgcolor=(0, 0, 0), size=(800, 800))
        mlab.clf()


        connections = []
        connections_alkyl = []
        connections_pep = []
        for i in range(nmol):
            for bond in bonds_permol:
                X1 = positions[ frame, bond[0] + num_atoms*i ]
                X2 = positions[ frame, bond[1] + num_atoms*i ]
                if not np.linalg.norm(X1-X2) > 0.5*min([Lx,Ly,Lz]):
                    connections = connections + [bond+num_atoms*i]
                    if set(bond) in bonds_alkyl:
                        connections_alkyl += [bond+num_atoms*i]
                    else:
                        connections_pep   += [bond+num_atoms*i]
                    
        
        X = positions[frame]

        connections = np.array(connections)
        connections_alkyl = np.array(connections_alkyl)
        connections_pep = np.array(connections_pep)
        

        x=X[:,0]; y=X[:,1]; z=X[:,2]
        src = mlab.pipeline.scalar_scatter(x, y, z)
        src.mlab_source.dataset.lines = connections_alkyl
        src.update()
        mlab.pipeline.surface(src, color=tuple(np.array(alkyl_color[:3])/255),
                opacity=alkyl_color[3], line_width=2)

        
        src = mlab.pipeline.scalar_scatter(x, y, z)
        src.mlab_source.dataset.lines = connections_pep
        src.update()
        mlab.pipeline.surface(src, color=tuple(np.array(pep_color[:3])/255),
                opacity=pep_color[3], line_width=2)
        

        if draw_box:
            box_color = [0,0,0,0.6]#[204,204,204, 0.7]
            mlab_box(box_color, box)
            mlab_drawaxes(box_color, box)
            
        mlab.view(azimuth=35, elevation=35, roll=0, distance=40, focalpoint=None)
        fig.scene.parallel_projection = True # orthogonal projection
        
        
        mlab.savefig(sim_path+'%04d.png'%index)
        mlab.close()





    for i,frame in enumerate(range(0,num_frames,10)):
        dump_frame(frame,i)
           

    
    os.system(f'ffmpeg -framerate 10 -i {sim_path}/%04d.png -s:v 1080x1080 -c:v libx264 -profile:v high -crf 20 -pix_fmt yuv420p \
        {filename}')
    
    os.system('rm %s'%sim_path+'0*.png')






def run_animation(box):
    

    
    bonds_perPA = get_bonds_per_molecule(job.fn('PA.itp'))
    num_atoms   = np.max(bonds_perPA)+1
    positions   = get_positions(job.fn('PA_water_min.gro'), job.fn('PA_water_eq.trr'), num_atoms)

    nmol = statepoint['nmol']

    mol_color = [84,39,143,0.6] #[117,107,177, 0.6]
    light_green = [161,215,106]
    yellow = [255,255,191]
    purple = [118,42,131]
    pink = [197,27,125]
    orange = [255,127,0]
    mol_color = [84,39,143,0.6]
    alkyl_color = purple+[0.6]
    pep_color = light_green+[0.6]

    # TODO: make this more generic for different alkyl lengths
    bonds_alkyl = [{0,1},{1,2},{2,3}]


    Lx = box['Lx']
    Ly = box['Ly']
    Lz = box['Lz']    


    fig=mlab.figure(1, bgcolor=(1, 1, 1), fgcolor=(0, 0, 0), size=(800, 800))
    mlab.clf()


    def draw_frame(frame):
        
        connections = []
        connections_alkyl = []
        connections_pep = []
        for i in range(nmol):
            for bond in bonds_perPA:
                X1 = positions[ frame, bond[0] + num_atoms*i ]
                X2 = positions[ frame, bond[1] + num_atoms*i ]
                if not np.linalg.norm(X1-X2) > 0.5*min([statepoint['Lx'],statepoint['Ly'],statepoint['Lz']]):
                    connections = connections + [bond+num_atoms*i]
                    if set(bond) in bonds_alkyl:
                        connections_alkyl += [bond+num_atoms*i]
                    else:
                        connections_pep   += [bond+num_atoms*i]

        
        X = positions[frame]
        connections = np.array(connections)
        connections_alkyl = np.array(connections_alkyl)
        connections_pep = np.array(connections_pep)


        x=X[:,0]; y=X[:,1]; z=X[:,2]
        src_alkyl = mlab.pipeline.scalar_scatter(x, y, z)
        src_alkyl.mlab_source.dataset.lines = connections_alkyl
        src_alkyl.update()
        surf_alkyl = mlab.pipeline.surface(src_alkyl, color=tuple(np.array(alkyl_color[:3])/255),
                opacity=alkyl_color[3], line_width=1)

        src_pep = mlab.pipeline.scalar_scatter(x, y, z)
        src_pep.mlab_source.dataset.lines = connections_pep
        src_pep.update()
        surf_pep = mlab.pipeline.surface(src_pep, color=tuple(np.array(pep_color[:3])/255),
                opacity=pep_color[3], line_width=1)


        box_color = [0,0,0,0.6]#[204,204,204, 0.7]
        mlab_box(box_color, box)
        mlab_drawaxes(box_color, box)
        mlab.view(azimuth=35, elevation=35, roll=0, distance=40, focalpoint=None)
        fig.scene.parallel_projection = True # orthogonal projection
    
        return surf_alkyl, surf_pep, src_alkyl, src_pep


    
       
    def update_frame(frame,surf_alkyl,surf_pep,src_alkyl,src_pep):
        
        connections = []
        connections_alkyl = []
        connections_pep = []
        for i in range(nmol):
            for bond in bonds_perPA:
                X1 = positions[ frame, bond[0] + num_atoms*i ]
                X2 = positions[ frame, bond[1] + num_atoms*i ]
                if not np.linalg.norm(X1-X2) > 0.5*min([statepoint['Lx'],statepoint['Ly'],statepoint['Lz']]):
                    connections = connections + [bond+num_atoms*i]
                    if set(bond) in bonds_alkyl:
                        connections_alkyl += [bond+num_atoms*i]
                    else:
                        connections_pep   += [bond+num_atoms*i]

        
        X = positions[frame]
        connections = np.array(connections)
        connections_alkyl = np.array(connections_alkyl)
        connections_pep = np.array(connections_pep)

        x=X[:,0]; y=X[:,1]; z=X[:,2]
        surf_alkyl.mlab_source.reset(x=x,y=y,z=z)
        src_alkyl.mlab_source.dataset.lines = connections_alkyl
        
        surf_pep.mlab_source.reset(x=x,y=y,z=z)
        src_pep.mlab_source.dataset.lines = connections_pep

    




        
    num_frames = positions.shape[0]
    frames = np.arange(num_frames)
    args = draw_frame(0)
    


    
    @mlab.animate(delay=100, ui=True)
    def anim():
        frame=0
        while True:
            update_frame(frame,*args)
            if frame >= num_frames-1:
                frame = 0
            else:
                frame += 1
            yield

    anim()
    mlab.show()






def interactive_scene(itpfile, topfile, grofile, trrfile, bonds_alkyl, box, draw_box=True):
    # itpnames is the list of names of different molecules that     

    bonds_permol =[]
    num_atoms    =[]
    nmol         =[]
    positions    =[]

    
    itpname = os.path.basename(itpfile).strip('.itp')

    bonds_permol = get_bonds_per_molecule(itpfile)
    num_atoms    = get_num_atoms(itpfile)
    nmol         = get_num_molecules(topfile, itpname)
    positions    = get_positions(grofile, trrfile, (0,num_atoms*nmol))
    
    
    Lx = box['Lx']
    Ly = box['Ly']
    Lz = box['Lz']    
    
    num_frames = positions.shape[0]  
    

    mol_color = [84,39,143,0.6] #[117,107,177, 0.6]
    light_green = [161,215,106]
    yellow = [255, 217, 0]#[255,255,191]
    yellow2 = [255,255,204]
    purple = [118,42,131]
    pink = [197,27,125]
    orange = [255,127,0]
    blue = [146,197,222]
    red = [202,0,32]
    mol_color = [84,39,143,0.6]
    alkyl_color = purple+[0.6]
    pep_color = light_green+[0.6]

    colors = [yellow, blue, red, light_green]

    

    def draw_frame(frame, fig):

        connections = []
        connections_alkyl = []
        connections_pep = []
        for i in range(nmol):
            for bond in bonds_permol:
                X1 = positions[ frame, bond[0] + num_atoms*i ]
                X2 = positions[ frame, bond[1] + num_atoms*i ]
                if not np.linalg.norm(X1-X2) > 0.5*min( [Lx,Ly,Lz] ):
                    connections = connections + [bond+num_atoms*i]
                    if set(bond) in bonds_alkyl:
                        connections_alkyl += [bond+num_atoms*i]
                    else:
                        connections_pep   += [bond+num_atoms*i]

        
        X = positions[frame]
        connections = np.array(connections)
        connections_alkyl = np.array(connections_alkyl)
        connections_pep = np.array(connections_pep)


        x=X[:,0]; y=X[:,1]; z=X[:,2]
        src_alkyl = mlab.pipeline.scalar_scatter(x, y, z)
        src_alkyl.mlab_source.dataset.lines = connections_alkyl
        src_alkyl.update()
        surf_alkyl = mlab.pipeline.surface(src_alkyl, color=tuple(np.array(alkyl_color[:3])/255),
                opacity=alkyl_color[3], line_width=1., figure=fig)

        src_pep = mlab.pipeline.scalar_scatter(x, y, z)
        src_pep.mlab_source.dataset.lines = connections_pep
        src_pep.update()
        surf_pep = mlab.pipeline.surface(src_pep, color=tuple(np.array(pep_color[:3])/255),
                opacity=pep_color[3], line_width=2, figure=fig)


        
        if draw_box:
            box_color = [0,0,0,0.6]#[204,204,204, 0.7]
            mlab_box(box_color, box)
            mlab_drawaxes(box_color, box)
        
        mlab.view(azimuth=35, elevation=35, roll=0, distance=40, focalpoint=None)
        fig.scene.parallel_projection = True # orthogonal projection
        
        return surf_alkyl, surf_pep, src_alkyl, src_pep



    def update_frame(frame,surf_alkyl,surf_pep,src_alkyl,src_pep):
        
        connections = []
        connections_alkyl = []
        connections_pep = []
        for i in range(nmol):
            for bond in bonds_permol:
                X1 = positions[ frame, bond[0] + num_atoms*i ]
                X2 = positions[ frame, bond[1] + num_atoms*i ]
                if not np.linalg.norm(X1-X2) > 0.5*min( [Lx,Ly,Lz] ):
                    connections = connections + [bond+num_atoms*i]
                    if set(bond) in bonds_alkyl:
                        connections_alkyl += [bond+num_atoms*i]
                    else:
                        connections_pep   += [bond+num_atoms*i]

        
        X = positions[frame]
        connections = np.array(connections)
        connections_alkyl = np.array(connections_alkyl)
        connections_pep = np.array(connections_pep)

        x=X[:,0]; y=X[:,1]; z=X[:,2]
        surf_alkyl.mlab_source.reset(x=x,y=y,z=z)
        src_alkyl.mlab_source.dataset.lines = connections_alkyl
        
        surf_pep.mlab_source.reset(x=x,y=y,z=z)
        src_pep.mlab_source.dataset.lines = connections_pep



    class MyModel(HasTraits):
        # slider = Range(-10, num_frames, .1)
        slider = Range(0, num_frames-1, 0, mode='slider')
        scene  = Instance(MlabSceneModel, (), width=10)

        def __init__(self):
            HasTraits.__init__(self)
            self.args = draw_frame(0,fig=self.scene.mayavi_scene)


        @on_trait_change('slider')
        def slider_changed(self):
            update_frame(self.slider, *self.args)
        

        view = View(Item('scene'),
                    Group("slider"), width=400,
                    resizable=True)

                    
    
    mlab.figure(bgcolor=(1, 1, 1), size=(400, 400))
    mlab.gcf().scene.parallel_projection = True # orthogonal projection
    
    my_model = MyModel()
    my_model.configure_traits()
    
    




def interactive_scene_co(itpfiles, topfile, grofile, trrfile):
    # itpnames is the list of names of different molecules that     

    bonds_permol =[]
    num_atoms    =[]
    nmol         =[]
    positions    =[]

    start=0
    for i,itpfile in enumerate(itpfiles):
        itpname = os.path.basename(itpfile).replace('.itp','')
        bonds_permol += [get_bonds_per_molecule(itpfile)]
        num_atoms    += [get_num_atoms(itpfile)]
        nmol         += [get_num_molecules(topfile, itpname)]
        positions    += [get_positions(grofile, trrfile, (start,start+num_atoms[i]*nmol[i]))]
        start = num_atoms[i]*nmol[i]
    
    
    
    num_frames = positions[0].shape[0]  
        

    mol_color = [84,39,143,0.6] #[117,107,177, 0.6]
    light_green = [161,215,106]
    yellow = [255,255,191]
    purple = [118,42,131]
    pink = [197,27,125]
    orange = [255,127,0]
    blue = [158,154,200]
    red = [217,95,14]
    mol_color = [84,39,143,0.6]
    alkyl_color = purple+[0.6]
    pep_color = light_green+[0.6]

    colors = [blue+[0.6], red+[0.6], light_green+[0.6]]

    src  = [[]]*len(itpfile)
    surf = [[]]*len(itpfile)
    def draw_frame(frame, fig):

        for k in range(len(itpfiles)):
            positions_ = positions[k]
            bonds_permol_ = bonds_permol[k]
            num_atoms_ = num_atoms[k]
            nmol_ = nmol[k]
            color_ = colors[k]

            connections = []
            connections_alkyl = []
            connections_pep = []
            for i in range(nmol_):
                for bond in bonds_permol_:
                    X1 = positions_[ frame, bond[0] + num_atoms_*i ]
                    X2 = positions_[ frame, bond[1] + num_atoms_*i ]
                    if not np.linalg.norm(X1-X2) > 0.5*min( [Lx,Ly,Lz] ):
                        connections = connections + [bond+num_atoms_*i]
                        # if set(bond) in bonds_alkyl:
                        #     connections_alkyl += [bond+num_atoms*i]
                        # else:
                        #     connections_pep   += [bond+num_atoms*i]

            
            X = positions_[frame]
            connections = np.array(connections)
            # connections_alkyl = np.array(connections_alkyl)
            # connections_pep = np.array(connections_pep)


            x=X[:,0]; y=X[:,1]; z=X[:,2]
            src[k] = mlab.pipeline.scalar_scatter(x, y, z)
            src[k].mlab_source.dataset.lines = connections
            src[k].update()
            surf[k] = mlab.pipeline.surface(src[k], color=tuple(np.array(color_[:3])/255),
                    opacity=color_[3], line_width=1.0, figure=fig)

        
        

        box_color = [0,0,0,0.6]#[204,204,204, 0.7]
        mlab_box(box_color)
        mlab_drawaxes(box_color)
        mlab.view(azimuth=35, elevation=35, roll=0, distance=40, focalpoint=None)
        fig.scene.parallel_projection = True # orthogonal projection
        
        return surf, src



    def update_frame(frame,surf,src):
    
        for k in range(len(itpfiles)):
            positions_ = positions[k]
            bonds_permol_ = bonds_permol[k]
            num_atoms_ = num_atoms[k]
            nmol_ = nmol[k]
            color_ = colors[k]

            connections = []
            connections_alkyl = []
            connections_pep = []
            for i in range(nmol_):
                for bond in bonds_permol_:
                    X1 = positions_[ frame, bond[0] + num_atoms_*i ]
                    X2 = positions_[ frame, bond[1] + num_atoms_*i ]
                    if not np.linalg.norm(X1-X2) > 0.5*min( [Lx,Ly,Lz] ):
                        connections = connections + [bond+num_atoms_*i]
                        # if set(bond) in bonds_alkyl:
                        #     connections_alkyl += [bond+num_atoms*i]
                        # else:
                        #     connections_pep   += [bond+num_atoms*i]


        
            X = positions_[frame]
            connections = np.array(connections)
            # connections_alkyl = np.array(connections_alkyl)
            # connections_pep = np.array(connections_pep)

            x=X[:,0]; y=X[:,1]; z=X[:,2]
            surf[k].mlab_source.reset(x=x,y=y,z=z)
            src[k].mlab_source.dataset.lines = connections
            
        



    class MyModel(HasTraits):
        # slider = Range(-10, num_frames, .1)
        slider = Range(0, num_frames-1, 0, mode='slider')
        scene  = Instance(MlabSceneModel, (), width=10)

        def __init__(self):
            HasTraits.__init__(self)
            self.args = draw_frame(0,fig=self.scene.mayavi_scene)


        @on_trait_change('slider')
        def slider_changed(self):
            update_frame(self.slider, *self.args)
        

        view = View(Item('scene'),
                    Group("slider"), width=400,
                    resizable=True)

                    
    
    mlab.figure(bgcolor=(1, 1, 1), size=(400, 400))
    mlab.gcf().scene.parallel_projection = True # orthogonal projection

    my_model = MyModel()
    my_model.configure_traits()






