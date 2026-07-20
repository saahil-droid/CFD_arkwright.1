import numpy as np
from geometry import naca4_geometry



class Panel:
    def __init__(self, start_x, start_y, end_x, end_y):
        self.start_x, self.start_y = start_x, start_y
        self.end_x,   self.end_y   = end_x,   end_y

        # midpoint of the panel — where the boundary condition will be applied later
        self.midpoint_x = (self.start_x + self.end_x) / 2
        self.midpoint_y = (self.start_y + self.end_y) / 2

        # panel length 
        self.length = np.hypot(self.end_x - self.start_x, self.end_y - self.start_y)

        # angle the panel makes with the horizontal
        self.panel_angle = np.arctan2(self.end_y - self.start_y, self.end_x - self.start_x)

        # outward normal — perpendicular to the panel, pointing away from aerofoil
        #self.normal_x = np.cos(self.panel_angle + np.pi / 2)
        #self.normal_y = np.sin(self.panel_angle + np.pi / 2)

        # outward normal for clockwise airfoil ordering
        self.normal_x = np.sin(self.panel_angle)
        self.normal_y = -np.cos(self.panel_angle)


#now loops through it all
def make_panels(x_coords, y_coords):
    panels = []
    for i in range(len(x_coords) - 1):
        # create Panel from point i to point i+1
        panel = Panel(x_coords[i], y_coords[i], x_coords[i + 1], y_coords[i + 1])
        panels.append(panel)

    return panels


    

#test to see if it works - v3

if __name__ == "__main__":

    x_coords = [0, 1, 2, 3]
    y_coords = [0, 1, 1, 0]
    
    panels = make_panels(x_coords, y_coords) 

    for p in panels[:3]:
        print(f"midpoint ({p.midpoint_x:.3f}, {p.midpoint_y:.3f})  normal ({p.normal_x:.3f}, {p.normal_y:.3f})")

    
    

    airfoils = naca4_geometry("2412")

    x_coords = np.concatenate([airfoils['x_upper'], airfoils['x_lower'][::-1]])
    y_coords = np.concatenate([airfoils['y_upper'], airfoils['y_lower'][::-1]])
    #note that the ::-1 reverses the lower surface coordiates so that the panels go around the airfoil in a clockwise direction.
    #this is so that the normal vectors point out from the airfoil, which is important for the condition of the panel method -> this is seen in visualisation

    panels = make_panels(x_coords, y_coords)

    print("First 3 panels (if all good: should be upper surface, ny > 0):")
    for p in panels[:3]:
        print(f"  midpoint ({p.midpoint_x:.3f}, {p.midpoint_y:.3f})  normal ({p.normal_x:.3f}, {p.normal_y:.3f})")

    print("Last 3 panels (if all good: should be lower surface, ny < 0):")
    for p in panels[-3:]:
        print(f"  midpoint ({p.midpoint_x:.3f}, {p.midpoint_y:.3f})  normal ({p.normal_x:.3f}, {p.normal_y:.3f})")





    #visualisation of the panels. < --- this is just temporaty
    import matplotlib.pyplot as plt
    plt.figure()
    plt.plot(x_coords, y_coords, 'k-', label='Airfoil surface')
    for p in panels:
        plt.plot([p.start_x, p.end_x], [p.start_y, p.end_y], 'b-')  # panel line
        plt.arrow(p.midpoint_x, p.midpoint_y, 0.1 * p.normal_x, 0.1 * p.normal_y, head_width=0.05, head_length=0.1, fc='r', ec='r')  # normal vector
    plt.axis('equal')
    plt.title('Panels and Normals')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.legend()
    plt.grid()
    plt.show()      
