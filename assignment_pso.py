import numpy as np
import matplotlib.pyplot as plt
'''
Author: Mohammed Al-Jaff
Date: 2021-09-21
'''


### 1D functions

def generate_an_objective_function(x_min, x_max):
    """
    Objective function generator for Particle swarm optimizaiton task.
    Functions are randomly generated based on a sum of 3 exponential functions and a sinusoidal function. 
    
    Returns:
        A randomly generated vectorized scaler function in 1d that is. 
    
    Params: 
    - x_min should be a numpy scaler
    - x_max should be a numpy scaler. X_min should be striclty grater than x_min
    
    """
    
    if (x_max <= x_min):
        raise Exception('x_min should be strcitly greater than x_max.')
    
    interval_dist = np.abs(x_max - x_min)
#    print(interval_dist)
    m1 = np.random.randint(low= x_min, high = x_min + np.random.uniform(low=0.01, high=0.4)*interval_dist)
    v1 = np.random.uniform(low=1, high=30)
    a1 = np.random.uniform(high=10)
    
    m2 = np.random.randint(low= x_min, high = x_min + np.random.uniform(low=0.4, high=0.7)*interval_dist)
    v2 = np.random.uniform(low=1, high=30)
    a2 = np.random.uniform(high=10)
    
    m3 = np.random.randint(low= x_min, high = x_min + np.random.uniform(low=0.6, high=1)*interval_dist)
    v3 = np.random.uniform(low=1, high=30)
    a3 = np.random.uniform(high=10)
    
    p1 = lambda x: ((x-m1)/(v1))**2
    p2 = lambda x: ((x-m2)/(v2))**2
    p3 = lambda x: ((x-m3)/(v3))**2
    
    c1 = np.random.randint(low=0, high=10)
    
    objective_function = lambda x: -1*(a1 * np.exp(-p1(x)/2) + a2*np.exp(-p2(x)/2) +  a3*np.exp(-p3(x)/2) ) + c1*np.sin(x)/(1+np.abs(x)) + np.sin(x)/(1+np.abs(x)) - 2*np.cos(0.01*x)
                                                                                                                
    return objective_function




###### 1D PSO VIZUALIZATION 

from matplotlib.animation import FuncAnimation, FFMpegWriter
from IPython.display import HTML

def generate_1D_PSO_animation(swarm_object, N_iterations=100, file_name='1d_pso_animation'):
    
    s = swarm_object
    
    fig, ax = plt.subplots()

    fig.set_figheight(7)
    fig.set_figwidth(10)
    print('Generating 1D PSO animation.....')
    def animate(i):
        
        print(i)
        # plotting objective function
        x = np.linspace(s.x_min, s.x_max, 1000)
        y = s.objective_function(x)

        ax.clear()
        ax.plot(x,y)
        ax.set_xlim([s.x_min, s.x_max])

        for particle in s.swarm:
            ax.scatter(particle.x, particle.fx, label=particle.id)
            ax.text(particle.x, particle.fx*(1 + 0.1), particle.id , fontsize=12)

        ax.plot(s.global_best_x, s.global_best_fx, marker='X',  markersize=20)
        title_text = 'c1: ' + str(s.c1) + ',     c2: ' + str(s.c2) + '    -    Step: ' + str(s.step_i)
        
        ax.set_title(title_text)
        ax.text(s.global_best_x, s.global_best_fx - 0.1, 'Global Best' , fontsize=12)

        s.step()


    ani = FuncAnimation(fig, animate, frames=N_iterations, interval=50)
    

    
    f = file_name+ ".mp4"
    writervideo = FFMpegWriter(fps=10) 
    ani.save(f, writer=writervideo)
    print('1D PSO animation done...')

    


def generate_2D_PSO_animation(swarm_object, N_iterations=100, file_name='1d_pso_animation'):
    
    s = swarm_object
    
    fig, ax = plt.subplots()

    fig.set_figheight(7)
    fig.set_figwidth(10)
    print('Generating 2D PSO animation.....')
    def animate(i):
        print(i)
        # plotting objective function
        
        ax.clear()
        x, y = np.meshgrid(np.linspace(s.x_min, s.x_max, 500), np.linspace(s.y_min, s.y_max, 500))
        
        z = s.objective_function(x,y)
        ax.set_xlim([s.x_min, s.x_max])
        ax.set_ylim([s.y_min, s.y_max])

        ax.contourf(x, y, z, np.linspace(z.min(), z.max(), 100), cmap='BrBG', alpha=0.8)
        ax.grid()
        
        ax.set_xlabel('x')
        ax.set_ylabel('y')
        
        title_text = 'c1: ' + str(s.c1) + ',     c2: ' + str(s.c2) + '    -    Step: ' + str(s.step_i)
       
        title_text += f'  -- Current Global Best: x: {np.round(s.global_best_x,3)}, y: {np.round(s.global_best_y,3)} {s.global_best_fx}'
        ax.set_title(title_text)

        ax.plot(s.global_best_x, s.global_best_y,   marker='o',  markersize=10, color='orange')
        ax.text(s.global_best_x, s.global_best_y - 0.1, 'Global Best' , fontsize=12)


        for particle in s.swarm:
            ax.scatter(particle.x, particle.y, label=particle.id, marker='X', linewidths=6, color='r',  cmap='BrBG', alpha=0.8)
            ax.text(particle.x, particle.y, particle.id , fontsize=20)

        s.step()


    ani = FuncAnimation(fig, animate, frames=N_iterations, interval=50)
    

    
    f = file_name+ ".mp4"
    writervideo = FFMpegWriter(fps=10) 
    ani.save(f, writer=writervideo)
    print('2D PSO animation done...')







## 2D objective functions


# Himmelblau's functions
# f(x,y)=(x^{2}+y-11)^{2}+(x+y^{2}-7)^{2}.\quad 
himmelblau = lambda x, y: np.power(np.power(x,2) + y - 11,2) + np.power(x + np.power(y,2)-7, 2)

noisy_himmelblau = lambda x, y: np.power(np.power(x,2) + y - 11,2) + np.power(x + np.power(y,2)-7, 2) + 7*np.random.randn(*np.array(x).shape) 
    
    


# Drop-Wave function
def dropwave(x, y):
    nom = 1 + np.cos(12*np.sqrt(x**2 + y**2))
    denom = 0.5*(x**2 + y**2) +2
    return -nom/denom

def noisy_dropwave(x, y):
    nom = 1 + np.cos(12*np.sqrt(x**2 + y**2))
    denom = 0.5*(x**2 + y**2) +2
    return -nom/denom + 0.05*np.random.randn(*np.array(x).shape) 


# Hölder function
def holders_table(x,y):
    p = np.abs(1 - np.sqrt(x**2 + y**2)/np.pi)
    exp = np.sin(x)*np.cos(y)*np.exp(p)
    return -np.abs(exp)

def noisy_holders_table(x,y): 
    p = np.abs(1 - np.sqrt(x**2 + y**2)/np.pi)
    exp = np.sin(x)*np.cos(y)*np.exp(p)
    return -np.abs(exp) +  0.5*np.random.randn(*np.array(x).shape) 
    
    
# Levi_13 functions
def levi_13(x,y):
    term1 = np.sin(3*np.pi*x)**2
    term2 = (1+np.sin(3* np.pi *y)**2)*((x-1)**2)
    term3 = np.power(y-1, 2)*(1 + np.power(np.sin(2* np.pi * y), 2))

    return term1 + term2 + term3

def noisy_levi_13(x,y):
    term1 = np.sin(3*np.pi*x)**2
    term2 = (1+np.sin(3* np.pi *y)**2)*((x-1)**2)
    term3 = np.power(y-1, 2)*(1 + np.power(np.sin(2* np.pi * y), 2))

    return term1 + term2 + term3 + x*np.random.randn(*np.array(x).shape) 


# Modified Easom function
def easom_maj(x,y):    
    p1 = (x-np.pi)**2 + (y-np.pi)**2
    p2 = (x+3*np.pi)**2 + (y+3*np.pi)**2
    p3 = (x-np.pi)**2 + (y+np.pi)**2
    p3 = (x+np.pi)**2 + (y-np.pi)**2
    return -np.cos(x)*np.cos(y)*np.exp(-p1) - 4*np.sin(x)*np.sin(y)*np.exp(-p2) - np.sin(x)*np.cos(y)*np.exp(-p3) 


def noisy_easom_maj(x,y):    
    p1 = (x-np.pi)**2 + (y-np.pi)**2
    p2 = (x+3*np.pi)**2 + (y+3*np.pi)**2
    p3 = (x-np.pi)**2 + (y+np.pi)**2
    p3 = (x+np.pi)**2 + (y-np.pi)**2
    return -np.cos(x)*np.cos(y)*np.exp(-p1) - 4*np.sin(x)*np.sin(y)*np.exp(-p2) - np.sin(x)*np.cos(y)*np.exp(-p3) + 0.02*np.random.randn(*np.array(x).shape) 



# michalewicz
def michalewicz(x,y):    
    m=10
    p1 = np.sin(x)*np.power(np.sin((1* x**2)/np.pi), 2*m)
    p2 = np.sin(y)*np.power(np.sin((2* y**2)/np.pi), 2*m)
    return -p1-p2 

def noisy_michalewicz(x,y):    
    m=10
    p1 = np.sin(x)*np.power(np.sin((1* x**2)/np.pi), 2*m)
    p2 = np.sin(y)*np.power(np.sin((2* y**2)/np.pi), 2*m)
    return -p1-p2 + 0.05*np.random.randn(*np.array(x).shape) 
