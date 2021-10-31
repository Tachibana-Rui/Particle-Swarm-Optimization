import numpy as np
from assignment_pso import *
import matplotlib.pyplot as plt

class PSO():
	def __init__(self, objective_function, N_particles, x_min, x_max, v_max, c1,c2,w=1):
		self.objective_function = objective_function
		self.nr_particles = N_particles
		self.x_min = x_min
		self.x_max = x_max
		self.v_max = v_max
		self.step_i = 0 # internal step counter
		self.c1 = c1
		self.c2 = c2
		self.w = w

		# Init N paritcles in swarm
		self.nr_particles = N_particles
		self.swarm = [] # init empty list
		for i in range(self.nr_particles):
			p = self.Particle(x_min, x_max, v_max, i, objective_function, c1, c2, w) # Particle class
			self.swarm.append(p) # append newly created particle to swarm

		self.global_best_x, self.global_best_fx = self.get_current_best_x_and_fx()


#	def get_swarm_info_array(self):
		# ...
#		return np.array(swarm_info_array,dtype=object)


	def get_current_best_x_and_fx(self):
		# return the global best position x_best
		# and the value of the objective function value at position x_best 

		swarm_fx=[]
		for p in self.swarm:
			swarm_fx.append(p.fx)
		swarm_fx = np.array(swarm_fx,dtype=object)
		best_p_index = np.argmin(swarm_fx)
		current_best_x = self.swarm[best_p_index].x
		current_best_fx = swarm_fx[best_p_index]
		return current_best_x, current_best_fx


	def step(self):
		# for all particles, take a step

		for p in self.swarm:
			p.step(self.global_best_x)

		current_best_x, current_best_fx = self.get_current_best_x_and_fx()

		if current_best_fx < self.global_best_fx:
			print(f"New Global Best has been achieved")
			print(f"- x: {current_best_x} f(x): {current_best_fx}")

			self.global_best_x = current_best_x
			self.global_best_fx = current_best_fx
		self.step_i +=1

	def visualize_swarm(self, figsize=[10,10]):
		base_function_x = np.linspace(self.x_min,self.x_max,10000)
		base_function_fx = list(map(self.objective_function,base_function_x))
		plt.figure(figsize = figsize)
		plt.plot(base_function_x,base_function_fx)
		plt.xlabel('x')
		plt.ylabel('y=f(x)')
		plt.title(f'step {self.step_i}')
		swarm_id,swarm_x,swarm_fx = [],[],[]
		for p in self.swarm:
		#get_particle_info_array() returns a tuple with
		#(id, x, f(x), v_next, x_personal_best, f(x_personal_best))
			swarm_id.append(p.get_particle_info_array()[0])
			swarm_x.append(p.get_particle_info_array()[1])
			swarm_fx.append(p.get_particle_info_array()[2])
		plt.scatter(swarm_x,swarm_fx)
		for i in swarm_id: # Draw id of each point
			plt.annotate(str(i), xy = (swarm_x[i], swarm_fx[i]), xytext = (swarm_x[i], swarm_fx[i]+0.1)) 
		gb_index = np.argmin(np.array(swarm_fx,dtype=object))
		plt.annotate('global best', xy = (swarm_x[gb_index], swarm_fx[gb_index]), xytext = (swarm_x[gb_index], swarm_fx[gb_index]-0.1))
		# plot a straight dash line of global best
		plt.plot(base_function_x,list(map(lambda a:swarm_fx[gb_index],base_function_x)),linestyle='dashed')
		plt.show()

	def plot_swarm_trajectory_graph(self, fig_width=15, fig_height=15):
		# p.trajectory looks like
		# [(1.1,1.0),
		#  (2.6,3.3),
		#  (4.8,5.7)...]
		plt.figure(figsize=[fig_width,fig_height])
		plt.xlabel('number of iterations')
		plt.ylabel('particle x position')
		plt.title('Particle trajectories}')
		for p in self.swarm:
			# get slices of p.trajectory, and plot them
			plt.plot(list(range(self.step_i+1)),[p.trajectory[i][0] for i in range(len(p.trajectory))])
		plt.show()

	def run(self, nr_steps):
		for s in range(nr_steps):
			self.step()
		return self.global_best_x, self.global_best_fx

	class Particle():
		def __init__(self, x_min, x_max, v_max, particle_id, objective_function, c1, c2, w):
			self.id = particle_id

			self.objective_function = objective_function
			self.c1 = c1
			self.c2 = c2
			self.w = w

			self.x_min = x_min
			self.x_max = x_max
			self.v_max = v_max

			self.x = (x_max-x_min)*np.random.random()+x_min
			self.fx = objective_function(self.x)

			self.v_next = (np.random.random()*2-1)*v_max #range:[-v_max,v_max)

			self.personal_best_x = self.x
			self.personal_best_fx = self.fx

			self.trajectory = [(self.x,self.fx)]

		def get_particle_info_array(self):
			# Return array with
			# [id, x, f(x), v_next, x_personal_best, f(x_personal_best)]

			return self.id,self.x,self.fx,self.personal_best_x,self.personal_best_fx


		def step(self,global_best_x):
			# update v_next
			r1 = np.random.uniform(size=1)
			r2 = np.random.uniform(size=1)
			# temporary v
			v_temp = self.w*self.v_next + self.c1*r1*(self.personal_best_x-self.x) + self.c2*r2*(global_best_x-self.x)
			if abs(v_temp) <= self.v_max: #should be less or equal
				self.v_next = v_temp 
			else:
				if v_temp >= 0:
					self.v_next = self.v_max
				else:
					self.v_next = -self.v_max

			# update x
			# when it jumps outside the search space, an intuitive way is to let self.x=x_max or self.x=x_mix
			# but this maybe will make a particle sticks to the boundary for several steps
			# A better way is to let the particle bounces off when it hits the boundary
			x_temp = self.x + self.v_next
			if x_temp <= self.x_min:
				self.v_next = -self.v_next
				self.x = self.x_min-(x_temp-self.x_min)
			elif x_temp >= self.x_max:
				self.v_next = -self.v_next
				self.x = self.x_max-(x_temp-self.x_max)
			else:
				self.x = x_temp

			# evaluate f(x)
			self.fx = self.objective_function(self.x)
			self.trajectory.append((self.x,self.fx))

			# update personal_best_x if case
			if self.fx > self.personal_best_fx:
				self.personal_best_x = self.x
				self.personal_best_fx = self.fx

if __name__ == '__main__':
#	Here is just for testing. You should run it in jupyter notebook.
#	PSO(objective_function, N_particles, x_min, x_max, v_max, c1,c2, w)


	obj_f = generate_an_objective_function(0,1000)
	my_pso = PSO(obj_f,10,0,1000,1,1,1,1)
#	x_pso, f_pso = my_pso.run(nr_steps = 1000)

#	my_pso.visualize_swarm()
	my_pso.plot_swarm_trajectory_graph()
	print(f'The coordinate of optimal point that have found so far is ({x_pso},{f_pso})')

	obj_f = generate_an_objective_function(-500,500)
	my_pso = PSO(obj_f,10,-500,500,1,1,1,1)
	x_pso, f_pso = my_pso.run(nr_steps = 1000)
	my_pso.plot_swarm_trajectory_graph()
	print(f'The coordinate of optimal point that have found so far is ({x_pso},{f_pso})')

#	generate_1D_PSO_animation(my_pso, N_iterations=200, file_name='1d_pso_animation')
