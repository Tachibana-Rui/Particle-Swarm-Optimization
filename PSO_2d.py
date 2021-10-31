import numpy as np
import assignment_pso as ap
import matplotlib.pyplot as plt


class PSO():
	def __init__(self, objective_function, N_particles, x_min, x_max, y_min, y_max, v_max, c1,c2,w=1):
		self.objective_function = objective_function
		self.nr_particles = N_particles
		self.x_min = x_min
		self.x_max = x_max
		self.y_min = y_min
		self.y_max = y_max

		self.v_max = v_max
		self.step_i = 0 # internal step counter
		self.c1 = c1
		self.c2 = c2
		self.w = w

		# used in autorun(), an internal step counter that records
		# the number of steps that the global optimal has not updated.
		self.stagnation_steps = 0 

		# Init N paritcles in swarm
		self.nr_particles = N_particles
		self.swarm = [] # init empty list
		for i in range(self.nr_particles):
			p = self.Particle(x_min, x_max, y_min, y_max, v_max, i, objective_function, c1, c2, w) # Particle class
			self.swarm.append(p) # append newly created particle to swarm

		self.global_best_x,self.global_best_y, self.global_best_fx = self.get_current_best_xy_and_fx()


#	def get_swarm_info_array(self):
		# ...
#		return np.array(swarm_info_array,dtype=object)


	def get_current_best_xy_and_fx(self):
		# return the global best position x_best
		# and the value of the objective function value at position x_best 

		swarm_fx=[]
		for p in self.swarm:
			swarm_fx.append(p.fx)
		swarm_fx = np.array(swarm_fx,dtype=object)
		best_p_index = np.argmin(swarm_fx)
		current_best_x = self.swarm[best_p_index].x
		current_best_y = self.swarm[best_p_index].y
		current_best_fx = swarm_fx[best_p_index]
		return current_best_x,current_best_y, current_best_fx


	def step(self):
		# for all particles, take a step

		for p in self.swarm:
			p.step(self.global_best_x,self.global_best_y)

		current_best_x,current_best_y, current_best_fx = self.get_current_best_xy_and_fx()

		if current_best_fx < self.global_best_fx:
			print(f"New Global Best has been achieved")
			print(f"- x,y: {current_best_x},{current_best_y} f(x): {current_best_fx}")

			self.global_best_x = current_best_x
			self.global_best_y = current_best_y
			self.global_best_fx = current_best_fx
			self.stagnation_steps = 0 #reset the counter when a new global_best found
		else:
			self.stagnation_steps += 1 
	
		self.step_i +=1



	def visualize_swarm(self, figsize=[10,10]):
		base_function_x = np.linspace(self.x_min,self.x_max,1000)
		base_function_y = np.linspace(self.y_min,self.y_max,1000)
#		coordinates = list(zip(base_function_x,base_function_y))
		X,Y = np.meshgrid(base_function_x,base_function_y)
		Z = self.objective_function(X,Y)


		plt.figure(figsize=figsize)
		plt.contourf(X,Y,Z,np.linspace(Z.min(), Z.max(), 100), cmap='BrBG', alpha=0.8)

		plt.xlabel('x')
		plt.ylabel('y')
		plt.title(f'step {self.step_i}')

		swarm_id,swarm_x,swarm_y,swarm_fx = [],[],[],[]
		for p in self.swarm:
		#get_particle_info_array() returns a tuple with
		#(id, x, f(x), v_next, x_personal_best, f(x_personal_best))
			swarm_id.append(p.get_particle_info_array()[0])
			swarm_x.append(p.get_particle_info_array()[1])
			swarm_y.append(p.get_particle_info_array()[2])
			swarm_fx.append(p.get_particle_info_array()[3])
		plt.scatter(swarm_x,swarm_y)

		for i in swarm_id: # Draw id of each point
			plt.annotate(str(i), xy = (swarm_x[i], swarm_y[i]), xytext = (swarm_x[i], swarm_y[i]+0.1)) 

		plt.plot(self.global_best_x,self.global_best_y,label='Global Best',marker='X',markersize=30,color='orange')
#		gb_index = np.argmin(np.array(swarm_fx,dtype=object))
#		plt.annotate('global best', xy = (swarm_x[gb_index], swarm_y[gb_index]), xytext = (swarm_x[gb_index], swarm_y[gb_index]-0.1))
		# plot a straight dash line of global best
#		plt.plot(base_function_x,list(map(lambda a:swarm_fx[gb_index],base_function_x)),linestyle='dashed')

		plt.show()
		

	def plot_swarm_trajectory_graph(self, fig_width=15, fig_height=15):
		fig, ax = plt.subplots()
		fig.set_figwidth(fig_width)
		fig.set_figheight(fig_height)

		for particle in self.swarm:
			trajectory_array = np.array(particle.trajectory, dtype=np.float)

			x = trajectory_array[:,0]
			y = trajectory_array[:,1]

			u = np.diff(x)
			v = np.diff(y)
			pos_x = x[:-1] + u/2
			pos_y = y[:-1] + v/2
			norm = np.sqrt(u**2+v**2)

			ax.plot(x,y, marker="o")
			ax.quiver(pos_x, pos_y, u/norm, v/norm, angles="xy", pivot="mid")

		# plot global best (x,y)
		ax.plot(self.global_best_x,self.global_best_y,
		label='Global Best',marker='X',markersize=30,color='orange')
		ax.text(self.global_best_x,
		self.global_best_y*(1-0.01),'Global Best',fontsize=30)

		plt.grid()
		plt.xlabel('x')
		plt.ylabel('y')
		plt.title(f"Global best found a x:{self.global_best_x}, y:{self.global_best_y} - f(x,y)= {self.global_best_fx}")
		plt.show()

	def run(self, nr_steps):
		for s in range(nr_steps):
			self.step()
		return self.global_best_x, self.global_best_y, self.global_best_fx

	# According to some research, the probability of the global best has not changed for 9 steps, is super rare.
	# So, we can use this as stopping criteria. When the stagnation_steps = 10, it returns the global best to the user.
	def autorun(self):
		max_iteration = 200000 
		for s in range(max_iteration):
			self.step()			
			if self.stagnation_steps >9: 
				break
		else:
			print('Maximial iteration reached!')
		return self.global_best_x, self.global_best_y, self.global_best_fx


	class Particle():
		def __init__(self, x_min, x_max, y_min, y_max,v_max, particle_id, objective_function, c1, c2, w):
			self.id = particle_id

			self.objective_function = objective_function
			self.c1 = c1
			self.c2 = c2
			self.w = w

			self.x_min = x_min
			self.x_max = x_max
			self.y_min = y_min
			self.y_max = y_max

			self.v_max = v_max

			self.x = (x_max-x_min)*np.random.random()+x_min
			self.y = (y_max-y_min)*np.random.random()+y_min
			self.fx = objective_function(self.x,self.y)

			self.v_next = [(np.random.random()*2-1)*v_max,(np.random.random()*2-1)*v_max] # speed vector, range [-v_max,v_max)

			self.personal_best_x = self.x
			self.personal_best_y = self.y
			self.personal_best_fx = self.fx

			self.trajectory = [(self.x,self.y)]

		def get_particle_info_array(self):
			return self.id,self.x,self.y,self.fx,self.personal_best_x,self.personal_best_y,self.personal_best_fx


		def step(self,global_best_x,global_best_y):
			# update v_next
			r1 = np.random.uniform(size=1)
			r2 = np.random.uniform(size=1)
			r3 = np.random.uniform(size=1)
			r4 = np.random.uniform(size=1)

			# calculate a temporary velocity of next step, then compare it with v_max
			
			v_temp_x = self.w*self.v_next[0] + self.c1*r1*(self.personal_best_x-self.x) + self.c2*r2*(global_best_x-self.x)
			v_temp_y = self.w*self.v_next[1] + self.c1*r3*(self.personal_best_y-self.y) + self.c2*r4*(global_best_y-self.y)
			if abs(v_temp_x) <= self.v_max: #should be less or equal
				self.v_next[0] = v_temp_x 
			else:
				if v_temp_x >= 0:
					self.v_next[0] = self.v_max
				else:
					self.v_next[0] = -self.v_max

			if abs(v_temp_y) <= self.v_max: #should be less or equal
				self.v_next[1] = v_temp_y 
			else:
				if v_temp_x >= 0:
					self.v_next[1] = self.v_max
				else:
					self.v_next[1] = -self.v_max

			# update x
			# when it jumps outside the search space, an intuitive way is to let self.x=x_max or self.x=x_mix
			# but this maybe will make a particle sticks to the boundary for several steps
			# A better way is to let the particle bounces off when it hits the boundary
			# Assume that particles and bounds are rigid, no energy loss in collision
			# Velocity changes only with its direction of corresponding dimension
			# Hard to explain self.x = self.x_min-(x_temp-self.x_min), 
			# but if you draw particles reflection on a piece of paper, you will understand it

			x_temp = self.x + self.v_next[0]
			if x_temp <= self.x_min:
				self.v_next[0] = -self.v_next[0]
				self.x = self.x_min-(x_temp-self.x_min)
			elif x_temp >= self.x_max:
				self.v_next[0] = -self.v_next[0]
				self.x = self.x_max-(x_temp-self.x_max)
			else:
				self.x = x_temp

			# update y, same as update x
			y_temp = self.y + self.v_next[1]
			if y_temp <= self.y_min:
				self.v_next[1] = -self.v_next[1]
				self.y = self.y_min-(y_temp-self.y_min)
			elif y_temp >= self.y_max:
				self.v_next[1] = -self.v_next[1]
				self.y = self.y_max-(y_temp-self.y_max)
			else:
				self.y = y_temp

			# evaluate f(x)
			self.fx = self.objective_function(self.x,self.y)
			self.trajectory.append((self.x,self.y)) # a trajectory list of tuples

			# update personal_best_x if case
			if self.fx > self.personal_best_fx:
				self.personal_best_x = self.x
				self.personal_best_y = self.y
				self.personal_best_fx = self.fx

if __name__ == '__main__':

#	my_pso = PSO(objective_function, N_particles, x_min, x_max, y_min, y_max, v_max, c1,c2,w=1)
	obj_f = ap.dropwave
	my_pso = PSO(obj_f,20,-5,5,-5,5,0.5,1,1,1)
#	x_pso, y_pso, f_pso= my_pso.run(nr_steps = 100)
	x_pso, y_pso, f_pso = my_pso.autorun()
#	my_pso.visualize_swarm()
#	my_pso.plot_swarm_trajectory_graph()
#	print(f'The coordinate of optimal point that have found so far is ({x_pso},{y_pso},{f_pso})')
	ap.generate_2D_PSO_animation(my_pso,  N_iterations=200,  file_name='2d_pso_animation')


#	x_pso, f_pso = my_pso.run(nr_steps = 1000)
#	my_pso.visualize_swarm()
	

