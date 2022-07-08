class EpsilonProcessor(object):
	def __init__(self, epsilon_initial=1, epsilon_minimum=0.05, epsilon_decay_rate=0.998):
		super(EpsilonProcessor, self).__init__()
		self.epsilon = epsilon_initial
		self.epsilon_minimum = epsilon_minimum
		self.epsilon_decay_rate = epsilon_decay_rate

	def get_epsilon(self):
		return self.epsilon

	def epsilon_decay(self):
		self.epsilon *= self.epsilon_decay_rate
		self.epsilon = max(self.epsilon, self.epsilon_minimum)
		return
'''
obj = EpsilonProcessor(epsilon_initial=1.0, epsilon_minimum=0.05, epsilon_decay_rate=0.999)
for i in range(2000):
	obj.epsilon_decay()
	epsilon = obj.get_epsilon()
	print(i)
	print(epsilon)
'''
