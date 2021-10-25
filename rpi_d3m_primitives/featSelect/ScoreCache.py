class ScoreCache:
	def __init__(self):
		self.parent_cache = dict()
		self.child_cache = dict()
		self.joint_cache = dict()

	def print(self):
		print("____CACHE_____")
		print("parents->(parent_states,parent_states_counts)")
		print(self.parent_cache)
		print("child->child_states_counts")
		print(self.child_cache)
		print("(child,hashed(parents)->k2")
		print(self.joint_cache)