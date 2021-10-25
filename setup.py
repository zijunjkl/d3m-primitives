from setuptools import setup, find_packages

setup(
	name='rpi_d3m_primitives',  # This is the name of your PyPI-package.
	version='0.2.7',  # Update the version number for new releases
	author='Naiyu Yin, Yuru Wang, Zijun Cui, Qiang Ji',
	author_email='yinn2@rpi.edu',
	url='https://github.com/zijun-rpi/d3m-primitives.git',
	description='RPI primitives for D3M submission.',
	platforms=['Linux', 'MacOS'],
        keywords = 'd3m_primitive',
	entry_points = {
		'd3m.primitives': [
			'feature_selection.simultaneous_markov_blanket.AutoRPI = rpi_d3m_primitives.STMBplus_auto:STMBplus_auto',
            'feature_selection.joint_mutual_information.AutoRPI = rpi_d3m_primitives.JMIplus_auto:JMIplus_auto',
			'feature_selection.score_based_markov_blanket.RPI = rpi_d3m_primitives.S2TMBplus:S2TMBplus',
            'classification.tree_augmented_naive_bayes.BayesianInfRPI = rpi_d3m_primitives.TreeAugmentedNB_BayesianInf:TreeAugmentedNB_BayesianInf',
            #'classification.naive_bayes.BayesianInfRPI = rpi_d3m_primitives.NaiveBayes_BayesianInf:NaiveBayes_BayesianInf',
			],
	},
	install_requires=[
		'd3m', 'pgmpy', 'networkx', 'numpy', 'scipy', 'pandas', 'torch', 'pyparsing', 'statsmodels', 'tqdm', 'joblib'
	],
	packages=find_packages()
)
