# CAREEN
Plugin for CloudCompare developed within the research project CAREEN for 3D point cloud processing in heritage constructions

FOLDERS STRUCTURE:

- assets: Pictures for "About" tab.

- configs: Yaml file for execute ".exe" files.

- geometric-based_methods: Scripts files and executables of geometric based methods
	- "analysis_of_arches.py"
	- "analysis_of_deflections.py"
	- "analysis_of_inclinations.py"
	- "analysis_of_vaults.py"
	- "geometrical_features.py" and the executable "jakteristics-0.6.0"

- main_module: Scripts files with functions imported to other scripts.
	- "main.py"
	- "main_gui.py"
	- "ransac.py"

- other_methods: Scripts files of other methods.
	- "anisotropic_denoising.py"
	- "potree_converter.py"
	- "voxelize.py"

- point_cloud_examples: ".bin" files with point cloud examples to train the different methods.

- pre-requisites: Requirements of libraries versions and CloudCompare-PythonPlugin install package.

- radiometric-based_methods: Scripts files of radiometic based methods
	- "color_conversion.py"
	- "stadistical_features.py"

- segmentation_methods: Scripts files and executables of segmentation methods
	- "deep_learning_segmentation.py" and the executable "point_transformer" 
	- "supervised_machine_learning.py" and the executables "optimal_flow-0.1.11", "scikit-learn-1.3.2" y "tpot-0.12.1"
	- "unsupervised_machine_learning.py" and the executable "scikit-learn-1.3.2"

