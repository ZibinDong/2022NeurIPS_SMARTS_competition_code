WARNING: Your kernel does not support swap limit capabilities or the cgroup is not mounted. Memory limited without swap.
Error: unknown flag: --version

  WARNING: Retrying (Retry(total=4, connect=None, read=None, redirect=None, status=None)) after connection broken by 'ConnectTimeoutError(<pip._vendor.urllib3.connection.HTTPSConnection object at 0x7fa381a6f9d0>, 'Connection to files.pythonhosted.org timed out. (connect timeout=15)')': /packages/a2/dc/53ce10a508df291c973293535536961fdc014de088cf0c9534165af880d5/yattag-1.14.0.tar.gz

[notice] A new release of pip available: 22.2.2 -> 22.3
[notice] To update, run: pip install --upgrade pip

[notice] A new release of pip available: 22.2.2 -> 22.3
[notice] To update, run: pip install --upgrade pip

[notice] A new release of pip available: 22.2.2 -> 22.3
[notice] To update, run: pip install --upgrade pip
ERROR: pip's dependency resolver does not currently take into account all the packages that are installed. This behaviour is the source of the following dependency conflicts.
tensorflow 2.4.0 requires absl-py~=0.10, but you have absl-py 1.1.0 which is incompatible.
tensorflow 2.4.0 requires grpcio~=1.32.0, but you have grpcio 1.47.0 which is incompatible.
tensorflow 2.4.0 requires numpy~=1.19.2, but you have numpy 1.23.1 which is incompatible.
tensorflow 2.4.0 requires six~=1.15.0, but you have six 1.16.0 which is incompatible.
tensorflow 2.4.0 requires tensorboard~=2.4, but you have tensorboard 2.2.0 which is incompatible.
tensorflow 2.4.0 requires typing-extensions~=3.7.4, but you have typing-extensions 4.3.0 which is incompatible.

[notice] A new release of pip available: 22.2.2 -> 22.3
[notice] To update, run: pip install --upgrade pip
/opt/.venv/lib/python3.8/site-packages/trimesh/curvature.py:12: DeprecationWarning: Please use `coo_matrix` from the `scipy.sparse` namespace, the `scipy.sparse.coo` namespace is deprecated.
  from scipy.sparse.coo import coo_matrix
/opt/.venv/lib/python3.8/site-packages/trimesh/curvature.py:12: DeprecationWarning: Please use `coo_matrix` from the `scipy.sparse` namespace, the `scipy.sparse.coo` namespace is deprecated.
  from scipy.sparse.coo import coo_matrix
/opt/.venv/lib/python3.8/site-packages/trimesh/curvature.py:12: DeprecationWarning: Please use `coo_matrix` from the `scipy.sparse` namespace, the `scipy.sparse.coo` namespace is deprecated.
  from scipy.sparse.coo import coo_matrix
/opt/.venv/lib/python3.8/site-packages/trimesh/curvature.py:12: DeprecationWarning: Please use `coo_matrix` from the `scipy.sparse` namespace, the `scipy.sparse.coo` namespace is deprecated.
  from scipy.sparse.coo import coo_matrix
Process SpawnProcess-2:
Traceback (most recent call last):
  File "/usr/lib/python3.8/multiprocessing/process.py", line 315, in _bootstrap
    self.run()
  File "/usr/lib/python3.8/multiprocessing/process.py", line 108, in run
    self._target(*self._args, **self._kwargs)
  File "/tmp/codalab/tmpiG5gNF/run/program/evaluate.py", line 349, in _worker
    observations = env.reset()
  File "/tmp/codalab/tmpiG5gNF/run/input/res/wrappers.py", line 442, in reset
    raw_obs = self.env.reset()
  File "/tmp/codalab/tmpiG5gNF/run/input/res/wrappers.py", line 69, in reset
    return super().reset()
  File "/opt/.venv/lib/python3.8/site-packages/gym/core.py", line 279, in reset
    return self.observation(observation)
  File "/tmp/codalab/tmpiG5gNF/run/input/res/wrappers.py", line 262, in observation
    EnvInfo_is_goal = self.cal_goal_lane(np_obs, raw_obs, lane_index, masked_all_lane_indeces).reshape(3, 3)
  File "/tmp/codalab/tmpiG5gNF/run/input/res/wrappers.py", line 92, in cal_goal_lane
    b = a[3]
IndexError: index 3 is out of bounds for axis 0 with size 1
ERROR:/tmp/codalab/tmpiG5gNF/run/program/evaluate.py:Evaluation failed due to error. Attempting retry: evaluate.<locals>.ProcessContext(process_builder=functools.partial(<function evaluate.<locals>.process_builder_func at 0x7f1168b7e8b0>, env_ctor=functools.partial(<function _make_env at 0x7f1319b38a60>, env_type='bubble_env_contrib:bubble_env-v0', scenario=None, shared_configs={'action_space': 'TargetPose', 'img_meters': 64, 'img_pixels': 256, 'sumo_headless': True}, seed=49, wrapper_ctors=<function submitted_wrappers at 0x7f1317b7c430>), policy_type=<class 'policy.Policy'>), env_name='bubble_env_6', retries=1, env_retries=10, last_reply=5625596.722337075)
/opt/.venv/lib/python3.8/site-packages/trimesh/curvature.py:12: DeprecationWarning: Please use `coo_matrix` from the `scipy.sparse` namespace, the `scipy.sparse.coo` namespace is deprecated.
  from scipy.sparse.coo import coo_matrix
/opt/.venv/lib/python3.8/site-packages/trimesh/curvature.py:12: DeprecationWarning: Please use `coo_matrix` from the `scipy.sparse` namespace, the `scipy.sparse.coo` namespace is deprecated.
  from scipy.sparse.coo import coo_matrix
/opt/.venv/lib/python3.8/site-packages/trimesh/curvature.py:12: DeprecationWarning: Please use `coo_matrix` from the `scipy.sparse` namespace, the `scipy.sparse.coo` namespace is deprecated.
  from scipy.sparse.coo import coo_matrix
/opt/.venv/lib/python3.8/site-packages/trimesh/curvature.py:12: DeprecationWarning: Please use `coo_matrix` from the `scipy.sparse` namespace, the `scipy.sparse.coo` namespace is deprecated.
  from scipy.sparse.coo import coo_matrix
/opt/.venv/lib/python3.8/site-packages/trimesh/curvature.py:12: DeprecationWarning: Please use `coo_matrix` from the `scipy.sparse` namespace, the `scipy.sparse.coo` namespace is deprecated.
  from scipy.sparse.coo import coo_matrix
/opt/.venv/lib/python3.8/site-packages/trimesh/curvature.py:12: DeprecationWarning: Please use `coo_matrix` from the `scipy.sparse` namespace, the `scipy.sparse.coo` namespace is deprecated.
  from scipy.sparse.coo import coo_matrix
/opt/.venv/lib/python3.8/site-packages/trimesh/curvature.py:12: DeprecationWarning: Please use `coo_matrix` from the `scipy.sparse` namespace, the `scipy.sparse.coo` namespace is deprecated.
  from scipy.sparse.coo import coo_matrix
/opt/.venv/lib/python3.8/site-packages/trimesh/curvature.py:12: DeprecationWarning: Please use `coo_matrix` from the `scipy.sparse` namespace, the `scipy.sparse.coo` namespace is deprecated.
  from scipy.sparse.coo import coo_matrix
/opt/.venv/lib/python3.8/site-packages/trimesh/curvature.py:12: DeprecationWarning: Please use `coo_matrix` from the `scipy.sparse` namespace, the `scipy.sparse.coo` namespace is deprecated.
  from scipy.sparse.coo import coo_matrix
/opt/.venv/lib/python3.8/site-packages/trimesh/curvature.py:12: DeprecationWarning: Please use `coo_matrix` from the `scipy.sparse` namespace, the `scipy.sparse.coo` namespace is deprecated.
  from scipy.sparse.coo import coo_matrix
/opt/.venv/lib/python3.8/site-packages/trimesh/curvature.py:12: DeprecationWarning: Please use `coo_matrix` from the `scipy.sparse` namespace, the `scipy.sparse.coo` namespace is deprecated.
  from scipy.sparse.coo import coo_matrix
/opt/.venv/lib/python3.8/site-packages/trimesh/curvature.py:12: DeprecationWarning: Please use `coo_matrix` from the `scipy.sparse` namespace, the `scipy.sparse.coo` namespace is deprecated.
  from scipy.sparse.coo import coo_matrix
/opt/.venv/lib/python3.8/site-packages/trimesh/curvature.py:12: DeprecationWarning: Please use `coo_matrix` from the `scipy.sparse` namespace, the `scipy.sparse.coo` namespace is deprecated.
  from scipy.sparse.coo import coo_matrix
/opt/.venv/lib/python3.8/site-packages/trimesh/curvature.py:12: DeprecationWarning: Please use `coo_matrix` from the `scipy.sparse` namespace, the `scipy.sparse.coo` namespace is deprecated.
  from scipy.sparse.coo import coo_matrix
/opt/.venv/lib/python3.8/site-packages/trimesh/curvature.py:12: DeprecationWarning: Please use `coo_matrix` from the `scipy.sparse` namespace, the `scipy.sparse.coo` namespace is deprecated.
  from scipy.sparse.coo import coo_matrix
/opt/.venv/lib/python3.8/site-packages/trimesh/curvature.py:12: DeprecationWarning: Please use `coo_matrix` from the `scipy.sparse` namespace, the `scipy.sparse.coo` namespace is deprecated.
  from scipy.sparse.coo import coo_matrix
/opt/.venv/lib/python3.8/site-packages/trimesh/curvature.py:12: DeprecationWarning: Please use `coo_matrix` from the `scipy.sparse` namespace, the `scipy.sparse.coo` namespace is deprecated.
  from scipy.sparse.coo import coo_matrix
/opt/.venv/lib/python3.8/site-packages/trimesh/curvature.py:12: DeprecationWarning: Please use `coo_matrix` from the `scipy.sparse` namespace, the `scipy.sparse.coo` namespace is deprecated.
  from scipy.sparse.coo import coo_matrix
/opt/.venv/lib/python3.8/site-packages/trimesh/curvature.py:12: DeprecationWarning: Please use `coo_matrix` from the `scipy.sparse` namespace, the `scipy.sparse.coo` namespace is deprecated.
  from scipy.sparse.coo import coo_matrix
/opt/.venv/lib/python3.8/site-packages/trimesh/curvature.py:12: DeprecationWarning: Please use `coo_matrix` from the `scipy.sparse` namespace, the `scipy.sparse.coo` namespace is deprecated.
  from scipy.sparse.coo import coo_matrix
/opt/.venv/lib/python3.8/site-packages/trimesh/curvature.py:12: DeprecationWarning: Please use `coo_matrix` from the `scipy.sparse` namespace, the `scipy.sparse.coo` namespace is deprecated.
  from scipy.sparse.coo import coo_matrix
/opt/.venv/lib/python3.8/site-packages/trimesh/curvature.py:12: DeprecationWarning: Please use `coo_matrix` from the `scipy.sparse` namespace, the `scipy.sparse.coo` namespace is deprecated.
  from scipy.sparse.coo import coo_matrix
/opt/.venv/lib/python3.8/site-packages/trimesh/curvature.py:12: DeprecationWarning: Please use `coo_matrix` from the `scipy.sparse` namespace, the `scipy.sparse.coo` namespace is deprecated.
  from scipy.sparse.coo import coo_matrix
/opt/.venv/lib/python3.8/site-packages/trimesh/curvature.py:12: DeprecationWarning: Please use `coo_matrix` from the `scipy.sparse` namespace, the `scipy.sparse.coo` namespace is deprecated.
  from scipy.sparse.coo import coo_matrix
/opt/.venv/lib/python3.8/site-packages/trimesh/curvature.py:12: DeprecationWarning: Please use `coo_matrix` from the `scipy.sparse` namespace, the `scipy.sparse.coo` namespace is deprecated.
  from scipy.sparse.coo import coo_matrix
/opt/.venv/lib/python3.8/site-packages/trimesh/curvature.py:12: DeprecationWarning: Please use `coo_matrix` from the `scipy.sparse` namespace, the `scipy.sparse.coo` namespace is deprecated.
  from scipy.sparse.coo import coo_matrix
/opt/.venv/lib/python3.8/site-packages/trimesh/curvature.py:12: DeprecationWarning: Please use `coo_matrix` from the `scipy.sparse` namespace, the `scipy.sparse.coo` namespace is deprecated.
  from scipy.sparse.coo import coo_matrix
/opt/.venv/lib/python3.8/site-packages/trimesh/curvature.py:12: DeprecationWarning: Please use `coo_matrix` from the `scipy.sparse` namespace, the `scipy.sparse.coo` namespace is deprecated.
  from scipy.sparse.coo import coo_matrix
/opt/.venv/lib/python3.8/site-packages/trimesh/curvature.py:12: DeprecationWarning: Please use `coo_matrix` from the `scipy.sparse` namespace, the `scipy.sparse.coo` namespace is deprecated.
  from scipy.sparse.coo import coo_matrix
/opt/.venv/lib/python3.8/site-packages/trimesh/curvature.py:12: DeprecationWarning: Please use `coo_matrix` from the `scipy.sparse` namespace, the `scipy.sparse.coo` namespace is deprecated.
  from scipy.sparse.coo import coo_matrix
/opt/.venv/lib/python3.8/site-packages/trimesh/curvature.py:12: DeprecationWarning: Please use `coo_matrix` from the `scipy.sparse` namespace, the `scipy.sparse.coo` namespace is deprecated.
  from scipy.sparse.coo import coo_matrix
/opt/.venv/lib/python3.8/site-packages/trimesh/curvature.py:12: DeprecationWarning: Please use `coo_matrix` from the `scipy.sparse` namespace, the `scipy.sparse.coo` namespace is deprecated.
  from scipy.sparse.coo import coo_matrix
/opt/.venv/lib/python3.8/site-packages/trimesh/curvature.py:12: DeprecationWarning: Please use `coo_matrix` from the `scipy.sparse` namespace, the `scipy.sparse.coo` namespace is deprecated.
  from scipy.sparse.coo import coo_matrix
Warning: Shape for junction 'gneJ11' has distance 27.04 to its given position.
/opt/.venv/lib/python3.8/site-packages/trimesh/curvature.py:12: DeprecationWarning: Please use `coo_matrix` from the `scipy.sparse` namespace, the `scipy.sparse.coo` namespace is deprecated.
  from scipy.sparse.coo import coo_matrix
/opt/.venv/lib/python3.8/site-packages/trimesh/curvature.py:12: DeprecationWarning: Please use `coo_matrix` from the `scipy.sparse` namespace, the `scipy.sparse.coo` namespace is deprecated.
  from scipy.sparse.coo import coo_matrix
/opt/.venv/lib/python3.8/site-packages/trimesh/curvature.py:12: DeprecationWarning: Please use `coo_matrix` from the `scipy.sparse` namespace, the `scipy.sparse.coo` namespace is deprecated.
  from scipy.sparse.coo import coo_matrix
/opt/.venv/lib/python3.8/site-packages/trimesh/curvature.py:12: DeprecationWarning: Please use `coo_matrix` from the `scipy.sparse` namespace, the `scipy.sparse.coo` namespace is deprecated.
  from scipy.sparse.coo import coo_matrix
WARNING:SMARTS:Attempted to perform actions on non-existing agent, Agent_1 
WARNING:SMARTS:Attempted to perform actions on non-existing agent, Agent_1 
WARNING:SMARTS:Attempted to perform actions on non-existing agent, Agent_1 
WARNING:SMARTS:Attempted to perform actions on non-existing agent, Agent_1 
WARNING:SMARTS:Attempted to perform actions on non-existing agent, Agent_1 
WARNING:SMARTS:Attempted to perform actions on non-existing agent, Agent_1 
WARNING:SMARTS:Attempted to perform actions on non-existing agent, Agent_1 
WARNING:SMARTS:Attempted to perform actions on non-existing agent, Agent_1 
WARNING:SMARTS:Attempted to perform actions on non-existing agent, Agent_1 
WARNING:SMARTS:Attempted to perform actions on non-existing agent, Agent_1 
WARNING:SMARTS:Attempted to perform actions on non-existing agent, Agent_1 
WARNING:SMARTS:Attempted to perform actions on non-existing agent, Agent_1 
WARNING:SMARTS:Attempted to perform actions on non-existing agent, Agent_1 
WARNING:SMARTS:Attempted to perform actions on non-existing agent, Agent_1 
WARNING:SMARTS:Attempted to perform actions on non-existing agent, Agent_1 
WARNING:SMARTS:Attempted to perform actions on non-existing agent, Agent_1 
WARNING:SMARTS:Attempted to perform actions on non-existing agent, Agent_1 
WARNING:SMARTS:Attempted to perform actions on non-existing agent, Agent_1 
WARNING:SMARTS:Attempted to perform actions on non-existing agent, Agent_1 
/opt/.venv/lib/python3.8/site-packages/trimesh/curvature.py:12: DeprecationWarning: Please use `coo_matrix` from the `scipy.sparse` namespace, the `scipy.sparse.coo` namespace is deprecated.
  from scipy.sparse.coo import coo_matrix
WARNING:SMARTS:Attempted to perform actions on non-existing agent, Agent_0 
/opt/.venv/lib/python3.8/site-packages/trimesh/curvature.py:12: DeprecationWarning: Please use `coo_matrix` from the `scipy.sparse` namespace, the `scipy.sparse.coo` namespace is deprecated.
  from scipy.sparse.coo import coo_matrix
/opt/.venv/lib/python3.8/site-packages/trimesh/curvature.py:12: DeprecationWarning: Please use `coo_matrix` from the `scipy.sparse` namespace, the `scipy.sparse.coo` namespace is deprecated.
  from scipy.sparse.coo import coo_matrix
Warning: Shape for junction 'gneJ11' has distance 20.57 to its given position.
/opt/.venv/lib/python3.8/site-packages/trimesh/curvature.py:12: DeprecationWarning: Please use `coo_matrix` from the `scipy.sparse` namespace, the `scipy.sparse.coo` namespace is deprecated.
  from scipy.sparse.coo import coo_matrix
ERROR:root:TraCI has disconnected with: connection closed by SUMO
WARNING:SumoTrafficSimulation:attempting to transfer SUMO vehicles to other providers...
WARNING:SMARTS:Attempted to perform actions on non-existing agent, Agent_1 
/usr/lib/python3.8/subprocess.py:946: ResourceWarning: subprocess 2929819 is still running
  _warn("subprocess %s is still running" % self.pid,
ResourceWarning: Enable tracemalloc to get the object allocation traceback
ERROR:root:TraCI has disconnected with: connection closed by SUMO
WARNING:SumoTrafficSimulation:attempting to transfer SUMO vehicles to other providers...
WARNING:SMARTS:Attempted to perform actions on non-existing agent, Agent_1 
/usr/lib/python3.8/subprocess.py:946: ResourceWarning: subprocess 3033018 is still running
  _warn("subprocess %s is still running" % self.pid,
ResourceWarning: Enable tracemalloc to get the object allocation traceback
/opt/.venv/lib/python3.8/site-packages/trimesh/curvature.py:12: DeprecationWarning: Please use `coo_matrix` from the `scipy.sparse` namespace, the `scipy.sparse.coo` namespace is deprecated.
  from scipy.sparse.coo import coo_matrix
/opt/.venv/lib/python3.8/site-packages/trimesh/curvature.py:12: DeprecationWarning: Please use `coo_matrix` from the `scipy.sparse` namespace, the `scipy.sparse.coo` namespace is deprecated.
  from scipy.sparse.coo import coo_matrix
WARNING:SMARTS:Attempted to perform actions on non-existing agent, Agent_1 
WARNING:SMARTS:Attempted to perform actions on non-existing agent, Agent_1 
WARNING:SMARTS:Attempted to perform actions on non-existing agent, Agent_1 
WARNING:SMARTS:Attempted to perform actions on non-existing agent, Agent_1 
WARNING:SMARTS:Attempted to perform actions on non-existing agent, Agent_1 
WARNING:SMARTS:Attempted to perform actions on non-existing agent, Agent_0 
WARNING:SMARTS:Attempted to perform actions on non-existing agent, Agent_1 
WARNING:SMARTS:Attempted to perform actions on non-existing agent, Agent_1 
WARNING:SMARTS:Attempted to perform actions on non-existing agent, Agent_1 
WARNING:SMARTS:Attempted to perform actions on non-existing agent, Agent_1 
WARNING:SMARTS:Attempted to perform actions on non-existing agent, Agent_1 
WARNING:SMARTS:Attempted to perform actions on non-existing agent, Agent_1 
WARNING:SMARTS:Attempted to perform actions on non-existing agent, Agent_1 
/opt/.venv/lib/python3.8/site-packages/trimesh/curvature.py:12: DeprecationWarning: Please use `coo_matrix` from the `scipy.sparse` namespace, the `scipy.sparse.coo` namespace is deprecated.
  from scipy.sparse.coo import coo_matrix
/opt/.venv/lib/python3.8/site-packages/trimesh/curvature.py:12: DeprecationWarning: Please use `coo_matrix` from the `scipy.sparse` namespace, the `scipy.sparse.coo` namespace is deprecated.
  from scipy.sparse.coo import coo_matrix
WARNING:SMARTS:Attempted to perform actions on non-existing agent, Agent_1 
WARNING:SMARTS:Attempted to perform actions on non-existing agent, Agent_1 
WARNING:SMARTS:Attempted to perform actions on non-existing agent, Agent_1 
/opt/.venv/lib/python3.8/site-packages/trimesh/curvature.py:12: DeprecationWarning: Please use `coo_matrix` from the `scipy.sparse` namespace, the `scipy.sparse.coo` namespace is deprecated.
  from scipy.sparse.coo import coo_matrix
WARNING:SMARTS:Attempted to perform actions on non-existing agent, Agent_1 
/opt/.venv/lib/python3.8/site-packages/trimesh/curvature.py:12: DeprecationWarning: Please use `coo_matrix` from the `scipy.sparse` namespace, the `scipy.sparse.coo` namespace is deprecated.
  from scipy.sparse.coo import coo_matrix
/opt/.venv/lib/python3.8/site-packages/trimesh/curvature.py:12: DeprecationWarning: Please use `coo_matrix` from the `scipy.sparse` namespace, the `scipy.sparse.coo` namespace is deprecated.
  from scipy.sparse.coo import coo_matrix
/opt/.venv/lib/python3.8/site-packages/trimesh/curvature.py:12: DeprecationWarning: Please use `coo_matrix` from the `scipy.sparse` namespace, the `scipy.sparse.coo` namespace is deprecated.
  from scipy.sparse.coo import coo_matrix
Warning: Shape for junction 'gneJ11' has distance 27.04 to its given position.
WARNING:SMARTS:Attempted to perform actions on non-existing agent, Agent_1 
WARNING:SMARTS:Attempted to perform actions on non-existing agent, Agent_0 
WARNING:SMARTS:Attempted to perform actions on non-existing agent, Agent_0 
WARNING:SMARTS:Attempted to perform actions on non-existing agent, Agent_1 
WARNING:SMARTS:Attempted to perform actions on non-existing agent, Agent_1 
/opt/.venv/lib/python3.8/site-packages/trimesh/curvature.py:12: DeprecationWarning: Please use `coo_matrix` from the `scipy.sparse` namespace, the `scipy.sparse.coo` namespace is deprecated.
  from scipy.sparse.coo import coo_matrix
/opt/.venv/lib/python3.8/site-packages/trimesh/curvature.py:12: DeprecationWarning: Please use `coo_matrix` from the `scipy.sparse` namespace, the `scipy.sparse.coo` namespace is deprecated.
  from scipy.sparse.coo import coo_matrix
WARNING:SMARTS:Attempted to perform actions on non-existing agent, Agent_1 
WARNING:SMARTS:Attempted to perform actions on non-existing agent, Agent_1 
WARNING:SMARTS:Attempted to perform actions on non-existing agent, Agent_0 
WARNING:SMARTS:Attempted to perform actions on non-existing agent, Agent_1 
WARNING:SMARTS:Attempted to perform actions on non-existing agent, Agent_1 
/opt/.venv/lib/python3.8/site-packages/trimesh/curvature.py:12: DeprecationWarning: Please use `coo_matrix` from the `scipy.sparse` namespace, the `scipy.sparse.coo` namespace is deprecated.
  from scipy.sparse.coo import coo_matrix
/opt/.venv/lib/python3.8/site-packages/trimesh/curvature.py:12: DeprecationWarning: Please use `coo_matrix` from the `scipy.sparse` namespace, the `scipy.sparse.coo` namespace is deprecated.
  from scipy.sparse.coo import coo_matrix
Warning: Shape for junction 'gneJ11' has distance 20.57 to its given position.
WARNING:SMARTS:Attempted to perform actions on non-existing agent, Agent_1 
WARNING:SMARTS:Attempted to perform actions on non-existing agent, Agent_1 
WARNING:SMARTS:Attempted to perform actions on non-existing agent, Agent_1 
WARNING:SMARTS:Attempted to perform actions on non-existing agent, Agent_1 
WARNING:SMARTS:Attempted to perform actions on non-existing agent, Agent_1 
WARNING:SMARTS:Attempted to perform actions on non-existing agent, Agent_0 
WARNING:SMARTS:Attempted to perform actions on non-existing agent, Agent_1 
WARNING:SMARTS:Attempted to perform actions on non-existing agent, Agent_1 
WARNING:SMARTS:Attempted to perform actions on non-existing agent, Agent_1 
WARNING:SMARTS:Attempted to perform actions on non-existing agent, Agent_1 
WARNING:SMARTS:Attempted to perform actions on non-existing agent, Agent_1 
WARNING:SMARTS:Attempted to perform actions on non-existing agent, Agent_1 
/opt/.venv/lib/python3.8/site-packages/trimesh/curvature.py:12: DeprecationWarning: Please use `coo_matrix` from the `scipy.sparse` namespace, the `scipy.sparse.coo` namespace is deprecated.
  from scipy.sparse.coo import coo_matrix
/opt/.venv/lib/python3.8/site-packages/trimesh/curvature.py:12: DeprecationWarning: Please use `coo_matrix` from the `scipy.sparse` namespace, the `scipy.sparse.coo` namespace is deprecated.
  from scipy.sparse.coo import coo_matrix
WARNING:SMARTS:Attempted to perform actions on non-existing agent, Agent_1 
WARNING:SMARTS:Attempted to perform actions on non-existing agent, Agent_1 
WARNING:SMARTS:Attempted to perform actions on non-existing agent, Agent_1 
/opt/.venv/lib/python3.8/site-packages/trimesh/curvature.py:12: DeprecationWarning: Please use `coo_matrix` from the `scipy.sparse` namespace, the `scipy.sparse.coo` namespace is deprecated.
  from scipy.sparse.coo import coo_matrix
WARNING:SMARTS:Attempted to perform actions on non-existing agent, Agent_1 
/opt/.venv/lib/python3.8/site-packages/trimesh/curvature.py:12: DeprecationWarning: Please use `coo_matrix` from the `scipy.sparse` namespace, the `scipy.sparse.coo` namespace is deprecated.
  from scipy.sparse.coo import coo_matrix
WARNING:SMARTS:Attempted to perform actions on non-existing agent, Agent_0 
WARNING:SMARTS:Attempted to perform actions on non-existing agent, Agent_1 
WARNING:SMARTS:Attempted to perform actions on non-existing agent, Agent_0 
WARNING:SMARTS:Attempted to perform actions on non-existing agent, Agent_1 
WARNING:SMARTS:Attempted to perform actions on non-existing agent, Agent_1 
WARNING:SMARTS:Attempted to perform actions on non-existing agent, Agent_1 
WARNING:SMARTS:Attempted to perform actions on non-existing agent, Agent_0 
WARNING:SMARTS:Attempted to perform actions on non-existing agent, Agent_0 
WARNING:SMARTS:Attempted to perform actions on non-existing agent, Agent_1 
WARNING:SMARTS:Attempted to perform actions on non-existing agent, Agent_1 
WARNING:SMARTS:Attempted to perform actions on non-existing agent, Agent_1 
WARNING:SMARTS:Attempted to perform actions on non-existing agent, Agent_1 
WARNING:SMARTS:Attempted to perform actions on non-existing agent, Agent_0 
WARNING:SMARTS:Attempted to perform actions on non-existing agent, Agent_1 
/opt/.venv/lib/python3.8/site-packages/trimesh/curvature.py:12: DeprecationWarning: Please use `coo_matrix` from the `scipy.sparse` namespace, the `scipy.sparse.coo` namespace is deprecated.
  from scipy.sparse.coo import coo_matrix
WARNING:SMARTS:Attempted to perform actions on non-existing agent, Agent_1 
/opt/.venv/lib/python3.8/site-packages/trimesh/curvature.py:12: DeprecationWarning: Please use `coo_matrix` from the `scipy.sparse` namespace, the `scipy.sparse.coo` namespace is deprecated.
  from scipy.sparse.coo import coo_matrix
/opt/.venv/lib/python3.8/site-packages/trimesh/curvature.py:12: DeprecationWarning: Please use `coo_matrix` from the `scipy.sparse` namespace, the `scipy.sparse.coo` namespace is deprecated.
  from scipy.sparse.coo import coo_matrix
WARNING:SMARTS:Attempted to perform actions on non-existing agent, Agent_0 
/opt/.venv/lib/python3.8/site-packages/trimesh/curvature.py:12: DeprecationWarning: Please use `coo_matrix` from the `scipy.sparse` namespace, the `scipy.sparse.coo` namespace is deprecated.
  from scipy.sparse.coo import coo_matrix
WARNING:SMARTS:Attempted to perform actions on non-existing agent, Agent_0 
WARNING:SMARTS:Attempted to perform actions on non-existing agent, Agent_0 
WARNING:SMARTS:Attempted to perform actions on non-existing agent, Agent_1 
WARNING:SMARTS:Attempted to perform actions on non-existing agent, Agent_1 
WARNING:SMARTS:Attempted to perform actions on non-existing agent, Agent_0 
WARNING:SMARTS:Attempted to perform actions on non-existing agent, Agent_1 
WARNING:SMARTS:Attempted to perform actions on non-existing agent, Agent_0 
ERROR:root:TraCI has disconnected with: connection closed by SUMO
WARNING:SumoTrafficSimulation:attempting to transfer SUMO vehicles to other providers...
WARNING:SMARTS:Attempted to perform actions on non-existing agent, Agent_0 
/usr/lib/python3.8/subprocess.py:946: ResourceWarning: subprocess 211226 is still running
  _warn("subprocess %s is still running" % self.pid,
ResourceWarning: Enable tracemalloc to get the object allocation traceback
WARNING:SMARTS:Attempted to perform actions on non-existing agent, Agent_1 
WARNING:SMARTS:Attempted to perform actions on non-existing agent, Agent_0 
WARNING:SMARTS:Attempted to perform actions on non-existing agent, Agent_0 
WARNING:SMARTS:Attempted to perform actions on non-existing agent, Agent_0 
WARNING:SMARTS:Attempted to perform actions on non-existing agent, Agent_0 
WARNING:SMARTS:Attempted to perform actions on non-existing agent, Agent_1 
/opt/.venv/lib/python3.8/site-packages/trimesh/curvature.py:12: DeprecationWarning: Please use `coo_matrix` from the `scipy.sparse` namespace, the `scipy.sparse.coo` namespace is deprecated.
  from scipy.sparse.coo import coo_matrix
/opt/.venv/lib/python3.8/site-packages/trimesh/curvature.py:12: DeprecationWarning: Please use `coo_matrix` from the `scipy.sparse` namespace, the `scipy.sparse.coo` namespace is deprecated.
  from scipy.sparse.coo import coo_matrix
Warning: Shape for junction 'gneJ11' has distance 27.04 to its given position.
WARNING:SMARTS:Attempted to perform actions on non-existing agent, Agent_1 
WARNING:SMARTS:Attempted to perform actions on non-existing agent, Agent_1 
WARNING:SMARTS:Attempted to perform actions on non-existing agent, Agent_0 
WARNING:SMARTS:Attempted to perform actions on non-existing agent, Agent_1 
/opt/.venv/lib/python3.8/site-packages/trimesh/curvature.py:12: DeprecationWarning: Please use `coo_matrix` from the `scipy.sparse` namespace, the `scipy.sparse.coo` namespace is deprecated.
  from scipy.sparse.coo import coo_matrix
/opt/.venv/lib/python3.8/site-packages/trimesh/curvature.py:12: DeprecationWarning: Please use `coo_matrix` from the `scipy.sparse` namespace, the `scipy.sparse.coo` namespace is deprecated.
  from scipy.sparse.coo import coo_matrix
Warning: Shape for junction 'gneJ11' has distance 20.57 to its given position.
WARNING:SMARTS:Attempted to perform actions on non-existing agent, Agent_0 
WARNING:SMARTS:Attempted to perform actions on non-existing agent, Agent_1 
WARNING:SMARTS:Attempted to perform actions on non-existing agent, Agent_0 
WARNING:SMARTS:Attempted to perform actions on non-existing agent, Agent_1 
WARNING:SMARTS:Attempted to perform actions on non-existing agent, Agent_0 
WARNING:SMARTS:Attempted to perform actions on non-existing agent, Agent_1 
WARNING:SMARTS:Attempted to perform actions on non-existing agent, Agent_1 
WARNING:SMARTS:Attempted to perform actions on non-existing agent, Agent_0 
WARNING:SMARTS:Attempted to perform actions on non-existing agent, Agent_1 
WARNING:SMARTS:Attempted to perform actions on non-existing agent, Agent_1 
/opt/.venv/lib/python3.8/site-packages/trimesh/curvature.py:12: DeprecationWarning: Please use `coo_matrix` from the `scipy.sparse` namespace, the `scipy.sparse.coo` namespace is deprecated.
  from scipy.sparse.coo import coo_matrix
/opt/.venv/lib/python3.8/site-packages/trimesh/curvature.py:12: DeprecationWarning: Please use `coo_matrix` from the `scipy.sparse` namespace, the `scipy.sparse.coo` namespace is deprecated.
  from scipy.sparse.coo import coo_matrix
WARNING:SMARTS:Attempted to perform actions on non-existing agent, Agent_0 
WARNING:SMARTS:Attempted to perform actions on non-existing agent, Agent_1 
WARNING:SMARTS:Attempted to perform actions on non-existing agent, Agent_1 
/opt/.venv/lib/python3.8/site-packages/trimesh/curvature.py:12: DeprecationWarning: Please use `coo_matrix` from the `scipy.sparse` namespace, the `scipy.sparse.coo` namespace is deprecated.
  from scipy.sparse.coo import coo_matrix
/opt/.venv/lib/python3.8/site-packages/trimesh/curvature.py:12: DeprecationWarning: Please use `coo_matrix` from the `scipy.sparse` namespace, the `scipy.sparse.coo` namespace is deprecated.
  from scipy.sparse.coo import coo_matrix
/opt/.venv/lib/python3.8/site-packages/trimesh/curvature.py:12: DeprecationWarning: Please use `coo_matrix` from the `scipy.sparse` namespace, the `scipy.sparse.coo` namespace is deprecated.
  from scipy.sparse.coo import coo_matrix
/opt/.venv/lib/python3.8/site-packages/trimesh/curvature.py:12: DeprecationWarning: Please use `coo_matrix` from the `scipy.sparse` namespace, the `scipy.sparse.coo` namespace is deprecated.
  from scipy.sparse.coo import coo_matrix
Process SpawnProcess-39:
Traceback (most recent call last):
  File "/usr/lib/python3.8/multiprocessing/process.py", line 315, in _bootstrap
    self.run()
  File "/usr/lib/python3.8/multiprocessing/process.py", line 108, in run
    self._target(*self._args, **self._kwargs)
  File "/tmp/codalab/tmpiG5gNF/run/program/evaluate.py", line 354, in _worker
    observations, rewards, dones, infos = env.step(actions)
  File "/tmp/codalab/tmpiG5gNF/run/input/res/wrappers.py", line 447, in step
    else: wrapped_act = self.pack_action(raw_act)
  File "/tmp/codalab/tmpiG5gNF/run/input/res/wrappers.py", line 563, in pack_action
    st = np.where(all_lane_indeces == self.env.target_lane_index[agent_id])[0][0].item()
IndexError: index 0 is out of bounds for axis 0 with size 0
Exception ignored in: <function SMARTS.__del__ at 0x7f9ce9055af0>
Traceback (most recent call last):
  File "/opt/.venv/lib/python3.8/site-packages/smarts/core/smarts.py", line 856, in __del__
TypeError: 'NoneType' object is not callable
ERROR:/tmp/codalab/tmpiG5gNF/run/program/evaluate.py:Evaluation failed due to error. Attempting retry: evaluate.<locals>.ProcessContext(process_builder=functools.partial(<function evaluate.<locals>.process_builder_func at 0x7f1168b7b3a0>, env_ctor=functools.partial(<function _make_env at 0x7f1319b38a60>, env_type='smarts.env:multi-scenario-v0', scenario='/tmp/codalab/tmpiG5gNF/run/input/ref/eval_scenarios/merge/single_agent/low_volume/3lane_curve_agents_1', shared_configs={'action_space': 'TargetPose', 'img_meters': 64, 'img_pixels': 256, 'sumo_headless': True}, seed=43, wrapper_ctors=<function submitted_wrappers at 0x7f1317b7c430>), policy_type=<class 'policy.Policy'>), env_name='/tmp/codalab/tmpiG5gNF/run/input/ref/eval_scenarios/merge/single_agent/low_volume/3lane_curve_agents_1', retries=1, env_retries=2, last_reply=5631773.446136682)
/opt/.venv/lib/python3.8/site-packages/trimesh/curvature.py:12: DeprecationWarning: Please use `coo_matrix` from the `scipy.sparse` namespace, the `scipy.sparse.coo` namespace is deprecated.
  from scipy.sparse.coo import coo_matrix
/opt/.venv/lib/python3.8/site-packages/trimesh/curvature.py:12: DeprecationWarning: Please use `coo_matrix` from the `scipy.sparse` namespace, the `scipy.sparse.coo` namespace is deprecated.
  from scipy.sparse.coo import coo_matrix
/opt/.venv/lib/python3.8/site-packages/trimesh/curvature.py:12: DeprecationWarning: Please use `coo_matrix` from the `scipy.sparse` namespace, the `scipy.sparse.coo` namespace is deprecated.
  from scipy.sparse.coo import coo_matrix
/opt/.venv/lib/python3.8/site-packages/trimesh/curvature.py:12: DeprecationWarning: Please use `coo_matrix` from the `scipy.sparse` namespace, the `scipy.sparse.coo` namespace is deprecated.
  from scipy.sparse.coo import coo_matrix
/opt/.venv/lib/python3.8/site-packages/trimesh/curvature.py:12: DeprecationWarning: Please use `coo_matrix` from the `scipy.sparse` namespace, the `scipy.sparse.coo` namespace is deprecated.
  from scipy.sparse.coo import coo_matrix
/opt/.venv/lib/python3.8/site-packages/trimesh/curvature.py:12: DeprecationWarning: Please use `coo_matrix` from the `scipy.sparse` namespace, the `scipy.sparse.coo` namespace is deprecated.
  from scipy.sparse.coo import coo_matrix
Warning: Shape for junction 'gneJ11' has distance 27.04 to its given position.
/opt/.venv/lib/python3.8/site-packages/trimesh/curvature.py:12: DeprecationWarning: Please use `coo_matrix` from the `scipy.sparse` namespace, the `scipy.sparse.coo` namespace is deprecated.
  from scipy.sparse.coo import coo_matrix
/opt/.venv/lib/python3.8/site-packages/trimesh/curvature.py:12: DeprecationWarning: Please use `coo_matrix` from the `scipy.sparse` namespace, the `scipy.sparse.coo` namespace is deprecated.
  from scipy.sparse.coo import coo_matrix
Warning: Shape for junction 'gneJ11' has distance 20.57 to its given position.
/opt/.venv/lib/python3.8/site-packages/trimesh/curvature.py:12: DeprecationWarning: Please use `coo_matrix` from the `scipy.sparse` namespace, the `scipy.sparse.coo` namespace is deprecated.
  from scipy.sparse.coo import coo_matrix
/opt/.venv/lib/python3.8/site-packages/trimesh/curvature.py:12: DeprecationWarning: Please use `coo_matrix` from the `scipy.sparse` namespace, the `scipy.sparse.coo` namespace is deprecated.
  from scipy.sparse.coo import coo_matrix
/opt/.venv/lib/python3.8/site-packages/trimesh/curvature.py:12: DeprecationWarning: Please use `coo_matrix` from the `scipy.sparse` namespace, the `scipy.sparse.coo` namespace is deprecated.
  from scipy.sparse.coo import coo_matrix
/opt/.venv/lib/python3.8/site-packages/trimesh/curvature.py:12: DeprecationWarning: Please use `coo_matrix` from the `scipy.sparse` namespace, the `scipy.sparse.coo` namespace is deprecated.
  from scipy.sparse.coo import coo_matrix
/opt/.venv/lib/python3.8/site-packages/trimesh/curvature.py:12: DeprecationWarning: Please use `coo_matrix` from the `scipy.sparse` namespace, the `scipy.sparse.coo` namespace is deprecated.
  from scipy.sparse.coo import coo_matrix
/opt/.venv/lib/python3.8/site-packages/trimesh/curvature.py:12: DeprecationWarning: Please use `coo_matrix` from the `scipy.sparse` namespace, the `scipy.sparse.coo` namespace is deprecated.
  from scipy.sparse.coo import coo_matrix
/opt/.venv/lib/python3.8/site-packages/trimesh/curvature.py:12: DeprecationWarning: Please use `coo_matrix` from the `scipy.sparse` namespace, the `scipy.sparse.coo` namespace is deprecated.
  from scipy.sparse.coo import coo_matrix
/opt/.venv/lib/python3.8/site-packages/trimesh/curvature.py:12: DeprecationWarning: Please use `coo_matrix` from the `scipy.sparse` namespace, the `scipy.sparse.coo` namespace is deprecated.
  from scipy.sparse.coo import coo_matrix
/opt/.venv/lib/python3.8/site-packages/trimesh/curvature.py:12: DeprecationWarning: Please use `coo_matrix` from the `scipy.sparse` namespace, the `scipy.sparse.coo` namespace is deprecated.
  from scipy.sparse.coo import coo_matrix
ERROR:/tmp/codalab/tmpiG5gNF/run/program/evaluate.py:Evaluation expired due to no response in 480s: evaluate.<locals>.ProcessContext(process_builder=functools.partial(<function evaluate.<locals>.process_builder_func at 0x7f1168b7af70>, env_ctor=functools.partial(<function _make_env at 0x7f1319b38a60>, env_type='smarts.env:multi-scenario-v0', scenario='/tmp/codalab/tmpiG5gNF/run/input/ref/eval_scenarios/merge/single_agent/mid_volume/3lane_curve_agents_1', shared_configs={'action_space': 'TargetPose', 'img_meters': 64, 'img_pixels': 256, 'sumo_headless': True}, seed=43, wrapper_ctors=<function submitted_wrappers at 0x7f1317b7c430>), policy_type=<class 'policy.Policy'>), env_name='/tmp/codalab/tmpiG5gNF/run/input/ref/eval_scenarios/merge/single_agent/mid_volume/3lane_curve_agents_1', retries=1, env_retries=2, last_reply=5634365.106444345)
/opt/.venv/lib/python3.8/site-packages/trimesh/curvature.py:12: DeprecationWarning: Please use `coo_matrix` from the `scipy.sparse` namespace, the `scipy.sparse.coo` namespace is deprecated.
  from scipy.sparse.coo import coo_matrix
/opt/.venv/lib/python3.8/site-packages/trimesh/curvature.py:12: DeprecationWarning: Please use `coo_matrix` from the `scipy.sparse` namespace, the `scipy.sparse.coo` namespace is deprecated.
  from scipy.sparse.coo import coo_matrix
WARNING:SMARTS:Attempted to perform actions on non-existing agent, Agent_0 
WARNING:SMARTS:Attempted to perform actions on non-existing agent, Agent_0 
WARNING:SMARTS:Attempted to perform actions on non-existing agent, Agent_0 
WARNING:SMARTS:Attempted to perform actions on non-existing agent, Agent_0 
WARNING:SMARTS:Attempted to perform actions on non-existing agent, Agent_0 
/opt/.venv/lib/python3.8/site-packages/trimesh/curvature.py:12: DeprecationWarning: Please use `coo_matrix` from the `scipy.sparse` namespace, the `scipy.sparse.coo` namespace is deprecated.
  from scipy.sparse.coo import coo_matrix
/opt/.venv/lib/python3.8/site-packages/trimesh/curvature.py:12: DeprecationWarning: Please use `coo_matrix` from the `scipy.sparse` namespace, the `scipy.sparse.coo` namespace is deprecated.
  from scipy.sparse.coo import coo_matrix
WARNING:SMARTS:Attempted to perform actions on non-existing agent, Agent_0 
WARNING:SMARTS:Attempted to perform actions on non-existing agent, Agent_0 
WARNING:SMARTS:Attempted to perform actions on non-existing agent, Agent_0 
WARNING:SMARTS:Attempted to perform actions on non-existing agent, Agent_0 
WARNING:SMARTS:Attempted to perform actions on non-existing agent, Agent_2 
WARNING:SMARTS:Attempted to perform actions on non-existing agent, Agent_0 
WARNING:SMARTS:Attempted to perform actions on non-existing agent, Agent_0 
WARNING:SMARTS:Attempted to perform actions on non-existing agent, Agent_0 
WARNING:SMARTS:Attempted to perform actions on non-existing agent, Agent_0 
/opt/.venv/lib/python3.8/site-packages/trimesh/curvature.py:12: DeprecationWarning: Please use `coo_matrix` from the `scipy.sparse` namespace, the `scipy.sparse.coo` namespace is deprecated.
  from scipy.sparse.coo import coo_matrix
/opt/.venv/lib/python3.8/site-packages/trimesh/curvature.py:12: DeprecationWarning: Please use `coo_matrix` from the `scipy.sparse` namespace, the `scipy.sparse.coo` namespace is deprecated.
  from scipy.sparse.coo import coo_matrix
WARNING:SMARTS:Attempted to perform actions on non-existing agent, Agent_0 
WARNING:SMARTS:Attempted to perform actions on non-existing agent, Agent_2 
WARNING:SMARTS:Attempted to perform actions on non-existing agent, Agent_0 
WARNING:SMARTS:Attempted to perform actions on non-existing agent, Agent_0 
WARNING:SMARTS:Attempted to perform actions on non-existing agent, Agent_0 
WARNING:SMARTS:Attempted to perform actions on non-existing agent, Agent_0 
WARNING:SMARTS:Attempted to perform actions on non-existing agent, Agent_0 
WARNING:SMARTS:Attempted to perform actions on non-existing agent, Agent_0 
WARNING:SMARTS:Attempted to perform actions on non-existing agent, Agent_0 
WARNING:SMARTS:Attempted to perform actions on non-existing agent, Agent_1 
WARNING:SMARTS:Attempted to perform actions on non-existing agent, Agent_2 
WARNING:SMARTS:Attempted to perform actions on non-existing agent, Agent_0 
WARNING:SMARTS:Attempted to perform actions on non-existing agent, Agent_0 
WARNING:SMARTS:Attempted to perform actions on non-existing agent, Agent_0 
WARNING:SMARTS:Attempted to perform actions on non-existing agent, Agent_0 
WARNING:SMARTS:Attempted to perform actions on non-existing agent, Agent_0 
WARNING:SMARTS:Attempted to perform actions on non-existing agent, Agent_0 
WARNING:SMARTS:Attempted to perform actions on non-existing agent, Agent_0 
/opt/.venv/lib/python3.8/site-packages/trimesh/curvature.py:12: DeprecationWarning: Please use `coo_matrix` from the `scipy.sparse` namespace, the `scipy.sparse.coo` namespace is deprecated.
  from scipy.sparse.coo import coo_matrix
WARNING:SMARTS:Attempted to perform actions on non-existing agent, Agent_2 
WARNING:SMARTS:Attempted to perform actions on non-existing agent, Agent_0 
WARNING:SMARTS:Attempted to perform actions on non-existing agent, Agent_0 
/opt/.venv/lib/python3.8/site-packages/trimesh/curvature.py:12: DeprecationWarning: Please use `coo_matrix` from the `scipy.sparse` namespace, the `scipy.sparse.coo` namespace is deprecated.
  from scipy.sparse.coo import coo_matrix
WARNING:SMARTS:Attempted to perform actions on non-existing agent, Agent_0 
WARNING:SMARTS:Attempted to perform actions on non-existing agent, Agent_0 
WARNING:SMARTS:Attempted to perform actions on non-existing agent, Agent_2 
WARNING:SMARTS:Attempted to perform actions on non-existing agent, Agent_1 
WARNING:SMARTS:Attempted to perform actions on non-existing agent, Agent_0 
WARNING:SMARTS:Attempted to perform actions on non-existing agent, Agent_0 
WARNING:SMARTS:Attempted to perform actions on non-existing agent, Agent_2 
WARNING:SMARTS:Attempted to perform actions on non-existing agent, Agent_0 
WARNING:SMARTS:Attempted to perform actions on non-existing agent, Agent_0 
WARNING:SMARTS:Attempted to perform actions on non-existing agent, Agent_2 
WARNING:SMARTS:Attempted to perform actions on non-existing agent, Agent_0 
WARNING:SMARTS:Attempted to perform actions on non-existing agent, Agent_0 
WARNING:SMARTS:Attempted to perform actions on non-existing agent, Agent_0 
WARNING:SMARTS:Attempted to perform actions on non-existing agent, Agent_2 
WARNING:SMARTS:Attempted to perform actions on non-existing agent, Agent_0 
WARNING:SMARTS:Attempted to perform actions on non-existing agent, Agent_0 
WARNING:SMARTS:Attempted to perform actions on non-existing agent, Agent_0 
WARNING:SMARTS:Attempted to perform actions on non-existing agent, Agent_2 
WARNING:SMARTS:Attempted to perform actions on non-existing agent, Agent_0 
WARNING:SMARTS:Attempted to perform actions on non-existing agent, Agent_0 
WARNING:SMARTS:Attempted to perform actions on non-existing agent, Agent_0 
WARNING:SMARTS:Attempted to perform actions on non-existing agent, Agent_2 
WARNING:SMARTS:Attempted to perform actions on non-existing agent, Agent_0 
WARNING:SMARTS:Attempted to perform actions on non-existing agent, Agent_0 
WARNING:SMARTS:Attempted to perform actions on non-existing agent, Agent_2 
WARNING:SMARTS:Attempted to perform actions on non-existing agent, Agent_0 
WARNING:SMARTS:Attempted to perform actions on non-existing agent, Agent_0 
/opt/.venv/lib/python3.8/site-packages/trimesh/curvature.py:12: DeprecationWarning: Please use `coo_matrix` from the `scipy.sparse` namespace, the `scipy.sparse.coo` namespace is deprecated.
  from scipy.sparse.coo import coo_matrix
/opt/.venv/lib/python3.8/site-packages/trimesh/curvature.py:12: DeprecationWarning: Please use `coo_matrix` from the `scipy.sparse` namespace, the `scipy.sparse.coo` namespace is deprecated.
  from scipy.sparse.coo import coo_matrix
WARNING:SMARTS:Attempted to perform actions on non-existing agent, Agent_2 
WARNING:SMARTS:Attempted to perform actions on non-existing agent, Agent_0 
WARNING:SMARTS:Attempted to perform actions on non-existing agent, Agent_2 
WARNING:SMARTS:Attempted to perform actions on non-existing agent, Agent_0 
WARNING:SMARTS:Attempted to perform actions on non-existing agent, Agent_1 
WARNING:SMARTS:Attempted to perform actions on non-existing agent, Agent_2 
WARNING:SMARTS:Attempted to perform actions on non-existing agent, Agent_2 
WARNING:SMARTS:Attempted to perform actions on non-existing agent, Agent_0 
WARNING:SMARTS:Attempted to perform actions on non-existing agent, Agent_2 
WARNING:SMARTS:Attempted to perform actions on non-existing agent, Agent_0 
/opt/.venv/lib/python3.8/site-packages/trimesh/curvature.py:12: DeprecationWarning: Please use `coo_matrix` from the `scipy.sparse` namespace, the `scipy.sparse.coo` namespace is deprecated.
  from scipy.sparse.coo import coo_matrix
/opt/.venv/lib/python3.8/site-packages/trimesh/curvature.py:12: DeprecationWarning: Please use `coo_matrix` from the `scipy.sparse` namespace, the `scipy.sparse.coo` namespace is deprecated.
  from scipy.sparse.coo import coo_matrix
/opt/.venv/lib/python3.8/site-packages/trimesh/curvature.py:12: DeprecationWarning: Please use `coo_matrix` from the `scipy.sparse` namespace, the `scipy.sparse.coo` namespace is deprecated.
  from scipy.sparse.coo import coo_matrix
WARNING:SMARTS:Attempted to perform actions on non-existing agent, Agent_2 
WARNING:SMARTS:Attempted to perform actions on non-existing agent, Agent_0 
WARNING:SMARTS:Attempted to perform actions on non-existing agent, Agent_2 
WARNING:SMARTS:Attempted to perform actions on non-existing agent, Agent_2 
WARNING:SMARTS:Attempted to perform actions on non-existing agent, Agent_0 
WARNING:SMARTS:Attempted to perform actions on non-existing agent, Agent_1 
WARNING:SMARTS:Attempted to perform actions on non-existing agent, Agent_2 
WARNING:SMARTS:Attempted to perform actions on non-existing agent, Agent_2 
WARNING:SMARTS:Attempted to perform actions on non-existing agent, Agent_0 
WARNING:SMARTS:Attempted to perform actions on non-existing agent, Agent_2 
WARNING:SMARTS:Attempted to perform actions on non-existing agent, Agent_0 
/opt/.venv/lib/python3.8/site-packages/trimesh/curvature.py:12: DeprecationWarning: Please use `coo_matrix` from the `scipy.sparse` namespace, the `scipy.sparse.coo` namespace is deprecated.
  from scipy.sparse.coo import coo_matrix
/opt/.venv/lib/python3.8/site-packages/trimesh/curvature.py:12: DeprecationWarning: Please use `coo_matrix` from the `scipy.sparse` namespace, the `scipy.sparse.coo` namespace is deprecated.
  from scipy.sparse.coo import coo_matrix
WARNING:SMARTS:Attempted to perform actions on non-existing agent, Agent_2 
WARNING:SMARTS:Attempted to perform actions on non-existing agent, Agent_0 
WARNING:SMARTS:Attempted to perform actions on non-existing agent, Agent_2 
WARNING:SMARTS:Attempted to perform actions on non-existing agent, Agent_0 
WARNING:SMARTS:Attempted to perform actions on non-existing agent, Agent_0 
WARNING:SMARTS:Attempted to perform actions on non-existing agent, Agent_2 
WARNING:SMARTS:Attempted to perform actions on non-existing agent, Agent_2 
WARNING:SMARTS:Attempted to perform actions on non-existing agent, Agent_0 
WARNING:SMARTS:Attempted to perform actions on non-existing agent, Agent_0 
WARNING:SMARTS:Attempted to perform actions on non-existing agent, Agent_2 
WARNING:SMARTS:Attempted to perform actions on non-existing agent, Agent_2 
WARNING:SMARTS:Attempted to perform actions on non-existing agent, Agent_0 
/opt/.venv/lib/python3.8/site-packages/trimesh/curvature.py:12: DeprecationWarning: Please use `coo_matrix` from the `scipy.sparse` namespace, the `scipy.sparse.coo` namespace is deprecated.
  from scipy.sparse.coo import coo_matrix
ERROR:/tmp/codalab/tmpiG5gNF/run/program/evaluate.py:Evaluation expired due to no response in 480s: evaluate.<locals>.ProcessContext(process_builder=functools.partial(<function evaluate.<locals>.process_builder_func at 0x7f1168b7aaf0>, env_ctor=functools.partial(<function _make_env at 0x7f1319b38a60>, env_type='smarts.env:multi-scenario-v0', scenario='/tmp/codalab/tmpiG5gNF/run/input/ref/eval_scenarios/straight/single_agent/cruise/low_volume/3lane_curve_agents_1', shared_configs={'action_space': 'TargetPose', 'img_meters': 64, 'img_pixels': 256, 'sumo_headless': True}, seed=43, wrapper_ctors=<function submitted_wrappers at 0x7f1317b7c430>), policy_type=<class 'policy.Policy'>), env_name='/tmp/codalab/tmpiG5gNF/run/input/ref/eval_scenarios/straight/single_agent/cruise/low_volume/3lane_curve_agents_1', retries=1, env_retries=2, last_reply=5637234.731203052)
/opt/.venv/lib/python3.8/site-packages/trimesh/curvature.py:12: DeprecationWarning: Please use `coo_matrix` from the `scipy.sparse` namespace, the `scipy.sparse.coo` namespace is deprecated.
  from scipy.sparse.coo import coo_matrix
/opt/.venv/lib/python3.8/site-packages/trimesh/curvature.py:12: DeprecationWarning: Please use `coo_matrix` from the `scipy.sparse` namespace, the `scipy.sparse.coo` namespace is deprecated.
  from scipy.sparse.coo import coo_matrix
WARNING:SMARTS:Attempted to perform actions on non-existing agent, Agent_1 
WARNING:SMARTS:Attempted to perform actions on non-existing agent, Agent_2 
WARNING:SMARTS:Attempted to perform actions on non-existing agent, Agent_2 
WARNING:SMARTS:Attempted to perform actions on non-existing agent, Agent_0 
WARNING:SMARTS:Attempted to perform actions on non-existing agent, Agent_1 
WARNING:SMARTS:Attempted to perform actions on non-existing agent, Agent_2 
WARNING:SMARTS:Attempted to perform actions on non-existing agent, Agent_2 
WARNING:SMARTS:Attempted to perform actions on non-existing agent, Agent_0 
/opt/.venv/lib/python3.8/site-packages/trimesh/curvature.py:12: DeprecationWarning: Please use `coo_matrix` from the `scipy.sparse` namespace, the `scipy.sparse.coo` namespace is deprecated.
  from scipy.sparse.coo import coo_matrix
/opt/.venv/lib/python3.8/site-packages/trimesh/curvature.py:12: DeprecationWarning: Please use `coo_matrix` from the `scipy.sparse` namespace, the `scipy.sparse.coo` namespace is deprecated.
  from scipy.sparse.coo import coo_matrix
WARNING:SMARTS:Attempted to perform actions on non-existing agent, Agent_2 
WARNING:SMARTS:Attempted to perform actions on non-existing agent, Agent_1 
WARNING:SMARTS:Attempted to perform actions on non-existing agent, Agent_0 
WARNING:SMARTS:Attempted to perform actions on non-existing agent, Agent_2 
/opt/.venv/lib/python3.8/site-packages/trimesh/curvature.py:12: DeprecationWarning: Please use `coo_matrix` from the `scipy.sparse` namespace, the `scipy.sparse.coo` namespace is deprecated.
  from scipy.sparse.coo import coo_matrix
WARNING:SMARTS:Attempted to perform actions on non-existing agent, Agent_1 
WARNING:SMARTS:Attempted to perform actions on non-existing agent, Agent_2 
/opt/.venv/lib/python3.8/site-packages/trimesh/curvature.py:12: DeprecationWarning: Please use `coo_matrix` from the `scipy.sparse` namespace, the `scipy.sparse.coo` namespace is deprecated.
  from scipy.sparse.coo import coo_matrix
/opt/.venv/lib/python3.8/site-packages/trimesh/curvature.py:12: DeprecationWarning: Please use `coo_matrix` from the `scipy.sparse` namespace, the `scipy.sparse.coo` namespace is deprecated.
  from scipy.sparse.coo import coo_matrix
WARNING:SMARTS:Attempted to perform actions on non-existing agent, Agent_2 
WARNING:SMARTS:Attempted to perform actions on non-existing agent, Agent_0 
WARNING:SMARTS:Attempted to perform actions on non-existing agent, Agent_0 
ERROR:/tmp/codalab/tmpiG5gNF/run/program/evaluate.py:Evaluation expired due to no response in 480s: evaluate.<locals>.ProcessContext(process_builder=functools.partial(<function evaluate.<locals>.process_builder_func at 0x7f1168b7a940>, env_ctor=functools.partial(<function _make_env at 0x7f1319b38a60>, env_type='smarts.env:multi-scenario-v0', scenario='/tmp/codalab/tmpiG5gNF/run/input/ref/eval_scenarios/straight/single_agent/cruise/high_volume/3lane_curve_agents_1', shared_configs={'action_space': 'TargetPose', 'img_meters': 64, 'img_pixels': 256, 'sumo_headless': True}, seed=43, wrapper_ctors=<function submitted_wrappers at 0x7f1317b7c430>), policy_type=<class 'policy.Policy'>), env_name='/tmp/codalab/tmpiG5gNF/run/input/ref/eval_scenarios/straight/single_agent/cruise/high_volume/3lane_curve_agents_1', retries=1, env_retries=2, last_reply=5638185.72820011)
/opt/.venv/lib/python3.8/site-packages/trimesh/curvature.py:12: DeprecationWarning: Please use `coo_matrix` from the `scipy.sparse` namespace, the `scipy.sparse.coo` namespace is deprecated.
  from scipy.sparse.coo import coo_matrix
WARNING:SMARTS:Attempted to perform actions on non-existing agent, Agent_2 
WARNING:SMARTS:Attempted to perform actions on non-existing agent, Agent_2 
WARNING:SMARTS:Attempted to perform actions on non-existing agent, Agent_0 
WARNING:SMARTS:Attempted to perform actions on non-existing agent, Agent_2 
WARNING:SMARTS:Attempted to perform actions on non-existing agent, Agent_0 
WARNING:SMARTS:Attempted to perform actions on non-existing agent, Agent_2 
WARNING:SMARTS:Attempted to perform actions on non-existing agent, Agent_0 
/opt/.venv/lib/python3.8/site-packages/trimesh/curvature.py:12: DeprecationWarning: Please use `coo_matrix` from the `scipy.sparse` namespace, the `scipy.sparse.coo` namespace is deprecated.
  from scipy.sparse.coo import coo_matrix
ERROR:/tmp/codalab/tmpiG5gNF/run/program/evaluate.py:Evaluation expired due to no response in 480s: evaluate.<locals>.ProcessContext(process_builder=functools.partial(<function evaluate.<locals>.process_builder_func at 0x7f1168b7a790>, env_ctor=functools.partial(<function _make_env at 0x7f1319b38a60>, env_type='smarts.env:multi-scenario-v0', scenario='/tmp/codalab/tmpiG5gNF/run/input/ref/eval_scenarios/straight/single_agent/overtake/low_volume/2lane_slow_agents_1', shared_configs={'action_space': 'TargetPose', 'img_meters': 64, 'img_pixels': 256, 'sumo_headless': True}, seed=43, wrapper_ctors=<function submitted_wrappers at 0x7f1317b7c430>), policy_type=<class 'policy.Policy'>), env_name='/tmp/codalab/tmpiG5gNF/run/input/ref/eval_scenarios/straight/single_agent/overtake/low_volume/2lane_slow_agents_1', retries=1, env_retries=2, last_reply=5638809.815227882)
/opt/.venv/lib/python3.8/site-packages/trimesh/curvature.py:12: DeprecationWarning: Please use `coo_matrix` from the `scipy.sparse` namespace, the `scipy.sparse.coo` namespace is deprecated.
  from scipy.sparse.coo import coo_matrix
/opt/.venv/lib/python3.8/site-packages/trimesh/curvature.py:12: DeprecationWarning: Please use `coo_matrix` from the `scipy.sparse` namespace, the `scipy.sparse.coo` namespace is deprecated.
  from scipy.sparse.coo import coo_matrix
/opt/.venv/lib/python3.8/site-packages/trimesh/curvature.py:12: DeprecationWarning: Please use `coo_matrix` from the `scipy.sparse` namespace, the `scipy.sparse.coo` namespace is deprecated.
  from scipy.sparse.coo import coo_matrix
/opt/.venv/lib/python3.8/site-packages/trimesh/curvature.py:12: DeprecationWarning: Please use `coo_matrix` from the `scipy.sparse` namespace, the `scipy.sparse.coo` namespace is deprecated.
  from scipy.sparse.coo import coo_matrix
ERROR:/tmp/codalab/tmpiG5gNF/run/program/evaluate.py:Evaluation expired due to no response in 480s: evaluate.<locals>.ProcessContext(process_builder=functools.partial(<function evaluate.<locals>.process_builder_func at 0x7f1168b7a670>, env_ctor=functools.partial(<function _make_env at 0x7f1319b38a60>, env_type='smarts.env:multi-scenario-v0', scenario='/tmp/codalab/tmpiG5gNF/run/input/ref/eval_scenarios/straight/single_agent/overtake/low_volume/3lane_curve_agents_1', shared_configs={'action_space': 'TargetPose', 'img_meters': 64, 'img_pixels': 256, 'sumo_headless': True}, seed=43, wrapper_ctors=<function submitted_wrappers at 0x7f1317b7c430>), policy_type=<class 'policy.Policy'>), env_name='/tmp/codalab/tmpiG5gNF/run/input/ref/eval_scenarios/straight/single_agent/overtake/low_volume/3lane_curve_agents_1', retries=1, env_retries=2, last_reply=5639146.166603714)
/opt/.venv/lib/python3.8/site-packages/trimesh/curvature.py:12: DeprecationWarning: Please use `coo_matrix` from the `scipy.sparse` namespace, the `scipy.sparse.coo` namespace is deprecated.
  from scipy.sparse.coo import coo_matrix
/opt/.venv/lib/python3.8/site-packages/trimesh/curvature.py:12: DeprecationWarning: Please use `coo_matrix` from the `scipy.sparse` namespace, the `scipy.sparse.coo` namespace is deprecated.
  from scipy.sparse.coo import coo_matrix
/opt/.venv/lib/python3.8/site-packages/trimesh/curvature.py:12: DeprecationWarning: Please use `coo_matrix` from the `scipy.sparse` namespace, the `scipy.sparse.coo` namespace is deprecated.
  from scipy.sparse.coo import coo_matrix
/opt/.venv/lib/python3.8/site-packages/trimesh/curvature.py:12: DeprecationWarning: Please use `coo_matrix` from the `scipy.sparse` namespace, the `scipy.sparse.coo` namespace is deprecated.
  from scipy.sparse.coo import coo_matrix
/opt/.venv/lib/python3.8/site-packages/trimesh/curvature.py:12: DeprecationWarning: Please use `coo_matrix` from the `scipy.sparse` namespace, the `scipy.sparse.coo` namespace is deprecated.
  from scipy.sparse.coo import coo_matrix
ERROR:/tmp/codalab/tmpiG5gNF/run/program/evaluate.py:Evaluation expired due to no response in 480s: evaluate.<locals>.ProcessContext(process_builder=functools.partial(<function evaluate.<locals>.process_builder_func at 0x7f1168b7a4c0>, env_ctor=functools.partial(<function _make_env at 0x7f1319b38a60>, env_type='smarts.env:multi-scenario-v0', scenario='/tmp/codalab/tmpiG5gNF/run/input/ref/eval_scenarios/straight/single_agent/overtake/mid_volume/2lane_agents_1', shared_configs={'action_space': 'TargetPose', 'img_meters': 64, 'img_pixels': 256, 'sumo_headless': True}, seed=43, wrapper_ctors=<function submitted_wrappers at 0x7f1317b7c430>), policy_type=<class 'policy.Policy'>), env_name='/tmp/codalab/tmpiG5gNF/run/input/ref/eval_scenarios/straight/single_agent/overtake/mid_volume/2lane_agents_1', retries=1, env_retries=2, last_reply=5639846.078407058)
/opt/.venv/lib/python3.8/site-packages/trimesh/curvature.py:12: DeprecationWarning: Please use `coo_matrix` from the `scipy.sparse` namespace, the `scipy.sparse.coo` namespace is deprecated.
  from scipy.sparse.coo import coo_matrix
/opt/.venv/lib/python3.8/site-packages/trimesh/curvature.py:12: DeprecationWarning: Please use `coo_matrix` from the `scipy.sparse` namespace, the `scipy.sparse.coo` namespace is deprecated.
  from scipy.sparse.coo import coo_matrix
/opt/.venv/lib/python3.8/site-packages/trimesh/curvature.py:12: DeprecationWarning: Please use `coo_matrix` from the `scipy.sparse` namespace, the `scipy.sparse.coo` namespace is deprecated.
  from scipy.sparse.coo import coo_matrix
/opt/.venv/lib/python3.8/site-packages/trimesh/curvature.py:12: DeprecationWarning: Please use `coo_matrix` from the `scipy.sparse` namespace, the `scipy.sparse.coo` namespace is deprecated.
  from scipy.sparse.coo import coo_matrix
/opt/.venv/lib/python3.8/site-packages/trimesh/curvature.py:12: DeprecationWarning: Please use `coo_matrix` from the `scipy.sparse` namespace, the `scipy.sparse.coo` namespace is deprecated.
  from scipy.sparse.coo import coo_matrix
WARNING:SMARTS:could not find a provider to assume control of vehicle actor-car--2361632437743525594-6 with role=Social after being relinquished.  removing it.
WARNING:SMARTS:could not find a provider to assume control of vehicle actor-car--2361632437743525594-24 with role=Social after being relinquished.  removing it.
ERROR:/tmp/codalab/tmpiG5gNF/run/program/evaluate.py:Evaluation expired due to no response in 480s: evaluate.<locals>.ProcessContext(process_builder=functools.partial(<function evaluate.<locals>.process_builder_func at 0x7f1168b7a310>, env_ctor=functools.partial(<function _make_env at 0x7f1319b38a60>, env_type='smarts.env:multi-scenario-v0', scenario='/tmp/codalab/tmpiG5gNF/run/input/ref/eval_scenarios/straight/single_agent/overtake/mid_volume/3lane_curve_agents_1', shared_configs={'action_space': 'TargetPose', 'img_meters': 64, 'img_pixels': 256, 'sumo_headless': True}, seed=43, wrapper_ctors=<function submitted_wrappers at 0x7f1317b7c430>), policy_type=<class 'policy.Policy'>), env_name='/tmp/codalab/tmpiG5gNF/run/input/ref/eval_scenarios/straight/single_agent/overtake/mid_volume/3lane_curve_agents_1', retries=1, env_retries=2, last_reply=5640527.911812813)
/opt/.venv/lib/python3.8/site-packages/trimesh/curvature.py:12: DeprecationWarning: Please use `coo_matrix` from the `scipy.sparse` namespace, the `scipy.sparse.coo` namespace is deprecated.
  from scipy.sparse.coo import coo_matrix
/opt/.venv/lib/python3.8/site-packages/trimesh/curvature.py:12: DeprecationWarning: Please use `coo_matrix` from the `scipy.sparse` namespace, the `scipy.sparse.coo` namespace is deprecated.
  from scipy.sparse.coo import coo_matrix
WARNING:SMARTS:could not find a provider to assume control of vehicle actor-car--2361632437743525594-76 with role=Social after being relinquished.  removing it.
WARNING:SMARTS:could not find a provider to assume control of vehicle actor-car--2361632437743525594-15 with role=Social after being relinquished.  removing it.
WARNING:SMARTS:could not find a provider to assume control of vehicle actor-car--2361632437743525594-16 with role=Social after being relinquished.  removing it.
WARNING:SMARTS:could not find a provider to assume control of vehicle actor-car--2361632437743525594-146 with role=Social after being relinquished.  removing it.
/opt/.venv/lib/python3.8/site-packages/trimesh/curvature.py:12: DeprecationWarning: Please use `coo_matrix` from the `scipy.sparse` namespace, the `scipy.sparse.coo` namespace is deprecated.
  from scipy.sparse.coo import coo_matrix
/opt/.venv/lib/python3.8/site-packages/trimesh/curvature.py:12: DeprecationWarning: Please use `coo_matrix` from the `scipy.sparse` namespace, the `scipy.sparse.coo` namespace is deprecated.
  from scipy.sparse.coo import coo_matrix
/opt/.venv/lib/python3.8/site-packages/trimesh/curvature.py:12: DeprecationWarning: Please use `coo_matrix` from the `scipy.sparse` namespace, the `scipy.sparse.coo` namespace is deprecated.
  from scipy.sparse.coo import coo_matrix
/opt/.venv/lib/python3.8/site-packages/trimesh/curvature.py:12: DeprecationWarning: Please use `coo_matrix` from the `scipy.sparse` namespace, the `scipy.sparse.coo` namespace is deprecated.
  from scipy.sparse.coo import coo_matrix
WARNING:SMARTS:could not find a provider to assume control of vehicle actor-car--2361632437743525594-155 with role=Social after being relinquished.  removing it.
WARNING:SMARTS:could not find a provider to assume control of vehicle actor-car--2361632437743525594-182 with role=Social after being relinquished.  removing it.
WARNING:SMARTS:could not find a provider to assume control of vehicle actor-car--2361632437743525594-225 with role=Social after being relinquished.  removing it.
WARNING:SMARTS:could not find a provider to assume control of vehicle actor-car--2361632437743525594-233 with role=Social after being relinquished.  removing it.
WARNING:SMARTS:could not find a provider to assume control of vehicle actor-car--2361632437743525594-291 with role=Social after being relinquished.  removing it.
WARNING:SMARTS:could not find a provider to assume control of vehicle actor-car--2361632437743525594-290 with role=Social after being relinquished.  removing it.
ERROR:/tmp/codalab/tmpiG5gNF/run/program/evaluate.py:Evaluation expired due to no response in 480s: evaluate.<locals>.ProcessContext(process_builder=functools.partial(<function evaluate.<locals>.process_builder_func at 0x7f1168b7a1f0>, env_ctor=functools.partial(<function _make_env at 0x7f1319b38a60>, env_type='smarts.env:multi-scenario-v0', scenario='/tmp/codalab/tmpiG5gNF/run/input/ref/eval_scenarios/straight/single_agent/overtake/mid_volume/4lane_slow_agents_1', shared_configs={'action_space': 'TargetPose', 'img_meters': 64, 'img_pixels': 256, 'sumo_headless': True}, seed=43, wrapper_ctors=<function submitted_wrappers at 0x7f1317b7c430>), policy_type=<class 'policy.Policy'>), env_name='/tmp/codalab/tmpiG5gNF/run/input/ref/eval_scenarios/straight/single_agent/overtake/mid_volume/4lane_slow_agents_1', retries=1, env_retries=2, last_reply=5640834.313718971)
/opt/.venv/lib/python3.8/site-packages/trimesh/curvature.py:12: DeprecationWarning: Please use `coo_matrix` from the `scipy.sparse` namespace, the `scipy.sparse.coo` namespace is deprecated.
  from scipy.sparse.coo import coo_matrix
/opt/.venv/lib/python3.8/site-packages/trimesh/curvature.py:12: DeprecationWarning: Please use `coo_matrix` from the `scipy.sparse` namespace, the `scipy.sparse.coo` namespace is deprecated.
  from scipy.sparse.coo import coo_matrix
/opt/.venv/lib/python3.8/site-packages/trimesh/curvature.py:12: DeprecationWarning: Please use `coo_matrix` from the `scipy.sparse` namespace, the `scipy.sparse.coo` namespace is deprecated.
  from scipy.sparse.coo import coo_matrix
/opt/.venv/lib/python3.8/site-packages/trimesh/curvature.py:12: DeprecationWarning: Please use `coo_matrix` from the `scipy.sparse` namespace, the `scipy.sparse.coo` namespace is deprecated.
  from scipy.sparse.coo import coo_matrix
WARNING:SMARTS:could not find a provider to assume control of vehicle actor-car--2361632437743525594-377 with role=Social after being relinquished.  removing it.
/opt/.venv/lib/python3.8/site-packages/trimesh/curvature.py:12: DeprecationWarning: Please use `coo_matrix` from the `scipy.sparse` namespace, the `scipy.sparse.coo` namespace is deprecated.
  from scipy.sparse.coo import coo_matrix
/opt/.venv/lib/python3.8/site-packages/trimesh/curvature.py:12: DeprecationWarning: Please use `coo_matrix` from the `scipy.sparse` namespace, the `scipy.sparse.coo` namespace is deprecated.
  from scipy.sparse.coo import coo_matrix
/opt/.venv/lib/python3.8/site-packages/trimesh/curvature.py:12: DeprecationWarning: Please use `coo_matrix` from the `scipy.sparse` namespace, the `scipy.sparse.coo` namespace is deprecated.
  from scipy.sparse.coo import coo_matrix
/opt/.venv/lib/python3.8/site-packages/trimesh/curvature.py:12: DeprecationWarning: Please use `coo_matrix` from the `scipy.sparse` namespace, the `scipy.sparse.coo` namespace is deprecated.
  from scipy.sparse.coo import coo_matrix
/opt/.venv/lib/python3.8/site-packages/trimesh/curvature.py:12: DeprecationWarning: Please use `coo_matrix` from the `scipy.sparse` namespace, the `scipy.sparse.coo` namespace is deprecated.
  from scipy.sparse.coo import coo_matrix
/opt/.venv/lib/python3.8/site-packages/trimesh/curvature.py:12: DeprecationWarning: Please use `coo_matrix` from the `scipy.sparse` namespace, the `scipy.sparse.coo` namespace is deprecated.
  from scipy.sparse.coo import coo_matrix
WARNING:SMARTS:could not find a provider to assume control of vehicle actor-car--2361632437743525594-461 with role=Social after being relinquished.  removing it.
WARNING:SMARTS:Attempted to perform actions on non-existing agent, Agent_0 
WARNING:SMARTS:could not find a provider to assume control of vehicle actor-car--2361632437743525594-498 with role=Social after being relinquished.  removing it.
/opt/.venv/lib/python3.8/site-packages/trimesh/curvature.py:12: DeprecationWarning: Please use `coo_matrix` from the `scipy.sparse` namespace, the `scipy.sparse.coo` namespace is deprecated.
  from scipy.sparse.coo import coo_matrix
/opt/.venv/lib/python3.8/site-packages/trimesh/curvature.py:12: DeprecationWarning: Please use `coo_matrix` from the `scipy.sparse` namespace, the `scipy.sparse.coo` namespace is deprecated.
  from scipy.sparse.coo import coo_matrix
WARNING:SMARTS:Attempted to perform actions on non-existing agent, Agent_0 
/opt/.venv/lib/python3.8/site-packages/trimesh/curvature.py:12: DeprecationWarning: Please use `coo_matrix` from the `scipy.sparse` namespace, the `scipy.sparse.coo` namespace is deprecated.
  from scipy.sparse.coo import coo_matrix
/opt/.venv/lib/python3.8/site-packages/trimesh/curvature.py:12: DeprecationWarning: Please use `coo_matrix` from the `scipy.sparse` namespace, the `scipy.sparse.coo` namespace is deprecated.
  from scipy.sparse.coo import coo_matrix
/opt/.venv/lib/python3.8/site-packages/trimesh/curvature.py:12: DeprecationWarning: Please use `coo_matrix` from the `scipy.sparse` namespace, the `scipy.sparse.coo` namespace is deprecated.
  from scipy.sparse.coo import coo_matrix
/opt/.venv/lib/python3.8/site-packages/trimesh/curvature.py:12: DeprecationWarning: Please use `coo_matrix` from the `scipy.sparse` namespace, the `scipy.sparse.coo` namespace is deprecated.
  from scipy.sparse.coo import coo_matrix
/opt/.venv/lib/python3.8/site-packages/trimesh/curvature.py:12: DeprecationWarning: Please use `coo_matrix` from the `scipy.sparse` namespace, the `scipy.sparse.coo` namespace is deprecated.
  from scipy.sparse.coo import coo_matrix
/opt/.venv/lib/python3.8/site-packages/trimesh/curvature.py:12: DeprecationWarning: Please use `coo_matrix` from the `scipy.sparse` namespace, the `scipy.sparse.coo` namespace is deprecated.
  from scipy.sparse.coo import coo_matrix
/opt/.venv/lib/python3.8/site-packages/trimesh/curvature.py:12: DeprecationWarning: Please use `coo_matrix` from the `scipy.sparse` namespace, the `scipy.sparse.coo` namespace is deprecated.
  from scipy.sparse.coo import coo_matrix
/opt/.venv/lib/python3.8/site-packages/trimesh/curvature.py:12: DeprecationWarning: Please use `coo_matrix` from the `scipy.sparse` namespace, the `scipy.sparse.coo` namespace is deprecated.
  from scipy.sparse.coo import coo_matrix
/opt/.venv/lib/python3.8/site-packages/trimesh/curvature.py:12: DeprecationWarning: Please use `coo_matrix` from the `scipy.sparse` namespace, the `scipy.sparse.coo` namespace is deprecated.
  from scipy.sparse.coo import coo_matrix
/opt/.venv/lib/python3.8/site-packages/trimesh/curvature.py:12: DeprecationWarning: Please use `coo_matrix` from the `scipy.sparse` namespace, the `scipy.sparse.coo` namespace is deprecated.
  from scipy.sparse.coo import coo_matrix
/opt/.venv/lib/python3.8/site-packages/trimesh/curvature.py:12: DeprecationWarning: Please use `coo_matrix` from the `scipy.sparse` namespace, the `scipy.sparse.coo` namespace is deprecated.
  from scipy.sparse.coo import coo_matrix
/opt/.venv/lib/python3.8/site-packages/trimesh/curvature.py:12: DeprecationWarning: Please use `coo_matrix` from the `scipy.sparse` namespace, the `scipy.sparse.coo` namespace is deprecated.
  from scipy.sparse.coo import coo_matrix
WARNING:SMARTS:Attempted to perform actions on non-existing agent, Agent_0 
/opt/.venv/lib/python3.8/site-packages/trimesh/curvature.py:12: DeprecationWarning: Please use `coo_matrix` from the `scipy.sparse` namespace, the `scipy.sparse.coo` namespace is deprecated.
  from scipy.sparse.coo import coo_matrix
/opt/.venv/lib/python3.8/site-packages/trimesh/curvature.py:12: DeprecationWarning: Please use `coo_matrix` from the `scipy.sparse` namespace, the `scipy.sparse.coo` namespace is deprecated.
  from scipy.sparse.coo import coo_matrix
/opt/.venv/lib/python3.8/site-packages/trimesh/curvature.py:12: DeprecationWarning: Please use `coo_matrix` from the `scipy.sparse` namespace, the `scipy.sparse.coo` namespace is deprecated.
  from scipy.sparse.coo import coo_matrix
/opt/.venv/lib/python3.8/site-packages/trimesh/curvature.py:12: DeprecationWarning: Please use `coo_matrix` from the `scipy.sparse` namespace, the `scipy.sparse.coo` namespace is deprecated.
  from scipy.sparse.coo import coo_matrix
/opt/.venv/lib/python3.8/site-packages/trimesh/curvature.py:12: DeprecationWarning: Please use `coo_matrix` from the `scipy.sparse` namespace, the `scipy.sparse.coo` namespace is deprecated.
  from scipy.sparse.coo import coo_matrix
/opt/.venv/lib/python3.8/site-packages/trimesh/curvature.py:12: DeprecationWarning: Please use `coo_matrix` from the `scipy.sparse` namespace, the `scipy.sparse.coo` namespace is deprecated.
  from scipy.sparse.coo import coo_matrix
/opt/.venv/lib/python3.8/site-packages/trimesh/curvature.py:12: DeprecationWarning: Please use `coo_matrix` from the `scipy.sparse` namespace, the `scipy.sparse.coo` namespace is deprecated.
  from scipy.sparse.coo import coo_matrix
/opt/.venv/lib/python3.8/site-packages/trimesh/curvature.py:12: DeprecationWarning: Please use `coo_matrix` from the `scipy.sparse` namespace, the `scipy.sparse.coo` namespace is deprecated.
  from scipy.sparse.coo import coo_matrix
/opt/.venv/lib/python3.8/site-packages/trimesh/curvature.py:12: DeprecationWarning: Please use `coo_matrix` from the `scipy.sparse` namespace, the `scipy.sparse.coo` namespace is deprecated.
  from scipy.sparse.coo import coo_matrix
/opt/.venv/lib/python3.8/site-packages/trimesh/curvature.py:12: DeprecationWarning: Please use `coo_matrix` from the `scipy.sparse` namespace, the `scipy.sparse.coo` namespace is deprecated.
  from scipy.sparse.coo import coo_matrix
/opt/.venv/lib/python3.8/site-packages/trimesh/curvature.py:12: DeprecationWarning: Please use `coo_matrix` from the `scipy.sparse` namespace, the `scipy.sparse.coo` namespace is deprecated.
  from scipy.sparse.coo import coo_matrix
/opt/.venv/lib/python3.8/site-packages/trimesh/curvature.py:12: DeprecationWarning: Please use `coo_matrix` from the `scipy.sparse` namespace, the `scipy.sparse.coo` namespace is deprecated.
  from scipy.sparse.coo import coo_matrix
/opt/.venv/lib/python3.8/site-packages/trimesh/curvature.py:12: DeprecationWarning: Please use `coo_matrix` from the `scipy.sparse` namespace, the `scipy.sparse.coo` namespace is deprecated.
  from scipy.sparse.coo import coo_matrix
/opt/.venv/lib/python3.8/site-packages/trimesh/curvature.py:12: DeprecationWarning: Please use `coo_matrix` from the `scipy.sparse` namespace, the `scipy.sparse.coo` namespace is deprecated.
  from scipy.sparse.coo import coo_matrix
/opt/.venv/lib/python3.8/site-packages/trimesh/curvature.py:12: DeprecationWarning: Please use `coo_matrix` from the `scipy.sparse` namespace, the `scipy.sparse.coo` namespace is deprecated.
  from scipy.sparse.coo import coo_matrix
/opt/.venv/lib/python3.8/site-packages/trimesh/curvature.py:12: DeprecationWarning: Please use `coo_matrix` from the `scipy.sparse` namespace, the `scipy.sparse.coo` namespace is deprecated.
  from scipy.sparse.coo import coo_matrix
/opt/.venv/lib/python3.8/site-packages/trimesh/curvature.py:12: DeprecationWarning: Please use `coo_matrix` from the `scipy.sparse` namespace, the `scipy.sparse.coo` namespace is deprecated.
  from scipy.sparse.coo import coo_matrix
/opt/.venv/lib/python3.8/site-packages/trimesh/curvature.py:12: DeprecationWarning: Please use `coo_matrix` from the `scipy.sparse` namespace, the `scipy.sparse.coo` namespace is deprecated.
  from scipy.sparse.coo import coo_matrix
/opt/.venv/lib/python3.8/site-packages/trimesh/curvature.py:12: DeprecationWarning: Please use `coo_matrix` from the `scipy.sparse` namespace, the `scipy.sparse.coo` namespace is deprecated.
  from scipy.sparse.coo import coo_matrix
/opt/.venv/lib/python3.8/site-packages/trimesh/curvature.py:12: DeprecationWarning: Please use `coo_matrix` from the `scipy.sparse` namespace, the `scipy.sparse.coo` namespace is deprecated.
  from scipy.sparse.coo import coo_matrix
/opt/.venv/lib/python3.8/site-packages/trimesh/curvature.py:12: DeprecationWarning: Please use `coo_matrix` from the `scipy.sparse` namespace, the `scipy.sparse.coo` namespace is deprecated.
  from scipy.sparse.coo import coo_matrix
/opt/.venv/lib/python3.8/site-packages/trimesh/curvature.py:12: DeprecationWarning: Please use `coo_matrix` from the `scipy.sparse` namespace, the `scipy.sparse.coo` namespace is deprecated.
  from scipy.sparse.coo import coo_matrix
/opt/.venv/lib/python3.8/site-packages/trimesh/curvature.py:12: DeprecationWarning: Please use `coo_matrix` from the `scipy.sparse` namespace, the `scipy.sparse.coo` namespace is deprecated.
  from scipy.sparse.coo import coo_matrix
/opt/.venv/lib/python3.8/site-packages/trimesh/curvature.py:12: DeprecationWarning: Please use `coo_matrix` from the `scipy.sparse` namespace, the `scipy.sparse.coo` namespace is deprecated.
  from scipy.sparse.coo import coo_matrix
/opt/.venv/lib/python3.8/site-packages/trimesh/curvature.py:12: DeprecationWarning: Please use `coo_matrix` from the `scipy.sparse` namespace, the `scipy.sparse.coo` namespace is deprecated.
  from scipy.sparse.coo import coo_matrix
/opt/.venv/lib/python3.8/site-packages/trimesh/curvature.py:12: DeprecationWarning: Please use `coo_matrix` from the `scipy.sparse` namespace, the `scipy.sparse.coo` namespace is deprecated.
  from scipy.sparse.coo import coo_matrix
/opt/.venv/lib/python3.8/site-packages/trimesh/curvature.py:12: DeprecationWarning: Please use `coo_matrix` from the `scipy.sparse` namespace, the `scipy.sparse.coo` namespace is deprecated.
  from scipy.sparse.coo import coo_matrix
/opt/.venv/lib/python3.8/site-packages/trimesh/curvature.py:12: DeprecationWarning: Please use `coo_matrix` from the `scipy.sparse` namespace, the `scipy.sparse.coo` namespace is deprecated.
  from scipy.sparse.coo import coo_matrix
/opt/.venv/lib/python3.8/site-packages/trimesh/curvature.py:12: DeprecationWarning: Please use `coo_matrix` from the `scipy.sparse` namespace, the `scipy.sparse.coo` namespace is deprecated.
  from scipy.sparse.coo import coo_matrix
/opt/.venv/lib/python3.8/site-packages/trimesh/curvature.py:12: DeprecationWarning: Please use `coo_matrix` from the `scipy.sparse` namespace, the `scipy.sparse.coo` namespace is deprecated.
  from scipy.sparse.coo import coo_matrix
/opt/.venv/lib/python3.8/site-packages/trimesh/curvature.py:12: DeprecationWarning: Please use `coo_matrix` from the `scipy.sparse` namespace, the `scipy.sparse.coo` namespace is deprecated.
  from scipy.sparse.coo import coo_matrix
/opt/.venv/lib/python3.8/site-packages/trimesh/curvature.py:12: DeprecationWarning: Please use `coo_matrix` from the `scipy.sparse` namespace, the `scipy.sparse.coo` namespace is deprecated.
  from scipy.sparse.coo import coo_matrix
/opt/.venv/lib/python3.8/site-packages/trimesh/curvature.py:12: DeprecationWarning: Please use `coo_matrix` from the `scipy.sparse` namespace, the `scipy.sparse.coo` namespace is deprecated.
  from scipy.sparse.coo import coo_matrix
/opt/.venv/lib/python3.8/site-packages/trimesh/curvature.py:12: DeprecationWarning: Please use `coo_matrix` from the `scipy.sparse` namespace, the `scipy.sparse.coo` namespace is deprecated.
  from scipy.sparse.coo import coo_matrix
WARNING:SMARTS:Attempted to perform actions on non-existing agent, Agent_0 
/opt/.venv/lib/python3.8/site-packages/trimesh/curvature.py:12: DeprecationWarning: Please use `coo_matrix` from the `scipy.sparse` namespace, the `scipy.sparse.coo` namespace is deprecated.
  from scipy.sparse.coo import coo_matrix
/opt/.venv/lib/python3.8/site-packages/trimesh/curvature.py:12: DeprecationWarning: Please use `coo_matrix` from the `scipy.sparse` namespace, the `scipy.sparse.coo` namespace is deprecated.
  from scipy.sparse.coo import coo_matrix
/opt/.venv/lib/python3.8/site-packages/trimesh/curvature.py:12: DeprecationWarning: Please use `coo_matrix` from the `scipy.sparse` namespace, the `scipy.sparse.coo` namespace is deprecated.
  from scipy.sparse.coo import coo_matrix
/opt/.venv/lib/python3.8/site-packages/trimesh/curvature.py:12: DeprecationWarning: Please use `coo_matrix` from the `scipy.sparse` namespace, the `scipy.sparse.coo` namespace is deprecated.
  from scipy.sparse.coo import coo_matrix
WARNING:SMARTS:Attempted to perform actions on non-existing agent, Agent_1 
/opt/.venv/lib/python3.8/site-packages/trimesh/curvature.py:12: DeprecationWarning: Please use `coo_matrix` from the `scipy.sparse` namespace, the `scipy.sparse.coo` namespace is deprecated.
  from scipy.sparse.coo import coo_matrix
/opt/.venv/lib/python3.8/site-packages/trimesh/curvature.py:12: DeprecationWarning: Please use `coo_matrix` from the `scipy.sparse` namespace, the `scipy.sparse.coo` namespace is deprecated.
  from scipy.sparse.coo import coo_matrix
/opt/.venv/lib/python3.8/site-packages/trimesh/curvature.py:12: DeprecationWarning: Please use `coo_matrix` from the `scipy.sparse` namespace, the `scipy.sparse.coo` namespace is deprecated.
  from scipy.sparse.coo import coo_matrix
/opt/.venv/lib/python3.8/site-packages/trimesh/curvature.py:12: DeprecationWarning: Please use `coo_matrix` from the `scipy.sparse` namespace, the `scipy.sparse.coo` namespace is deprecated.
  from scipy.sparse.coo import coo_matrix
/opt/.venv/lib/python3.8/site-packages/trimesh/curvature.py:12: DeprecationWarning: Please use `coo_matrix` from the `scipy.sparse` namespace, the `scipy.sparse.coo` namespace is deprecated.
  from scipy.sparse.coo import coo_matrix
/opt/.venv/lib/python3.8/site-packages/trimesh/curvature.py:12: DeprecationWarning: Please use `coo_matrix` from the `scipy.sparse` namespace, the `scipy.sparse.coo` namespace is deprecated.
  from scipy.sparse.coo import coo_matrix
/opt/.venv/lib/python3.8/site-packages/trimesh/curvature.py:12: DeprecationWarning: Please use `coo_matrix` from the `scipy.sparse` namespace, the `scipy.sparse.coo` namespace is deprecated.
  from scipy.sparse.coo import coo_matrix
/opt/.venv/lib/python3.8/site-packages/trimesh/curvature.py:12: DeprecationWarning: Please use `coo_matrix` from the `scipy.sparse` namespace, the `scipy.sparse.coo` namespace is deprecated.
  from scipy.sparse.coo import coo_matrix
WARNING:SMARTS:Attempted to perform actions on non-existing agent, Agent_1 
/opt/.venv/lib/python3.8/site-packages/trimesh/curvature.py:12: DeprecationWarning: Please use `coo_matrix` from the `scipy.sparse` namespace, the `scipy.sparse.coo` namespace is deprecated.
  from scipy.sparse.coo import coo_matrix
/opt/.venv/lib/python3.8/site-packages/trimesh/curvature.py:12: DeprecationWarning: Please use `coo_matrix` from the `scipy.sparse` namespace, the `scipy.sparse.coo` namespace is deprecated.
  from scipy.sparse.coo import coo_matrix
/opt/.venv/lib/python3.8/site-packages/trimesh/curvature.py:12: DeprecationWarning: Please use `coo_matrix` from the `scipy.sparse` namespace, the `scipy.sparse.coo` namespace is deprecated.
  from scipy.sparse.coo import coo_matrix
/opt/.venv/lib/python3.8/site-packages/trimesh/curvature.py:12: DeprecationWarning: Please use `coo_matrix` from the `scipy.sparse` namespace, the `scipy.sparse.coo` namespace is deprecated.
  from scipy.sparse.coo import coo_matrix
/opt/.venv/lib/python3.8/site-packages/trimesh/curvature.py:12: DeprecationWarning: Please use `coo_matrix` from the `scipy.sparse` namespace, the `scipy.sparse.coo` namespace is deprecated.
  from scipy.sparse.coo import coo_matrix
/opt/.venv/lib/python3.8/site-packages/trimesh/curvature.py:12: DeprecationWarning: Please use `coo_matrix` from the `scipy.sparse` namespace, the `scipy.sparse.coo` namespace is deprecated.
  from scipy.sparse.coo import coo_matrix
/opt/.venv/lib/python3.8/site-packages/trimesh/curvature.py:12: DeprecationWarning: Please use `coo_matrix` from the `scipy.sparse` namespace, the `scipy.sparse.coo` namespace is deprecated.
  from scipy.sparse.coo import coo_matrix
/opt/.venv/lib/python3.8/site-packages/trimesh/curvature.py:12: DeprecationWarning: Please use `coo_matrix` from the `scipy.sparse` namespace, the `scipy.sparse.coo` namespace is deprecated.
  from scipy.sparse.coo import coo_matrix
/opt/.venv/lib/python3.8/site-packages/trimesh/curvature.py:12: DeprecationWarning: Please use `coo_matrix` from the `scipy.sparse` namespace, the `scipy.sparse.coo` namespace is deprecated.
  from scipy.sparse.coo import coo_matrix
/opt/.venv/lib/python3.8/site-packages/trimesh/curvature.py:12: DeprecationWarning: Please use `coo_matrix` from the `scipy.sparse` namespace, the `scipy.sparse.coo` namespace is deprecated.
  from scipy.sparse.coo import coo_matrix
/opt/.venv/lib/python3.8/site-packages/trimesh/curvature.py:12: DeprecationWarning: Please use `coo_matrix` from the `scipy.sparse` namespace, the `scipy.sparse.coo` namespace is deprecated.
  from scipy.sparse.coo import coo_matrix
/opt/.venv/lib/python3.8/site-packages/trimesh/curvature.py:12: DeprecationWarning: Please use `coo_matrix` from the `scipy.sparse` namespace, the `scipy.sparse.coo` namespace is deprecated.
  from scipy.sparse.coo import coo_matrix
Process SpawnProcess-115:
Traceback (most recent call last):
  File "/usr/lib/python3.8/multiprocessing/process.py", line 315, in _bootstrap
    self.run()
  File "/usr/lib/python3.8/multiprocessing/process.py", line 108, in run
    self._target(*self._args, **self._kwargs)
  File "/tmp/codalab/tmpiG5gNF/run/program/evaluate.py", line 354, in _worker
    observations, rewards, dones, infos = env.step(actions)
  File "/tmp/codalab/tmpiG5gNF/run/input/res/wrappers.py", line 447, in step
    else: wrapped_act = self.pack_action(raw_act)
  File "/tmp/codalab/tmpiG5gNF/run/input/res/wrappers.py", line 481, in pack_action
    last_correct_wp_pos = self.env.last_correct_wp_pos[agent_id]
KeyError: 'Agent_0'
Exception ignored in: <function SMARTS.__del__ at 0x7f7533852b80>
Traceback (most recent call last):
  File "/opt/.venv/lib/python3.8/site-packages/smarts/core/smarts.py", line 856, in __del__
TypeError: 'NoneType' object is not callable
ERROR:/tmp/codalab/tmpiG5gNF/run/program/evaluate.py:Evaluation failed due to error. Attempting retry: evaluate.<locals>.ProcessContext(process_builder=functools.partial(<function evaluate.<locals>.process_builder_func at 0x7f1168b70820>, env_ctor=functools.partial(<function _make_env at 0x7f1319b38a60>, env_type='smarts.env:multi-scenario-v0', scenario='/tmp/codalab/tmpiG5gNF/run/input/ref/eval_scenarios/naturalistic/rt2-agents_1', shared_configs={'action_space': 'TargetPose', 'img_meters': 64, 'img_pixels': 256, 'sumo_headless': True}, seed=43, wrapper_ctors=<function submitted_wrappers at 0x7f1317b7c430>), policy_type=<class 'policy.Policy'>), env_name='/tmp/codalab/tmpiG5gNF/run/input/ref/eval_scenarios/naturalistic/rt2-agents_1', retries=1, env_retries=2, last_reply=5641800.24601372)
/opt/.venv/lib/python3.8/site-packages/trimesh/curvature.py:12: DeprecationWarning: Please use `coo_matrix` from the `scipy.sparse` namespace, the `scipy.sparse.coo` namespace is deprecated.
  from scipy.sparse.coo import coo_matrix
/opt/.venv/lib/python3.8/site-packages/trimesh/curvature.py:12: DeprecationWarning: Please use `coo_matrix` from the `scipy.sparse` namespace, the `scipy.sparse.coo` namespace is deprecated.
  from scipy.sparse.coo import coo_matrix
/opt/.venv/lib/python3.8/site-packages/trimesh/curvature.py:12: DeprecationWarning: Please use `coo_matrix` from the `scipy.sparse` namespace, the `scipy.sparse.coo` namespace is deprecated.
  from scipy.sparse.coo import coo_matrix
/opt/.venv/lib/python3.8/site-packages/trimesh/curvature.py:12: DeprecationWarning: Please use `coo_matrix` from the `scipy.sparse` namespace, the `scipy.sparse.coo` namespace is deprecated.
  from scipy.sparse.coo import coo_matrix
/opt/.venv/lib/python3.8/site-packages/trimesh/curvature.py:12: DeprecationWarning: Please use `coo_matrix` from the `scipy.sparse` namespace, the `scipy.sparse.coo` namespace is deprecated.
  from scipy.sparse.coo import coo_matrix
/opt/.venv/lib/python3.8/site-packages/trimesh/curvature.py:12: DeprecationWarning: Please use `coo_matrix` from the `scipy.sparse` namespace, the `scipy.sparse.coo` namespace is deprecated.
  from scipy.sparse.coo import coo_matrix
/opt/.venv/lib/python3.8/site-packages/trimesh/curvature.py:12: DeprecationWarning: Please use `coo_matrix` from the `scipy.sparse` namespace, the `scipy.sparse.coo` namespace is deprecated.
  from scipy.sparse.coo import coo_matrix
/opt/.venv/lib/python3.8/site-packages/trimesh/curvature.py:12: DeprecationWarning: Please use `coo_matrix` from the `scipy.sparse` namespace, the `scipy.sparse.coo` namespace is deprecated.
  from scipy.sparse.coo import coo_matrix
/opt/.venv/lib/python3.8/site-packages/trimesh/curvature.py:12: DeprecationWarning: Please use `coo_matrix` from the `scipy.sparse` namespace, the `scipy.sparse.coo` namespace is deprecated.
  from scipy.sparse.coo import coo_matrix
/opt/.venv/lib/python3.8/site-packages/trimesh/curvature.py:12: DeprecationWarning: Please use `coo_matrix` from the `scipy.sparse` namespace, the `scipy.sparse.coo` namespace is deprecated.
  from scipy.sparse.coo import coo_matrix
/opt/.venv/lib/python3.8/site-packages/trimesh/curvature.py:12: DeprecationWarning: Please use `coo_matrix` from the `scipy.sparse` namespace, the `scipy.sparse.coo` namespace is deprecated.
  from scipy.sparse.coo import coo_matrix
/opt/.venv/lib/python3.8/site-packages/trimesh/curvature.py:12: DeprecationWarning: Please use `coo_matrix` from the `scipy.sparse` namespace, the `scipy.sparse.coo` namespace is deprecated.
  from scipy.sparse.coo import coo_matrix
/opt/.venv/lib/python3.8/site-packages/trimesh/curvature.py:12: DeprecationWarning: Please use `coo_matrix` from the `scipy.sparse` namespace, the `scipy.sparse.coo` namespace is deprecated.
  from scipy.sparse.coo import coo_matrix
/opt/.venv/lib/python3.8/site-packages/trimesh/curvature.py:12: DeprecationWarning: Please use `coo_matrix` from the `scipy.sparse` namespace, the `scipy.sparse.coo` namespace is deprecated.
  from scipy.sparse.coo import coo_matrix
/opt/.venv/lib/python3.8/site-packages/trimesh/curvature.py:12: DeprecationWarning: Please use `coo_matrix` from the `scipy.sparse` namespace, the `scipy.sparse.coo` namespace is deprecated.
  from scipy.sparse.coo import coo_matrix
/opt/.venv/lib/python3.8/site-packages/trimesh/curvature.py:12: DeprecationWarning: Please use `coo_matrix` from the `scipy.sparse` namespace, the `scipy.sparse.coo` namespace is deprecated.
  from scipy.sparse.coo import coo_matrix
/opt/.venv/lib/python3.8/site-packages/trimesh/curvature.py:12: DeprecationWarning: Please use `coo_matrix` from the `scipy.sparse` namespace, the `scipy.sparse.coo` namespace is deprecated.
  from scipy.sparse.coo import coo_matrix
/opt/.venv/lib/python3.8/site-packages/trimesh/curvature.py:12: DeprecationWarning: Please use `coo_matrix` from the `scipy.sparse` namespace, the `scipy.sparse.coo` namespace is deprecated.
  from scipy.sparse.coo import coo_matrix
/opt/.venv/lib/python3.8/site-packages/trimesh/curvature.py:12: DeprecationWarning: Please use `coo_matrix` from the `scipy.sparse` namespace, the `scipy.sparse.coo` namespace is deprecated.
  from scipy.sparse.coo import coo_matrix
/opt/.venv/lib/python3.8/site-packages/trimesh/curvature.py:12: DeprecationWarning: Please use `coo_matrix` from the `scipy.sparse` namespace, the `scipy.sparse.coo` namespace is deprecated.
  from scipy.sparse.coo import coo_matrix
/opt/.venv/lib/python3.8/site-packages/trimesh/curvature.py:12: DeprecationWarning: Please use `coo_matrix` from the `scipy.sparse` namespace, the `scipy.sparse.coo` namespace is deprecated.
  from scipy.sparse.coo import coo_matrix
/opt/.venv/lib/python3.8/site-packages/trimesh/curvature.py:12: DeprecationWarning: Please use `coo_matrix` from the `scipy.sparse` namespace, the `scipy.sparse.coo` namespace is deprecated.
  from scipy.sparse.coo import coo_matrix
/opt/.venv/lib/python3.8/site-packages/trimesh/curvature.py:12: DeprecationWarning: Please use `coo_matrix` from the `scipy.sparse` namespace, the `scipy.sparse.coo` namespace is deprecated.
  from scipy.sparse.coo import coo_matrix
/opt/.venv/lib/python3.8/site-packages/trimesh/curvature.py:12: DeprecationWarning: Please use `coo_matrix` from the `scipy.sparse` namespace, the `scipy.sparse.coo` namespace is deprecated.
  from scipy.sparse.coo import coo_matrix
/opt/.venv/lib/python3.8/site-packages/trimesh/curvature.py:12: DeprecationWarning: Please use `coo_matrix` from the `scipy.sparse` namespace, the `scipy.sparse.coo` namespace is deprecated.
  from scipy.sparse.coo import coo_matrix
/opt/.venv/lib/python3.8/site-packages/trimesh/curvature.py:12: DeprecationWarning: Please use `coo_matrix` from the `scipy.sparse` namespace, the `scipy.sparse.coo` namespace is deprecated.
  from scipy.sparse.coo import coo_matrix
/opt/.venv/lib/python3.8/site-packages/trimesh/curvature.py:12: DeprecationWarning: Please use `coo_matrix` from the `scipy.sparse` namespace, the `scipy.sparse.coo` namespace is deprecated.
  from scipy.sparse.coo import coo_matrix
/opt/.venv/lib/python3.8/site-packages/trimesh/curvature.py:12: DeprecationWarning: Please use `coo_matrix` from the `scipy.sparse` namespace, the `scipy.sparse.coo` namespace is deprecated.
  from scipy.sparse.coo import coo_matrix
/opt/.venv/lib/python3.8/site-packages/trimesh/curvature.py:12: DeprecationWarning: Please use `coo_matrix` from the `scipy.sparse` namespace, the `scipy.sparse.coo` namespace is deprecated.
  from scipy.sparse.coo import coo_matrix
Process SpawnProcess-130:
Traceback (most recent call last):
  File "/usr/lib/python3.8/multiprocessing/process.py", line 315, in _bootstrap
    self.run()
  File "/usr/lib/python3.8/multiprocessing/process.py", line 108, in run
    self._target(*self._args, **self._kwargs)
  File "/tmp/codalab/tmpiG5gNF/run/program/evaluate.py", line 349, in _worker
    observations = env.reset()
  File "/tmp/codalab/tmpiG5gNF/run/input/res/wrappers.py", line 442, in reset
    raw_obs = self.env.reset()
  File "/tmp/codalab/tmpiG5gNF/run/input/res/wrappers.py", line 69, in reset
    return super().reset()
  File "/opt/.venv/lib/python3.8/site-packages/gym/core.py", line 279, in reset
    return self.observation(observation)
  File "/tmp/codalab/tmpiG5gNF/run/input/res/wrappers.py", line 262, in observation
    EnvInfo_is_goal = self.cal_goal_lane(np_obs, raw_obs, lane_index, masked_all_lane_indeces).reshape(3, 3)
  File "/tmp/codalab/tmpiG5gNF/run/input/res/wrappers.py", line 92, in cal_goal_lane
    b = a[3]
IndexError: index 3 is out of bounds for axis 0 with size 1
ERROR:/tmp/codalab/tmpiG5gNF/run/program/evaluate.py:Evaluation failed due to error. Attempting retry: evaluate.<locals>.ProcessContext(process_builder=functools.partial(<function evaluate.<locals>.process_builder_func at 0x7f1168b7e8b0>, env_ctor=functools.partial(<function _make_env at 0x7f1319b38a60>, env_type='bubble_env_contrib:bubble_env-v0', scenario=None, shared_configs={'action_space': 'TargetPose', 'img_meters': 64, 'img_pixels': 256, 'sumo_headless': True}, seed=50, wrapper_ctors=<function submitted_wrappers at 0x7f1317b7c430>), policy_type=<class 'policy.Policy'>), env_name='bubble_env_6', retries=2, env_retries=10, last_reply=5642036.587259179)
/opt/.venv/lib/python3.8/site-packages/trimesh/curvature.py:12: DeprecationWarning: Please use `coo_matrix` from the `scipy.sparse` namespace, the `scipy.sparse.coo` namespace is deprecated.
  from scipy.sparse.coo import coo_matrix
/opt/.venv/lib/python3.8/site-packages/trimesh/curvature.py:12: DeprecationWarning: Please use `coo_matrix` from the `scipy.sparse` namespace, the `scipy.sparse.coo` namespace is deprecated.
  from scipy.sparse.coo import coo_matrix
/opt/.venv/lib/python3.8/site-packages/trimesh/curvature.py:12: DeprecationWarning: Please use `coo_matrix` from the `scipy.sparse` namespace, the `scipy.sparse.coo` namespace is deprecated.
  from scipy.sparse.coo import coo_matrix
/opt/.venv/lib/python3.8/site-packages/trimesh/curvature.py:12: DeprecationWarning: Please use `coo_matrix` from the `scipy.sparse` namespace, the `scipy.sparse.coo` namespace is deprecated.
  from scipy.sparse.coo import coo_matrix
/opt/.venv/lib/python3.8/site-packages/trimesh/curvature.py:12: DeprecationWarning: Please use `coo_matrix` from the `scipy.sparse` namespace, the `scipy.sparse.coo` namespace is deprecated.
  from scipy.sparse.coo import coo_matrix
/opt/.venv/lib/python3.8/site-packages/trimesh/curvature.py:12: DeprecationWarning: Please use `coo_matrix` from the `scipy.sparse` namespace, the `scipy.sparse.coo` namespace is deprecated.
  from scipy.sparse.coo import coo_matrix
/opt/.venv/lib/python3.8/site-packages/trimesh/curvature.py:12: DeprecationWarning: Please use `coo_matrix` from the `scipy.sparse` namespace, the `scipy.sparse.coo` namespace is deprecated.
  from scipy.sparse.coo import coo_matrix
/opt/.venv/lib/python3.8/site-packages/trimesh/curvature.py:12: DeprecationWarning: Please use `coo_matrix` from the `scipy.sparse` namespace, the `scipy.sparse.coo` namespace is deprecated.
  from scipy.sparse.coo import coo_matrix
Process SpawnProcess-132:
Traceback (most recent call last):
  File "/usr/lib/python3.8/multiprocessing/process.py", line 315, in _bootstrap
    self.run()
  File "/usr/lib/python3.8/multiprocessing/process.py", line 108, in run
    self._target(*self._args, **self._kwargs)
  File "/tmp/codalab/tmpiG5gNF/run/program/evaluate.py", line 354, in _worker
    observations, rewards, dones, infos = env.step(actions)
  File "/tmp/codalab/tmpiG5gNF/run/input/res/wrappers.py", line 447, in step
    else: wrapped_act = self.pack_action(raw_act)
  File "/tmp/codalab/tmpiG5gNF/run/input/res/wrappers.py", line 563, in pack_action
    st = np.where(all_lane_indeces == self.env.target_lane_index[agent_id])[0][0].item()
IndexError: index 0 is out of bounds for axis 0 with size 0
Exception ignored in: <function SMARTS.__del__ at 0x7fac6223daf0>
Traceback (most recent call last):
  File "/opt/.venv/lib/python3.8/site-packages/smarts/core/smarts.py", line 856, in __del__
TypeError: 'NoneType' object is not callable
ERROR:/tmp/codalab/tmpiG5gNF/run/program/evaluate.py:Evaluation failed due to error. Attempting retry: evaluate.<locals>.ProcessContext(process_builder=functools.partial(<function evaluate.<locals>.process_builder_func at 0x7f1168b7af70>, env_ctor=functools.partial(<function _make_env at 0x7f1319b38a60>, env_type='smarts.env:multi-scenario-v0', scenario='/tmp/codalab/tmpiG5gNF/run/input/ref/eval_scenarios/merge/single_agent/mid_volume/3lane_curve_agents_1', shared_configs={'action_space': 'TargetPose', 'img_meters': 64, 'img_pixels': 256, 'sumo_headless': True}, seed=44, wrapper_ctors=<function submitted_wrappers at 0x7f1317b7c430>), policy_type=<class 'policy.Policy'>), env_name='/tmp/codalab/tmpiG5gNF/run/input/ref/eval_scenarios/merge/single_agent/mid_volume/3lane_curve_agents_1', retries=2, env_retries=2, last_reply=5642219.904706328)
/opt/.venv/lib/python3.8/site-packages/trimesh/curvature.py:12: DeprecationWarning: Please use `coo_matrix` from the `scipy.sparse` namespace, the `scipy.sparse.coo` namespace is deprecated.
  from scipy.sparse.coo import coo_matrix
Process SpawnProcess-131:
Traceback (most recent call last):
  File "/usr/lib/python3.8/multiprocessing/process.py", line 315, in _bootstrap
    self.run()
  File "/usr/lib/python3.8/multiprocessing/process.py", line 108, in run
    self._target(*self._args, **self._kwargs)
  File "/tmp/codalab/tmpiG5gNF/run/program/evaluate.py", line 354, in _worker
    observations, rewards, dones, infos = env.step(actions)
  File "/tmp/codalab/tmpiG5gNF/run/input/res/wrappers.py", line 447, in step
    else: wrapped_act = self.pack_action(raw_act)
  File "/tmp/codalab/tmpiG5gNF/run/input/res/wrappers.py", line 563, in pack_action
    st = np.where(all_lane_indeces == self.env.target_lane_index[agent_id])[0][0].item()
IndexError: index 0 is out of bounds for axis 0 with size 0
Exception ignored in: <function SMARTS.__del__ at 0x7f2267864af0>
Traceback (most recent call last):
  File "/opt/.venv/lib/python3.8/site-packages/smarts/core/smarts.py", line 856, in __del__
TypeError: 'NoneType' object is not callable
ERROR:/tmp/codalab/tmpiG5gNF/run/program/evaluate.py:Evaluation failed due to error. Attempting retry: evaluate.<locals>.ProcessContext(process_builder=functools.partial(<function evaluate.<locals>.process_builder_func at 0x7f1168b7b3a0>, env_ctor=functools.partial(<function _make_env at 0x7f1319b38a60>, env_type='smarts.env:multi-scenario-v0', scenario='/tmp/codalab/tmpiG5gNF/run/input/ref/eval_scenarios/merge/single_agent/low_volume/3lane_curve_agents_1', shared_configs={'action_space': 'TargetPose', 'img_meters': 64, 'img_pixels': 256, 'sumo_headless': True}, seed=44, wrapper_ctors=<function submitted_wrappers at 0x7f1317b7c430>), policy_type=<class 'policy.Policy'>), env_name='/tmp/codalab/tmpiG5gNF/run/input/ref/eval_scenarios/merge/single_agent/low_volume/3lane_curve_agents_1', retries=2, env_retries=2, last_reply=5642930.814647401)
/opt/.venv/lib/python3.8/site-packages/trimesh/curvature.py:12: DeprecationWarning: Please use `coo_matrix` from the `scipy.sparse` namespace, the `scipy.sparse.coo` namespace is deprecated.
  from scipy.sparse.coo import coo_matrix
/opt/.venv/lib/python3.8/site-packages/trimesh/curvature.py:12: DeprecationWarning: Please use `coo_matrix` from the `scipy.sparse` namespace, the `scipy.sparse.coo` namespace is deprecated.
  from scipy.sparse.coo import coo_matrix
ERROR:/tmp/codalab/tmpiG5gNF/run/program/evaluate.py:Evaluation expired due to no response in 480s: evaluate.<locals>.ProcessContext(process_builder=functools.partial(<function evaluate.<locals>.process_builder_func at 0x7f1168b7a790>, env_ctor=functools.partial(<function _make_env at 0x7f1319b38a60>, env_type='smarts.env:multi-scenario-v0', scenario='/tmp/codalab/tmpiG5gNF/run/input/ref/eval_scenarios/straight/single_agent/overtake/low_volume/2lane_slow_agents_1', shared_configs={'action_space': 'TargetPose', 'img_meters': 64, 'img_pixels': 256, 'sumo_headless': True}, seed=44, wrapper_ctors=<function submitted_wrappers at 0x7f1317b7c430>), policy_type=<class 'policy.Policy'>), env_name='/tmp/codalab/tmpiG5gNF/run/input/ref/eval_scenarios/straight/single_agent/overtake/low_volume/2lane_slow_agents_1', retries=2, env_retries=2, last_reply=5643180.238062117)
/opt/.venv/lib/python3.8/site-packages/trimesh/curvature.py:12: DeprecationWarning: Please use `coo_matrix` from the `scipy.sparse` namespace, the `scipy.sparse.coo` namespace is deprecated.
  from scipy.sparse.coo import coo_matrix
/opt/.venv/lib/python3.8/site-packages/trimesh/curvature.py:12: DeprecationWarning: Please use `coo_matrix` from the `scipy.sparse` namespace, the `scipy.sparse.coo` namespace is deprecated.
  from scipy.sparse.coo import coo_matrix
/opt/.venv/lib/python3.8/site-packages/trimesh/curvature.py:12: DeprecationWarning: Please use `coo_matrix` from the `scipy.sparse` namespace, the `scipy.sparse.coo` namespace is deprecated.
  from scipy.sparse.coo import coo_matrix
/opt/.venv/lib/python3.8/site-packages/trimesh/curvature.py:12: DeprecationWarning: Please use `coo_matrix` from the `scipy.sparse` namespace, the `scipy.sparse.coo` namespace is deprecated.
  from scipy.sparse.coo import coo_matrix
ERROR:/tmp/codalab/tmpiG5gNF/run/program/evaluate.py:Evaluation expired due to no response in 480s: evaluate.<locals>.ProcessContext(process_builder=functools.partial(<function evaluate.<locals>.process_builder_func at 0x7f1168b7a670>, env_ctor=functools.partial(<function _make_env at 0x7f1319b38a60>, env_type='smarts.env:multi-scenario-v0', scenario='/tmp/codalab/tmpiG5gNF/run/input/ref/eval_scenarios/straight/single_agent/overtake/low_volume/3lane_curve_agents_1', shared_configs={'action_space': 'TargetPose', 'img_meters': 64, 'img_pixels': 256, 'sumo_headless': True}, seed=44, wrapper_ctors=<function submitted_wrappers at 0x7f1317b7c430>), policy_type=<class 'policy.Policy'>), env_name='/tmp/codalab/tmpiG5gNF/run/input/ref/eval_scenarios/straight/single_agent/overtake/low_volume/3lane_curve_agents_1', retries=2, env_retries=2, last_reply=5643891.281726914)
/opt/.venv/lib/python3.8/site-packages/trimesh/curvature.py:12: DeprecationWarning: Please use `coo_matrix` from the `scipy.sparse` namespace, the `scipy.sparse.coo` namespace is deprecated.
  from scipy.sparse.coo import coo_matrix
/opt/.venv/lib/python3.8/site-packages/trimesh/curvature.py:12: DeprecationWarning: Please use `coo_matrix` from the `scipy.sparse` namespace, the `scipy.sparse.coo` namespace is deprecated.
  from scipy.sparse.coo import coo_matrix
Process SpawnProcess-140:
Traceback (most recent call last):
  File "/usr/lib/python3.8/multiprocessing/process.py", line 315, in _bootstrap
    self.run()
  File "/usr/lib/python3.8/multiprocessing/process.py", line 108, in run
    self._target(*self._args, **self._kwargs)
  File "/tmp/codalab/tmpiG5gNF/run/program/evaluate.py", line 354, in _worker
    observations, rewards, dones, infos = env.step(actions)
  File "/tmp/codalab/tmpiG5gNF/run/input/res/wrappers.py", line 447, in step
    else: wrapped_act = self.pack_action(raw_act)
  File "/tmp/codalab/tmpiG5gNF/run/input/res/wrappers.py", line 481, in pack_action
    last_correct_wp_pos = self.env.last_correct_wp_pos[agent_id]
KeyError: 'Agent_0'
Exception ignored in: <function SMARTS.__del__ at 0x7f462eda3b80>
Traceback (most recent call last):
  File "/opt/.venv/lib/python3.8/site-packages/smarts/core/smarts.py", line 856, in __del__
TypeError: 'NoneType' object is not callable
ERROR:/tmp/codalab/tmpiG5gNF/run/program/evaluate.py:Evaluation failed due to error. Attempting retry: evaluate.<locals>.ProcessContext(process_builder=functools.partial(<function evaluate.<locals>.process_builder_func at 0x7f1168b70820>, env_ctor=functools.partial(<function _make_env at 0x7f1319b38a60>, env_type='smarts.env:multi-scenario-v0', scenario='/tmp/codalab/tmpiG5gNF/run/input/ref/eval_scenarios/naturalistic/rt2-agents_1', shared_configs={'action_space': 'TargetPose', 'img_meters': 64, 'img_pixels': 256, 'sumo_headless': True}, seed=44, wrapper_ctors=<function submitted_wrappers at 0x7f1317b7c430>), policy_type=<class 'policy.Policy'>), env_name='/tmp/codalab/tmpiG5gNF/run/input/ref/eval_scenarios/naturalistic/rt2-agents_1', retries=2, env_retries=2, last_reply=5643912.893876047)
/opt/.venv/lib/python3.8/site-packages/trimesh/curvature.py:12: DeprecationWarning: Please use `coo_matrix` from the `scipy.sparse` namespace, the `scipy.sparse.coo` namespace is deprecated.
  from scipy.sparse.coo import coo_matrix
Process SpawnProcess-141:
Traceback (most recent call last):
  File "/usr/lib/python3.8/multiprocessing/process.py", line 315, in _bootstrap
    self.run()
  File "/usr/lib/python3.8/multiprocessing/process.py", line 108, in run
    self._target(*self._args, **self._kwargs)
  File "/tmp/codalab/tmpiG5gNF/run/program/evaluate.py", line 349, in _worker
    observations = env.reset()
  File "/tmp/codalab/tmpiG5gNF/run/input/res/wrappers.py", line 442, in reset
    raw_obs = self.env.reset()
  File "/tmp/codalab/tmpiG5gNF/run/input/res/wrappers.py", line 69, in reset
    return super().reset()
  File "/opt/.venv/lib/python3.8/site-packages/gym/core.py", line 279, in reset
    return self.observation(observation)
  File "/tmp/codalab/tmpiG5gNF/run/input/res/wrappers.py", line 262, in observation
    EnvInfo_is_goal = self.cal_goal_lane(np_obs, raw_obs, lane_index, masked_all_lane_indeces).reshape(3, 3)
  File "/tmp/codalab/tmpiG5gNF/run/input/res/wrappers.py", line 92, in cal_goal_lane
    b = a[3]
IndexError: index 3 is out of bounds for axis 0 with size 1
ERROR:/tmp/codalab/tmpiG5gNF/run/program/evaluate.py:Evaluation failed due to error. Attempting retry: evaluate.<locals>.ProcessContext(process_builder=functools.partial(<function evaluate.<locals>.process_builder_func at 0x7f1168b7e8b0>, env_ctor=functools.partial(<function _make_env at 0x7f1319b38a60>, env_type='bubble_env_contrib:bubble_env-v0', scenario=None, shared_configs={'action_space': 'TargetPose', 'img_meters': 64, 'img_pixels': 256, 'sumo_headless': True}, seed=51, wrapper_ctors=<function submitted_wrappers at 0x7f1317b7c430>), policy_type=<class 'policy.Policy'>), env_name='bubble_env_6', retries=3, env_retries=10, last_reply=5643921.948441664)
ERROR:/tmp/codalab/tmpiG5gNF/run/program/evaluate.py:Scoring skipped for evaluation because retries expended: evaluate.<locals>.ProcessContext(process_builder=functools.partial(<function evaluate.<locals>.process_builder_func at 0x7f1168b7af70>, env_ctor=functools.partial(<function _make_env at 0x7f1319b38a60>, env_type='smarts.env:multi-scenario-v0', scenario='/tmp/codalab/tmpiG5gNF/run/input/ref/eval_scenarios/merge/single_agent/mid_volume/3lane_curve_agents_1', shared_configs={'action_space': 'TargetPose', 'img_meters': 64, 'img_pixels': 256, 'sumo_headless': True}, seed=44, wrapper_ctors=<function submitted_wrappers at 0x7f1317b7c430>), policy_type=<class 'policy.Policy'>), env_name='/tmp/codalab/tmpiG5gNF/run/input/ref/eval_scenarios/merge/single_agent/mid_volume/3lane_curve_agents_1', retries=2, env_retries=2, last_reply=5642219.904706328)
ERROR:root:Scoring skipped due to errors!