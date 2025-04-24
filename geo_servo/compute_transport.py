import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import scienceplots
import ott
from ott.geometry import costs, pointcloud
from ott import utils
from ott.problems.linear import linear_problem
from ott.solvers.linear import sinkhorn
from ott.tools import plot
from tqdm import trange, tqdm

plt.style.use(['science', 'notebook'])

if __name__ == "__main__":
    
    target_pose = jnp.load('../notebooks/data/target_square_hole.npz')
    initial_pose = jnp.load('../notebooks/data/initial_pose_1.npz')
    
    x = jnp.arange(0, 480, 1)
    y = jnp.arange(0, 640, 1)
    X, Y = jnp.meshgrid(x, y)
    
    target_cloud = 0.01*target_pose['depth_image']
    target_cloud[target_cloud>100] = 0.0
    target_pose_se3 = target_pose['pose']

    # Initial pose
    initial_cloud = 0.01*initial_pose['depth_image']
    initial_pose_se3 = initial_pose['pose']
    
    g_0 = jnp.vstack([X.ravel(), Y.ravel(), initial_cloud.T.ravel()]).T
    g_1 = jnp.vstack([X.ravel(), Y.ravel(), target_cloud.T.ravel()]).T
    geom = pointcloud.PointCloud(g_0[0:30000,:], g_1[0:30000,:], epsilon=1e-3)
    
    tau = 0.999
    ot_prob = linear_problem.LinearProblem(geom, tau_a=tau, tau_b=tau)
    with tqdm() as pbar:
        progress_fn = utils.tqdm_progress_fn(pbar)
        solver = sinkhorn.Sinkhorn(progress_fn=progress_fn)
        ot = jax.jit(solver)(ot_prob)
        
    print(
        " Sinkhorn has converged: ",
        ot.converged,
        "\n",
        "Error upon last iteration: ",
        ot.errors[(ot.errors > -1)][-1],
        "\n",
        "Sinkhorn required ",
        jnp.sum(ot.errors > -1),
        " iterations to converge. \n",
        "Entropy regularized OT cost: ",
        ot.ent_reg_cost,
        "\n",
        "OT cost (without entropy): ",
        jnp.sum(ot.matrix * ot.geom.cost_matrix),
    )