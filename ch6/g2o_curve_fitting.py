# import numpy as np
# import g2o
# import matplotlib.pyplot as plt
# import time
#
#
# class BundleAdjustment(g2o.SparseOptimizer):
#     def __init__(self, ):
#         super().__init__()
#         solver = g2o.BlockSolverSE3(g2o.LinearSolverCSparseSE3())
#         solver = g2o.OptimizationAlgorithmLevenberg(solver)
#         super().set_algorithm(solver)
#
#     def optimize(self, max_iterations=10):
#         super().initialize_optimization()
#         super().optimize(max_iterations)
#
#     def add_pose(self, pose_id, pose, cam, fixed=False):
#         sbacam = g2o.SBACam(pose.orientation(), pose.position())
#         sbacam.set_cam(cam.fx, cam.fy, cam.cx, cam.cy, cam.baseline)
#
#         v_se3 = g2o.VertexCam()
#         v_se3.set_id(pose_id * 2)   # internal id
#         v_se3.set_estimate(sbacam)
#         v_se3.set_fixed(fixed)
#         super().add_vertex(v_se3)
#
#     def add_point(self, point_id, point, fixed=False, marginalized=True):
#         v_p = g2o.VertexSBAPointXYZ()
#         v_p.set_id(point_id * 2 + 1)
#         v_p.set_estimate(point)
#         v_p.set_marginalized(marginalized)
#         v_p.set_fixed(fixed)
#         super().add_vertex(v_p)
#
#     def add_edge(self, point_id, pose_id,
#             measurement,
#             information=np.identity(2),
#             robust_kernel=g2o.RobustKernelHuber(np.sqrt(5.991))):   # 95% CI
#
#         edge = g2o.EdgeProjectP2MC()
#         edge.set_vertex(0, self.vertex(point_id * 2 + 1))
#         edge.set_vertex(1, self.vertex(pose_id * 2))
#         edge.set_measurement(measurement)   # projection
#         edge.set_information(information)
#
#         if robust_kernel is not None:
#             edge.set_robust_kernel(robust_kernel)
#         super().add_edge(edge)
#
#     def get_pose(self, pose_id):
#         return self.vertex(pose_id * 2).estimate()
#
#     def get_point(self, point_id):
#         return self.vertex(point_id * 2 + 1).estimate()
#
#
# a = 1
# b = 2
# c = 1
#
# w_sigma = 1
#
# x_data = np.array(list(range(100))) / 100
# y_data = np.exp(a * x_data ** 2 + b * x_data + c) +\
#          np.random.normal(scale=w_sigma, size=x_data.shape)
#
# tic = time.time()
#
# # solver = BundleAdjustment()
#
# sparse_optimizer = g2o.SparseOptimizer()
#
# solver = g2o.BlockSolverSE3(g2o.LinearSolverCSparseSE3())
# solver = g2o.OptimizationAlgorithmLevenberg(solver)
# sparse_optimizer.set_algorithm(solver)
#
# point = g2o.VertexSBAPointXYZ()
# point.set_id(0)
# point.set_estimate(np.array([[0], [0], [0]]))
# sparse_optimizer.add_vertex(point)
#
# i = 0
# for x, y in zip(x_data, y_data):
#     edge = g2o.BaseEdge_1_double()
#     edge.set_vertex(0, self.vertex(point_id * 2 + 1))
#     edge.set_vertex(1, self.vertex(pose_id * 2))
#     edge.set_measurement(measurement)  # projection
#     edge.set_information(information)
#
#     if robust_kernel is not None:
#         edge.set_robust_kernel(robust_kernel)
#     super().add_edge(edge)
# solver.optimize(100)
# plsq = solver.get_point(0)
# print('用时：', time.time() - tic, 's')
# print('拟合结果：', plsq)
# plt.ion()
# plt.scatter(x_data, y_data, c='k')
# plt.plot(x_data, np.exp(plsq[0] * x_data ** 2 + plsq[1] * x_data + plsq[2]), c='r')
# plt.ioff()
# plt.waitforbuttonpress()
#
