# import os
# from dotenv import load_dotenv
# from dask.distributed import Client, LocalCluster
# from ai4good.utils.logger_util import get_logger
#
#
# logger = get_logger(__name__,'DEBUG')
# load_dotenv()
#
#
# # initialise dask distributed client
# def dask_client() -> Client:
#     global _client
#     # client can only have one thread due to scipy ode solver constraints
#     # https://docs.scipy.org/doc/scipy/reference/generated/scipy.integrate.ode.html
#     if _client is None:
#         if ("DASK_SCHEDULER_HOST" not in os.environ) :
#             logger.warn("No Dask Sceduler host specified in .env, Running Dask locally ...")
#             cluster = LocalCluster(n_workers=4, threads_per_worker=1)
#             _client = Client(cluster)
#         elif (os.environ.get("DASK_SCHEDULER_HOST")=="127.0.0.1") :
#             logger.info("Running Dask locally ...")
#             cluster = LocalCluster(n_workers=4, threads_per_worker=1)
#             _client = Client(cluster)
#         elif (os.environ.get("DASK_SCHEDULER_HOST")=='') :
#             logger.warn("No Dask Sceduler host specified in .env, Running Dask locally ...")
#             cluster = LocalCluster(n_workers=4, threads_per_worker=1)
#             _client = Client(cluster)
#         else :
#             logger.info("Running Dask Distributed using Dask Scheduler ["+os.environ.get("DASK_SCHEDULER_HOST")+"] ...")
#             _client = Client(os.environ.get("DASK_SCHEDULER_HOST")+":"+os.environ.get("DASK_SCHEDULER_PORT"))
#
#     return _client
#
# _client = None  # Needs lazy init
#
# client = dask_client()
#
#
# def square(x):
#     return x ** 2
#
#
# def neg(x):
#     return -x
#
#
# A = client.map(square, range(10))
# B = client.map(neg, A)
# total = client.submit(sum, B)
# result = total.result()
# assert result == -285
# outcome = client.gather(A)
# assert outcome == [0, 1, 4, 9, 16, 25, 36, 49, 64, 81]
#
#
