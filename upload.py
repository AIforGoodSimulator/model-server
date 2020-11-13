from dotenv import load_dotenv
import redis
import socket
import os
from dask.distributed import Client

load_dotenv()

c = Client(os.environ.get("DASK_SCHEDULER_HOST")+":"+os.environ.get("DASK_SCHEDULER_PORT"))
c.upload_file('upload.txt')

def install():
    import os
    os.system("find . -name 'upload.txt' -exec sh -c 'unzip -o -d `dirname {}` {}' ';'")

c.run(install)
