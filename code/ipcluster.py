import os
import subprocess
import time
from typing import List

import ipyparallel as ipp


class IPCluster:
    def __init__(
        self, profile="default", n=1, init=False, ip=None, location=None, engines=None
    ):
        command = f"ipcluster start --profile={profile} -n {n} --daemonize=True"

        if init is True:
            command += " --init"
        if ip is not None:
            command += f" --ip={ip}"
        if location is not None:
            command += f" --location={location}"
        if engines is not None:
            command += f" --engines={engines}"

        self._start_commmand = command
        self._n = n
        self._profile = profile
        self._started = False
        self._connected = False

    def start(self):
        try:
            subprocess.run(
                self._start_commmand, shell=True, check=True, capture_output=True
            )
            self._started = True
        except subprocess.CalledProcessError as err:
            print("command::\t", err.cmd)
            print("stdout:\t", err.stdout)
            print("stderr:\t", err.stderr)

    def _hostname_and_cuda_visible_devices(self):
        dview = self.rc[:]
        dview.block = True

        command = """
        import os
        
        CUDA_VISIBLE_DEVICES = os.getenv("CUDA_VISIBLE_DEVICES")
        
        if CUDA_VISIBLE_DEVICES:
            cuda_devices = list(int(i) for i in os.getenv("CUDA_VISIBLE_DEVICES",).split(","))
        else:
            cuda_devices = []
        
        hostname = os.uname()[1]
        """
        dview.execute(command)

        res = dview.pull(("cuda_devices", "hostname"))

        self.hostnames = tuple([el[1] for el in res])
        self.cuda_visible_devices = tuple([tuple(el[0]) for el in res])

    def connect(self, max_waiting_time=300):
        if self._started is False:
            raise RuntimeError("start the ipcluster")
        rc = ipp.Client(profile=self._profile)

        start_time = time.time()

        while len(rc.ids) < self._n:
            time.sleep(1)
            if time.time() - start_time > max_waiting_time:
                raise RuntimeError(
                    f"The engines were not ready after {max_waiting_time} seconds."
                )
        self.rc = rc
        self._connected = True

        self._hostname_and_cuda_visible_devices()

    def stop(self):
        if self._connected:
            self.rc.shutdown(hub=True)
        else:
            command = f"ipcluster stop --profile={self._profile}"

            try:
                subprocess.run(command, shell=True, check=True, capture_output=True)
            except subprocess.CalledProcessError as err:
                print("command::\t", err.cmd)
                print("stdout:\t", err.stdout)
                print("stderr:\t", err.stderr)
                raise err
        self._started = False
        self._connected = False

    def assign_cuda_device(self, use_gpu: bool = True) -> List[int]:
        """
        At most one device is assigned per engine (worker).
        
        """
        if use_gpu is False:
            cuda_devices = [-1] * len(self.rc.ids)
        else:
            cuda_devices = []
            cuda_devices_host = {}

            for i, el in enumerate(self.hostnames):
                if el not in cuda_devices_host:
                    cuda_devices_host[el] = list(self.cuda_visible_devices[i])
                cuda_device_id = (
                    cuda_devices_host[el].pop(0) if cuda_devices_host[el] else -1
                )
                cuda_devices.append(cuda_device_id)

        self.cuda_devices = tuple(cuda_devices)

        return self.cuda_devices


class LeonhardIPCluster(IPCluster):
    def __init__(self):
        profile = "LSB" + os.getenv("LSB_BATCH_JID")
        n = int(os.getenv("LSB_MAX_NUM_PROCESSORS"))
        init = True
        ip = '"*"'
        location = "$(hostname)"
        engines = "MPI"
        super().__init__(
            profile=profile, n=n, init=init, ip=ip, location=location, engines=engines,
        )
