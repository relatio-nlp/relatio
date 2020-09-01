from binascii import Error
import os
import subprocess
import time

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
            res = subprocess.run(
                self._start_commmand, shell=True, check=True, capture_output=True
            )
            self._started = True
        except subprocess.CalledProcessError as err:
            print(res.stdout)
            print(res.stderr)
            raise err

    def connect(self, max_waiting_time=300):
        if self._started is False:
            raise Error("start the ipcluster")
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

    def stop(self):
        if self._connected:
            self.rc.shutdown(hub=True)
        else:
            command = f"ipcluster stop --profile={self._profile}"

        try:
            res = subprocess.run(command, shell=True, check=True, capture_output=True)
        except subprocess.CalledProcessError as err:
            print(res.stdout)
            print(res.stderr)
            raise err

    def assign_cuda_device_ids(self, no_gpu=False):
        """
        At most one device is assigned per engine (worker).
        
        """
        if no_gpu is True:
            return [-1] * len(self.rc.ids)

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

        cuda_device_ids = []
        cuda_devices_host = {}

        for el in res:
            if not el[1] in cuda_devices_host:
                cuda_devices_host[el[1]] = el[0]
            cuda_device_id = (
                cuda_devices_host[el[1]].pop(0) if cuda_devices_host[el[1]] else -1
            )
            cuda_device_ids.append(cuda_device_id)

        return cuda_device_ids


class LeonhardIPCluster(IPCluster):
    def __init__(
        self,
        profile="LSB" + os.getenv("LSB_BATCH_JID"),
        n=int(os.getenv("LSB_MAX_NUM_PROCESSORS")),
        init=True,
        ip='"*"',
        location="$(hostname)",
        engines="MPI",
    ):
        super().__init__(
            profile=profile, n=n, init=init, ip=ip, location=location, engines=engines,
        )
