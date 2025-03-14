# BSD 3- Clause License Copyright (c) 2023, Tecorigin Co., Ltd. All rights
# reserved.
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
# Redistributions of source code must retain the above copyright notice,
# this list of conditions and the following disclaimer.
# Redistributions in binary form must reproduce the above copyright notice,
# this list of conditions and the following disclaimer in the documentation
# and/or other materials provided with the distribution.
# Neither the name of the copyright holder nor the names of its contributors
# may be used to endorse or promote products derived from this software
# without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
# ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE
# LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
# CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
# SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
# INTERRUPTION)
# HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT,
# STRICT LIABILITY,OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)  ARISING IN ANY
# WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY
# OF SUCH DAMAGE.


import os, sys
import time

sys.path.append(os.getcwd())
log_dir = os.path.join(os.path.dirname(__file__), "logs")
if not os.path.exists(log_dir): os.mkdir(log_dir)

import subprocess
from rag.common.utils import settings
from rag.common.utils import logger


SUCCESS_FLAG = "Application startup complete"
ERROR_FLAG = "Fail to start application"


def main():
    ########################################################
    # 启动Api Server
    ########################################################
    cmd = ["python", "server/main.py", "--create_tables"]
    t1 = time.time()
    p_load_api_server = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    while p_load_api_server.poll() is None:
        line = p_load_api_server.stdout.readline().decode('utf-8')
        if SUCCESS_FLAG in line:
            logger.info(f"Starting Api Server cost {time.time() - t1} seconds")
            break
        elif ERROR_FLAG in line:
            raise RuntimeError("Fail to start Api Server")

    # ########################################################
    # # 启动WebUI
    # ########################################################
    cmd = ["streamlit", "run",
           "server/web_app.py",
           "--server.port", str(settings.server.web_server_port)]

    subprocess.run(cmd)


if __name__ == "__main__":

    main()





