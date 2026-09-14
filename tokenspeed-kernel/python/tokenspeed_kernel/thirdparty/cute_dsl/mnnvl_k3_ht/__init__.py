# Copyright (c) 2026 by FlashInfer team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# Copyright (c) 2026 LightSeek Foundation
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

"""Native-H3584 specialization of FlashInfer's MNNVL HT protocol."""

from tokenspeed_kernel.thirdparty.cute_dsl.mnnvl_k3_ht.device_kernel import (
    K3H3584MoeFinalizeAllReduceRMSNormHTDeviceKernel,
)
from tokenspeed_kernel.thirdparty.cute_dsl.mnnvl_k3_ht.protocol import (
    K3_HT_ALL_REDUCE_GB300_TP8_H3584,
    K3_HT_FINALIZE_GB300_TP8_H3584_K16,
    K3H3584HTProtocol,
)

__all__ = [
    "K3H3584HTProtocol",
    "K3H3584MoeFinalizeAllReduceRMSNormHTDeviceKernel",
    "K3_HT_ALL_REDUCE_GB300_TP8_H3584",
    "K3_HT_FINALIZE_GB300_TP8_H3584_K16",
]
