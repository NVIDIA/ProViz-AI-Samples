# SPDX-FileCopyrightText: Copyright (c) 2021 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: MIT
#
# Permission is hereby granted, free of charge, to any person obtaining a
# copy of this software and associated documentation files (the "Software"),
# to deal in the Software without restriction, including without limitation
# the rights to use, copy, modify, merge, publish, distribute, sublicense,
# and/or sell copies of the Software, and to permit persons to whom the
# Software is furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL
# THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
# FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
# DEALINGS IN THE SOFTWARE.

import sys

import onnx
import onnx_graphsurgeon as gs

# Load ONNX model and create IR
graph = gs.import_onnx(onnx.load(str(sys.argv[1])))

# Find all BatchNormalization nodes
bn_nodes = [node for node in graph.nodes if node.op == "BatchNormalization"]

# Disconnect BatchNormalization nodes and insert MyPlugin nodes
for node in bn_nodes:
    inputs = node.inputs[:1]
    outputs = node.outputs[:]

    node.inputs[0].outputs.remove(node)
    node.outputs[0].inputs.remove(node)

    graph.layer(op="MyPlugin", inputs=inputs, outputs=outputs, attrs={"num_inputs": 1})

# Remove dangling BatchNormalization nodes
graph.cleanup()

# Save modified ONNX to disk
onnx.save(gs.export_onnx(graph), str(sys.argv[2]))
