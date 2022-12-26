import time

from openvino.offline_transformations import serialize
from openvino.runtime import Core
import torch
# ie = Core()
# onnx_model_path = "resnet18.onnx"
# model_onnx = ie.read_model(model=onnx_model_path)
# # compiled_model_onnx = ie.compile_model(model=model_onnx, device_name="CPU")
#
# serialize(model=model_onnx, xml_path="exported_onnx_model.xml", bin_path="exported_onnx_model.bin",version="IR_V11")



# ie = Core()
# classification_model_xml = "exported_onnx_model.xml"
# model = ie.read_model(model=classification_model_xml)
# model.inputs[0].any_name

x = torch.randn(32, 3, 224, 224).numpy()


from openvino.runtime import Core

ie = Core()
classification_model_xml = "exported_onnx_model.xml"
model = ie.read_model(model=classification_model_xml)
compiled_model = ie.compile_model(model=model, device_name="CPU")
input_layer = next(iter(compiled_model.inputs))
output_layer = next(iter(compiled_model.outputs))

s= time.time()
result = compiled_model([x])[output_layer]
e = time.time()
print(e-s)