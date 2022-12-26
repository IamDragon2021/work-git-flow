
import torch
import torchvision
import pdb
import time
from tqdm import tqdm
import numpy as np

from openvino.runtime import Core,Layout,PartialShape,Type
import numpy as np
import onnxruntime as rt
import time


def convert_resnet18_torchscript():
    """
    将 resnet18 转为 TorchScript 模型格式
    """
    # An instance of your model.
    model = torchvision.models.resnet18(pretrained=False)

    # Switch the model to eval model
    model.eval()

    # An example input you would normally provide to your model's forward() method.
    example = torch.rand(1, 3, 224, 224)
    input = torch.rand(32, 3, 224, 224)
    e = time.time()
    output = model(input)
    s = time.time()

    print(s-e)
    # Use torch.jit.trace to generate a torch.jit.ScriptModule via tracing.
    trace_model = torch.jit.trace(model, example) # torch.jit.ScriptModule

    raw_output = model(example)
    trace_model_output = trace_model(example)
    np.testing.assert_allclose(raw_output.detach().numpy(), trace_model_output.detach().numpy())
    # Save the TorchScript model
    trace_model.save("resnet18_traced_model.pt")


    # Use torch.jit.trace to generate a torch.jit.ScriptModule via script.
    script_model = torch.jit.script(model)
    script_model_output = script_model(example)
    np.testing.assert_allclose(raw_output.detach().numpy(), script_model_output.detach().numpy())
    # Save the TorchScript model
    script_model.save("resnet18_script_model.pt")

convert_resnet18_torchscript()
import torch

MODEL_ONNX_PATH = "resnet18.onnx"
OPERATOR_EXPORT_TYPE = torch._C._onnx.OperatorExportTypes.ONNX
model = torchvision.models.resnet18(pretrained=False)
model.eval()
org_dummy_input = torch.rand(1, 3, 224, 224)
torch.onnx.export(model,
                org_dummy_input,
                MODEL_ONNX_PATH,
                verbose=True,
                operator_export_type=OPERATOR_EXPORT_TYPE,
                opset_version=12,
                input_names=['inputs'],
                output_names=['outputs'],
                do_constant_folding=True,
                dynamic_axes={"inputs": {0: "batch_size"}, "outputs": {0: "batch_size"}}
                )



x = torch.randn(32, 3, 224, 224).numpy()

sess = rt.InferenceSession(MODEL_ONNX_PATH, None)
input_name = sess.get_inputs()[0].name
out_name = sess.get_outputs()[0].name
s = time.time()
pred_onx = sess.run([out_name], {input_name: x})
e = time.time()
print('onnx_runtime推理时间：',e-s)



ie = Core()
device = ie.available_devices

model = ie.read_model(MODEL_ONNX_PATH)

input_layer=model.input(0)
output_layer = model.output(0)
compiled_model = ie.compile_model(model,'CPU')

ss = time.time()
result = compiled_model.infer_new_request({0:x})
ee = time.time()
print('onnx推理时间：',ee-ss)

# IR 模型
ie = Core()
classification_model_xml = "exported_onnx_model.xml"
model = ie.read_model(model=classification_model_xml)
compiled_model = ie.compile_model(model=model, device_name="CPU")
input_layer = next(iter(compiled_model.inputs))
output_layer = next(iter(compiled_model.outputs))

s= time.time()
result = compiled_model([x])[output_layer]
e = time.time()
print('IR 模型推理时间：',e-s)