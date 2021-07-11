from onnxruntime.transformers import optimizer
import torch

def convert_and_optimize(model,path,use_fp16):
    input_ids = torch.ones(1,32, dtype=torch.int64,requires_grad=False).cuda()
    attention_mask = torch.ones(1,32,dtype=torch.int64,requires_grad=False).cuda()
    input_names = ["input_ids", "attention_mask"]
    output_names = ["contextual"]

    onnx_dummy_input = {"input_ids":input_ids, "attention_mask":attention_mask}

    torch.onnx.export(model, onnx_dummy_input, path, 
                input_names = ["input_ids", "attention_mask"],
                output_names = output_names,
                dynamic_axes = {"input_ids":[0,1], "attention_mask":[0,1],"contextual":[0,1]},verbose=False,opset_version=11)
    del onnx_dummy_input

    #optimized_model = optimizer.optimize_model(path,use_gpu=True,opt_level=99, model_type='bert', num_heads=12, hidden_size=768)
    optimized_model = optimizer.optimize_model(path, model_type='bert', num_heads=12, hidden_size=768)
    if use_fp16:
        optimized_model.convert_model_float32_to_float16()
    optimized_model.save_model_to_file(path)
    #print("Fully optimized:",optimized_model.is_fully_optimized())
    del optimized_model

    # Load the ONNX model
    #model = onnx.load(os.path.join(run_folder,"test-out-optimized.onnx"))
    # Check that the IR is well formed
    #onnx.checker.check_model(model)
    # Print a human readable representation of the graph
    #onnx.helper.printable_graph(model.graph)
